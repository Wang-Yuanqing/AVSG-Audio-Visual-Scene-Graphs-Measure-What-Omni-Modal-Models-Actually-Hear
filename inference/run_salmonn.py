# Public-release adaptation: paths/imports only; campaign model and decoding logic retained.
"""Run video-SALMONN 2+ 7B on the AVSG benchmark (repo inference path, single GPU).

video-SALMONN 2+ (Qwen2.5-VL-7B fork + Whisper audio, tsinghua-ee/video-SALMONN2_plus_7B_full
— the full merged weights, NOT the LoRA-adapter repo) is an audio-visual captioning
specialist. Inference follows the repo's own video_SALMONN2_plus/inference.py recipe
exactly:
  * official eval settings from the README: max_frames 768, max_pixels 61250,
    base_interval 0.1, min_frames 16
  * the repo dataset's `_get_item` does ALL preprocessing (frame sampling + whisper
    features + A/V token interleaving + M-RoPE position_ids); run_test=True truncates
    input_ids right after `<|im_start|>assistant\n` — the canonical generation prompt
  * liger kernels patched in like inference.py does; attn_implementation=sdpa
    (no flash-attn wheels for EL8 glibc 2.28 — same as every model env here)
  * official generation is GREEDY (do_sample=False) per inference.py

Deviations from the demo, both deliberate:
  * the AVSG SYSTEM prompt is prepended to the user turn (their preprocessing hardcodes
    the system message "You are a helpful assistant."; content parity is what the shared
    prompt requires)
  * house cap max_new_tokens=4096 + the loop safety-net retry (repetition_penalty) on
    cap-hit outputs

Same interface as every other model: SHARED prompt.py / parse_output.py and the same
output files the evaluator consumes:
  * audit  : results/video-salmonn2plus-7b_<setting>_raw.jsonl       {key, text, ...}
  * scored : results/verified/video-salmonn2plus-7b_<setting>.jsonl  {filename, nodes, edges, valid}

Run via launch_salmonn.sh (env + caches + LD_LIBRARY_PATH for torchcodec), e.g.
  ./launch_salmonn.sh --clips-file ../minicpm/smoke_clips.txt --settings raw --out-dir smoke
  ./launch_salmonn.sh --order name
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)                      # .../avsg_eval
REPO = os.environ.get("SALMONN_REPO", os.path.join(EVAL_ROOT, "external", "video-SALMONN", "video_SALMONN2_plus"))
sys.path.insert(0, os.path.join(EVAL_ROOT, "evaluation"))                          # prompt.py, parse_output.py, vocab.py
sys.path.insert(0, REPO)                               # qwenvl.*

import prompt as P                                     # noqa: E402
from parse_output import parse                         # noqa: E402

MODEL_TAG = "video-salmonn2plus-7b"


def load_gold(gold_path):
    cap = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                cap[r["filename"]] = r.get("caption", "")
    return cap


def clip_duration(fn):
    try:
        a, b = fn[:-4].rsplit("_", 2)[-2:]
        return float(b) - float(a)
    except Exception:
        return float("inf")


def done_filenames(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["filename"])
                except Exception:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Local official checkpoint directory")
    ap.add_argument("--clips-dir", default=os.path.join(EVAL_ROOT, "clips"))
    ap.add_argument("--gold", default=os.path.join(EVAL_ROOT, "data", "gold", "1k_result.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(EVAL_ROOT, "outputs", "inference"))
    ap.add_argument("--settings", default="raw,caption")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--order", choices=["short", "long", "name"], default="name")
    ap.add_argument("--clips", default="", help="comma-separated filenames; overrides order/limit")
    ap.add_argument("--clips-file", default="", help="file with one filename per line (or commas)")
    # generation (official inference.py: GREEDY; house cap 4096 + loop retry)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--sample", action="store_true", help="do_sample=True (official is greedy)")
    ap.add_argument("--temperature", type=float, default=0.7, help="only with --sample")
    ap.add_argument("--loop-retry-rep", type=float, default=1.1,
                    help="on cap-hit output, retry once with this repetition_penalty; 0 = off")
    # multimodal (official README eval settings)
    ap.add_argument("--max-frames", type=int, default=768)
    ap.add_argument("--min-frames", type=int, default=16)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--max-pixels", type=int, default=61250)
    ap.add_argument("--no-liger", action="store_true", help="skip liger kernel patching (debug)")
    ap.add_argument("--no-audio", action="store_true", help="use_audio=False (debug)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    os.makedirs(os.path.join(args.out_dir, "verified"), exist_ok=True)
    use_audio = not args.no_audio

    cap = load_gold(args.gold)
    on_disk = {fn for fn in os.listdir(args.clips_dir) if fn.endswith(".mp4")}
    clips = [fn for fn in cap if fn in on_disk]
    want = [c.strip() for c in args.clips.split(",") if c.strip()]
    if args.clips_file:
        with open(args.clips_file, encoding="utf-8") as f:
            want += [c.strip() for line in f for c in line.replace(",", "\n").splitlines() if c.strip()]
    if want:
        clips = [fn for fn in want if fn in cap and fn in on_disk]
    else:
        if args.order == "name":
            clips.sort()
        else:
            clips.sort(key=clip_duration, reverse=(args.order == "long"))
        if args.limit:
            clips = clips[:args.limit]

    audit_path = {s: os.path.join(args.out_dir, f"{MODEL_TAG}_{s}_raw.jsonl") for s in settings}
    verif_path = {s: os.path.join(args.out_dir, "verified", f"{MODEL_TAG}_{s}.jsonl") for s in settings}
    done = {s: (set() if args.no_resume else done_filenames(verif_path[s])) for s in settings}
    for s in settings:
        print(f"[run] {s}: {len(done[s])} already done (resume)", flush=True)
    todo = [fn for fn in clips if any(fn not in done[s] for s in settings)]
    print(f"[run] {len(clips)} clips ({len(todo)} todo), settings={settings}, audio={use_audio}, "
          f"max_frames={args.max_frames}, max_pixels={args.max_pixels}, "
          f"greedy={not args.sample}, max_new={args.max_new_tokens}", flush=True)
    if not todo:
        print("[run] nothing to do.", flush=True)
        return

    # ---- load model (once), mirroring repo inference.py ----
    import torch
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    from qwenvl.train.argument import DataArguments

    if not args.no_liger:
        # inference.py's apply_liger_kernel_to_qwen2_5_vl(), verbatim effect
        from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
        from qwenvl.model import modeling_qwen2_5_vl
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
        print("[run] liger kernels patched (rope, rms_norm, swiglu)", flush=True)
    from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
    from qwenvl.data.dataset import make_supervised_data_module

    print(f"[run] loading video-SALMONN 2+ 7B from {args.model_path} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=131072, padding_side="right", use_fast=False)
    model = video_SALMONN2_plus.from_pretrained(
        args.model_path, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, device_map="cpu")
    model.cuda().eval()

    data_args = DataArguments()
    data_args.video_max_frames = args.max_frames
    data_args.video_min_frames = args.min_frames
    data_args.base_interval = args.interval
    data_args.max_pixels = args.max_pixels
    data_args.video_max_frame_pixels = args.max_pixels
    data_args.run_test = True
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(args.model_path)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length)
    data_args.model_type = "qwen2.5vl"
    test_data = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)["train_dataset"]
    print(f"[run] model ready in {time.time() - t0:.0f}s (bf16, sdpa, "
          f"liger={'off' if args.no_liger else 'on'})", flush=True)

    audit_fh = {s: open(audit_path[s], "a", encoding="utf-8") for s in settings}
    verif_fh = {s: open(verif_path[s], "a", encoding="utf-8") for s in settings}

    def emit(fn, setting, text, n_units, secs, capped, retried, err=None):
        ans = (text or "").strip()
        p = parse(ans) if ans else {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}
        audit_fh[setting].write(json.dumps(
            {"key": fn, "setting": setting, "text": text, "n_units": n_units,
             "finish_reason": ("length" if capped else "stop") if not err else "error",
             "retried": retried, "secs": secs, "error": err}, ensure_ascii=False) + "\n")
        verif_fh[setting].write(json.dumps(
            {"filename": fn, "nodes": p["nodes"], "edges": p["edges"], "valid": p["valid"]},
            ensure_ascii=False) + "\n")
        done[setting].add(fn)

    def build_inputs(video_path, setting, fn):
        """Repo `_get_item` does all preprocessing; system prompt rides in the user turn."""
        user_text = P.build_user_text(cap[fn] if setting == "caption" else None)
        input_dict = {
            "video": video_path,
            "use_audio": use_audio,
            "conversations": [
                {"from": "human", "value": "<video>\n" + P.SYSTEM_PROMPT + "\n\n" + user_text},
                {"from": "gpt", "value": ""},
            ],
        }
        d = test_data._get_item(input_dict)
        assert d.get("should_use", True) and d.get("video") == video_path, "dataset returned fallback sample"
        n_units = int(d["input_ids"].shape[1])
        d = {k: v.to("cuda") for k, v in d.items() if isinstance(v, torch.Tensor)}
        return d, n_units

    @torch.inference_mode()
    def run_one(inputs, rep=None):
        gk = {"max_new_tokens": args.max_new_tokens, "do_sample": args.sample}
        if args.sample:
            gk["temperature"] = args.temperature
        if rep and rep != 1.0:
            gk["repetition_penalty"] = float(rep)
        t1 = time.time()
        out = model.generate(**inputs, **gk)
        new = out[0, inputs["input_ids"].shape[1]:]
        capped = new.shape[0] >= args.max_new_tokens - 4
        text = tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.strip(), round(time.time() - t1, 1), capped

    n_done, t_run = 0, time.time()
    for fn in todo:
        vp = os.path.join(args.clips_dir, fn)
        for s in settings:
            if fn in done[s]:
                continue
            try:
                inputs, n_units = build_inputs(vp, s, fn)
                text, secs, capped = run_one(inputs)
                retried = False
                if capped and args.loop_retry_rep > 0:
                    print(f"[loop] {fn} [{s}] hit cap -> retry rep={args.loop_retry_rep}", flush=True)
                    t2, s2, c2 = run_one(inputs, rep=args.loop_retry_rep)
                    if not c2:
                        text, secs, capped, retried = t2, s2, False, True
                emit(fn, s, text, n_units, secs, capped, retried)
                del inputs
            except Exception as e:
                print(f"[err] {fn} [{s}]: {str(e)[:200]}", flush=True)
                emit(fn, s, "", 0, 0, False, False, str(e)[:200])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for s in settings:
            audit_fh[s].flush(); verif_fh[s].flush()
        n_done += 1
        if n_done % 5 == 0 or n_done == len(todo):
            rate = n_done / max(time.time() - t_run, 1e-6)
            print(f"[run] {n_done}/{len(todo)} clips  ({rate*60:.1f} clips/min)", flush=True)

    for s in settings:
        audit_fh[s].close(); verif_fh[s].close()
    print(f"[run] done in {(time.time() - t_run)/60:.1f} min. "
          f"verified -> {os.path.join(args.out_dir, 'verified')}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
