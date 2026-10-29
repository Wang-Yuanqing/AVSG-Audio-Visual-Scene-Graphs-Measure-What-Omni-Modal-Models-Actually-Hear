# Public-release adaptation: paths/imports only; campaign model and decoding logic retained.
"""Run NVIDIA OmniVinci on the AVSG benchmark (HuggingFace Transformers, single GPU).

OmniVinci is a VILA/NVILA-based omni-modal model: it does NOT run on vLLM, so this
mirrors run_qwen.py's *interface* (same prompt.py, parse_output.py, and output files
that the evaluator consumes) but drives the model's native HF generate() API.

For each requested setting (`raw` = video+audio only, `caption` = +reference caption)
it writes, under the selected output directory/:
  * audit  : results/omnivinci_<setting>_raw.jsonl       {key, text, n_gen_tokens, ...}
  * scored : results/verified/omnivinci_<setting>.jsonl  {filename, nodes, edges, valid}

Notes vs the Qwen runner:
  * Audio is fed time-aligned IN the video (load_audio_in_video=True) — the whole point.
  * The big SYSTEM_PROMPT is folded into the single user turn: the model's chat examples
    only use a user turn, so this guarantees the identical instructions reach the model
    regardless of whether its template supports a system role (content is unchanged).
  * Greedy decoding (do_sample=False) for a reproducible benchmark number. OmniVinci is
    not a "thinking" model, so there is no <think> loop / huge token budget to guard.
  * One clip at a time (no paged-attention batching); both settings done back-to-back so
    the decoded mp4 stays warm. Incremental + resumable: a re-run skips finished clips.

Run via launch_omnivinci.sh (sets env + caches), e.g.
  ./launch_omnivinci.sh --limit 3 --order short      # smoke test
  ./launch_omnivinci.sh                               # full 1024 x {raw,caption}
"""

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)                      # .../avsg_eval
sys.path.insert(0, os.path.join(EVAL_ROOT, "evaluation"))                          # prompt.py, parse_output.py, vocab.py

import prompt as P                                     # noqa: E402
from parse_output import parse                         # noqa: E402

MODEL_TAG = "omnivinci"                                 # -> results/verified/<tag>_<setting>.jsonl


# --------------------------------------------------------------------------- clips / gold
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


# --------------------------------------------------------------------------- output
def strip_thinking(text):
    if text and "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return (text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Local official checkpoint directory")
    ap.add_argument("--clips-dir", default=os.path.join(EVAL_ROOT, "clips"))
    ap.add_argument("--gold", default=os.path.join(EVAL_ROOT, "data", "gold", "1k_result.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(EVAL_ROOT, "outputs", "inference"))
    ap.add_argument("--settings", default="raw,caption")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--order", choices=["short", "long", "name"], default="short")
    ap.add_argument("--clips", default="", help="comma-separated filenames; overrides order/limit")
    # generation
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--do-sample", action="store_true", help="default greedy (reproducible)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    # anti-degeneration: greedy decoding on this model loops in the free-text "attribute"
    # field; a repetition penalty breaks the loop without touching the (shared) prompt.
    # TUNED (finetune_sweep.py): 1.10 maximizes node/edge structure while staying loop-safe
    # in the sweep; 1.3 over-suppresses structure (edges -60%), 1.05 broadens loops.
    ap.add_argument("--repetition-penalty", type=float, default=1.10)
    # loop safety-net: penalty is per-clip chaotic, so any fixed value can loop on SOME
    # clip at 1024-clip scale. On finish_reason=length, retry that clip once at this
    # higher penalty (1.3 gave 0 loops across all 2048 in the original full run).
    ap.add_argument("--loop-retry-rep", type=float, default=1.3,
                    help="0 disables the loop safety net")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0,
                    help="0 = off; avoid small values, they ban legit recurring JSON n-grams")
    # multimodal budget
    ap.add_argument("--num-frames", type=int, default=32, help="video frames sampled per clip")
    ap.add_argument("--audio-length", default="max_3600", help="audio_chunk_length config")
    ap.add_argument("--no-audio", action="store_true", help="disable audio-in-video (debug)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    # Use the historical fixed benchmark prompt.
    PROMPT = P

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    os.makedirs(os.path.join(args.out_dir, "verified"), exist_ok=True)
    load_audio = not args.no_audio

    cap = load_gold(args.gold)
    on_disk = {fn for fn in os.listdir(args.clips_dir) if fn.endswith(".mp4")}
    clips = [fn for fn in cap if fn in on_disk]
    missing = [fn for fn in cap if fn not in on_disk]
    if args.clips:
        want = [c.strip() for c in args.clips.split(",") if c.strip()]
        clips = [fn for fn in want if fn in cap and fn in on_disk]
    else:
        if args.order == "name":
            clips.sort()
        else:
            clips.sort(key=clip_duration, reverse=(args.order == "long"))
        if args.limit:
            clips = clips[:args.limit]
    print(f"[run] {len(clips)} clips ({len(missing)} gold clips missing on disk), "
          f"settings={settings}, audio_in_video={load_audio}, frames={args.num_frames}", flush=True)

    audit_path = {s: os.path.join(args.out_dir, f"{MODEL_TAG}_{s}_raw.jsonl") for s in settings}
    verif_path = {s: os.path.join(args.out_dir, "verified", f"{MODEL_TAG}_{s}.jsonl") for s in settings}
    done = {s: (set() if args.no_resume else done_filenames(verif_path[s])) for s in settings}
    for s in settings:
        print(f"[run] {s}: {len(done[s])} already done (resume)", flush=True)
    todo = [fn for fn in clips if any(fn not in done[s] for s in settings)]
    if not todo:
        print("[run] nothing to do.", flush=True)
        return

    # ---- load model (once) ----
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    if args.model_path not in sys.path:
        sys.path.insert(0, args.model_path)          # let the checkpoint's remote code resolve
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"[run] loading OmniVinci from {args.model_path} ...", flush=True)
    t0 = time.time()
    AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                                      torch_dtype=dtype, device_map="auto",
                                      low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # multimodal config (mirror example_infer.py)
    model.config.load_audio_in_video = load_audio
    processor.config.load_audio_in_video = load_audio
    if args.num_frames > 0:
        model.config.num_video_frames = args.num_frames
        processor.config.num_video_frames = args.num_frames
    if args.audio_length != "-1":
        model.config.audio_chunk_length = args.audio_length
        processor.config.audio_chunk_length = args.audio_length

    gen_cfg = model.default_generation_config
    # sampling/length knobs applied once; repetition_penalty is set PER CALL (run_one)
    # so the loop safety-net can retry a looped clip at a higher penalty.
    base_gk = {"max_new_tokens": args.max_new_tokens, "max_length": 99999999,
               "do_sample": bool(args.do_sample)}
    if args.do_sample:
        base_gk["temperature"] = args.temperature
        base_gk["top_p"] = args.top_p
    if args.no_repeat_ngram_size > 0:
        base_gk["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    gen_cfg.update(**base_gk)

    device = getattr(model, "device", None) or next(model.parameters()).device
    print(f"[run] model ready in {time.time() - t0:.0f}s on {device} "
          f"(dtype={args.dtype}, greedy={not args.do_sample})", flush=True)

    audit_fh = {s: open(audit_path[s], "a", encoding="utf-8") for s in settings}
    verif_fh = {s: open(verif_path[s], "a", encoding="utf-8") for s in settings}

    def emit(fn, setting, text, n_gen, finish, err=None):
        ans = strip_thinking(text) if text else ""
        p = parse(ans) if text else {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}
        audit_fh[setting].write(json.dumps(
            {"key": fn, "setting": setting, "text": text, "finish_reason": finish,
             "n_gen_tokens": n_gen, "error": err}, ensure_ascii=False) + "\n")
        verif_fh[setting].write(json.dumps(
            {"filename": fn, "nodes": p["nodes"], "edges": p["edges"], "valid": p["valid"]},
            ensure_ascii=False) + "\n")
        done[setting].add(fn)

    @torch.inference_mode()
    def run_one(video_path, fn, setting, rep):
        """Generate the AVSG JSON for one clip+setting at repetition_penalty=rep;
        returns (text, n_gen, finish)."""
        caption = cap[fn] if setting == "caption" else None
        # Proper system/user split (the processor + qwen2 template support a system role),
        # matching how Qwen consumed the identical prompt — content is unchanged, only the
        # role wrapper, so the benchmark stays comparable.
        conversation = [
            {"role": "system", "content": PROMPT.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": PROMPT.build_user_text(caption)}]}]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor([text])
        if getattr(inputs, "input_ids", None) is not None:
            inputs.input_ids = inputs.input_ids.to(device)
        gen_cfg.repetition_penalty = float(rep) if rep and rep != 1.0 else 1.0
        out_ids = model.generate(input_ids=inputs.input_ids,
                                 media=getattr(inputs, "media", None),
                                 media_config=getattr(inputs, "media_config", None),
                                 generation_config=gen_cfg)
        n_gen = int(out_ids.shape[-1])
        resp = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        finish = "length" if n_gen >= args.max_new_tokens else "stop"
        return resp, n_gen, finish

    n_done = 0
    t_run = time.time()
    for fn in todo:
        vp = os.path.join(args.clips_dir, fn)
        for s in settings:
            if fn in done[s]:
                continue
            try:
                resp, n_gen, finish = run_one(vp, fn, s, args.repetition_penalty)
                # loop safety-net: a looped clip (finish=length) is retried once at a
                # higher, empirically loop-free penalty; keep it only if it escaped.
                if finish == "length" and args.loop_retry_rep > 0 \
                        and args.loop_retry_rep != args.repetition_penalty:
                    print(f"[loop] {fn} [{s}] hit cap at rep={args.repetition_penalty} "
                          f"-> retry at rep={args.loop_retry_rep}", flush=True)
                    r2, n2, f2 = run_one(vp, fn, s, args.loop_retry_rep)
                    if f2 != "length":
                        resp, n_gen, finish = r2, n2, f2
                emit(fn, s, resp, n_gen, finish)
            except Exception as e:
                print(f"[err] {fn} [{s}]: {str(e)[:200]}", flush=True)
                emit(fn, s, "", 0, "generate_error", str(e)[:200])
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
    print("[run] score with:  python -m evaluation.score --predictions YOUR_VERIFIED_FILE --split full", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
