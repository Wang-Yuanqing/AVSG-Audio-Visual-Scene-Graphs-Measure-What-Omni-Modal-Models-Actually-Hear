# Public-release adaptation: paths/imports only; campaign model and decoding logic retained.
"""Run openbmb/MiniCPM-o 4.5 on the AVSG benchmark (HuggingFace Transformers, single GPU).

MiniCPM-o 4.5 (9B: Qwen3-8B LLM + SigLip2 vision + Whisper-medium audio) takes video+audio
natively via its half-duplex "omni mode": 1 fps frames and 1 s audio chunks interleaved in a
single user turn (`get_video_frame_audio_segments` + `model.chat(..., omni_mode=True)`).
vLLM's documented path has no combined video+audio request, so this drives the reference HF
implementation directly — same precedent as the OmniVinci runner.

Same interface as every other model: the SHARED prompt.py / parse_output.py, and the same
output files the evaluator consumes:
  * audit  : results/minicpm-o-4.5_<setting>_raw.jsonl       {key, text, ...}
  * scored : results/verified/minicpm-o-4.5_<setting>.jsonl  {filename, nodes, edges, valid}

Notes:
  * TEXT output only: init_tts=False (TTS head never loaded), generate_audio=False.
  * enable_thinking=False (the model's default mode) + official omni sampling
    (do_sample, temperature 0.7) per the model card example.
  * max_inp_length raised from the 8192 default to fit long clips (LLM context is 40960);
    frames+audio are extracted ONCE per clip and reused for both settings.
  * Loop safety-net (house style): if generation hits the token cap (proxy for the
    repetition-loop failure seen on every model), retry once with repetition_penalty.
Run via launch_minicpm.sh (env + caches), e.g.
  ./launch_minicpm.sh --clips-file smoke_clips.txt --settings raw --out-dir smoke  # smoke
  ./launch_minicpm.sh --order name                                                 # full run
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

MODEL_TAG = "minicpm-o-4.5"


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
    ap.add_argument("--order", choices=["short", "long", "name"], default="name")
    ap.add_argument("--clips", default="", help="comma-separated filenames; overrides order/limit")
    ap.add_argument("--clips-file", default="", help="file with one filename per line (or commas)")
    # generation (official omni-mode example: do_sample, temp 0.7, enable_thinking=False)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--greedy", action="store_true", help="do_sample=False (debug only)")
    ap.add_argument("--thinking", action="store_true", help="enable_thinking=True (off = model default)")
    ap.add_argument("--max-inp-length", type=int, default=32768,
                    help="input-token cap inside chat(); ctx is 40960, leave generation room")
    ap.add_argument("--loop-retry-rep", type=float, default=1.1,
                    help="on cap-hit output, retry once with this repetition_penalty; 0 = off")
    # multimodal
    ap.add_argument("--stack-frames", type=int, default=1, help="1 = standard 1 fps omni mode")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="if >0, evenly subsample frames above this count (ALL audio kept)")
    ap.add_argument("--no-audio", action="store_true", help="frames only (debug)")
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
    print(f"[run] {len(clips)} clips, settings={settings}, audio={use_audio}, "
          f"thinking={args.thinking}, temp={args.temperature}, max_new={args.max_new_tokens}", flush=True)

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
    from transformers import AutoModel, AutoTokenizer
    from minicpmo.utils import get_video_frame_audio_segments

    print(f"[run] loading MiniCPM-o 4.5 from {args.model_path} ...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        init_vision=True, init_audio=True, init_tts=False)   # text out only — never load TTS
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[run] model ready in {time.time() - t0:.0f}s "
          f"(bf16, sdpa, thinking={args.thinking})", flush=True)

    audit_fh = {s: open(audit_path[s], "a", encoding="utf-8") for s in settings}
    verif_fh = {s: open(verif_path[s], "a", encoding="utf-8") for s in settings}

    def emit(fn, setting, text, n_units, secs, capped, retried, err=None):
        ans = strip_thinking(text) if text else ""
        p = parse(ans) if ans else {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}
        audit_fh[setting].write(json.dumps(
            {"key": fn, "setting": setting, "text": text, "n_units": n_units,
             "finish_reason": ("length" if capped else "stop") if not err else "error",
             "retried": retried, "secs": secs, "error": err}, ensure_ascii=False) + "\n")
        verif_fh[setting].write(json.dumps(
            {"filename": fn, "nodes": p["nodes"], "edges": p["edges"], "valid": p["valid"]},
            ensure_ascii=False) + "\n")
        done[setting].add(fn)

    def build_contents(video_path):
        """frames + audio chunks, interleaved (official omni recipe); extracted once/clip."""
        frames, audio_segs, stacked = get_video_frame_audio_segments(
            video_path, stack_frames=args.stack_frames)
        n = len(frames)
        keep = set(range(n))
        if args.max_frames and n > args.max_frames:           # cap frames, keep ALL audio
            step = n / args.max_frames
            keep = {int(i * step) for i in range(args.max_frames)}
        contents = []
        for i in range(n):
            if i in keep:
                contents.append(frames[i])
                if stacked is not None and stacked[i] is not None:
                    contents.append(stacked[i])
            if use_audio and audio_segs is not None and i < len(audio_segs):
                contents.append(audio_segs[i])
        return contents, n

    def approx_cap_hit(text):
        """chat() returns text only; treat a long non-terminated JSON as a cap hit."""
        ids = tokenizer(text, add_special_tokens=False).input_ids
        return len(ids) >= args.max_new_tokens - 8

    @torch.inference_mode()
    def run_one(contents, fn, setting, rep=None):
        user_text = P.build_user_text(cap[fn] if setting == "caption" else None)
        msgs = [
            {"role": "system", "content": [P.SYSTEM_PROMPT]},
            {"role": "user", "content": contents + [user_text]},
        ]
        gk = {}
        if rep and rep != 1.0:
            gk["repetition_penalty"] = float(rep)
        t1 = time.time()
        res = model.chat(
            msgs=msgs, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            max_inp_length=args.max_inp_length,
            do_sample=not args.greedy,
            temperature=(args.temperature if not args.greedy else None),
            enable_thinking=args.thinking,
            omni_mode=True, max_slice_nums=1,
            generate_audio=False, use_tts_template=True,
            **gk)
        text = res if isinstance(res, str) else str(res)
        return text, round(time.time() - t1, 1)

    n_done, t_run = 0, time.time()
    for fn in todo:
        vp = os.path.join(args.clips_dir, fn)
        contents = None
        for s in settings:
            if fn in done[s]:
                continue
            try:
                if contents is None:
                    contents, n_units = build_contents(vp)
                text, secs = run_one(contents, fn, s)
                capped, retried = approx_cap_hit(text), False
                if capped and args.loop_retry_rep > 0:
                    print(f"[loop] {fn} [{s}] hit cap -> retry rep={args.loop_retry_rep}", flush=True)
                    t2, s2 = run_one(contents, fn, s, rep=args.loop_retry_rep)
                    if not approx_cap_hit(t2):
                        text, secs, capped, retried = t2, s2, False, True
                emit(fn, s, text, n_units, secs, capped, retried)
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
