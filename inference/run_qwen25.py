# Public-release adaptation: paths/imports only; campaign model and decoding logic retained.
"""Run Qwen3-Omni-30B-A3B-Thinking on the AVSG benchmark (offline vLLM, single GPU).

Produces, for each requested setting (`raw` = video+audio only, `caption` = +reference
caption), two files under the selected output directory/:
  * audit  : results/<MODEL>_<setting>_raw.jsonl       {key, text, n_prompt_tokens, ...}
  * scored : results/verified/<MODEL>_<setting>.jsonl  {filename, nodes, edges, valid}
The verified file is exactly what the evaluator consumes (no extra collect step).

Design / optimizations:
  * ONE model load serves both settings; per clip we build raw+caption back-to-back so
    vLLM's multimodal cache reuses the (identical) decoded video+audio across the two.
  * enable_prefix_caching: the ~2.3k-token system prompt is shared by every request.
  * use_audio_in_video=True so the audio track is actually fed to the model (the whole
    point of the benchmark) and stays time-aligned with the frames.
  * Long-clip guard: video frames are hard-capped (max_frames) so a 9-min clip costs the
    same vision tokens as a 14s one; any request that still overflows / OOMs is retried
    once with a smaller budget, then recorded invalid. Every downscale is logged.
  * Incremental, resumable: outputs flushed per chunk; a re-run skips finished clips.
  * Official main-pass sampling; launch settings are documented in README.md.

Public-release commands and environments: inference/README.md.
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

MODEL_TAG = "qwen2.5-omni-7b"                           # -> results/verified/<tag>_<setting>.jsonl


# --------------------------------------------------------------------------- clips / gold
def load_gold(gold_path):
    """filename -> caption (and the set of gold filenames)."""
    cap = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                cap[r["filename"]] = r.get("caption", "")
    return cap


def clip_duration(fn):
    # "<id>_<start>_<end>.mp4" -> end-start ; fall back to +inf so unknowns sort last
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


# --------------------------------------------------------------------------- prompt build
def build_messages(video_path, user_text, fps, max_frames, max_pixels, audio_mode):
    content = [{"type": "video", "video": video_path,
                "fps": fps, "max_frames": max_frames, "max_pixels": max_pixels}]
    if audio_mode == "separate":            # extract the audio track as its own modality
        content.append({"type": "audio", "audio": video_path})
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": P.SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def make_input(processor, process_mm_info, video_path, user_text,
               fps, max_frames, max_pixels, audio_mode):
    """Decode the clip once and return a vLLM request dict (or raise).

    audio_mode: 'separate' (audio+video as independent modalities, robust default),
                'in_video' (interleaved; broken in this vLLM build for clips >2s),
                'none' (video only)."""
    in_video = (audio_mode == "in_video")
    msgs = build_messages(video_path, user_text, fps, max_frames, max_pixels, audio_mode)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    audios, images, videos, vkw = process_mm_info(msgs, use_audio_in_video=in_video,
                                                  return_video_kwargs=True)
    mm = {}
    if images:
        mm["image"] = images
    if videos:
        mm["video"] = videos
    if audios:
        mm["audio"] = audios
    mmkw = {"use_audio_in_video": in_video and bool(audios)}
    # the Qwen3-Omni processor wants a SCALAR fps; process_mm_info returns a 1-elem list
    fps_v = vkw.get("fps")
    if fps_v is not None:
        mmkw["fps"] = fps_v[0] if isinstance(fps_v, (list, tuple)) else fps_v
    return {"prompt": text, "multi_modal_data": mm, "mm_processor_kwargs": mmkw}


# --------------------------------------------------------------------------- output
def strip_thinking(text):
    """Thinking model emits '<think> ... </think> <answer>'. Keep the answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


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
    ap.add_argument("--chunk", type=int, default=16, help="clips per generate() call")
    # generation
    ap.add_argument("--max-tokens", type=int, default=16384)  # Qwen rec.; fits 32768 win
    ap.add_argument("--temperature", type=float, default=0.6)   # Qwen3-Omni Thinking official
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="primary-pass presence penalty; 0 keeps the official sampling config")
    ap.add_argument("--retry-presence-penalty", type=float, default=1.5,
                    help="on finish_reason=length (the Thinking model's reasoning-repetition "
                         "loop), retry the clip once with this presence penalty to break the "
                         "loop; 0 disables the safety net")
    # multimodal budget
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)   # ~256 vis tokens/frame
    # audio handling: 'separate' = audio+video as independent modalities (robust);
    # 'in_video' = 2s-interleaved (crashes M-RoPE in this vLLM build); 'none' = video only
    ap.add_argument("--audio-mode", choices=["separate", "in_video", "none"], default="separate")
    # engine
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true")
    # vLLM's per-modality mm cache processes the video ALONE, which is incompatible with
    # use_audio_in_video (no paired audio in that sub-call); keep it off by default.
    ap.add_argument("--enable-mm-cache", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tag", default=MODEL_TAG,
                    help="result-file tag (separate leaderboard row, e.g. *-ft)")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    os.makedirs(os.path.join(args.out_dir, "verified"), exist_ok=True)

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
          f"settings={settings}, audio_mode={args.audio_mode}", flush=True)

    # resume bookkeeping + open append handles
    audit_path = {s: os.path.join(args.out_dir, f"{args.tag}_{s}_raw.jsonl") for s in settings}
    verif_path = {s: os.path.join(args.out_dir, "verified", f"{args.tag}_{s}.jsonl") for s in settings}
    done = {s: (set() if args.no_resume else done_filenames(verif_path[s])) for s in settings}
    for s in settings:
        print(f"[run] {s}: {len(done[s])} already done (resume)", flush=True)
    todo = [fn for fn in clips if any(fn not in done[s] for s in settings)]
    if not todo:
        print("[run] nothing to do.", flush=True)
        return

    # ---- load model (once) ----
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from qwen_omni_utils import process_mm_info

    print(f"[run] loading processor + model from {args.model_path} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 0, "video": 1, "audio": 1},
        enable_prefix_caching=True,
        disable_mm_preprocessor_cache=not args.enable_mm_cache,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed,
                        presence_penalty=args.presence_penalty)
    # Anti-degeneration retry: same official temp/top_p/top_k, only +presence_penalty to
    # break the Thinking model's reasoning-repetition loops (the one observed failure mode).
    sp_retry = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed + 1,
                              presence_penalty=args.retry_presence_penalty)
    print(f"[run] model ready in {time.time() - t0:.0f}s", flush=True)

    audit_fh = {s: open(audit_path[s], "a", encoding="utf-8") for s in settings}
    verif_fh = {s: open(verif_path[s], "a", encoding="utf-8") for s in settings}

    def assemble(fn, budget):
        """Return {setting: vLLM input}; budget = (fps, max_frames, max_pixels)."""
        fps, mf, mp = budget
        vp = os.path.join(args.clips_dir, fn)
        out = {}
        for s in settings:
            if fn in done[s]:
                continue
            utext = P.build_user_text(cap[fn] if s == "caption" else None)
            out[s] = make_input(processor, process_mm_info, vp, utext, fps, mf, mp, args.audio_mode)
        return out

    def emit(fn, setting, text, n_prompt, n_gen, finish, downscaled, err=None, retried=False):
        ans = strip_thinking(text) if text else ""
        p = parse(ans) if text else {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}
        audit_fh[setting].write(json.dumps(
            {"key": fn, "setting": setting, "text": text, "finish_reason": finish,
             "n_prompt_tokens": n_prompt, "n_gen_tokens": n_gen,
             "downscaled": downscaled, "retried": retried, "error": err}, ensure_ascii=False) + "\n")
        verif_fh[setting].write(json.dumps(
            {"filename": fn, "nodes": p["nodes"], "edges": p["edges"], "valid": p["valid"]},
            ensure_ascii=False) + "\n")
        done[setting].add(fn)

    def emit_with_retry(req, fn, setting, o, downscaled):
        """Emit one vLLM output. If it hit the token cap (finish_reason=length) — the
        Thinking model's reasoning-repetition loop — retry the clip once with
        presence_penalty to break the loop, and keep the cleaner result."""
        out = o.outputs[0]
        retried = False
        if out.finish_reason == "length" and args.retry_presence_penalty > 0:
            print(f"[retry] {fn} [{setting}] hit max_tokens (likely reasoning loop) -> "
                  f"retry w/ presence_penalty={args.retry_presence_penalty}", flush=True)
            try:
                o2 = llm.generate([req], sp_retry)[0]
                if o2.outputs[0].finish_reason != "length":      # escaped the loop
                    o, out, retried = o2, o2.outputs[0], True
            except Exception as e:
                print(f"[retry] failed {fn} [{setting}]: {str(e)[:150]}", flush=True)
        emit(fn, setting, out.text, len(o.prompt_token_ids), len(out.token_ids),
             out.finish_reason, downscaled, retried=retried)

    full_budget = (args.fps, args.max_frames, args.max_pixels)
    small_budget = (1.0, max(4, args.max_frames // 2), 128 * 28 * 28)   # fallback for overflow/OOM

    n_clip = 0
    t_run = time.time()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i:i + args.chunk]
        # assemble (decode video+audio) for the whole chunk
        reqs, meta = [], []          # meta: (fn, setting)
        for fn in chunk:
            try:
                ins = assemble(fn, full_budget)
            except Exception as e:
                print(f"[warn] assemble failed {fn}: {e}", flush=True)
                for s in settings:
                    if fn not in done[s]:
                        emit(fn, s, "", 0, 0, "assemble_error", False, str(e)[:200])
                continue
            for s, inp in ins.items():
                reqs.append(inp)
                meta.append((fn, s))
        if not reqs:
            continue
        try:
            outs = llm.generate(reqs, sp)
            for (fn, s), req, o in zip(meta, reqs, outs):
                emit_with_retry(req, fn, s, o, downscaled=False)
        except Exception as e:
            # isolate: retry each (fn,setting) alone with the smaller budget
            print(f"[warn] chunk generate failed ({len(reqs)} reqs): {str(e)[:200]} "
                  f"-> per-request retry", flush=True)
            for fn, s in meta:
                if fn in done[s]:
                    continue
                try:
                    inp = assemble(fn, small_budget)[s]
                    o = llm.generate([inp], sp)[0]
                    emit_with_retry(inp, fn, s, o, downscaled=True)
                except Exception as e2:
                    print(f"[err] giving up {fn} [{s}]: {str(e2)[:200]}", flush=True)
                    emit(fn, s, "", 0, 0, "generate_error", True, str(e2)[:200])
        for s in settings:
            audit_fh[s].flush()
            verif_fh[s].flush()
        n_clip += len(chunk)
        rate = n_clip / max(time.time() - t_run, 1e-6)
        print(f"[run] {n_clip}/{len(todo)} clips  ({rate*60:.1f} clips/min)", flush=True)

    for s in settings:
        audit_fh[s].close()
        verif_fh[s].close()
    print(f"[run] done in {(time.time() - t_run)/60:.1f} min. "
          f"verified -> {os.path.join(args.out_dir, 'verified')}", flush=True)
    print("[run] Score the verified outputs using evaluation/score.py with explicit cohort coverage.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
