# Public-release adaptation: paths/imports only; campaign model and decoding logic retained.
"""Run NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 on the AVSG benchmark.

Nemotron-Omni is a Mamba2-Transformer hybrid MoE (C-RADIO vision + Parakeet audio) and
is NOT supported by the old vLLM in the qwen container — it needs vLLM >=0.20. NVIDIA's
supported path is a vLLM OpenAI server (`vllm serve ... --reasoning-parser nemotron_v3`)
that loads video+audio itself from a file URL. This client hits that server.

Same as every other model: uses the SHARED prompt.py, parse_output.py, and writes the same
result files the evaluator consumes. Audio is fed via use_audio_in_video=True (the whole
point of the AVSG benchmark) — the server extracts the audio track from each mp4.

  results/<MODEL>_<setting>_raw.jsonl        audit   {key,text,finish_reason,...}
  results/verified/<MODEL>_<setting>.jsonl   scored  {filename,nodes,edges,valid}
"""
import argparse, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EVAL_ROOT, "evaluation"))
import prompt as P                                     # noqa: E402  (SHARED prompt)
from parse_output import parse                         # noqa: E402

MODEL_TAG = "nemotron-3-nano-omni"


def clip_duration(fn):
    try:
        a, b = fn[:-4].rsplit("_", 2)[-2:]
        return float(b) - float(a)
    except Exception:
        return float("inf")


def load_gold(gold_path):
    cap = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                cap[r["filename"]] = r.get("caption", "")
    return cap


def done_filenames(path):
    out = set()
    if os.path.exists(path):
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
    """Nemotron reasoning model: '<think> ... </think> answer'. Keep the answer."""
    if text and "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return (text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default=os.path.join(EVAL_ROOT, "clips"))
    ap.add_argument("--gold", default=os.path.join(EVAL_ROOT, "data", "gold", "1k_result.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(EVAL_ROOT, "outputs", "inference"))
    ap.add_argument("--settings", default="raw,caption")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--order", choices=["short", "long", "name"], default="short")
    ap.add_argument("--clips", default="")
    # server
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--served-model", default="nemotron")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel requests (server batches)")
    ap.add_argument("--timeout", type=float, default=600.0)
    # generation
    ap.add_argument("--thinking", action="store_true",
                    help="enable reasoning mode (slow; default is instruct/non-thinking)")
    ap.add_argument("--reasoning-budget", type=int, default=4096,
                    help="thinking mode: cap reasoning tokens (keeps wall-clock bounded)")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.2)   # instruct-mode default
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1)
    # multimodal
    ap.add_argument("--no-audio", action="store_true", help="disable use_audio_in_video (debug)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tag", default=MODEL_TAG,
                    help="result-file tag (separate leaderboard row, e.g. *-think)")
    args = ap.parse_args()

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    os.makedirs(os.path.join(args.out_dir, "verified"), exist_ok=True)
    use_audio = not args.no_audio

    cap = load_gold(args.gold)
    on_disk = {fn for fn in os.listdir(args.clips_dir) if fn.endswith(".mp4")}
    clips = [fn for fn in cap if fn in on_disk]
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

    if args.thinking:
        # reasoning_budget tokens of CoT + ~2048 grace/answer; official thinking sampling
        max_tokens = args.reasoning_budget + 2048
        temperature, top_p, top_k = 0.6, 0.95, -1
    else:
        max_tokens, temperature, top_p, top_k = args.max_tokens, args.temperature, args.top_p, args.top_k
    print(f"[run] {len(clips)} clips, settings={settings}, audio_in_video={use_audio}, "
          f"thinking={args.thinking}, max_tokens={max_tokens}, concurrency={args.concurrency}", flush=True)

    audit_path = {s: os.path.join(args.out_dir, f"{args.tag}_{s}_raw.jsonl") for s in settings}
    verif_path = {s: os.path.join(args.out_dir, "verified", f"{args.tag}_{s}.jsonl") for s in settings}
    done = {s: (set() if args.no_resume else done_filenames(verif_path[s])) for s in settings}
    for s in settings:
        print(f"[run] {s}: {len(done[s])} already done (resume)", flush=True)

    jobs = [(fn, s) for fn in clips for s in settings if fn not in done[s]]
    if not jobs:
        print("[run] nothing to do.", flush=True)
        return

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)

    ctk = {"enable_thinking": bool(args.thinking)}
    if args.thinking:
        ctk["reasoning_budget"] = args.reasoning_budget
    extra_body = {"chat_template_kwargs": ctk}
    if use_audio:
        extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}
    if top_k and top_k > 0:
        extra_body["top_k"] = top_k

    def infer(fn, setting):
        video_url = "file://" + os.path.abspath(os.path.join(args.clips_dir, fn))
        user_text = P.build_user_text(cap[fn] if setting == "caption" else None)
        messages = [
            {"role": "system", "content": P.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": user_text}]},
        ]
        t0 = time.time()
        resp = client.chat.completions.create(
            model=args.served_model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, extra_body=extra_body)
        ch = resp.choices[0]
        text = ch.message.content or ""
        # some builds put the CoT in a separate reasoning_content field
        rc = getattr(ch.message, "reasoning_content", None)
        return {"text": text, "reasoning": rc, "finish": ch.finish_reason,
                "n_gen": (resp.usage.completion_tokens if resp.usage else None),
                "secs": round(time.time() - t0, 1)}

    audit_fh = {s: open(audit_path[s], "a", encoding="utf-8") for s in settings}
    verif_fh = {s: open(verif_path[s], "a", encoding="utf-8") for s in settings}

    def emit(fn, setting, res, err=None):
        text = res["text"] if res else ""
        ans = strip_thinking(text) if text else ""
        p = parse(ans) if ans else {"nodes": [], "edges": [], "valid": False, "oov_nodes": 0}
        audit_fh[setting].write(json.dumps(
            {"key": fn, "setting": setting, "text": text,
             "reasoning": (res or {}).get("reasoning"),
             "finish_reason": (res or {}).get("finish"),
             "n_gen_tokens": (res or {}).get("n_gen"), "secs": (res or {}).get("secs"),
             "error": err}, ensure_ascii=False) + "\n")
        verif_fh[setting].write(json.dumps(
            {"filename": fn, "nodes": p["nodes"], "edges": p["edges"], "valid": p["valid"]},
            ensure_ascii=False) + "\n")
        done[setting].add(fn)

    n = 0
    t_run = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut = {ex.submit(infer, fn, s): (fn, s) for (fn, s) in jobs}
        for f in as_completed(fut):
            fn, s = fut[f]
            try:
                emit(fn, s, f.result())
            except Exception as e:
                print(f"[err] {fn} [{s}]: {str(e)[:200]}", flush=True)
                emit(fn, s, None, str(e)[:200])
            n += 1
            if n % 10 == 0 or n == len(jobs):
                for s2 in settings:
                    audit_fh[s2].flush(); verif_fh[s2].flush()
                rate = n / max(time.time() - t_run, 1e-6)
                print(f"[run] {n}/{len(jobs)} ({rate*60:.1f}/min)", flush=True)

    for s in settings:
        audit_fh[s].close(); verif_fh[s].close()
    print(f"[run] done in {(time.time()-t_run)/60:.1f} min -> {os.path.join(args.out_dir,'verified')}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
