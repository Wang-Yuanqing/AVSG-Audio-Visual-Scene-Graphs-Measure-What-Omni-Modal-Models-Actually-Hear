# Public-release adaptation: paths/imports only; campaign model and decoding logic retained.
"""Run sensenova/InteractiveOmni-8B on the AVSG benchmark (HF trust_remote_code, single GPU).

InteractiveOmni-8B (InternViT-300M vision + Whisper-style audio encoder + Qwen3-8B LLM)
takes images/video-frames and audio in one turn. Its stock `chat()` has two demo-level
limits that would cripple AVSG: (1) one audio item per turn and (2) `process_audio()`
hard-truncates audio at 30 s (one Whisper window). The underlying plumbing is fully
multi-audio (N `<audio>` placeholders, features concatenated), so this runner rebuilds
chat()'s prompt assembly faithfully and feeds the FULL clip audio as consecutive <=30 s
chunks — the model hears everything, matching what the other benchmark models get.

Vision recipe: `load_video` frames (default 32, the remote-code default) at 1 tile per
frame (InternVL video convention, 256 tokens/frame). The README chat example is 8 frames
x 12 tiles for generic use; for minutes-long clips 32x1 keeps temporal coverage within
context. Both knobs are CLI-exposed (--frames / --max-patch).

Same interface as every other model: SHARED prompt.py / parse_output.py and the same
output files the evaluator consumes:
  * audit  : results/interactiveomni-8b_<setting>_raw.jsonl       {key, text, ...}
  * scored : results/verified/interactiveomni-8b_<setting>.jsonl  {filename, nodes, edges, valid}

Notes:
  * TEXT output only: generate_audio=False (voicelm/flow/hifigan heads never run).
  * Official sampling per README: do_sample=True, no explicit temperature.
  * Loop safety-net (house style): exact cap detection on returned new-token count;
    on cap hit retry once with repetition_penalty.
Run via launch_interomni.sh (env + caches), e.g.
  ./launch_interomni.sh --clips-file ../minicpm/smoke_clips.txt --settings raw --out-dir smoke
  ./launch_interomni.sh --order name
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.dirname(HERE)                      # .../avsg_eval
sys.path.insert(0, os.path.join(EVAL_ROOT, "evaluation"))                          # prompt.py, parse_output.py, vocab.py

import prompt as P                                     # noqa: E402
from parse_output import parse                         # noqa: E402

MODEL_TAG = "interactiveomni-8b"
SR = 16000
CHUNK = 30 * SR                                        # process_audio()'s hard cap
MIN_TAIL = SR // 10                                    # drop <0.1 s tail chunks


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


def load_wav(path):
    """Full clip audio as float32 mono 16 kHz via ffmpeg (empty array if no audio track)."""
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)


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
    # generation (README: do_sample=True, nothing else pinned; house cap 4096)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0, help=">0 passes temperature")
    ap.add_argument("--greedy", action="store_true", help="do_sample=False (debug only)")
    ap.add_argument("--loop-retry-rep", type=float, default=1.1,
                    help="on cap-hit output, retry once with this repetition_penalty; 0 = off")
    # multimodal
    ap.add_argument("--frames", type=int, default=32, help="load_video num_segments")
    ap.add_argument("--max-patch", type=int, default=1,
                    help="tiles per frame (InternVL video convention = 1; README images = 12)")
    ap.add_argument("--attn", choices=["flash_attention_2", "sdpa", "eager"], default="",
                    help="default: flash_attention_2 if importable else sdpa")
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

    audit_path = {s: os.path.join(args.out_dir, f"{MODEL_TAG}_{s}_raw.jsonl") for s in settings}
    verif_path = {s: os.path.join(args.out_dir, "verified", f"{MODEL_TAG}_{s}.jsonl") for s in settings}
    done = {s: (set() if args.no_resume else done_filenames(verif_path[s])) for s in settings}
    for s in settings:
        print(f"[run] {s}: {len(done[s])} already done (resume)", flush=True)
    todo = [fn for fn in clips if any(fn not in done[s] for s in settings)]
    print(f"[run] {len(clips)} clips ({len(todo)} todo), settings={settings}, audio={use_audio}, "
          f"frames={args.frames}x{args.max_patch}tile, max_new={args.max_new_tokens}", flush=True)
    if not todo:
        print("[run] nothing to do.", flush=True)
        return

    # ---- load model (once) ----
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    attn = args.attn
    if not attn:
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
    print(f"[run] loading InteractiveOmni-8B from {args.model_path} (attn={attn}) ...", flush=True)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    # config.json ships llm_config.attn_implementation=flash_attention_2 and the remote
    # code builds Qwen3ForCausalLM(config.llm_config) directly, so the override must go
    # through the `_attn_implementation` property (a plain `.attn_implementation`
    # attribute is never read and the load crashes on flash-attn-less nodes).
    cfg.llm_config._attn_implementation = attn
    cfg.vision_config.use_flash_attn = (attn == "flash_attention_2")
    model = AutoModel.from_pretrained(
        args.model_path, config=cfg, trust_remote_code=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    mm = sys.modules[type(model).__module__]           # remote-code module: token constants
    print(f"[run] model ready in {time.time() - t0:.0f}s (bf16, {attn}, "
          f"num_image_token={model.num_image_token})", flush=True)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(mm.IMG_CONTEXT_TOKEN)
    model.audio_context_token_id = tokenizer.convert_tokens_to_ids(mm.AUDIO_CONTEXT_TOKEN)
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
    start_token_id = tokenizer.convert_tokens_to_ids(["<|im_start|>"])[0]

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

    def build_contents(video_path):
        """Frames -> tiled pixel_values; full audio -> <=30 s whisper-window chunks.
        Extracted once per clip and reused for both settings."""
        frames = model.load_video(video_path, num_segments=args.frames)
        dynamic_images, dynamic_nums = [], []
        for fr in frames:
            tiles = model.load_image(fr, max_num=args.max_patch)
            dynamic_images += tiles
            dynamic_nums.append(len(tiles))
        pixel_values = torch.stack([model.transform(im) for im in dynamic_images])
        audio_feats = []                                # list of (values, len_after_cnn, token_num)
        if use_audio:
            wav = load_wav(video_path)
            for i in range(0, len(wav), CHUNK):
                chunk = wav[i:i + CHUNK]
                if len(chunk) < MIN_TAIL and i > 0:
                    break
                d = model.process_audio(chunk, return_tensors="pt", sampling_rate=SR)
                if int(d["audio_token_num"].item()) <= 0:
                    continue
                audio_feats.append((d["input_features"], d["audio_len_after_cnn"], d["audio_token_num"]))
        return (pixel_values, dynamic_nums, audio_feats), len(frames) + len(audio_feats)

    @torch.inference_mode()
    def run_one(contents, fn, setting, rep=None):
        pixel_values, dynamic_nums, audio_feats = contents
        user_text = P.build_user_text(cap[fn] if setting == "caption" else None)
        text = "<image>\n" * len(dynamic_nums) + "<audio>\n" * len(audio_feats) + user_text

        template = mm.get_conv_template(model.template)
        template.set_system_message(P.SYSTEM_PROMPT)
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        import re
        if audio_feats:
            audio_tokens_iter = iter(
                mm.AUDIO_START_TOKEN + mm.AUDIO_CONTEXT_TOKEN * int(t.item()) + mm.AUDIO_END_TOKEN
                for _, _, t in audio_feats)
            query = re.sub(r"<audio>", lambda m: next(audio_tokens_iter), query)
        image_tokens_iter = iter(
            mm.IMG_START_TOKEN + mm.IMG_CONTEXT_TOKEN * model.num_image_token * n + mm.IMG_END_TOKEN
            for n in dynamic_nums)
        query = re.sub(r"<image>", lambda m: next(image_tokens_iter), query)

        inputs = tokenizer(query, return_tensors="pt", add_special_tokens=False)
        gk = {"max_new_tokens": args.max_new_tokens, "do_sample": not args.greedy,
              "eos_token_id": eos_token_id}
        if args.temperature > 0 and not args.greedy:
            gk["temperature"] = args.temperature
        if rep and rep != 1.0:
            gk["repetition_penalty"] = float(rep)
        if audio_feats:
            av = torch.cat([v for v, _, _ in audio_feats], dim=0).to(dtype=model.dtype).cuda()
            alac = torch.stack([l for _, l, _ in audio_feats], dim=0)
            atn = torch.stack([t for _, _, t in audio_feats], dim=0)
        else:
            av = alac = atn = None
        t1 = time.time()
        out, _, _ = model.generate(
            pixel_values=pixel_values.to(torch.bfloat16).cuda(),
            audio_values=av, audio_len_after_cnn=alac, audio_token_num=atn,
            input_ids=inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            generate_audio=False, start_token_id=start_token_id,
            **gk)
        capped = out.shape[1] >= args.max_new_tokens - 4
        text_out = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
        text_out = text_out.split("<|im_end|>")[0].replace("<|endoftext|>", "").strip()
        return text_out, round(time.time() - t1, 1), capped

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
                text, secs, capped = run_one(contents, fn, s)
                retried = False
                if capped and args.loop_retry_rep > 0:
                    print(f"[loop] {fn} [{s}] hit cap -> retry rep={args.loop_retry_rep}", flush=True)
                    t2, s2, c2 = run_one(contents, fn, s, rep=args.loop_retry_rep)
                    if not c2:
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
