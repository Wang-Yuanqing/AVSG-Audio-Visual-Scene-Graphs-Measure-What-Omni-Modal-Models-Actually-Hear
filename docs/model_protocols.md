# Recorded model protocols

This table describes the configurations evaluated in the paper. All models receive the fixed rubric prompt with three worked examples from `evaluation/prompt.py`. Raw input excludes the reference caption; the caption condition appends it. Input/API protocols differ between model families.

| Configuration | Media path | Main-pass decoding |
|---|---|---|
| Gemini 2.5 Pro / 3.1 Pro / 3.5 Flash | MP4 through Vertex batch API; API-side sampling | JSON response MIME type; no explicit temperature, top-p or output-token override |
| Qwen3-Omni Thinking | vLLM V0; 2 fps, at most 16 frames, 200,704 pixels/frame; interleaved in-video audio | T=.6, top-p=.95, top-k=20; 16,384 output tokens |
| Qwen3-Omni Instruct | Same media path | Same sampling; 8,192 output tokens |
| Qwen2.5-Omni | vLLM V0; 2 fps, at most 16 frames; full audio supplied separately | T=.6, top-p=.95, top-k=20; 4,096 output tokens |
| Nemotron Instruct | vLLM 0.20 server; 2 fps, at most 32 frames, in-video audio | T=.2, top-p=1, top-k=1; 3,072 output tokens |
| Nemotron Thinking | Same server/media path | T=.6, top-p=.95; no top-k override; 16,384 reasoning tokens and 18,432 total output cap |
| MiniCPM-o 4.5 | 1 fps / 1 s audio interleave, 16 kHz mono, one slice/frame; input cap 32,768 tokens | Sampling, T=.7; 4,096 output tokens |
| OmniVinci | 32 frames, audio enabled with `audio_chunk_length=max_3600` | Greedy, repetition penalty 1.10; 1,024 output tokens |
| InteractiveOmni | 32 frames, one tile/frame; full 16 kHz mono audio split into windows of at most 30 s | Sampling with no temperature override; 4,096 output tokens |
| video-SALMONN 2+ | 16–768 frames, interval parameter .1, pixel cap 61,250; 30 s Whisper audio windows | Greedy; 4,096 output tokens |

Official checkpoints and their native/reference serving paths were used. Unspecified sampling settings follow the checkpoint or API configuration. The recovered OmniVinci runner explicitly defaults to FP16 loading; this is distinct from a checkpoint's stored precision. Model comparisons are scoped to these configurations, not a controlled isolation of architecture or reasoning tokens.

Decode-only recovery changes sampling/penalties or token budgets on failed records while keeping the task prompt fixed. For example, Nemotron thinking recovery used presence penalties 1.5 then 2.0 and allowed 16,384 + 4,096 output tokens. The [runner README](../inference/README.md) explains which recovery paths are included and distinguishes main-pass reruns from reproduction using the final saved predictions.

For audio-removal experiments, use identical clips with no audio input (or audio-stripped MP4s for Gemini) and the same task prompt. The AV-minus-V score difference measures score sensitivity under this intervention. It is not a count or fraction of relations that were perceived acoustically. Full-corpus experiments cover 1,024 clips; LoRA comparisons use the fixed 197-clip test set.
