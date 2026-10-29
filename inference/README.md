# Inference

Seven recovered campaign runners cover the nine open configurations. They preserve the model-specific media preprocessing, prompt composition, decoding and built-in retries. Public-release changes make checkpoint/media/output paths portable, import the released shared prompt/parser and remove an unused OmniVinci diagnostic prompt option. Source and adapted-file hashes are recorded in `source_manifest.json`.

Run from the repository root. Each model needs its **own compatible GPU environment**, official checkpoint and locally restored MP4 clips. The evaluation-only requirements are not a model inference environment. All runners support `--help` without loading a model. Their released entry points were checked on CPU; no GPU inference or training was rerun for this release. Exact paper results are reproducible from the saved predictions even when a historical checkpoint/API is unavailable.

## Main-pass commands

Use the full checkpoint directory for `--model-path` and select a fresh output directory. Qwen requires the recorded vLLM V0 multimodal path (`VLLM_USE_V1=0`). On Windows, use a supported Linux/CUDA environment for model execution.

```sh
# Qwen3-Omni Thinking (vLLM 0.9.3 campaign environment)
VLLM_USE_V1=0 python inference/run_qwen.py --model-path /path/to/Qwen3-Omni-Thinking --clips-dir clips --audio-mode in_video --settings raw,caption --max-tokens 16384 --out-dir outputs/qwen3_think

# Qwen3-Omni Instruct: same runner and prompt, different checkpoint and output cap
VLLM_USE_V1=0 python inference/run_qwen.py --model-path /path/to/Qwen3-Omni-Instruct --clips-dir clips --audio-mode in_video --settings raw,caption --max-tokens 8192 --tag qwen3-omni-30b-instruct --out-dir outputs/qwen3_instruct

# Qwen2.5-Omni: full audio supplied separately
VLLM_USE_V1=0 python inference/run_qwen25.py --model-path /path/to/Qwen2.5-Omni-7B --clips-dir clips --audio-mode separate --settings raw,caption --max-tokens 4096 --out-dir outputs/qwen25

python inference/run_minicpm.py --model-path /path/to/MiniCPM-o-4.5 --clips-dir clips --settings raw,caption --out-dir outputs/minicpm
python inference/run_omnivinci.py --model-path /path/to/OmniVinci --clips-dir clips --settings raw,caption --repetition-penalty 1.10 --num-frames 32 --loop-retry-rep 1.3 --out-dir outputs/omnivinci
python inference/run_interomni.py --model-path /path/to/InteractiveOmni-8B --clips-dir clips --settings raw,caption --attn sdpa --out-dir outputs/interactiveomni

# Point SALMONN_REPO at the upstream directory that contains qwenvl/.
SALMONN_REPO=/path/to/video_SALMONN2_plus python inference/run_salmonn.py --model-path /path/to/video-SALMONN-2plus --clips-dir clips --settings raw,caption --out-dir outputs/salmonn
```

For a smoke run, add `--limit 1 --settings raw`. Production runs resume by skipping completed filenames. An input-list restriction (`--clips`, or `--clips-file` where supported) is available for held-out evaluation. Use `python inference/run_MODEL.py --help` for exact options.

### Nemotron server/client

Use the recorded **vLLM 0.20** environment with Nemotron support. Start the server in one terminal (server and client must resolve the same absolute media paths):

```sh
vllm serve /path/to/Nemotron-3-Nano-Omni --served-model-name nemotron --trust-remote-code --tensor-parallel-size 1 --max-model-len 40960 --max-num-seqs 8 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"video":1,"image":1,"audio":1}' --media-io-kwargs '{"video":{"fps":2,"num_frames":32}}' --allowed-local-media-path /absolute/path/to/clips --reasoning-parser nemotron_v3 --host 127.0.0.1 --port 8000
```

Then run the local client:

```sh
python inference/run_nemotron.py --clips-dir /absolute/path/to/clips --settings raw,caption --out-dir outputs/nemotron_instruct
python inference/run_nemotron.py --clips-dir /absolute/path/to/clips --settings raw,caption --thinking --reasoning-budget 16384 --tag nemotron-3-nano-omni-think --out-dir outputs/nemotron_think
```

The default endpoint is `http://localhost:8000/v1`; its `EMPTY` API-key value is a local-server placeholder, not a credential. For a remote authenticated endpoint, adapt deployment authentication without committing a token.

## Environments and protocol

The following pins come from the saved launch/setup scripts, not a fresh compatibility claim for current packages. Separate environments avoid their conflicting dependencies. Checkpoint remote code and upstream model packages are not vendored here.

| Runner | Recorded environment notes |
|---|---|
| Qwen3 / Qwen2.5 | Campaign vLLM 0.9.3 container, V0 engine; `qwen_omni_utils`, Transformers and FFmpeg; multimodal model support must match the container/checkpoint. Installing an arbitrary current vLLM build is not the recorded environment. |
| Nemotron | vLLM 0.20 server with `nemotron_v3` reasoning parser; Python client uses `openai`. |
| MiniCPM-o | torch/torchaudio 2.8.0, torchvision 0.23.0, Transformers 4.51.0, `minicpmo-utils[all]>=1.0.5`, librosa, soundfile, Pillow and FFmpeg. |
| InteractiveOmni | torch/torchaudio 2.8.0, torchvision 0.23.0, Transformers 4.51.0, NumPy 1.26.4, decord 0.6.0; accelerate, einops, timm, omegaconf, scipy, diffusers, onnxruntime, librosa, soundfile and Pillow. Campaign used SDPA. |
| video-SALMONN 2+ | torch/torchaudio 2.7.1, torchvision 0.22.1, triton 3.3.1, Transformers 4.51.3, tokenizers 0.21.0, accelerate 1.7.0, peft 0.15.2, liger_kernel 0.5.10, decord 0.6.0, NumPy 1.24.4, torchcodec 0.4.0; upstream `qwenvl` source, librosa/soundfile and FFmpeg. Campaign used SDPA. |
| OmniVinci | torch 2.3.0, torchvision 0.18.0, flash-attn 2.5.8, NumPy 1.26.4, protobuf 3.20.x, pydantic 1.10.22, triton 3.1.0; upstream llava/VILA package, s2wrapper, audio libraries and checkpoint-supplied Transformers modeling patch. The recovered runner loads FP16 by default. |

The detailed [model protocol table](../docs/model_protocols.md) distinguishes media budgets, main-pass settings and recovery. GPU kernels, model remote code and batch sampling can affect reruns; the saved-prediction release fixes the paper's evaluated outputs.

## Video-only runs and recovery

Use a **separate output directory** and `--settings raw`. Qwen uses `--audio-mode none`; the other open runners use `--no-audio`. The task prompt is unchanged. Gemini receives copies remuxed without audio, e.g. `ffmpeg -i input.mp4 -c:v copy -an output.mp4`; keep the original clip filename in its separate video-only directory.

The recovered runners include their original built-in cap/parse retries. The campaign additionally used decode-only recovery sweeps on selected failed records. Those orchestration scripts and every retry history are not included in this initial public release; a single main pass is not guaranteed to produce the exact final healed corpus. The published parsed predictions are the authoritative inputs for reproducing the paper's scores. The released final parser retains OOV labels and includes the campaign's multi-object merge, bare-key repair and prefix-salvage behavior.

## Gemini batch requests

```sh
python inference/build_gemini_requests.py --gcs-dir gs://YOUR_BUCKET/clips --output-dir outputs/gemini_requests
```

This emits raw/caption Vertex batch JSONL for all 1,024 clips using the exact saved system prompt and JSON response MIME type, with no vocabulary enums or explicit sampling/token-limit overrides. Submit requests through your own Vertex project with the intended model identifier. Billing, credentials, cloud resources and model availability are configured outside this repository; no automatic job submission occurs. For video-only requests, use a different `--gcs-dir` containing audio-stripped clips and retain the raw file.

Responses must be mapped back by the request `key` (gold filename), parsed with `evaluation/parse_output.py`, and checked for complete cohort coverage before scoring. Each open runner already writes this schema under its output directory's `verified/` subfolder:

```json
{"filename":"clip.mp4","nodes":[],"edges":[],"valid":false}
```

Score one complete run with `python -m evaluation.score --predictions outputs/qwen3_think/verified/qwen3-omni-30b-think_raw.jsonl --split full`. A 197-clip held-out run uses `--split test`. Missing clips must be resolved or explicitly represented as failed predictions; the scorer rejects incomplete coverage.
