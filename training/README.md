# LoRA adaptation

These recipes transcribe the saved campaign launches for Qwen2.5-Omni-7B and Qwen3-Omni-30B-Instruct. They use the official **source-video-disjoint split: 827 train clips from 364 videos, 197 test clips from 91 videos**. The held-out numbers must not be compared directly with the full-corpus 1,024-clip leaderboard.

Install a separate environment with **ms-swift** and the relevant model dependencies. The original command-line options are preserved below; a complete historical ms-swift environment lock was not recovered. This release validates training-record reconstruction and script syntax, not a fresh GPU training run. Use an ms-swift version supporting `qwen2_5_omni` / `qwen3_omni_moe` and these options.

## Prepare records

From the repository root, after restoring media under `clips/`:

```sh
python training/prepare_data.py --model qwen25 --clips-dir clips --output outputs/train_qwen25.jsonl
python training/prepare_data.py --model qwen3 --clips-dir clips --output outputs/train_qwen3.jsonl
```

The builder preserves the exact shared benchmark system prompt, raw user text, saved split order and gold target serialization. Both 827-record outputs were compared with the archived training files: messages, target strings, order and media basenames match exactly (only the local media directory changes). See `reconstruction_check.json`. Qwen2.5 uses `<video><audio>` and supplies the same MP4 separately in `videos` and `audios`; Qwen3 uses `<video>` with in-video audio. No reference caption is appended. `--skip-media-check` can generate records for inspection without media, but does not make them trainable.

## Train

```sh
bash training/train_lora.sh qwen25 /path/to/Qwen2.5-Omni-7B outputs/train_qwen25.jsonl outputs/lora_qwen25
bash training/train_lora.sh qwen3 /path/to/Qwen3-Omni-Instruct outputs/train_qwen3.jsonl outputs/lora_qwen3
```

Shared settings: LoRA rank 16, alpha 32, dropout 0.05; 2 epochs; learning rate `1e-4`; batch size 1 and gradient accumulation 8; maximum sequence length 8,192; BF16 and SDPA; gradient checkpointing; vision encoder and aligner frozen. The media environment uses 2 fps, at most 16 frames and 200,704 pixels per frame. The script preserves the launch's default target-module selection rather than inventing an explicit replacement.

Set `RESUME_FROM_CHECKPOINT` to a saved checkpoint path to resume. Merge a selected adapter into a **new** output directory explicitly:

```sh
swift export --adapters /path/to/checkpoint --merge_lora true --output_dir outputs/merged_qwen25
```

Model and adapter weights are not bundled. Evaluation uses the corresponding [inference runner](../inference/README.md), the same input/sampling protocol as the untuned model, and the 197 test filenames in `data/splits/`. Use `--tag` on Qwen runners to distinguish tuned outputs. For the video-only condition, pass `--audio-mode none`. The saved tuned AV/V predictions are already included in `results/predictions/finetuned/` and `finetuned_noaudio/` for GPU-free analysis.
