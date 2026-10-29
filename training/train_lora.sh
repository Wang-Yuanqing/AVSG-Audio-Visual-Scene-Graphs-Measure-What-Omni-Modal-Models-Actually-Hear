#!/usr/bin/env bash
# Portable transcription of the two campaign ms-swift launch recipes.
# Usage: bash training/train_lora.sh qwen25|qwen3 CHECKPOINT DATA_JSONL OUTPUT_DIR
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 qwen25|qwen3 CHECKPOINT DATA_JSONL OUTPUT_DIR" >&2
  exit 2
fi
model_kind=$1
model_path=$2
dataset_path=$3
output_path=$4
case "$model_kind" in
  qwen25) model_type=qwen2_5_omni; export USE_AUDIO_IN_VIDEO=false ;;
  qwen3) model_type=qwen3_omni_moe; export USE_AUDIO_IN_VIDEO=true ;;
  *) echo "Expected qwen25 or qwen3" >&2; exit 2 ;;
esac
export FPS=2 FPS_MAX_FRAMES=16 VIDEO_MAX_PIXELS=200704
export ENABLE_AUDIO_OUTPUT=false TOKENIZERS_PARALLELISM=false
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
swift sft \
  --model "$model_path" --model_type "$model_type" \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --freeze_vit true --freeze_aligner true \
  --dataset "$dataset_path" --split_dataset_ratio 0 \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --num_train_epochs 2 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 --learning_rate 1e-4 \
  --max_length 8192 --gradient_checkpointing true \
  --logging_steps 5 --save_strategy steps --save_steps 50 --save_total_limit 2 \
  --output_dir "$output_path" "${resume_args[@]}"
