#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/processed/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-1.7b-thuqa-lora}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/eval-qwen3-1.7b}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

if [[ ! -s "$TRAIN_DATA" ]]; then
  echo "missing train data: $TRAIN_DATA" >&2
  echo "run scripts/build_dataset.py first." >&2
  exit 1
fi

if [[ ! -s "$EVAL_DATA" ]]; then
  echo "missing eval data: $EVAL_DATA" >&2
  echo "run scripts/build_dataset.py first." >&2
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "model=$MODEL_NAME"
echo "train=$TRAIN_DATA"
echo "eval=$EVAL_DATA"
echo "adapter_output=$OUTPUT_DIR"
echo "eval_output=$EVAL_OUTPUT_DIR"

python scripts/train_lora.py \
  --model-name "$MODEL_NAME" \
  --train-file "$TRAIN_DATA" \
  --eval-file "$EVAL_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing \
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
  --max-length "$MAX_LENGTH" \
  --num-train-epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA"

python scripts/evaluate.py \
  --model-name "$MODEL_NAME" \
  --adapter-dir "$OUTPUT_DIR" \
  --eval-file "$EVAL_DATA" \
  --output-dir "$EVAL_OUTPUT_DIR" \
  --bf16

echo "done."
echo "adapter: $OUTPUT_DIR"
echo "metrics: $EVAL_OUTPUT_DIR/metrics.json"
echo "chat:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python scripts/chat_lora.py --model-name $MODEL_NAME --adapter-dir $OUTPUT_DIR --load-in-4bit --bf16 --no-thinking"
