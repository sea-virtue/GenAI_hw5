#!/usr/bin/env bash
set -euo pipefail

INPUT_DATA="${INPUT_DATA:-data/processed/thu_qa_manual_all.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/processed/eval.jsonl}"

if [[ "${FORCE_CRAWL:-0}" == "1" ]]; then
  echo "FORCE_CRAWL is no longer part of the default curated-QA training pipeline." >&2
  echo "Run scripts/crawl_thu.py separately, then manually curate QA before build_dataset.py." >&2
  exit 1
else
  echo "reuse existing QA data: $INPUT_DATA"
fi

python scripts/build_dataset.py \
  --input "$INPUT_DATA" \
  --train-output "$TRAIN_DATA" \
  --eval-output "$EVAL_DATA" \
  --eval-size "${EVAL_SIZE:-50}" \
  --split-mode "${SPLIT_MODE:-qa-paraphrase}"

python scripts/train_lora.py \
  --model-name "${MODEL_NAME:-Qwen/Qwen3-0.6B}" \
  --train-file "$TRAIN_DATA" \
  --eval-file "$EVAL_DATA" \
  --output-dir "${OUTPUT_DIR:-outputs/qwen3-0.6b-thuqa-lora}" \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing

python scripts/evaluate.py \
  --model-name "${MODEL_NAME:-Qwen/Qwen3-0.6B}" \
  --adapter-dir "${OUTPUT_DIR:-outputs/qwen3-0.6b-thuqa-lora}" \
  --eval-file "$EVAL_DATA" \
  --output-dir outputs/eval \
  --bf16
