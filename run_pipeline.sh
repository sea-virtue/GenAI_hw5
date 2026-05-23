#!/usr/bin/env bash
set -euo pipefail

RAW_DATA="${RAW_DATA:-data/raw/thu_pages.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-data/processed/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/processed/eval.jsonl}"

if [[ "${FORCE_CRAWL:-0}" == "1" || ! -s "$RAW_DATA" ]]; then
  python scripts/crawl_thu.py \
    --seeds configs/crawl_seed_urls.txt \
    --output "$RAW_DATA" \
    --max-pages "${MAX_PAGES:-120}" \
    --max-depth "${MAX_DEPTH:-2}" \
    --delay "${CRAWL_DELAY:-0.8}"
else
  echo "reuse existing raw data: $RAW_DATA"
fi

python scripts/build_dataset.py \
  --input "$RAW_DATA" \
  --train-output "$TRAIN_DATA" \
  --eval-output "$EVAL_DATA" \
  --eval-size "${EVAL_SIZE:-50}"

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
