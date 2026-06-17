#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# =========================
# B01 eval parameters
# =========================

CHECKPOINT="outputs/b01_loso_unfreeze4/best.pt"
SPLIT="test"
OUTPUT="outputs/b01_loso_unfreeze4/test_metrics.json"
DEVICE="auto"

DATASET_NAME="AbstractTTS/IEMOCAP"
SPLIT_STRATEGY="loso"
TEST_SESSION="Ses05"
SAMPLING_RATE=16000
MAX_DURATION_SECONDS=12.0
VALIDATION_SIZE=0.1
TEST_SIZE=0.0
SEED=42
NUM_PROC=1

MAX_TRAIN_SAMPLES=""
MAX_VALIDATION_SAMPLES=""
MAX_TEST_SAMPLES=""

EVAL_BATCH_SIZE=8
NUM_WORKERS=4
PROGRESS_BAR=true

# =========================
# Do not edit below
# =========================

ARGS=(
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --output "$OUTPUT"
  --device "$DEVICE"
  --dataset-name "$DATASET_NAME"
  --split-strategy "$SPLIT_STRATEGY"
  --test-session "$TEST_SESSION"
  --sampling-rate "$SAMPLING_RATE"
  --max-duration-seconds "$MAX_DURATION_SECONDS"
  --validation-size "$VALIDATION_SIZE"
  --test-size "$TEST_SIZE"
  --seed "$SEED"
  --num-proc "$NUM_PROC"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --progress-bar "$PROGRESS_BAR"
)

if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_VALIDATION_SAMPLES" ]]; then
  ARGS+=(--max-validation-samples "$MAX_VALIDATION_SAMPLES")
fi
if [[ -n "$MAX_TEST_SAMPLES" ]]; then
  ARGS+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi

python -m scripts.evaluate_b0 "${ARGS[@]}"
