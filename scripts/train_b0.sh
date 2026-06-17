#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python not found. Activate your env or set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

# =========================
# B0 train parameters
# =========================

DATASET_NAME="AbstractTTS/IEMOCAP"
SPLIT_STRATEGY="random"
TEST_SESSION="Ses05"
SAMPLING_RATE=16000
MAX_DURATION_SECONDS=12.0
VALIDATION_SIZE=0.1
TEST_SIZE=0.1
SEED=42
NUM_PROC=1

# Set to an integer for smoke tests, or empty for full data.
MAX_TRAIN_SAMPLES=""
MAX_VALIDATION_SAMPLES=""
MAX_TEST_SAMPLES=""

ENCODER_NAME="microsoft/wavlm-base"
BASELINE_NAME="B0_utterance"
POOLING="mean"
FREEZE_ENCODER=true
TRAINABLE_ENCODER_LAYERS=0
DROPOUT=0.2
HIDDEN_DIM=256

OUTPUT_DIR="outputs/b0_utterance"
BATCH_SIZE=16
EVAL_BATCH_SIZE=8
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.01
EPOCHS=50
GRADIENT_ACCUMULATION_STEPS=1
MAX_GRAD_NORM=1.0
LR_SCHEDULER="cosine"   # linear, cosine, constant
WARMUP_RATIO=0.1
EARLY_STOPPING_PATIENCE=10  # 0 disables early stopping
EARLY_STOPPING_MIN_DELTA=0.001
NUM_WORKERS=2
DEVICE="cuda"

PROGRESS_BAR=true
PROGRESS_MININTERVAL=2.0
LOG_EVERY_STEPS=50
LOG_FILE="train.log"

USE_WANDB=true
WANDB_PROJECT="conversational-SER"
WANDB_RUN_NAME="b0-wavlm"
WANDB_ENTITY=""
WANDB_MODE="online"

# =========================
# Do not edit below
# =========================

ARGS=(
  --dataset-name "$DATASET_NAME"
  --split-strategy "$SPLIT_STRATEGY"
  --test-session "$TEST_SESSION"
  --sampling-rate "$SAMPLING_RATE"
  --max-duration-seconds "$MAX_DURATION_SECONDS"
  --validation-size "$VALIDATION_SIZE"
  --test-size "$TEST_SIZE"
  --seed "$SEED"
  --num-proc "$NUM_PROC"
  --baseline-name "$BASELINE_NAME"
  --encoder-name "$ENCODER_NAME"
  --pooling "$POOLING"
  --freeze-encoder "$FREEZE_ENCODER"
  --trainable-encoder-layers "$TRAINABLE_ENCODER_LAYERS"
  --dropout "$DROPOUT"
  --hidden-dim "$HIDDEN_DIM"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --max-grad-norm "$MAX_GRAD_NORM"
  --lr-scheduler "$LR_SCHEDULER"
  --warmup-ratio "$WARMUP_RATIO"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --progress-bar "$PROGRESS_BAR"
  --progress-mininterval "$PROGRESS_MININTERVAL"
  --log-every-steps "$LOG_EVERY_STEPS"
  --log-file "$LOG_FILE"
  --use-wandb "$USE_WANDB"
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
  --wandb-mode "$WANDB_MODE"
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
if [[ -n "$WANDB_ENTITY" ]]; then
  ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

"$PYTHON_BIN" -m scripts.train_b0 "${ARGS[@]}"
