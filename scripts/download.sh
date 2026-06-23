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
# Hugging Face download + eval parameters
# =========================

# Model/version folder inside the shared Hugging Face repo.
# Override examples:
#   ./scripts/download.sh --model wavlm_tim
#   MODEL_NAME=wavlm_tim ./scripts/download.sh
MODEL_NAME="${MODEL_NAME:-wavlm_tim}"

HF_REPO_ID="${HF_REPO_ID:-ngocbao05/ser}"
PATH_IN_REPO="${PATH_IN_REPO:-$MODEL_NAME}"
REPO_TYPE="${REPO_TYPE:-model}"
REVISION="${REVISION:-main}"

# Download target. Repository folders are preserved under this root.
# Final checkpoint path defaults to `results/wavlm_tim/best.pth`.
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-results}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$DOWNLOAD_ROOT/$MODEL_NAME}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pth}"

# Evaluation parameters.
# The bundled evaluator is for legacy B0 checkpoints. Keep download independent
# so WavLM/MAL/TIM artifact downloads do not fail after completing.
RUN_EVAL="${RUN_EVAL:-false}"
SPLIT="${SPLIT:-test}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$LOCAL_MODEL_DIR/${SPLIT}_metrics.json}"
DEVICE="${DEVICE:-auto}"

DATASET_NAME="${DATASET_NAME:-AbstractTTS/IEMOCAP}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-}"
TEST_SESSION="${TEST_SESSION:-Ses05}"
SAMPLING_RATE="${SAMPLING_RATE:-16000}"
MAX_DURATION_SECONDS="${MAX_DURATION_SECONDS:-12.0}"
VALIDATION_SIZE="${VALIDATION_SIZE:-0.1}"
TEST_SIZE="${TEST_SIZE:-0.1}"
SEED="${SEED:-42}"
NUM_PROC="${NUM_PROC:-1}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROGRESS_BAR="${PROGRESS_BAR:-true}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"

# =========================
# Do not edit below
# =========================

usage() {
  cat <<'EOF'
Usage:
  ./scripts/download.sh [--model MODEL_NAME] [--repo-id OWNER/REPO] [--download-root PATH]

Environment overrides:
  MODEL_NAME=wavlm_tim
  HF_REPO_ID=ngocbao05/ser
  PATH_IN_REPO=wavlm_tim
  DOWNLOAD_ROOT=results
  CHECKPOINT_NAME=best.pth
  SPLIT_STRATEGY=random
  TEST_SESSION=Ses05
  RUN_EVAL=false
  SPLIT=test
  DEVICE=auto
  FORCE_DOWNLOAD=false

Prerequisites:
  hf auth login
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_NAME="${2#MODEL_NAME=}"
      PATH_IN_REPO="$MODEL_NAME"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift 2
      ;;
    --model=*)
      MODEL_NAME="${1#--model=}"
      MODEL_NAME="${MODEL_NAME#MODEL_NAME=}"
      PATH_IN_REPO="$MODEL_NAME"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift
      ;;
    MODEL_NAME=*)
      MODEL_NAME="${1#MODEL_NAME=}"
      PATH_IN_REPO="$MODEL_NAME"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift
      ;;
    --repo-id)
      HF_REPO_ID="$2"
      shift 2
      ;;
    --repo-id=*)
      HF_REPO_ID="${1#--repo-id=}"
      shift
      ;;
    HF_REPO_ID=*)
      HF_REPO_ID="${1#HF_REPO_ID=}"
      shift
      ;;
    --path-in-repo)
      PATH_IN_REPO="$2"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$PATH_IN_REPO"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift 2
      ;;
    --path-in-repo=*)
      PATH_IN_REPO="${1#--path-in-repo=}"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$PATH_IN_REPO"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift
      ;;
    --download-root)
      DOWNLOAD_ROOT="$2"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift 2
      ;;
    --download-root=*)
      DOWNLOAD_ROOT="${1#--download-root=}"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift
      ;;
    --local-model-dir)
      LOCAL_MODEL_DIR="$2"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift 2
      ;;
    --local-model-dir=*)
      LOCAL_MODEL_DIR="${1#--local-model-dir=}"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift
      ;;
    --checkpoint-name)
      CHECKPOINT_NAME="$2"
      shift 2
      ;;
    --checkpoint-name=*)
      CHECKPOINT_NAME="${1#--checkpoint-name=}"
      shift
      ;;
    --split)
      SPLIT="$2"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift 2
      ;;
    --split=*)
      SPLIT="${1#--split=}"
      EVAL_OUTPUT="$LOCAL_MODEL_DIR/${SPLIT}_metrics.json"
      shift
      ;;
    --output)
      EVAL_OUTPUT="$2"
      shift 2
      ;;
    --output=*)
      EVAL_OUTPUT="${1#--output=}"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#--device=}"
      shift
      ;;
    --run-eval)
      if [[ $# -gt 1 && "$2" != --* ]]; then
        RUN_EVAL="$2"
        shift 2
      else
        RUN_EVAL=true
        shift
      fi
      ;;
    --run-eval=*)
      RUN_EVAL="${1#--run-eval=}"
      shift
      ;;
    --force-download)
      FORCE_DOWNLOAD=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$PATH_IN_REPO" == /* || "$PATH_IN_REPO" == *".."* ]]; then
  echo "PATH_IN_REPO must be a relative repository folder: $PATH_IN_REPO" >&2
  exit 2
fi

if [[ -z "$SPLIT_STRATEGY" ]]; then
  case "$MODEL_NAME" in
    b01)
      SPLIT_STRATEGY="loso"
      ;;
    *)
      SPLIT_STRATEGY="random"
      ;;
  esac
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with: python -m pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_ROOT"

DOWNLOAD_ARGS=(
  "$HF_REPO_ID"
  --repo-type "$REPO_TYPE"
  --revision "$REVISION"
  --local-dir "$DOWNLOAD_ROOT"
  --include "$PATH_IN_REPO/**"
  --exclude "$PATH_IN_REPO/wandb/**"
)
if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
  DOWNLOAD_ARGS+=(--force-download)
fi

echo "Downloading Hugging Face repo folder $HF_REPO_ID/$PATH_IN_REPO to $DOWNLOAD_ROOT/$PATH_IN_REPO"
hf download "${DOWNLOAD_ARGS[@]}"

CHECKPOINT="$LOCAL_MODEL_DIR/$CHECKPOINT_NAME"
if [[ ! -f "$CHECKPOINT" && -f "$DOWNLOAD_ROOT/$CHECKPOINT_NAME" ]]; then
  CHECKPOINT="$DOWNLOAD_ROOT/$CHECKPOINT_NAME"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found after download: $LOCAL_MODEL_DIR/$CHECKPOINT_NAME" >&2
  exit 1
fi

echo "Downloaded checkpoint: $CHECKPOINT"

if [[ "$RUN_EVAL" != "true" ]]; then
  exit 0
fi

EVAL_ARGS=(
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --output "$EVAL_OUTPUT"
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
  EVAL_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_VALIDATION_SAMPLES" ]]; then
  EVAL_ARGS+=(--max-validation-samples "$MAX_VALIDATION_SAMPLES")
fi
if [[ -n "$MAX_TEST_SAMPLES" ]]; then
  EVAL_ARGS+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi

echo "Evaluating checkpoint on split=$SPLIT"
"$PYTHON_BIN" -m scripts.evaluate_b0 "${EVAL_ARGS[@]}"
