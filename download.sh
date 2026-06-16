#!/usr/bin/env bash
set -euo pipefail

# =========================
# Hugging Face download + eval parameters
# =========================

# Model/version folder inside the shared Hugging Face repo.
# Override examples:
#   ./download.sh --model b0
#   MODEL_NAME=b0 ./download.sh
MODEL_NAME="${MODEL_NAME:-b0}"

HF_REPO_ID="${HF_REPO_ID:-ngocbao05/ser}"
PATH_IN_REPO="${PATH_IN_REPO:-$MODEL_NAME}"
REPO_TYPE="${REPO_TYPE:-model}"
REVISION="${REVISION:-main}"

# Download target. Final checkpoint path defaults to:
#   outputs/hf_checkpoints/<MODEL_NAME>/best.pt
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-outputs/hf_checkpoints}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$DOWNLOAD_ROOT/$MODEL_NAME}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"

# Evaluation parameters.
RUN_EVAL="${RUN_EVAL:-true}"
SPLIT="${SPLIT:-test}"
EVAL_OUTPUT="${EVAL_OUTPUT:-$LOCAL_MODEL_DIR/${SPLIT}_metrics.json}"
DEVICE="${DEVICE:-auto}"

DATASET_NAME="${DATASET_NAME:-AbstractTTS/IEMOCAP}"
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
  ./download.sh [--model MODEL_NAME] [--repo-id OWNER/REPO] [--download-root PATH]

Environment overrides:
  MODEL_NAME=b0
  HF_REPO_ID=ngocbao05/ser
  PATH_IN_REPO=b0
  DOWNLOAD_ROOT=outputs/hf_checkpoints
  RUN_EVAL=true
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

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with: python -m pip install huggingface_hub" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_ROOT"

DOWNLOAD_ARGS=(
  "$HF_REPO_ID"
  "$PATH_IN_REPO/"
  --repo-type "$REPO_TYPE"
  --revision "$REVISION"
  --local-dir "$DOWNLOAD_ROOT"
)
if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
  DOWNLOAD_ARGS+=(--force-download)
fi

echo "Downloading Hugging Face repo path $HF_REPO_ID/$PATH_IN_REPO to $DOWNLOAD_ROOT"
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
python evaluate_b0.py "${EVAL_ARGS[@]}"
