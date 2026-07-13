#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# =========================
# Hugging Face download parameters
# =========================

MODEL_NAME="${MODEL_NAME:-cdim}"
HF_REPO_ID="${HF_REPO_ID:-ngocbao05/ser}"
PATH_IN_REPO="${PATH_IN_REPO:-$MODEL_NAME}"
REPO_TYPE="${REPO_TYPE:-model}"
REVISION="${REVISION:-main}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-results}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-$DOWNLOAD_ROOT/$MODEL_NAME}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pth}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/download.sh [--model MODEL_NAME] [--repo-id OWNER/REPO] [--download-root PATH]

Environment overrides:
  MODEL_NAME=cdim
  HF_REPO_ID=ngocbao05/ser
  PATH_IN_REPO=cdim
  DOWNLOAD_ROOT=results
  CHECKPOINT_NAME=best.pth
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
      shift 2
      ;;
    --model=*)
      MODEL_NAME="${1#--model=}"
      MODEL_NAME="${MODEL_NAME#MODEL_NAME=}"
      PATH_IN_REPO="$MODEL_NAME"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      shift
      ;;
    MODEL_NAME=*)
      MODEL_NAME="${1#MODEL_NAME=}"
      PATH_IN_REPO="$MODEL_NAME"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
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
      shift 2
      ;;
    --path-in-repo=*)
      PATH_IN_REPO="${1#--path-in-repo=}"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$PATH_IN_REPO"
      shift
      ;;
    PATH_IN_REPO=*)
      PATH_IN_REPO="${1#PATH_IN_REPO=}"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$PATH_IN_REPO"
      shift
      ;;
    --download-root)
      DOWNLOAD_ROOT="$2"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      shift 2
      ;;
    --download-root=*)
      DOWNLOAD_ROOT="${1#--download-root=}"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      shift
      ;;
    DOWNLOAD_ROOT=*)
      DOWNLOAD_ROOT="${1#DOWNLOAD_ROOT=}"
      LOCAL_MODEL_DIR="$DOWNLOAD_ROOT/$MODEL_NAME"
      shift
      ;;
    --local-model-dir)
      LOCAL_MODEL_DIR="$2"
      shift 2
      ;;
    --local-model-dir=*)
      LOCAL_MODEL_DIR="${1#--local-model-dir=}"
      shift
      ;;
    LOCAL_MODEL_DIR=*)
      LOCAL_MODEL_DIR="${1#LOCAL_MODEL_DIR=}"
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
    CHECKPOINT_NAME=*)
      CHECKPOINT_NAME="${1#CHECKPOINT_NAME=}"
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
