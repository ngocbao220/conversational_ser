#!/usr/bin/env bash
set -euo pipefail

# =========================
# Hugging Face upload parameters
# =========================

# Model alias used to choose the folder inside the shared Hugging Face repo.
# Override examples:
#   ./upload.sh --model b0
#   MODEL_NAME=b0 ./upload.sh
MODEL_NAME="${MODEL_NAME:-b0}"

# Shared repo for all SER checkpoint versions.
HF_REPO_ID="${HF_REPO_ID:-ngocbao05/ser}"

# Upload the whole training output directory so the checkpoint stays together
# with run_config.json, history.json, train.log, and metrics if present.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/b0_utterance}"
PATH_IN_REPO="${PATH_IN_REPO:-$MODEL_NAME}"

REPO_TYPE="${REPO_TYPE:-model}"
PRIVATE="${PRIVATE:-false}"
REVISION="${REVISION:-main}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-upload ${MODEL_NAME} checkpoint}"

# =========================
# Do not edit below
# =========================

usage() {
  cat <<'EOF'
Usage:
  ./upload.sh [--model MODEL_NAME] [--repo-id OWNER/REPO] [--output-dir PATH]

Environment overrides:
  MODEL_NAME=b0
  HF_REPO_ID=ngocbao05/ser
  OUTPUT_DIR=outputs/b0_utterance
  PATH_IN_REPO=b0
  PRIVATE=false
  REVISION=main
  COMMIT_MESSAGE="upload b0 checkpoint"

Prerequisites:
  hf auth login
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_NAME="${2#MODEL_NAME=}"
      shift 2
      ;;
    --model=*)
      MODEL_NAME="${1#--model=}"
      MODEL_NAME="${MODEL_NAME#MODEL_NAME=}"
      shift
      ;;
    MODEL_NAME=*)
      MODEL_NAME="${1#MODEL_NAME=}"
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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#--output-dir=}"
      shift
      ;;
    OUTPUT_DIR=*)
      OUTPUT_DIR="${1#OUTPUT_DIR=}"
      shift
      ;;
    --path-in-repo)
      PATH_IN_REPO="$2"
      shift 2
      ;;
    --path-in-repo=*)
      PATH_IN_REPO="${1#--path-in-repo=}"
      shift
      ;;
    --private)
      if [[ $# -gt 1 && "$2" != --* ]]; then
        PRIVATE="$2"
        shift 2
      else
        PRIVATE=true
        shift
      fi
      ;;
    --private=*)
      PRIVATE="${1#--private=}"
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

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output directory not found: $OUTPUT_DIR" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with: python -m pip install huggingface_hub" >&2
  exit 1
fi

CREATE_ARGS=(--type "$REPO_TYPE")
if [[ "$PRIVATE" == "true" ]]; then
  CREATE_ARGS+=(--private)
fi

hf repo create "$HF_REPO_ID" "${CREATE_ARGS[@]}" --exist-ok

echo "Uploading $OUTPUT_DIR to Hugging Face repo $HF_REPO_ID:$PATH_IN_REPO"
hf upload \
  "$HF_REPO_ID" \
  "$OUTPUT_DIR" \
  "$PATH_IN_REPO" \
  --repo-type "$REPO_TYPE" \
  --revision "$REVISION" \
  --commit-message "$COMMIT_MESSAGE"

echo "Done: https://huggingface.co/$HF_REPO_ID"
