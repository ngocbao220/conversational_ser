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

CONFIG_PATH="${1:-${CONFIG_PATH:-configs/wavlm_baseline_no_mal_no_tim.yaml}}"

"$PYTHON_BIN" -m scripts.train_wavlm_baseline --config "$CONFIG_PATH"
