#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# =========================
# B0 inference parameters
# =========================

AUDIO_PATH="data/neutral.wav"
CHECKPOINT="outputs/b0_utterance/best.pt"
DEVICE="auto"

# =========================
# Do not edit below
# =========================

python -m scripts.infer_b0 \
  --audio "$AUDIO_PATH" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE"
