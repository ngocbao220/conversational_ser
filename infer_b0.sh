#!/usr/bin/env bash
set -euo pipefail

# =========================
# B0 inference parameters
# =========================

AUDIO_PATH="data/neutral.wav"
CHECKPOINT="outputs/b0_utterance/best.pt"
DEVICE="auto"

# =========================
# Do not edit below
# =========================

python infer_b0.py \
  --audio "$AUDIO_PATH" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE"
