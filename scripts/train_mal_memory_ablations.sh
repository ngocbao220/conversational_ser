#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/wavlm_mal_zero_state_loso.yaml"
  "configs/wavlm_mal_no_update_loso.yaml"
  "configs/wavlm_mal_shuffled_memory_loso.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python scripts/train_wavlm_mal.py --config "${config}"
done
