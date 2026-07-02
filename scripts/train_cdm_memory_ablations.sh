#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/wavlm_cdm_zero_state_loso.yaml"
  "configs/wavlm_cdm_no_update_loso.yaml"
  "configs/wavlm_cdm_shuffled_memory_loso.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_wavlm_cdm --config "${config}"
done
