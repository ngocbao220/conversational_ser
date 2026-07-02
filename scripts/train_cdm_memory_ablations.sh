#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/cdm_ablation_zero_state.yaml"
  "configs/cdm_ablation_no_update.yaml"
  "configs/cdm_ablation_shuffled_memory.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_wavlm_cdm --config "${config}"
done
