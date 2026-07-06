#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/cdm_ablation_zero.yaml"
  "configs/cdm_ablation_shuffled.yaml"
  "configs/cdm_ablation_full.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_cdm --config "${config}"
done
