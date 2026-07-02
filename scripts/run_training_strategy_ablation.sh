#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/training_strategy_cdm_then_cim.yaml"
  "configs/training_strategy_cim_then_cdm.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
