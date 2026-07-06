#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/feature_ablation_feature_4.yaml"
  "configs/feature_ablation_feature_8.yaml"
  "configs/feature_ablation_feature_8_turn.yaml"
  "configs/feature_ablation_feature_12.yaml"
  "configs/feature_ablation_feature_16.yaml"
  "configs/feature_ablation_feature_36.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
