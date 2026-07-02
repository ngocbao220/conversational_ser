#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/fusion_residual_gated.yaml"
  "configs/fusion_residual_sum.yaml"
  "configs/fusion_branch_sum.yaml"
  "configs/fusion_dialogue_only.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
