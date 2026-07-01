#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/dual_branch_fusion_residual_gated.yaml"
  "configs/dual_branch_fusion_residual_sum.yaml"
  "configs/dual_branch_fusion_branch_sum.yaml"
  "configs/dual_branch_fusion_dialogue_only.yaml"
  "configs/dual_branch_fusion_temporal_only.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python scripts/train_dual_branch.py --config "${config}"
done
