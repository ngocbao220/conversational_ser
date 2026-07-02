#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/architecture_residual_gate.yaml"
  "configs/architecture_residual_sum.yaml"
  "configs/architecture_branch_sum.yaml"
  "configs/architecture_branch_concat.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
