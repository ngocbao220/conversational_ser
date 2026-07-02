#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/cim_interaction4_branch_concat_no_relative_gap.yaml"
  "configs/cim_interaction4_branch_concat_no_overlap_ratio.yaml"
  "configs/cim_interaction4_branch_concat_no_speaker_switch.yaml"
  "configs/cim_interaction4_branch_concat_no_speaker_overlap_style.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
