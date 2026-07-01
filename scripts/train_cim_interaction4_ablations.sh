#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/cim_branch_concat_interaction4_no_relative_gap_loso.yaml"
  "configs/cim_branch_concat_interaction4_no_overlap_ratio_loso.yaml"
  "configs/cim_branch_concat_interaction4_no_speaker_switch_loso.yaml"
  "configs/cim_branch_concat_interaction4_no_speaker_overlap_style_loso.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python scripts/train_dual_branch.py --config "${config}"
done
