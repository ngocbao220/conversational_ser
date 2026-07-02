#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/cim_ablation_zero.yaml"
  "configs/cim_ablation_shuffled.yaml"
  "configs/cim_ablation_no_overlap_ratio.yaml"
  "configs/cim_ablation_no_relative_gap.yaml"
  "configs/cim_ablation_no_speaker_switch.yaml"
  "configs/cim_ablation_no_speaker_overlap_rate.yaml"
  "configs/cim_ablation_full.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
