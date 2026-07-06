#!/usr/bin/env bash
set -euo pipefail

declare -A TRAINERS=(
  ["configs/unfreeze4_ses5_wavlm_baseline.yaml"]="scripts.train_baseline"
  ["configs/unfreeze4_ses5_wavlm_cdm.yaml"]="scripts.train_cdm"
  ["configs/unfreeze4_ses5_wavlm_cim.yaml"]="scripts.train_dual_branch"
)

CONFIGS=(
  "configs/unfreeze4_ses5_wavlm_baseline.yaml"
  "configs/unfreeze4_ses5_wavlm_cdm.yaml"
  "configs/unfreeze4_ses5_wavlm_cim.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m "${TRAINERS[${config}]}" --config "${config}"
done
