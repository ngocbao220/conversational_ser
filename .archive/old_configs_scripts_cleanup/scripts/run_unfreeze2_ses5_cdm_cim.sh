#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/unfreeze2_ses5_wavlm_cdm_cim.yaml"
  "configs/unfreeze2_ses5_hubert_cdm_cim.yaml"
  "configs/unfreeze2_ses5_wav2vec_cdm_cim.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  python -m scripts.train_dual_branch --config "${config}"
done
