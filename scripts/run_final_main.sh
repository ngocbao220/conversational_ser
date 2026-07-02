#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/main_wavlm_baseline.yaml"
  "configs/main_wavlm_cdm.yaml"
  "configs/main_wavlm_cim.yaml"
  "configs/main_wavlm_cdm_cim.yaml"
  "configs/main_wav2vec_baseline.yaml"
  "configs/main_wav2vec_cdm.yaml"
  "configs/main_wav2vec_cim.yaml"
  "configs/main_wav2vec_cdm_cim.yaml"
  "configs/main_hubert_baseline.yaml"
  "configs/main_hubert_cdm.yaml"
  "configs/main_hubert_cim.yaml"
  "configs/main_hubert_cdm_cim.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  case "${config}" in
    *baseline*) python -m scripts.train_wavlm_baseline --config "${config}" ;;
    *cdm_cim*|*cim*) python -m scripts.train_dual_branch --config "${config}" ;;
    *cdm*) python -m scripts.train_wavlm_cdm --config "${config}" ;;
    *) echo "Unknown config type: ${config}" >&2; exit 1 ;;
  esac
done
