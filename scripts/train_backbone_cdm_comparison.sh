#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/backbone_wav2vec2_baseline_loso.yaml"
  "configs/backbone_wav2vec2_cdm_loso.yaml"
  "configs/backbone_hubert_baseline_loso.yaml"
  "configs/backbone_hubert_cdm_loso.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running ${config}"
  case "${config}" in
    *baseline*) python -m scripts.train_wavlm_baseline --config "${config}" ;;
    *cdm*) python -m scripts.train_wavlm_cdm --config "${config}" ;;
    *) echo "Unknown config type: ${config}" >&2; exit 1 ;;
  esac
done
