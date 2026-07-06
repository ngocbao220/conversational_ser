#!/usr/bin/env bash
set -euo pipefail

run_config() {
  local config="$1"
  case "$config" in
    *baseline.yaml) python -m scripts.train_baseline --config "$config" ;;
    *cdm_cim.yaml|*cim.yaml) python -m scripts.train_dual_branch --config "$config" ;;
    *cdm.yaml) python -m scripts.train_cdm --config "$config" ;;
    *) echo "Unknown config: $config" >&2; exit 1 ;;
  esac
}

CONFIGS=(
  "configs/strict4/main_wavlm_baseline.yaml"
  "configs/strict4/main_wavlm_cdm.yaml"
  "configs/strict4/main_wavlm_cim.yaml"
  "configs/strict4/main_wavlm_cdm_cim.yaml"
  "configs/strict4/main_hubert_baseline.yaml"
  "configs/strict4/main_hubert_cdm.yaml"
  "configs/strict4/main_hubert_cim.yaml"
  "configs/strict4/main_hubert_cdm_cim.yaml"
  "configs/strict4/main_wav2vec_baseline.yaml"
  "configs/strict4/main_wav2vec_cdm.yaml"
  "configs/strict4/main_wav2vec_cim.yaml"
  "configs/strict4/main_wav2vec_cdm_cim.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running $config"
  run_config "$config"
done
