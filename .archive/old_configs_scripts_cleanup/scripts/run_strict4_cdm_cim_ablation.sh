#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "configs/strict4/cdm_cim_ablation_wavlm_zero_dialogue_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wavlm_zero_interaction_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wavlm_shuffled_dialogue_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wavlm_shuffled_interaction_memory.yaml"
  "configs/strict4/cdm_cim_ablation_hubert_zero_dialogue_memory.yaml"
  "configs/strict4/cdm_cim_ablation_hubert_zero_interaction_memory.yaml"
  "configs/strict4/cdm_cim_ablation_hubert_shuffled_dialogue_memory.yaml"
  "configs/strict4/cdm_cim_ablation_hubert_shuffled_interaction_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wav2vec_zero_dialogue_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wav2vec_zero_interaction_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wav2vec_shuffled_dialogue_memory.yaml"
  "configs/strict4/cdm_cim_ablation_wav2vec_shuffled_interaction_memory.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Running $config"
  python -m scripts.train_dual_branch --config "$config"
done
