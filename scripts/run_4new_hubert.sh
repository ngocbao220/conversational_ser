#!/usr/bin/env bash
set -euo pipefail

# Sequential 4-new feature runs for hubert.
# Includes baseline, CDM, CIM full, and CIM ablations.

python -m scripts.train_baseline --config configs/correct_data/features/4new/hubert_baseline.yaml
python -m scripts.train_cdm --config configs/correct_data/features/4new/hubert_cdm.yaml
python -m scripts.train_cim --config configs/correct_data/features/4new/hubert_cim.yaml

python -m scripts.train_cim --config configs/correct_data/features/4new/hubert_cim_acoustic_only.yaml
python -m scripts.train_cim --config configs/correct_data/features/4new/hubert_cim_zero_temporal.yaml
python -m scripts.train_cim --config configs/correct_data/features/4new/hubert_cim_temporal_only.yaml
python -m scripts.train_cim --config configs/correct_data/features/4new/hubert_cim_shuffled_temporal.yaml
