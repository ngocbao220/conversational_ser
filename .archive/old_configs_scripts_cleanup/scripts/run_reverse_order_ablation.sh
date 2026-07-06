#!/usr/bin/env bash
set -euo pipefail

echo "Running CDM reverse-order memory ablation"
python -m scripts.train_cdm --config configs/cdm_ablation_reverse.yaml

echo "Running CIM reverse-order interaction ablation"
python -m scripts.train_dual_branch --config configs/cim_ablation_reverse.yaml
