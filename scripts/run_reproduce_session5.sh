#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

echo "[1/4] Train WavLM baseline on Session 5"
python -m scripts.train_baseline --config configs/reproduce_session5/wavlm_baseline_ses05.yaml

echo "[2/4] Train WavLM + CDM on Session 5"
python -m scripts.train_cdm --config configs/reproduce_session5/wavlm_cdm_ses05.yaml

echo "[3/4] Train WavLM + CDIM on Session 5"
python -m scripts.train_cdim --config configs/reproduce_session5/wavlm_cdim_ses05.yaml

echo "[4/4] Compare reproduced metrics with final_result"
python scripts/compare_reproduce_session5.py
