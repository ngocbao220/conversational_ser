# CDIM Conversational Speech Emotion Recognition

This repository contains the code, demo, and paper assets for SSL-based conversational speech emotion recognition with Conversational Dialogue Memory (CDM) and Conversational Dialogue-Interaction Memory (CDIM).

The current experiments use the corrected IEMOCAP label policy:

- target labels: `ang -> angry`, `hap + exc -> happy`, `neu -> neutral`, `sad -> sad`
- other IEMOCAP labels stay in the dialogue as `context` / `ignore_index=-100`
- temporal features and memory are computed on the full dialogue timeline
- loss and metrics are computed only on the 4 target emotion classes

## Repository Layout

```text
configs/correct_data/     LOSO configs for main and ablation runs
models/                   baseline, CDM, and CDIM model definitions
scripts/                  training, download, and figure-generation scripts
utils/                    dataset loading, features, metrics, and shared helpers
demo/                     browser demo with final predictions
reports/                  LaTeX paper source, figures, tables, slides, and PDF
notebooks/                analysis notebooks
```

Large local data and full training outputs are intentionally not part of the clean package:

```text
data/
results/
```

## Active Configs

Use only the cleaned config set:

```text
configs/correct_data/main/
configs/correct_data/cdim_ablations/
```

Outputs are written to:

```text
results/correct_data/main/
results/correct_data/cdim_ablations/
results/correct_data/shared_cache/
```

## Train Commands

Baseline:

```bash
python -m scripts.train_baseline --config configs/correct_data/main/wavlm_baseline.yaml
```

CDM:

```bash
python -m scripts.train_cdm --config configs/correct_data/main/wavlm_cdm.yaml
```

CDIM:

```bash
python -m scripts.train_cdim --config configs/correct_data/main/wavlm_cim.yaml
```

Swap `wavlm` with `hubert` or `wav2vec` for the other SSL backbones.

## Ablation Example

```bash
python -m scripts.train_cdim --config configs/correct_data/cdim_ablations/wavlm_zero_dialogue_memory.yaml
python -m scripts.train_cdim --config configs/correct_data/cdim_ablations/wavlm_zero_interaction_memory.yaml
python -m scripts.train_cdim --config configs/correct_data/cdim_ablations/wavlm_shuffled_dialogue_memory.yaml
python -m scripts.train_cdim --config configs/correct_data/cdim_ablations/wavlm_shuffled_interaction_memory.yaml
```

## Demo

The final demo is self-contained in `demo/` and loads `demo/demo_data.json`.

```bash
cd demo
python3 -m http.server 8000
```

Open `http://localhost:8000/index.html`.

## Report

The paper source is in `reports/`. Build it with:

```bash
cd reports
pdflatex -interaction=nonstopmode main.tex
```

The compiled PDF is `reports/main.pdf`.
