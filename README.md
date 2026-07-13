# CDIM Conversational Speech Emotion Recognition

This repository contains code for conversational speech emotion recognition on IEMOCAP with three model families:

- **Baseline**: frozen SSL utterance embeddings with a classifier.
- **CDM**: Conversational Dialogue Memory over acoustic utterance embeddings.
- **CDIM**: Conversational Dialogue-Interaction Memory over acoustic embeddings plus interaction features.

The experiments use the following four-class IEMOCAP label policy:

- target labels: `ang -> angry`, `hap + exc -> happy`, `neu -> neutral`, `sad -> sad`
- other IEMOCAP labels remain in the dialogue timeline as context with `ignore_index=-100`
- memory and temporal features are computed over the full dialogue
- loss and metrics are computed only on the four target emotion classes

## Repository Layout

```text
configs/                  Training configs
models/                   Baseline, CDM, and CDIM model definitions
scripts/                  Training, download, and utility scripts
utils/                    Dataset loading, temporal features, metrics, and shared helpers
```

Large local files are not included in the clean package:

```text
data/
results/
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want the scripts to download IEMOCAP through Kaggle/Hugging Face, make sure your credentials are available in the environment. You can also place the dataset manually under:

```text
data/iemocap
```

## What Gets Logged

Each training run writes a full config tree before training starts. If embedding precomputation is needed, it shows a progress bar. Epoch progress is controlled by the config key:

```yaml
training:
  progress_bar: true
```

Main configs currently keep this off for compact logs; set it to `true` when you want tqdm bars during full training.

## Single-Session Training

The trainer modules run one test session from `dataset.test_session` in the config. For WavLM Session 5:

```bash
python -m scripts.train_baseline --config configs/main/wavlm_baseline.yaml
python -m scripts.train_cdm --config configs/main/wavlm_cdm.yaml
python -m scripts.train_cdim --config configs/main/wavlm_cdim.yaml
```

Outputs are written under:

```text
results/main/wavlm/
```

Each run saves:

```text
config.json
train.log
best.pth
last.pth
metrics.json
predictions.csv
subset_metrics.json
```

## Full Cross-Session Training

Use `scripts.run_cross_session` to run all five IEMOCAP leave-one-session-out splits. The first argument is the trainer module, and `--config` is the base config.

WavLM:

```bash
python -m scripts.run_cross_session scripts.train_baseline --config configs/main/wavlm_baseline.yaml
python -m scripts.run_cross_session scripts.train_cdm --config configs/main/wavlm_cdm.yaml
python -m scripts.run_cross_session scripts.train_cdim --config configs/main/wavlm_cdim.yaml
```

HuBERT:

```bash
python -m scripts.run_cross_session scripts.train_baseline --config configs/main/hubert_baseline.yaml
python -m scripts.run_cross_session scripts.train_cdm --config configs/main/hubert_cdm.yaml
python -m scripts.run_cross_session scripts.train_cdim --config configs/main/hubert_cdim.yaml
```

Wav2Vec2:

```bash
python -m scripts.run_cross_session scripts.train_baseline --config configs/main/wav2vec_baseline.yaml
python -m scripts.run_cross_session scripts.train_cdm --config configs/main/wav2vec_cdm.yaml
python -m scripts.run_cross_session scripts.train_cdim --config configs/main/wav2vec_cdim.yaml
```

Cross-session outputs are written to:

```text
results/main/<backbone>/<model>/cross_session/<run_id>/
```

The summary files are:

```text
cross_session_summary.json
cross_session_metrics.csv
```

## CDIM Ablations

The WavLM Session 5 ablation configs are:

```bash
python -m scripts.train_cdim --config configs/ablations/wavlm_cdim_zero_temporal.yaml
python -m scripts.train_cdim --config configs/ablations/wavlm_cdim_shuffled_temporal.yaml
python -m scripts.train_cdim --config configs/ablations/wavlm_cdim_acoustic_only.yaml
python -m scripts.train_cdim --config configs/ablations/wavlm_cdim_temporal_only.yaml
python -m scripts.train_cdim --config configs/ablations/wavlm_cdim_temporal_sum.yaml
```

These write to:

```text
results/ablations/wavlm/cdim/
```
