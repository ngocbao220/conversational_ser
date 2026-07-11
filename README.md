# Conversational SER

This repository implements conversational speech emotion recognition on IEMOCAP with two model families:

- baseline mean-pooled SSL utterance embeddings
- CDIM: Conversational Dialogue-Interaction Memory, which concatenates acoustic embeddings with temporal interaction features before the memory/readout stage

The IEMOCAP label policy is:

- target labels: `ang -> angry`, `hap + exc -> happy`, `neu -> neutral`, `sad -> sad`
- other labels remain in the dialogue context with `ignore_index=-100`
- temporal features are computed on the full dialogue timeline
- loss and metrics are computed only on the four target emotion classes

## Repository Layout

```text
configs/main/  Main baseline and CDIM configs for WavLM, HuBERT, and Wav2Vec2
models/        Model definitions
scripts/       Training, evaluation, and table-building entry points
utils/         Dataset, feature, metric, and embedding utilities
```

Training outputs are written to:

```text
results/main/
results/shared_cache/
```

## Setup

```bash
pip install -r requirements.txt
```

Place IEMOCAP under `data/iemocap`. If the data is not present locally, the configs can use Kaggle auto-download (`sangayb/iemocap`), provided Kaggle credentials are configured.

## Training

The provided configs run leave-one-session-out evaluation across sessions 1-5.

Baseline:

```bash
python -m scripts.train_baseline --config configs/main/wavlm_baseline.yaml
```

CDIM:

```bash
python -m scripts.train_cdim --config configs/main/wavlm_cdim.yaml
```

Swap `wavlm` with `hubert` or `wav2vec` for the other SSL backbones.

You can also invoke the cross-session runner directly:

```bash
python -m scripts.run_cross_session --trainer cdim --config configs/main/wavlm_cdim.yaml
```

The `--trainer` value must be either `baseline` or `cdim`.


### Full Session 5 Runs

Use these configs when you want full training with IEMOCAP Session 5 as the held-out test session, without running all LOSO folds.

Run all main models:

```bash
./scripts/train_session5.sh main
```

Run all CDIM ablations:

```bash
./scripts/train_session5.sh cdim_ablation
```

Run everything:

```bash
./scripts/train_session5.sh all
```

Filter to one backbone or a subset of ablations:

```bash
BACKBONES="wavlm" ./scripts/train_session5.sh main
BACKBONES="wavlm" ABLATIONS="zero_temporal shuffled_temporal" ./scripts/train_session5.sh cdim_ablation
```

The full Session 5 configs live under:

```text
configs/session5/main/
configs/session5/cdim_ablation/
```

Training logs print the full run config as a tree before data preparation. Progress bars are enabled by default for embedding precompute, training, validation, and test passes.

## Evaluation

Each training run writes fold metrics, predictions, checkpoints, and confusion matrices under its configured output directory.

To evaluate saved `last.pth` checkpoints without retraining:

```bash
python -m scripts.evaluate --roots results/main --skip-existing --summary results/last_checkpoint_eval_summary.csv
```

To evaluate a single checkpoint:

```bash
python -m scripts.evaluate --config results/main/wavlm/cdim/cross_session/<run_name>/test_Ses05/config.json --checkpoint results/main/wavlm/cdim/cross_session/<run_name>/test_Ses05/last.pth
```

## Result Tables

After training, regenerate summary tables:

```bash
python -m scripts.build_result_tables --root results --out-dir results/tables
```

This writes:

```text
results/tables/main_results.*
```

The table is exported as CSV, Markdown, LaTeX, and PNG when `matplotlib` is available.

## Smoke Tests

These lightweight checks verify that the packaged command-line entry points load successfully:

```bash
python -m scripts.run_cross_session --help
python -m scripts.build_result_tables --help
python -m scripts.build_result_tables --root results --out-dir /tmp/conversational_ser_tables
```
