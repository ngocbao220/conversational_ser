# Verification Report: Three SER Experiments

Date: 2026-06-19

Final status: **VERIFICATION PASSED**

No critical failures were found. Static checks and minimal dry-runs passed. No project code was rewritten.

## PASS/FAIL Summary

| Section | Status | Notes |
|---|---|---|
| A. File structure | PASS | Expected configs, parser, models, train scripts, temporal utilities, and subset evaluator exist. |
| B. Dataset parser | PASS | Parser reads Kaggle IEMOCAP structure, annotations, transcripts, audio paths, metadata, and required label mapping. |
| C. LOSO split | PASS | `test_session` is configurable, default is 5, train excludes test session, validation is dialogue-level, and dry-run confirmed no dialogue overlap. |
| D. Exp 1 model behavior | PASS | Baseline forward accepts metadata but deletes it; no memory or temporal features are used. Default WavLM freeze behavior verified in dry-run with mocked WavLM. |
| E. Exp 2 model behavior | PASS | MAL uses fixed mean-pooled WavLM embeddings, zero temporal vectors, dialogue-level sequential processing, and read-before-write memory reset per dialogue. |
| F. Exp 3 model behavior | PASS | TIM uses 16 real temporal features, train-split-only normalization for continuous features, binary flags unnormalized, and causal speaker history. |
| G. Metrics | PASS | WA, UA, WF1, Macro-F1, per-class precision/recall/F1, and confusion matrices use correct sklearn definitions. |
| H. Checkpointing | PASS | All three train scripts select `best.pth` by validation UA and evaluate test only after model selection. |
| I. Output schema | PASS with note | Existing trained artifacts are under `outputs/hf_checkpoints/...`, not `results/...`; required files and prediction schemas were verified there. |
| J. Reproducibility | PASS | Seeds, CUDNN deterministic settings, parameter-count logging, config-controlled wandb, and tqdm progress are present. |
| K. Dry-run | PASS | Minimal mocked/model-only dry-run passed for Exp 1/2/3 logits, losses, temporal features, and LOSO split invariants. |

## Files Checked

- `instructions/verify_three_experiments.md`
- `configs/wavlm_baseline_no_mal_no_tim.yaml`
- `configs/wavlm_mal_no_tim.yaml`
- `configs/wavlm_tim.yaml`
- `utils/iemocap_kaggle.py`
- `utils/dialogue_embeddings.py`
- `utils/temporal_features.py`
- `utils/experiment_metrics.py`
- `utils/metrics.py`
- `models/wavlm_baseline.py`
- `models/wavlm_mal.py`
- `models/wavlm_tim.py`
- `scripts/train_wavlm_baseline.py`
- `scripts/train_wavlm_mal.py`
- `scripts/train_wavlm_tim.py`
- `scripts/evaluate_temporal_subsets.py`
- `outputs/hf_checkpoints/wavlm_baseline_no_mal_no_tim/*`
- `outputs/hf_checkpoints/wavlm_mal_no_tim/*`
- `outputs/hf_checkpoints/wavlm_tim/*`

## Commands Run

```bash
sed -n '1,320p' instructions/verify_three_experiments.md
git status --short
pgrep -fl train.py
pgrep -fl speech
python -m py_compile utils/iemocap_kaggle.py utils/metrics.py utils/experiment_metrics.py utils/dialogue_embeddings.py utils/temporal_features.py models/wavlm_baseline.py models/wavlm_mal.py models/wavlm_tim.py scripts/train_wavlm_baseline.py scripts/train_wavlm_mal.py scripts/train_wavlm_tim.py scripts/evaluate_temporal_subsets.py
find results outputs/hf_checkpoints -maxdepth 2 -type f \( -name 'metrics.json' -o -name 'predictions.csv' -o -name 'config.json' -o -name 'confusion_matrix.csv' -o -name 'confusion_matrix.png' -o -name 'best.pth' -o -name 'last.pth' -o -name 'temporal_feature_stats.json' -o -name 'subset_metrics.json' \)
head -n 1 outputs/hf_checkpoints/wavlm_baseline_no_mal_no_tim/predictions.csv
head -n 1 outputs/hf_checkpoints/wavlm_mal_no_tim/predictions.csv
head -n 1 outputs/hf_checkpoints/wavlm_tim/predictions.csv
/opt/anaconda3/envs/speech/bin/python -c '<minimal mocked dry-run for Exp 1/2/3 and LOSO split>'
/opt/anaconda3/envs/speech/bin/python -c '<artifact schema validation for outputs/hf_checkpoints>'
```

One attempted dry-run with the default `python` failed before project code ran because the base environment has `transformers` / `huggingface-hub` version mismatch. The same dry-run passed in `/opt/anaconda3/envs/speech/bin/python`.

## Detailed Findings

### A. File Structure

Expected implementation files are present:

- Configs: `configs/wavlm_baseline_no_mal_no_tim.yaml`, `configs/wavlm_mal_no_tim.yaml`, `configs/wavlm_tim.yaml`
- Dataset/parser: `utils/iemocap_kaggle.py`
- Models: `models/wavlm_baseline.py`, `models/wavlm_mal.py`, `models/wavlm_tim.py`
- Training scripts: `scripts/train_wavlm_baseline.py`, `scripts/train_wavlm_mal.py`, `scripts/train_wavlm_tim.py`
- Temporal utilities: `utils/temporal_features.py`
- Subset analysis: `scripts/evaluate_temporal_subsets.py`

### B. Dataset Parser

`utils/iemocap_kaggle.py` satisfies the checklist:

- `LABEL_NAMES = ["angry", "happy", "neutral", "sad"]`
- `LABEL2ID = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}`
- Raw labels map exactly as required: `ang -> angry`, `hap -> happy`, `exc -> happy`, `neu -> neutral`, `sad -> sad`.
- Unknown labels are discarded by `RAW_LABEL_MAP.get(...)` followed by `continue`.
- Emotion labels come from `dialog/EmoEvaluation/*.txt`, not transcript text.
- Transcripts are matched by `utterance_id`; audio paths are resolved as `Session*/sentences/wav/<dialogue_id>/<utterance_id>.wav`.
- Metadata includes session, dialogue, utterance, speaker, start/end time, label, raw label, and transcript.

### C. LOSO Split

`split_loso_by_dialogue(...)`:

- Uses configurable `test_session`.
- Defaults to session 5 through configs.
- Places only held-out session samples in test.
- Builds train/validation from non-test sessions.
- Selects validation by dialogue IDs, preventing train/validation utterance leakage.
- Dry-run with synthetic dialogues confirmed no dialogue overlap across train, validation, and test.

### D. Experiment 1: WavLM Baseline

`WavLMSERBaseline.forward(...)` accepts:

```python
forward(input_values, attention_mask=None, labels=None, **metadata)
```

The implementation deletes metadata with `del metadata`, so `dialogue_id`, `utterance_id`, `speaker_id`, `start_time`, and `end_time` are not used as model inputs.

Default config has:

- `freeze_wavlm: true`
- `unfreeze_last_n_layers: 0`
- `pooling: attentive_statistics`

Dry-run with a mocked WavLM confirmed default WavLM parameters were frozen and logits/loss were finite.

### E. Experiment 2: WavLM + MAL No TIM

`scripts/train_wavlm_mal.py` enforces:

- `use_temporal_features=false`
- `temporal_feature_mode=zero`
- `precompute.enabled=true`

`run_dialogue_epoch(...)` creates zero temporal vectors through `make_temporal_features(...)`. Dialogue embeddings are sorted by `(dialogue_id, start_time, end_time, utterance_id)` for ordering, and no duration/gap/overlap/interruption features are computed in the MAL path.

`MALMemoryModule` is causal read-before-write:

- It reads from current projected embedding plus previous state.
- It appends memory read before updating `state = GRUCell(...)`.
- Each dialogue call starts without passing a prior `initial_state`, so memory resets at dialogue boundary.

Speaker IDs are stored only in prediction rows; they are not fed into the model.

### F. Experiment 3: WavLM + TIM

`TEMPORAL_FEATURE_NAMES` contains the expected 16 features:

1. `duration`
2. `gap_prev`
3. `overlap_prev`
4. `overlap_ratio`
5. `is_overlap`
6. `is_interrupting_prev`
7. `speaker_switch`
8. `same_speaker`
9. `turn_index_norm`
10. `prev_gap_abs`
11. `short_response`
12. `long_pause`
13. `speaker_prev_overlap_rate`
14. `speaker_prev_mean_gap`
15. `speaker_prev_mean_duration`
16. `speaker_prev_turn_count_norm`

The checklist uses `abs_gap`; implementation names this equivalent feature `prev_gap_abs`. This is a naming mismatch only; the value is `abs(gap_prev)`.

Causality:

- For each utterance, feature computation uses current start/end time and `previous_row`.
- Speaker-history stats are read before the current utterance updates `speaker_history`.
- No future labels, future audio, or future timestamps are used.
- `turn_index_norm` uses `max_train_dialogue_length`, not full current dialogue length, avoiding future dialogue length leakage.

Normalization:

- `TemporalInteractionFeatureBuilder.fit(train_dialogues)` is called only on training dialogues.
- Continuous features are normalized using train split `mean` and `std`.
- Binary flags are excluded from normalization via `BINARY_TEMPORAL_FEATURES`.
- `temporal_feature_stats.json` is saved.

### G. Metrics

`utils/experiment_metrics.py` defines:

- WA as `accuracy_score`
- UA as `balanced_accuracy_score`
- WF1 as weighted `f1_score`
- Macro-F1 as macro `f1_score`
- Per-class precision/recall/F1 via `precision_recall_fscore_support`
- Confusion matrix with fixed label order

All three experiment scripts save `metrics.json`, `predictions.csv`, `confusion_matrix.csv`, and `confusion_matrix.png`.

### H. Checkpointing

All three scripts:

- Save `last.pth` each epoch with validation metrics.
- Update `best.pth` only when `val_metrics["UA"]` improves.
- Load `best.pth` after training.
- Evaluate test only after checkpoint selection.

No test metrics are used for model selection.

### I. Output Schema

Current trained artifacts were verified under:

- `outputs/hf_checkpoints/wavlm_baseline_no_mal_no_tim/`
- `outputs/hf_checkpoints/wavlm_mal_no_tim/`
- `outputs/hf_checkpoints/wavlm_tim/`

Each contains:

- `metrics.json`
- `predictions.csv`
- `config.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `best.pth`
- `last.pth`

TIM also contains:

- `temporal_feature_stats.json`
- `subset_metrics.json`

Prediction headers:

- Baseline/MAL: `dialogue_id, utterance_id, speaker_id, start_time, end_time, gold_label, pred_label, prob_angry, prob_happy, prob_neutral, prob_sad`
- TIM: includes the same required metadata/probability columns plus all temporal features.

Note: `results/wavlm_*` output directories were not populated at verification time. The configs point to `results/...`, but the completed checkpoint artifacts being inspected are under `outputs/hf_checkpoints/...`.

### J. Reproducibility

All three training scripts set:

- Python `random.seed`
- NumPy seed
- PyTorch CPU/GPU seed
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

They log total/trainable parameter counts. W&B is controlled by config, and tqdm progress bars are used/configurable.

### K. Dry-run

Dry-run in the `speech` environment passed:

- Exp 1 mocked WavLM forward: logits shape `(2, 4)`, finite loss, default WavLM frozen.
- Exp 2 MAL forward: logits shape `(3, 4)`, finite loss, zero temporal vector accepted.
- Exp 3 TIM forward: logits shape `(3, 4)`, finite loss, temporal feature shape `(3, 16)`.
- Synthetic LOSO split: test contains only session 5; train/validation exclude session 5; dialogue IDs are disjoint across all splits.

## Missing Files

No expected source files are missing.

Model artifacts are missing from `results/wavlm_*` because the available completed runs are stored under `outputs/hf_checkpoints/wavlm_*`. This is not a critical implementation failure.

## Implementation Mismatches

- TIM feature name `prev_gap_abs` differs from checklist wording `abs_gap`, but the computed value is the expected absolute previous gap.
- Existing trained artifacts are under `outputs/hf_checkpoints/...`, while configs use `results/...` output directories.

## Leakage Risk Assessment

Critical leakage checks passed:

- No utterance-level split leakage found in LOSO split logic.
- Exp 1 deletes metadata and uses audio only.
- Exp 2 uses start/end times only for sorting and does not compute real temporal features.
- Exp 3 feature builder is causal and fits normalization stats on train dialogues only.
- Test metrics are not used for checkpoint selection.

Residual low risk:

- Exp 2 and Exp 3 both sort dialogue utterances by `start_time`, which is acceptable for dialogue ordering but should be described explicitly in papers/reports.

## Metric Definition Mismatch

No mismatch found. WA/UA/WF1/Macro-F1 definitions match the checklist.

## Fairness Between Exp 2 and Exp 3

The comparison is acceptable:

- Both use fixed mean-pooled WavLM embeddings.
- Both use memory dimension 256, dropout 0.2, residual gate initialized at 0.0, AdamW, cosine scheduler, validation UA checkpoint selection.
- Exp 2 receives zero temporal vectors; Exp 3 receives real temporal interaction features.

One expected architectural difference remains: Exp 3 has a temporal encoder and concatenates temporal embeddings into memory input, while Exp 2 uses a zero temporal projection path. This is consistent with the experiment definition.

## Recommended Fixes

No critical fixes are required.

Recommended cleanup:

1. Align the TIM feature name with the checklist by either documenting `prev_gap_abs = abs_gap` or renaming it in a backward-compatible way.
2. Decide whether final trained artifacts should live under `results/...` or `outputs/hf_checkpoints/...` and keep docs/configs consistent.
3. Use the `speech` conda environment for verification/training; base `python` currently has a `transformers` dependency mismatch.

## Terminal Summary

`VERIFICATION PASSED`
