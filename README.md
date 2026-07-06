# Conversational SER

This workspace now uses the corrected IEMOCAP label policy:

- target labels: `ang -> angry`, `hap + exc -> happy`, `neu -> neutral`, `sad -> sad`
- other IEMOCAP labels stay in the dialogue as `context` / `ignore_index=-100`
- temporal features and memory are computed on the full dialogue timeline
- loss and metrics are computed only on the 4 target emotion classes

## Active Configs

Use only the cleaned config set:

```text
configs/correct_data/main/
configs/correct_data/cim_cdm_ablation/
```

Outputs are written to:

```text
results/correct_data/main/
results/correct_data/cim_cdm_ablation/
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

CIM:

```bash
python -m scripts.train_cim --config configs/correct_data/main/wavlm_cim.yaml
```

CDM+CIM:

```bash
python -m scripts.train_dual_branch --config configs/correct_data/main/wavlm_cdm_cim.yaml
```

Swap `wavlm` with `hubert` or `wav2vec` for the other SSL backbones.

## Ablation Example

```bash
python -m scripts.train_dual_branch --config configs/correct_data/cim_cdm_ablation/wavlm_zero_dialogue_memory.yaml
python -m scripts.train_dual_branch --config configs/correct_data/cim_cdm_ablation/wavlm_zero_interaction_memory.yaml
python -m scripts.train_dual_branch --config configs/correct_data/cim_cdm_ablation/wavlm_shuffled_dialogue_memory.yaml
python -m scripts.train_dual_branch --config configs/correct_data/cim_cdm_ablation/wavlm_shuffled_interaction_memory.yaml
```

Older launch scripts and legacy configs were moved to:

```text
.archive/old_configs_scripts_cleanup/
```
