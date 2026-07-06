#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAIN_CONFIGS = [
    "configs/strict4/main_hubert_baseline.yaml",
    "configs/strict4/main_hubert_cdm.yaml",
    "configs/strict4/main_hubert_cim.yaml",
    "configs/strict4/main_hubert_cdm_cim.yaml",
    "configs/strict4/main_wavlm_baseline.yaml",
    "configs/strict4/main_wavlm_cdm.yaml",
    "configs/strict4/main_wavlm_cim.yaml",
    "configs/strict4/main_wavlm_cdm_cim.yaml",
    "configs/strict4/main_wav2vec_baseline.yaml",
    "configs/strict4/main_wav2vec_cdm.yaml",
    "configs/strict4/main_wav2vec_cim.yaml",
    "configs/strict4/main_wav2vec_cdm_cim.yaml",
]

ABLATION_CONFIGS = [
    "configs/strict4/cdm_cim_ablation_hubert_zero_dialogue_memory.yaml",
    "configs/strict4/cdm_cim_ablation_hubert_zero_interaction_memory.yaml",
    "configs/strict4/cdm_cim_ablation_hubert_shuffled_dialogue_memory.yaml",
    "configs/strict4/cdm_cim_ablation_hubert_shuffled_interaction_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wavlm_zero_dialogue_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wavlm_zero_interaction_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wavlm_shuffled_dialogue_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wavlm_shuffled_interaction_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wav2vec_zero_dialogue_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wav2vec_zero_interaction_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wav2vec_shuffled_dialogue_memory.yaml",
    "configs/strict4/cdm_cim_ablation_wav2vec_shuffled_interaction_memory.yaml",
]

ATTENTIVE_SES5_CONFIGS = [
    "configs/strict4_attentive_ses5/wavlm_baseline.yaml",
    "configs/strict4_attentive_ses5/wavlm_cdm.yaml",
    "configs/strict4_attentive_ses5/wavlm_cdm_cim.yaml",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def infer_ssl(config_path: Path, config: dict[str, Any]) -> str:
    text = " ".join([config_path.name, str(config.get("model", {}).get("wavlm_model_name", ""))]).lower()
    if "hubert" in text:
        return "hubert"
    if "wav2vec" in text or "wav2vec2" in text:
        return "wav2vec"
    return "wavlm"


def infer_arch(config_path: Path) -> str:
    name = config_path.stem
    if "cdm_cim" in name:
        return "dual_branch"
    if name.endswith("_baseline") or "baseline" in name:
        return "baseline"
    if name.endswith("_cim") or "_cim_" in name:
        return "cim"
    if name.endswith("_cdm") or "_cdm_" in name:
        return "cdm"
    raise ValueError(f"Cannot infer trainer from config name: {config_path}")


def trainer_module(config_path: Path) -> str:
    arch = infer_arch(config_path)
    return {
        "baseline": "scripts.train_baseline",
        "cdm": "scripts.train_cdm",
        "cim": "scripts.train_cim",
        "dual_branch": "scripts.train_dual_branch",
    }[arch]


def result_subdir(config_path: Path, suite: str, config: dict[str, Any]) -> Path:
    original = Path(str(config["output_dir"]))
    parts = original.parts
    if len(parts) >= 2 and parts[0] == "results" and parts[1] == "strict4":
        return Path(*parts[2:])
    if len(parts) >= 2 and parts[0] == "results" and parts[1] == "strict4_attentive_ses5":
        return Path("attentive_ses5", *parts[2:])
    return Path(suite, config_path.stem)


def rewrite_config(
    source_path: Path,
    suite: str,
    run_id: str,
    output_root: Path,
    config_root: Path,
    wandb_mode: str,
) -> Path:
    source_abs = PROJECT_ROOT / source_path
    config = load_yaml(source_abs)
    ssl = infer_ssl(source_path, config)
    pooling = str(config.get("precompute", {}).get("pooling", "mean"))
    new_output_dir = output_root / run_id / result_subdir(source_path, suite, config)

    config["output_dir"] = str(new_output_dir)
    for key in ("experiment_name", "run_name"):
        if key in config and config[key] is not None:
            config[key] = f"{config[key]}__{run_id}"

    arch = infer_arch(source_path)
    if arch == "cim":
        config.setdefault("model", {})["use_temporal_features"] = True

    precompute = config.setdefault("precompute", {})
    precompute["cache_path"] = str(Path("results/context_timeline_v4/shared_cache") / f"{ssl}_{pooling}_embeddings.pt")
    precompute["force_recompute"] = False

    cross_session = config.get("cross_session")
    if isinstance(cross_session, dict) and cross_session.get("enabled", False):
        cross_session["run_name"] = run_id

    wandb = config.get("wandb")
    if isinstance(wandb, dict):
        if wandb.get("run_name") is not None:
            wandb["run_name"] = f"{wandb['run_name']}__{run_id}"
        if wandb_mode != "unchanged":
            if wandb_mode == "disabled":
                wandb["use_wandb"] = False
            else:
                wandb["use_wandb"] = True
                wandb["mode"] = wandb_mode

    relative_source = source_path.with_suffix(".yaml")
    if relative_source.parts and relative_source.parts[0] == "configs":
        relative_source = Path(*relative_source.parts[1:])
    generated_path = PROJECT_ROOT / config_root / run_id / relative_source
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return generated_path


def select_configs(suites: list[str], ssl_filter: str) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    expanded = ["main", "ablation", "attentive_ses5"] if "all" in suites else suites
    for suite in expanded:
        if suite == "main":
            paths = MAIN_CONFIGS
        elif suite == "ablation":
            paths = ABLATION_CONFIGS
        elif suite == "attentive_ses5":
            paths = ATTENTIVE_SES5_CONFIGS
        else:
            raise ValueError(f"Unknown suite: {suite}")
        for raw_path in paths:
            path = Path(raw_path)
            config = load_yaml(PROJECT_ROOT / path)
            if ssl_filter != "all" and infer_ssl(path, config) != ssl_filter:
                continue
            selected.append((suite, path))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-timeline-v4 experiments without overwriting old strict4 runs.")
    parser.add_argument("--suite", nargs="+", choices=["all", "main", "ablation", "attentive_ses5"], default=["all"])
    parser.add_argument("--ssl", choices=["all", "hubert", "wavlm", "wav2vec"], default="all")
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("ctx_v4_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", default="results/context_timeline_v4")
    parser.add_argument("--config-root", default="configs/generated/context_timeline_v4")
    parser.add_argument("--wandb-mode", choices=["unchanged", "disabled", "offline", "online"], default="unchanged")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    config_root = Path(args.config_root)
    jobs = select_configs(args.suite, args.ssl)
    if not jobs:
        raise SystemExit("No configs selected.")

    generated: list[tuple[Path, str]] = []
    for suite, source in jobs:
        generated_config = rewrite_config(
            source,
            suite=suite,
            run_id=args.run_id,
            output_root=output_root,
            config_root=config_root,
            wandb_mode=args.wandb_mode,
        )
        generated.append((generated_config, trainer_module(source)))

    print(f"run_id={args.run_id}")
    print(f"generated_configs={config_root / args.run_id}")
    print(f"output_root={output_root / args.run_id}")
    for config_path, module in generated:
        command = [sys.executable, "-m", module, "--config", str(config_path)]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
