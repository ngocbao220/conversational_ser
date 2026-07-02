from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

import yaml


TRAINER_MODULES = {
    "baseline": "scripts.train_wavlm_baseline",
    "cdm": "scripts.train_wavlm_cdm",
    "cim": "scripts.train_wavlm_cim",
    "dual_branch": "scripts.train_dual_branch",
}
METRIC_NAMES = ("WA", "UA", "WF1", "Macro-F1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def unique_run_dir(base_output_dir: Path, run_name: str | None) -> Path:
    root = base_output_dir / "cross_session"
    candidate_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / candidate_name
    if not candidate.exists():
        return candidate
    if run_name:
        raise FileExistsError(f"Cross-session run already exists: {candidate}")

    suffix = 2
    while (root / f"{candidate_name}_{suffix}").exists():
        suffix += 1
    return root / f"{candidate_name}_{suffix}"


def normalize_sessions(raw_sessions: Any) -> list[int]:
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError("cross_session.test_sessions must be a non-empty list of session numbers.")
    sessions = [int(session) for session in raw_sessions]
    if any(session < 1 or session > 5 for session in sessions):
        raise ValueError("cross_session.test_sessions may only contain IEMOCAP sessions 1 through 5.")
    if len(set(sessions)) != len(sessions):
        raise ValueError("cross_session.test_sessions must not contain duplicates.")
    return sessions


def fold_config(base_config: Mapping[str, Any], test_session: int, fold_dir: Path, run_name: str) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["output_dir"] = str(fold_dir)
    config["experiment_name"] = f"{base_config['experiment_name']}__test_Ses{test_session:02d}"
    config.setdefault("dataset", {})["test_session"] = test_session
    config.setdefault("cross_session", {})["enabled"] = False
    config["cross_session"]["parent_run_name"] = run_name

    wandb_cfg = config.get("wandb")
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("use_wandb", False):
        base_name = str(wandb_cfg.get("run_name", base_config["experiment_name"]))
        wandb_cfg["run_name"] = f"{base_name}__test_Ses{test_session:02d}"
    return config


def read_fold_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test", payload)
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for metric_name in METRIC_NAMES:
        values = [float(fold["metrics"][metric_name]) for fold in folds]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {metric_name} value found in cross-session metrics.")
        summary[metric_name] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def run_cross_session(trainer_module: str, config_path: str | Path) -> Path:
    if trainer_module not in TRAINER_MODULES.values():
        raise ValueError(f"Unsupported trainer module: {trainer_module}")

    base_config = load_config(config_path)
    dataset_name = str(base_config.get("dataset", {}).get("name", "iemocap")).lower()
    if dataset_name != "iemocap":
        raise ValueError(
            f"cross_session LOSO is only supported for IEMOCAP, got dataset.name={dataset_name!r}. "
            "For MELD, use the official train/dev/test split with cross_session.enabled=false."
        )
    cross_cfg = base_config.get("cross_session", {})
    if not bool(cross_cfg.get("enabled", False)):
        raise ValueError("Set cross_session.enabled=true before launching a cross-session run.")

    sessions = normalize_sessions(cross_cfg.get("test_sessions", [1, 2, 3, 4, 5]))
    run_dir = unique_run_dir(Path(base_config["output_dir"]), cross_cfg.get("run_name"))
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    run_name = run_dir.name
    folds: list[dict[str, Any]] = []

    for test_session in sessions:
        fold_dir = run_dir / f"test_Ses{test_session:02d}"
        child_config = fold_config(base_config, test_session, fold_dir, run_name)
        child_config_path = config_dir / f"test_Ses{test_session:02d}.yaml"
        child_config_path.write_text(yaml.safe_dump(child_config, sort_keys=False), encoding="utf-8")

        subprocess.run(
            [sys.executable, "-m", trainer_module, "--config", str(child_config_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )
        metrics_path = fold_dir / "metrics.json"
        folds.append(
            {
                "test_session": test_session,
                "output_dir": str(fold_dir),
                "metrics": read_fold_metrics(metrics_path),
            }
        )

        partial_summary = {
            "trainer_module": trainer_module,
            "run_name": run_name,
            "test_sessions": sessions,
            "folds": folds,
            "aggregate": summarize_folds(folds),
        }
        (run_dir / "cross_session_summary.json").write_text(
            json.dumps(partial_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return run_dir / "cross_session_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO cross-session training and aggregate mean +/- std metrics.")
    parser.add_argument("--trainer", choices=TRAINER_MODULES, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary_path = run_cross_session(TRAINER_MODULES[args.trainer], args.config)
    print(f"cross_session_summary={summary_path}")


if __name__ == "__main__":
    main()
