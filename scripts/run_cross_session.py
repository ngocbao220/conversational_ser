from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


SUMMARY_METRICS = ("WA", "UA", "WF1", "Macro-F1", "loss")


def _session_name(session: int | str) -> str:
    text = str(session)
    if text.startswith("Ses"):
        return text
    return f"Ses{int(text):02d}"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return payload


def _save_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _metric_stats(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _write_per_session_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = ["session", "output_dir", "best_epoch", *SUMMARY_METRICS]
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(str(value))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_child_config(config: dict[str, Any], session: int | str, output_dir: Path) -> dict[str, Any]:
    session_label = _session_name(session)
    child = json.loads(json.dumps(config))
    child.setdefault("dataset", {})["test_session"] = int(str(session_label).replace("Ses", ""))
    child.setdefault("cross_session", {})["enabled"] = False
    child["output_dir"] = str(output_dir)

    base_name = str(config.get("experiment_name") or config.get("run_name") or output_dir.parent.parent.name)
    child["experiment_name"] = f"{base_name}__test_{session_label}"
    child["run_name"] = child["experiment_name"]
    child.setdefault("wandb", {})["run_name"] = child["experiment_name"]
    return child


def run_cross_session(trainer_module: str, config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    cross_cfg = config.get("cross_session", {}) or {}
    test_sessions = cross_cfg.get("test_sessions") or [1, 2, 3, 4, 5]

    base_output_dir = Path(str(config["output_dir"]))
    run_name = cross_cfg.get("run_name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_output_dir / "cross_session" / str(run_name)
    configs_dir = run_dir / "configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_yaml(run_dir / "base_config.yaml", config)

    session_rows: list[dict[str, Any]] = []
    for session in test_sessions:
        session_label = _session_name(session)
        child_output_dir = run_dir / f"test_{session_label}"
        child_config = build_child_config(config, session, child_output_dir)
        child_config_path = configs_dir / f"test_{session_label}.yaml"
        _save_yaml(child_config_path, child_config)

        subprocess.run(
            [sys.executable, "-m", trainer_module, "--config", str(child_config_path)],
            check=True,
        )

        metrics_path = child_output_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected metrics file after training: {metrics_path}")
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {
            "session": session_label,
            "output_dir": str(child_output_dir),
            "best_epoch": metrics.get("best_epoch", metrics.get("epoch")),
        }
        for metric in SUMMARY_METRICS:
            if metric in metrics:
                row[metric] = metrics[metric]
        session_rows.append(row)

    summary = {
        "trainer_module": trainer_module,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "test_sessions": [_session_name(session) for session in test_sessions],
        "per_session": session_rows,
        "metrics": _metric_stats(session_rows),
    }
    summary_path = run_dir / "cross_session_summary.json"
    _save_json(summary_path, summary)
    _write_per_session_csv(run_dir / "cross_session_metrics.csv", session_rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO cross-session training for one trainer/config pair.")
    parser.add_argument("trainer_module")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary_path = run_cross_session(args.trainer_module, args.config)
    print(f"cross_session_summary={summary_path}")


if __name__ == "__main__":
    main()
