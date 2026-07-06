#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def discover_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("last.pth") if (path.parent / "config.json").exists())


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "root",
        "run_dir",
        "run_name",
        "architecture",
        "checkpoint_epoch",
        "WA",
        "UA",
        "WF1",
        "Macro-F1",
        "loss",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate last.pth for every run directory under one or more result roots.")
    parser.add_argument("roots", nargs="+", help="Result roots to scan, e.g. results/main results/cdm_cim_ablation.")
    parser.add_argument("--device", default=None, help="Optional device override passed to evaluate_run_checkpoint.py.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with eval_checkpoints/last/metrics.json.")
    parser.add_argument("--summary", default="results/_last_checkpoint_eval_summary.csv")
    args = parser.parse_args()

    run_dirs: list[tuple[Path, Path]] = []
    for root_arg in args.roots:
        root = Path(root_arg)
        for run_dir in discover_runs(root):
            run_dirs.append((root, run_dir))

    rows: list[dict] = []
    total = len(run_dirs)
    for index, (root, run_dir) in enumerate(run_dirs, start=1):
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "last.pth"
        metrics_path = run_dir / "eval_checkpoints" / "last" / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[{index}/{total}] skip existing {run_dir}", flush=True)
        else:
            cmd = [
                sys.executable,
                "scripts/evaluate_run_checkpoint.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
            ]
            if args.device:
                cmd.extend(["--device", args.device])
            print(f"[{index}/{total}] evaluate {run_dir}", flush=True)
            subprocess.run(cmd, check=True)

        if metrics_path.exists():
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "root": str(root),
                    "run_dir": str(run_dir),
                    "run_name": metrics.get("run_name", ""),
                    "architecture": metrics.get("architecture", ""),
                    "checkpoint_epoch": metrics.get("checkpoint_epoch", ""),
                    "WA": metrics.get("WA", ""),
                    "UA": metrics.get("UA", ""),
                    "WF1": metrics.get("WF1", ""),
                    "Macro-F1": metrics.get("Macro-F1", ""),
                    "loss": metrics.get("loss", ""),
                    "metrics_path": str(metrics_path),
                }
            )

    write_summary(rows, Path(args.summary))
    print(f"summary={args.summary} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
