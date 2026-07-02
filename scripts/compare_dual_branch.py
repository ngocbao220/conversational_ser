from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = ("WA", "UA", "WF1", "Macro-F1")
DEFAULT_RUNS = {
    "Baseline": Path("results/wavlm_baseline_no_cdm_no_cim/metrics.json"),
    "CDM": Path("results/wavlm_cdm_no_cim/metrics.json"),
    "CIM": Path("results/wavlm_cim/metrics.json"),
    "dual_branch": Path("results/dual_branch/metrics.json"),
}


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(payload: Mapping[str, Any], metric: str) -> float:
    if metric in payload:
        return float(payload[metric])
    test_payload = payload.get("test", {})
    if isinstance(test_payload, Mapping) and metric in test_payload:
        return float(test_payload[metric])
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Baseline, CDM, CIM and dual-branch metrics.")
    parser.add_argument("--output-dir", default="results/dual_branch")
    args = parser.parse_args()

    rows = []
    for name, path in DEFAULT_RUNS.items():
        if not path.exists():
            rows.append({"model": name, "metrics_path": str(path), "available": False})
            continue
        payload = read_json(path)
        row = {"model": name, "metrics_path": str(path), "available": True}
        for metric in METRICS:
            row[metric] = metric_value(payload, metric)
        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_metrics.csv"
    json_path = output_dir / "comparison_metrics.json"
    fieldnames = ["model", "available", *METRICS, "metrics_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
