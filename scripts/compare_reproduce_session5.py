from __future__ import annotations

import json
from pathlib import Path


METRICS = ("WA", "UA", "WF1", "Macro-F1")


RUNS = {
    "baseline": {
        "reproduced": Path("results/reproduce_session5/wavlm/baseline/metrics.json"),
        "reference": Path("results/final_result/wavlm/baseline/cross_session/run_20260706_120026/test_Ses05/metrics.json"),
    },
    "cdm": {
        "reproduced": Path("results/reproduce_session5/wavlm/cdm/metrics.json"),
        "reference": Path("results/final_result/wavlm/cdm/cross_session/run_20260706_120717/test_Ses05/metrics.json"),
    },
    "cdim": {
        "reproduced": Path("results/reproduce_session5/wavlm/cdim/metrics.json"),
        "reference": Path("results/final_result/wavlm/cim/cross_session/run_20260706_093905/test_Ses05/metrics.json"),
    },
}


def load_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {metric: float(payload[metric]) * 100.0 for metric in METRICS}


def main() -> None:
    rows: list[dict[str, object]] = []
    for model_name, paths in RUNS.items():
        reproduced = load_metrics(paths["reproduced"])
        reference = load_metrics(paths["reference"])
        for metric in METRICS:
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "reproduced": reproduced[metric],
                    "reference": reference[metric],
                    "delta": reproduced[metric] - reference[metric],
                }
            )

    output_path = Path("results/reproduce_session5/wavlm/reproduction_vs_final_result.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "metric", "reproduced", "reference", "delta"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["model"]),
                    str(row["metric"]),
                    f"{float(row['reproduced']):.6f}",
                    f"{float(row['reference']):.6f}",
                    f"{float(row['delta']):+.6f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"comparison_csv={output_path}")
    print(f"{'Model':<10} {'Metric':<8} {'Reproduced':>12} {'Final':>12} {'Delta':>12}")
    print("-" * 58)
    for row in rows:
        print(
            f"{row['model']:<10} {row['metric']:<8} "
            f"{float(row['reproduced']):>12.4f} "
            f"{float(row['reference']):>12.4f} "
            f"{float(row['delta']):>+12.4f}"
        )


if __name__ == "__main__":
    main()
