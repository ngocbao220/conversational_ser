from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from utils.experiment_metrics import compute_ser_metrics, save_json
from utils.iemocap_kaggle import LABEL2ID, LABEL_NAMES


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def read_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    targets = [LABEL2ID[str(row["gold_label"])] for row in rows]
    predictions = [LABEL2ID[str(row["pred_label"])] for row in rows]
    metrics = compute_ser_metrics(targets, predictions, LABEL_NAMES)
    metrics["num_samples"] = len(rows)
    return metrics


def evaluate_temporal_subsets(
    prediction_rows: Sequence[Mapping[str, Any]],
    strong_overlap_threshold: float = 0.25,
) -> Dict[str, Any]:
    subsets = {
        "no_overlap": [row for row in prediction_rows if _float(row, "overlap_prev") <= 0.0],
        "overlap": [row for row in prediction_rows if _float(row, "overlap_prev") > 0.0],
        "strong_overlap": [row for row in prediction_rows if _float(row, "overlap_ratio") >= strong_overlap_threshold],
        "short_response": [row for row in prediction_rows if _float(row, "short_response") >= 0.5],
        "long_pause": [row for row in prediction_rows if _float(row, "long_pause") >= 0.5],
        "speaker_switch": [row for row in prediction_rows if _float(row, "speaker_switch") >= 0.5],
    }
    return {name: metrics_for_rows(rows) for name, rows in subsets.items()}


def save_temporal_subset_metrics(
    predictions_path: str | Path,
    output_path: str | Path,
    strong_overlap_threshold: float = 0.25,
) -> Dict[str, Any]:
    rows = read_prediction_rows(predictions_path)
    metrics = evaluate_temporal_subsets(rows, strong_overlap_threshold=strong_overlap_threshold)
    save_json(output_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CIM predictions on temporal subsets.")
    parser.add_argument("--predictions", default="results/wavlm_cim/predictions.csv")
    parser.add_argument("--output", default="results/wavlm_cim/subset_metrics.json")
    parser.add_argument("--strong-overlap-threshold", type=float, default=0.25)
    args = parser.parse_args()
    save_temporal_subset_metrics(
        predictions_path=args.predictions,
        output_path=args.output,
        strong_overlap_threshold=args.strong_overlap_threshold,
    )


if __name__ == "__main__":
    main()
