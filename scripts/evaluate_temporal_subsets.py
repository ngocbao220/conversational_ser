from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.experiment_metrics import compute_ser_metrics
from utils.iemocap_kaggle import LABEL2ID, LABEL_NAMES


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metrics_for_temporal_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if str(row.get("gold_label")) in LABEL2ID]
    targets = [LABEL2ID[str(row["gold_label"])] for row in valid_rows]
    predictions = [LABEL2ID[str(row["pred_label"])] for row in valid_rows]
    metrics = compute_ser_metrics(targets, predictions, LABEL_NAMES)
    metrics["num_samples"] = len(valid_rows)
    metrics["num_context_only"] = len(rows) - len(valid_rows)
    return metrics


def evaluate_temporal_subsets(
    prediction_rows: Sequence[Mapping[str, Any]],
    strong_overlap_threshold: float = 0.25,
) -> dict[str, Any]:
    subsets = {
        "no_overlap": [row for row in prediction_rows if _float(row, "overlap_prev") <= 0.0],
        "overlap": [row for row in prediction_rows if _float(row, "overlap_prev") > 0.0],
        "strong_overlap": [
            row for row in prediction_rows if _float(row, "overlap_ratio") >= strong_overlap_threshold
        ],
        "short_response": [row for row in prediction_rows if _float(row, "short_response") >= 0.5],
        "long_gap": [row for row in prediction_rows if _float(row, "long_gap") >= 0.5],
        "speaker_switch": [row for row in prediction_rows if _float(row, "speaker_switch") >= 0.5],
        "same_speaker": [row for row in prediction_rows if _float(row, "speaker_switch") < 0.5],
    }
    return {name: metrics_for_temporal_rows(rows) for name, rows in subsets.items()}


def save_temporal_subset_metrics(
    output_path: str | Path,
    prediction_rows: Sequence[Mapping[str, Any]],
    strong_overlap_threshold: float = 0.25,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_temporal_subsets(
        prediction_rows,
        strong_overlap_threshold=strong_overlap_threshold,
    )
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
