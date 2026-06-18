from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_ser_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    label_names: Sequence[str],
    loss: float | None = None,
    epoch: int | None = None,
) -> Dict[str, Any]:
    target_array = np.asarray(targets, dtype=np.int64)
    pred_array = np.asarray(predictions, dtype=np.int64)
    label_ids = list(range(len(label_names)))
    if target_array.size == 0:
        zeros = {label: 0.0 for label in label_names}
        return {
            "epoch": epoch,
            "loss": loss,
            "WA": 0.0,
            "UA": 0.0,
            "WF1": 0.0,
            "Macro-F1": 0.0,
            "per-class precision": zeros,
            "per-class recall": zeros,
            "per-class F1": zeros,
            "confusion_matrix": np.zeros((len(label_names), len(label_names)), dtype=int).tolist(),
            "labels": list(label_names),
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_array,
        pred_array,
        labels=label_ids,
        zero_division=0,
    )
    return {
        "epoch": epoch,
        "loss": loss,
        "WA": float(accuracy_score(target_array, pred_array)),
        "UA": float(balanced_accuracy_score(target_array, pred_array)),
        "WF1": float(f1_score(target_array, pred_array, average="weighted", zero_division=0)),
        "Macro-F1": float(f1_score(target_array, pred_array, average="macro", zero_division=0)),
        "per-class precision": {label: float(value) for label, value in zip(label_names, precision)},
        "per-class recall": {label: float(value) for label, value in zip(label_names, recall)},
        "per-class F1": {label: float(value) for label, value in zip(label_names, f1)},
        "confusion_matrix": confusion_matrix(target_array, pred_array, labels=label_ids).tolist(),
        "labels": list(label_names),
    }


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], label_names: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dialogue_id",
        "utterance_id",
        "speaker_id",
        "start_time",
        "end_time",
        "gold_label",
        "pred_label",
        *[f"prob_{label}" for label in label_names],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_confusion_matrix_csv(path: str | Path, matrix: Sequence[Sequence[int]], label_names: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold/pred", *label_names])
        for label, row in zip(label_names, matrix):
            writer.writerow([label, *row])


def save_confusion_matrix_png(path: str | Path, matrix: Sequence[Sequence[int]], label_names: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_array = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix_array, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    for row_idx in range(matrix_array.shape[0]):
        for col_idx in range(matrix_array.shape[1]):
            ax.text(col_idx, row_idx, str(matrix_array[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
