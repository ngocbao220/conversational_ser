from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_names: List[str],
) -> Dict[str, object]:
    if targets.size == 0:
        return {
            "accuracy": 0.0,
            "wa": 0.0,
            "ua": 0.0,
            "uar": 0.0,
            "WA": 0.0,
            "UA": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "WF1": 0.0,
            "confusion_matrix": np.zeros((len(label_names), len(label_names)), dtype=int).tolist(),
            "labels": label_names,
        }

    label_ids = list(range(len(label_names)))
    wa = float(accuracy_score(targets, predictions))
    ua = float(recall_score(targets, predictions, labels=label_ids, average="macro", zero_division=0))
    wf1 = float(f1_score(targets, predictions, average="weighted", zero_division=0))
    return {
        "accuracy": wa,
        "wa": wa,
        "ua": ua,
        "uar": ua,
        "WA": wa,
        "UA": ua,
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": wf1,
        "WF1": wf1,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=label_ids).tolist(),
        "labels": label_names,
    }
