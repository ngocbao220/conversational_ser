from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from dataset import CANONICAL_LABELS


@torch.no_grad()
def predict_batches(model: torch.nn.Module, dataloader: Iterable[Mapping[str, torch.Tensor]], device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses: List[float] = []
    predictions: List[int] = []
    targets: List[int] = []
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        labels = batch["labels"].to(device)
        logits = model(
            input_values=batch["input_values"].to(device),
            attention_mask=batch.get("attention_mask", None).to(device) if batch.get("attention_mask", None) is not None else None,
        )
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return np.asarray(predictions), np.asarray(targets), mean_loss


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, object]:
    if targets.size == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix": np.zeros((len(CANONICAL_LABELS), len(CANONICAL_LABELS)), dtype=int).tolist(),
        }
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(CANONICAL_LABELS)))
        ).tolist(),
    }


def evaluate_model(model: torch.nn.Module, dataloader: Iterable[Mapping[str, torch.Tensor]], device: torch.device) -> Dict[str, object]:
    predictions, targets, loss = predict_batches(model, dataloader, device)
    metrics = compute_metrics(predictions, targets)
    metrics["loss"] = loss
    metrics["labels"] = CANONICAL_LABELS
    return metrics
