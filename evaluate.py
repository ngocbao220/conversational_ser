from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from tqdm.auto import tqdm

from dataset import CANONICAL_LABELS, SERDataCollator, load_iemocap_splits
from model import SERModel


@torch.no_grad()
def predict_batches(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    progress_bar: bool = False,
    description: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses: List[float] = []
    predictions: List[int] = []
    targets: List[int] = []
    criterion = torch.nn.CrossEntropyLoss()

    iterator = tqdm(dataloader, desc=description or "Evaluating", leave=False) if progress_bar else dataloader
    for batch in iterator:
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


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    progress_bar: bool = False,
    description: Optional[str] = None,
) -> Dict[str, object]:
    predictions, targets, loss = predict_batches(model, dataloader, device, progress_bar, description)
    metrics = compute_metrics(predictions, targets)
    metrics["loss"] = loss
    metrics["labels"] = CANONICAL_LABELS
    return metrics


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SER checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/ser_baseline/best.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config")
    if config is None:
        if args.config is None:
            raise ValueError("Checkpoint has no config. Pass --config explicitly.")
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    datasets = load_iemocap_splits(config)
    model_cfg = config["model"]
    audio_cfg = config["audio"]
    training_cfg = config["training"]
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["encoder_name"])
    collator = SERDataCollator(feature_extractor, sampling_rate=int(audio_cfg.get("sampling_rate", 16000)))
    dataloader = DataLoader(
        datasets[args.split],
        batch_size=int(training_cfg.get("eval_batch_size", 8)),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    model = SERModel(
        encoder_name=model_cfg["encoder_name"],
        num_labels=len(CANONICAL_LABELS),
        pooling=model_cfg.get("pooling", "mean"),
        freeze_encoder=False,
        dropout=float(model_cfg.get("dropout", 0.2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, dataloader, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
