from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from b0_config import add_dataset_args, str_to_bool
from b0_model import build_b0_model
from dataset import CANONICAL_LABELS, SERDataCollator, load_iemocap_splits
from metrics import classification_metrics


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_batches(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    progress_bar: bool = False,
    description: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses = []
    predictions = []
    targets = []
    criterion = torch.nn.CrossEntropyLoss()
    iterator = tqdm(
        dataloader,
        desc=description or "Evaluating",
        leave=False,
        dynamic_ncols=False,
        ascii=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        disable=not progress_bar,
    )

    with torch.no_grad():
        for batch in iterator:
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            )
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return np.asarray(predictions), np.asarray(targets), mean_loss


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    progress_bar: bool = False,
    description: Optional[str] = None
) -> Dict[str, object]:
    predictions, targets, loss = predict_batches(model, dataloader, device, progress_bar, description)
    metrics = classification_metrics(predictions, targets, CANONICAL_LABELS)
    metrics["loss"] = loss
    return metrics


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B0 utterance-level SER baseline.")
    parser.add_argument("--checkpoint", default="outputs/b0_utterance/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--progress-bar", type=str_to_bool, default=None)
    parser.add_argument("--progress-ncols", type=int, default=None)
    add_dataset_args(parser)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, resolve_device(args.device or "auto"))
    config = checkpoint["config"]
    # Dataset arguments are intentionally CLI-controlled for eval scripts.
    config["dataset"].update(
        {
            "name": args.dataset_name,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "num_proc": args.num_proc,
        }
    )
    for key, value in {
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "max_test_samples": args.max_test_samples,
    }.items():
        if value is not None:
            config["dataset"][key] = value
    config["audio"].update(
        {
            "sampling_rate": args.sampling_rate,
            "max_duration_seconds": args.max_duration_seconds,
        }
    )

    b0_cfg = config["baselines"]["b0"]
    training_cfg = b0_cfg.get("training", {})
    model_cfg = b0_cfg.get("model", {})
    audio_cfg = config.get("audio", {})
    device = resolve_device(args.device or str(training_cfg.get("device", "auto")))
    eval_batch_size = int(args.eval_batch_size or training_cfg.get("eval_batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0))
    progress_bar = bool(args.progress_bar if args.progress_bar is not None else config.get("logging", {}).get("progress_bar", True))

    datasets = load_iemocap_splits(config)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["encoder_name"])
    collator = SERDataCollator(feature_extractor, sampling_rate=int(audio_cfg.get("sampling_rate", 16000)))
    dataloader = DataLoader(
        datasets[args.split],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    model = build_b0_model(model_cfg, num_labels=len(CANONICAL_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(
        model,
        dataloader,
        device,
        progress_bar=progress_bar,
        description=f"B0 {args.split}"
    )

    output_path = Path(args.output or b0_cfg.get("metrics_path", f"outputs/b0_utterance/{args.split}_metrics.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
