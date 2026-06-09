from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

from dataset import CANONICAL_LABELS, SERDataCollator, load_iemocap_splits
from evaluate import evaluate_model
from model import SERModel


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: SERModel, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "labels": CANONICAL_LABELS,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SSL-based Speech Emotion Recognition baseline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    model_cfg = config["model"]
    audio_cfg = config["audio"]

    device = resolve_device(str(training_cfg.get("device", "auto")))
    output_dir = Path(training_cfg.get("output_dir", "outputs/ser_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_iemocap_splits(config)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["encoder_name"])
    collator = SERDataCollator(feature_extractor, sampling_rate=int(audio_cfg.get("sampling_rate", 16000)))

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=int(training_cfg.get("eval_batch_size", 8)),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    model = SERModel(
        encoder_name=model_cfg["encoder_name"],
        num_labels=len(CANONICAL_LABELS),
        pooling=model_cfg.get("pooling", "mean"),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    ).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(training_cfg.get("epochs", 5))
    accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    total_steps = max(1, math.ceil(len(train_loader) / accumulation_steps) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []

        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch.get("attention_mask", None).to(device) if batch.get("attention_mask", None) is not None else None,
            )
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            train_losses.append(float(loss.item() * accumulation_steps))

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(training_cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, float(training_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate_model(model, val_loader, device)
        epoch_log = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "validation": val_metrics,
        }
        history.append(epoch_log)
        print(json.dumps(epoch_log, ensure_ascii=False))

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            save_checkpoint(output_dir / "best.pt", model, config, val_metrics)

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
