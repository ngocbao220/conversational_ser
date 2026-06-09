from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

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


def init_wandb(config: Dict[str, Any], output_dir: Path):
    logging_cfg = config.get("logging", {})
    if not bool(logging_cfg.get("use_wandb", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("logging.use_wandb=true but wandb is not installed. Run `pip install wandb`.") from exc

    return wandb.init(
        project=logging_cfg.get("wandb_project", "ser-baseline"),
        name=logging_cfg.get("wandb_run_name"),
        entity=logging_cfg.get("wandb_entity"),
        config=config,
        dir=str(output_dir),
        mode=logging_cfg.get("wandb_mode", "online"),
    )


def wandb_log_payload(epoch_log: Dict[str, Any], learning_rate: float) -> Dict[str, Any]:
    validation = epoch_log["validation"]
    return {
        "epoch": epoch_log["epoch"],
        "train/loss": epoch_log["train_loss"],
        "train/learning_rate": learning_rate,
        "validation/loss": validation["loss"],
        "validation/accuracy": validation["accuracy"],
        "validation/macro_f1": validation["macro_f1"],
        "validation/weighted_f1": validation["weighted_f1"],
    }


def append_log_line(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def format_epoch_log(epoch_log: Dict[str, Any], learning_rate: float, best_macro_f1: float) -> str:
    validation = epoch_log["validation"]
    return (
        f"epoch={epoch_log['epoch']} "
        f"train_loss={epoch_log['train_loss']:.6f} "
        f"val_loss={validation['loss']:.6f} "
        f"val_acc={validation['accuracy']:.6f} "
        f"val_macro_f1={validation['macro_f1']:.6f} "
        f"val_weighted_f1={validation['weighted_f1']:.6f} "
        f"best_macro_f1={best_macro_f1:.6f} "
        f"lr={learning_rate:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SSL-based Speech Emotion Recognition baseline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    model_cfg = config["model"]
    audio_cfg = config["audio"]
    logging_cfg = config.get("logging", {})

    device = resolve_device(str(training_cfg.get("device", "auto")))
    output_dir = Path(training_cfg.get("output_dir", "outputs/ser_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / str(logging_cfg.get("log_file", "train.log"))
    with open(log_path, "w", encoding="utf-8"):
        pass
    progress_bar = bool(logging_cfg.get("progress_bar", True))
    wandb_run = init_wandb(config, output_dir)
    append_log_line(log_path, f"start run output_dir={output_dir} device={device}")

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
        append_log_line(log_path, f"start epoch={epoch}/{epochs}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []

        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs} train",
            leave=True,
            disable=not progress_bar,
        )
        for step, batch in enumerate(train_iterator, start=1):
            labels = batch["labels"].to(device)
            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=batch.get("attention_mask", None).to(device) if batch.get("attention_mask", None) is not None else None,
            )
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            loss_value = float(loss.item() * accumulation_steps)
            train_losses.append(loss_value)
            train_iterator.set_postfix(loss=f"{loss_value:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

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

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            progress_bar=progress_bar,
            description=f"Epoch {epoch}/{epochs} validation",
        )
        epoch_log = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "validation": val_metrics,
        }
        history.append(epoch_log)
        current_lr = scheduler.get_last_lr()[0]
        log_line = format_epoch_log(epoch_log, current_lr, max(best_macro_f1, float(val_metrics["macro_f1"])))
        print(log_line)
        append_log_line(log_path, log_line)

        if wandb_run is not None:
            wandb_run.log(wandb_log_payload(epoch_log, current_lr), step=epoch)

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            save_checkpoint(output_dir / "best.pt", model, config, val_metrics)
            append_log_line(log_path, f"saved best checkpoint path={output_dir / 'best.pt'} macro_f1={best_macro_f1:.6f}")

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    append_log_line(log_path, f"finished run best_macro_f1={best_macro_f1:.6f}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
