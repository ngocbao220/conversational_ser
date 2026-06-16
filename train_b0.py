from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup

from b0_config import add_b0_model_args, add_dataset_args, add_logging_args, add_training_args, build_b0_config
from b0_model import B0UtteranceClassifier, build_b0_model
from dataset import CANONICAL_LABELS, SERDataCollator, load_iemocap_splits
from evaluate_b0 import evaluate_model, resolve_device


def append_log_line(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")
        handle.flush()


def emit_log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    append_log_line(log_path, message)


def save_checkpoint(path: Path, model: B0UtteranceClassifier, config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline": "B0_utterance",
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "labels": CANONICAL_LABELS,
        },
        path,
    )


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


def progress_kwargs(logging_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "leave": False,
        "dynamic_ncols": False,
        "ascii": True,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train B0 utterance-level SER baseline.")
    add_dataset_args(parser)
    add_b0_model_args(parser)
    add_training_args(parser)
    add_logging_args(parser)
    args = parser.parse_args()
    config = build_b0_config(args)

    b0_cfg = config["baselines"]["b0"]
    model_cfg = b0_cfg["model"]
    training_cfg = b0_cfg["training"]
    audio_cfg = config.get("audio", {})
    logging_cfg = config.get("logging", {})

    output_dir = Path(training_cfg.get("output_dir", "outputs/b0_utterance"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / str(logging_cfg.get("log_file", "train.log"))
    log_path.write_text("", encoding="utf-8")

    device = resolve_device(str(training_cfg.get("device", "auto")))
    progress_bar = bool(logging_cfg.get("progress_bar", False))
    log_every_steps = int(logging_cfg.get("log_every_steps", 50))
    append_log_line(log_path, f"start baseline=B0_utterance output_dir={output_dir} device={device}")

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

    model = build_b0_model(model_cfg, num_labels=len(CANONICAL_LABELS)).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("B0 has no trainable parameters. Check classifier and freeze_encoder settings.")

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
            desc=f"B0 epoch {epoch}/{epochs} train",
            disable=not progress_bar,
            **progress_kwargs(logging_cfg),
        )
        for step, batch in enumerate(train_iterator, start=1):
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            logits = model(
                input_values=batch["input_values"].to(device),
                attention_mask=attention_mask.to(device) if attention_mask is not None else None,
            )
            loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            loss_value = float(loss.item() * accumulation_steps)
            train_losses.append(loss_value)
            current_lr = scheduler.get_last_lr()[0]
            if progress_bar:
                train_iterator.set_postfix(loss=f"{loss_value:.4f}", lr=f"{current_lr:.2e}")
            elif log_every_steps > 0 and (step == 1 or step % log_every_steps == 0 or step == len(train_loader)):
                recent_losses = train_losses[-log_every_steps:] if log_every_steps > 0 else train_losses
                mean_recent_loss = sum(recent_losses) / max(len(recent_losses), 1)
                emit_log(
                    log_path,
                    (
                        f"epoch={epoch}/{epochs} "
                        f"step={step}/{len(train_loader)} "
                        f"loss={mean_recent_loss:.6f} "
                        f"lr={current_lr:.6e}"
                    ),
                )

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
            description=f"B0 epoch {epoch}/{epochs} validation"
        )
        epoch_log = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "validation": val_metrics,
        }
        history.append(epoch_log)
        current_lr = scheduler.get_last_lr()[0]
        best_for_log = max(best_macro_f1, float(val_metrics["macro_f1"]))
        log_line = format_epoch_log(epoch_log, current_lr, best_for_log)
        emit_log(log_path, log_line)

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            save_checkpoint(output_dir / "best.pt", model, config, val_metrics)
            append_log_line(log_path, f"saved best checkpoint path={output_dir / 'best.pt'} macro_f1={best_macro_f1:.6f}")

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log_line(log_path, f"finished baseline=B0_utterance best_macro_f1={best_macro_f1:.6f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
