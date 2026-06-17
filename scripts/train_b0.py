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
from transformers import AutoFeatureExtractor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from models.b0 import B0UtteranceClassifier, build_b0_model
from scripts.evaluate_b0 import evaluate_model, resolve_device
from utils.config import add_b0_model_args, add_dataset_args, add_logging_args, add_training_args, build_b0_config
from utils.dataset import CANONICAL_LABELS, SERDataCollator, load_iemocap_splits


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


def build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Dict[str, Any], total_steps: int):
    warmup_steps = int(float(training_cfg.get("warmup_ratio", 0.0)) * total_steps)
    scheduler_name = str(training_cfg.get("lr_scheduler", "linear"))
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported lr_scheduler={scheduler_name!r}.")


def warmup_steps(training_cfg: Dict[str, Any], total_steps: int) -> int:
    return int(float(training_cfg.get("warmup_ratio", 0.0)) * total_steps)


def format_epoch_log(epoch_log: Dict[str, Any], learning_rate: float, best_macro_f1: float) -> str:
    validation = epoch_log["validation"]
    return (
        f"epoch={epoch_log['epoch']} "
        f"train_loss={epoch_log['train_loss']:.6f} "
        f"val_loss={validation['loss']:.6f} "
        f"val_WA={validation['WA']:.6f} "
        f"val_UA={validation['UA']:.6f} "
        f"val_macro_f1={validation['macro_f1']:.6f} "
        f"val_WF1={validation['WF1']:.6f} "
        f"best_macro_f1={best_macro_f1:.6f} "
        f"lr={learning_rate:.6e}"
    )


def init_wandb(config: Dict[str, Any], output_dir: Path):
    logging_cfg = config.get("logging", {})
    if not bool(logging_cfg.get("use_wandb", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("use_wandb=true but wandb is not installed. Run `pip install wandb`.") from exc

    return wandb.init(
        project=str(logging_cfg.get("wandb_project", "conversational-SER")),
        name=logging_cfg.get("wandb_run_name") or None,
        entity=logging_cfg.get("wandb_entity") or None,
        mode=str(logging_cfg.get("wandb_mode", "online")),
        dir=str(output_dir),
        config=config,
    )


def wandb_epoch_payload(epoch_log: Dict[str, Any], learning_rate: float, best_macro_f1: float) -> Dict[str, float]:
    validation = epoch_log["validation"]
    return {
        "epoch": float(epoch_log["epoch"]),
        "train/loss": float(epoch_log["train_loss"]),
        "train/learning_rate": float(learning_rate),
        "validation/loss": float(validation["loss"]),
        "validation/accuracy": float(validation["accuracy"]),
        "validation/wa": float(validation["wa"]),
        "validation/ua": float(validation["ua"]),
        "validation/uar": float(validation["uar"]),
        "validation/WA": float(validation["WA"]),
        "validation/UA": float(validation["UA"]),
        "validation/macro_f1": float(validation["macro_f1"]),
        "validation/weighted_f1": float(validation["weighted_f1"]),
        "validation/WF1": float(validation["WF1"]),
        "best/macro_f1": float(best_macro_f1),
    }


def parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def log_training_session(
    log_path: Path,
    config: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    total_steps: int | None = None,
) -> None:
    dataset_cfg = config["dataset"]
    audio_cfg = config["audio"]
    b0_cfg = config["baselines"]["b0"]
    model_cfg = b0_cfg["model"]
    training_cfg = b0_cfg["training"]
    logging_cfg = config["logging"]

    lines = [
        "training_session_start",
        f"baseline={b0_cfg['name']}",
        f"output_dir={output_dir}",
        f"best_checkpoint={output_dir / 'best.pt'}",
        f"last_checkpoint={output_dir / 'last.pt'}",
        f"device={device}",
        f"dataset={dataset_cfg['name']}",
        f"split_strategy={dataset_cfg.get('split_strategy', 'random')}",
        f"test_session={dataset_cfg.get('test_session', '')}",
        f"sampling_rate={audio_cfg['sampling_rate']}",
        f"max_duration_seconds={audio_cfg['max_duration_seconds']}",
        f"seed={dataset_cfg['seed']}",
        f"encoder={model_cfg['encoder_name']}",
        f"pooling={model_cfg['pooling']}",
        f"freeze_encoder={model_cfg['freeze_encoder']}",
        f"trainable_encoder_layers={model_cfg.get('trainable_encoder_layers', 0)}",
        f"hidden_dim={model_cfg['hidden_dim']}",
        f"dropout={model_cfg['dropout']}",
        f"batch_size={training_cfg['batch_size']}",
        f"eval_batch_size={training_cfg['eval_batch_size']}",
        f"epochs={training_cfg['epochs']}",
        f"learning_rate={training_cfg['learning_rate']}",
        f"weight_decay={training_cfg['weight_decay']}",
        f"lr_scheduler={training_cfg['lr_scheduler']}",
        f"warmup_ratio={training_cfg['warmup_ratio']}",
        f"warmup_steps={warmup_steps(training_cfg, total_steps) if total_steps is not None else 'unknown'}",
        f"gradient_accumulation_steps={training_cfg['gradient_accumulation_steps']}",
        f"early_stopping_patience={training_cfg['early_stopping_patience']}",
        f"early_stopping_min_delta={training_cfg['early_stopping_min_delta']}",
        f"progress_bar={logging_cfg['progress_bar']}",
        f"use_wandb={logging_cfg['use_wandb']}",
        f"wandb_project={logging_cfg['wandb_project']}",
        f"wandb_run_name={logging_cfg['wandb_run_name']}",
    ]
    for line in lines:
        emit_log(log_path, line)


def progress_kwargs(logging_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "leave": False,
        "dynamic_ncols": False,
        "ncols": 100,
        "mininterval": float(logging_cfg.get("progress_mininterval", 2.0)),
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
    wandb_run = init_wandb(config, output_dir)
    if wandb_run is not None:
        emit_log(log_path, f"wandb_initialized url={getattr(wandb_run, 'url', '')}")

    datasets = load_iemocap_splits(config)
    emit_log(
        log_path,
        (
            "dataset_loaded "
            f"train={len(datasets['train'])} "
            f"validation={len(datasets['validation'])} "
            f"test={len(datasets['test'])}"
        ),
    )
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
    param_counts = parameter_counts(model)
    trainable_ratio = param_counts["trainable"] / max(param_counts["total"], 1)
    emit_log(
        log_path,
        (
            "model_parameters "
            f"total={param_counts['total']:,} "
            f"trainable={param_counts['trainable']:,} "
            f"frozen={param_counts['frozen']:,} "
            f"trainable_ratio={trainable_ratio:.6f}"
        ),
    )
    if wandb_run is not None:
        wandb_run.summary["parameters/total"] = param_counts["total"]
        wandb_run.summary["parameters/trainable"] = param_counts["trainable"]
        wandb_run.summary["parameters/frozen"] = param_counts["frozen"]
        wandb_run.summary["parameters/trainable_ratio"] = trainable_ratio

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
    scheduler = build_scheduler(optimizer, training_cfg, total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    log_training_session(log_path, config, output_dir, device, total_steps=total_steps)

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
            description=f"B0 epoch {epoch}/{epochs} validation",
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
        if wandb_run is not None:
            wandb_run.log(wandb_epoch_payload(epoch_log, current_lr, best_for_log), step=epoch)

        save_checkpoint(output_dir / "last.pt", model, config, val_metrics)
        append_log_line(log_path, f"saved last checkpoint path={output_dir / 'last.pt'}")

        current_macro_f1 = float(val_metrics["macro_f1"])
        if current_macro_f1 > best_macro_f1 + early_stopping_min_delta:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, config, val_metrics)
            append_log_line(log_path, f"saved best checkpoint path={output_dir / 'best.pt'} macro_f1={best_macro_f1:.6f}")
        else:
            epochs_without_improvement += 1
            append_log_line(
                log_path,
                (
                    f"no improvement epochs_without_improvement={epochs_without_improvement} "
                    f"patience={early_stopping_patience}"
                ),
            )
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                emit_log(
                    log_path,
                    (
                        f"early_stopping triggered epoch={epoch} "
                        f"best_macro_f1={best_macro_f1:.6f} "
                        f"patience={early_stopping_patience}"
                    ),
                )
                break

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log_line(log_path, f"finished baseline=B0_utterance best_macro_f1={best_macro_f1:.6f}")
    if wandb_run is not None:
        wandb_run.summary["best_macro_f1"] = best_macro_f1
        wandb_run.summary["best_checkpoint"] = str(output_dir / "best.pt")
        wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
