from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from models.wavlm_baseline import WavLMSERBaseline, build_wavlm_ser_baseline
from utils.experiment_metrics import (
    compute_ser_metrics,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_json,
    save_predictions_csv,
)
from utils.iemocap_kaggle import (
    ID2LABEL,
    LABEL_NAMES,
    ConversationalSERCollator,
    ConversationalSERDataset,
    ConversationSERSample,
    discover_iemocap_samples,
    split_loso_by_dialogue,
)


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_optimizer(model: WavLMSERBaseline, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    classifier_lr = float(training_cfg.get("learning_rate_classifier", 1e-4))
    wavlm_lr = float(training_cfg.get("learning_rate_wavlm", 1e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))

    wavlm_params = [parameter for parameter in model.wavlm.parameters() if parameter.requires_grad]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("wavlm.")
    ]
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": classifier_lr, "name": "classifier"})
    if wavlm_params:
        param_groups.append({"params": wavlm_params, "lr": wavlm_lr, "name": "wavlm"})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any], total_steps: int):
    training_cfg = config["training"]
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler_name = str(training_cfg.get("scheduler", "cosine"))
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported scheduler={scheduler_name!r}.")


def init_wandb(config: Mapping[str, Any], output_dir: Path, log_path: Path):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("use_wandb", False)):
        return None
    try:
        import wandb

        return wandb.init(
            project=str(wandb_cfg.get("project", "conversational-SER")),
            name=str(wandb_cfg.get("run_name", config["experiment_name"])),
            entity=wandb_cfg.get("entity") or None,
            mode=str(wandb_cfg.get("mode", "online")),
            dir=str(output_dir),
            config=dict(config),
        )
    except Exception as exc:
        append_log(
            log_path,
            f"wandb disabled: {exc}. Set wandb.use_wandb=false in config to silence this message.",
        )
        return None


def make_dataloaders(config: Mapping[str, Any]):
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    samples = discover_iemocap_samples(
        dataset_cfg["iemocap_root"],
        auto_download=bool(dataset_cfg.get("auto_download", False)),
        kaggle_dataset=str(dataset_cfg.get("kaggle_dataset", "sangayb/iemocap")),
    )
    splits = split_loso_by_dialogue(
        samples,
        test_session=int(dataset_cfg.get("test_session", 5)),
        validation_ratio=float(dataset_cfg.get("validation_ratio", 0.1)),
        seed=int(config.get("seed", 42)),
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["wavlm_model_name"])
    collator = ConversationalSERCollator(feature_extractor, sampling_rate=int(dataset_cfg.get("sampling_rate", 16000)))

    def dataset_for(rows: Sequence[ConversationSERSample]) -> ConversationalSERDataset:
        return ConversationalSERDataset(
            rows,
            sampling_rate=int(dataset_cfg.get("sampling_rate", 16000)),
            max_duration_seconds=dataset_cfg.get("max_duration_seconds"),
        )

    train_loader = DataLoader(
        dataset_for(splits["train"]),
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        dataset_for(splits["validation"]),
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    test_loader = DataLoader(
        dataset_for(splits["test"]),
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 16))),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    return train_loader, val_loader, test_loader, splits


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def run_epoch(
    model: WavLMSERBaseline,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    max_grad_norm: float = 1.0,
    progress: bool = True,
    description: str = "",
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    prediction_rows: list[Dict[str, Any]] = []
    iterator = tqdm(dataloader, desc=description, disable=not progress, dynamic_ncols=True)

    for batch in iterator:
        batch = batch_to_device(batch, device)
        labels = batch["labels"]
        with torch.set_grad_enabled(is_train):
            output = model(
                input_values=batch["input_values"],
                attention_mask=batch.get("attention_mask"),
                labels=labels,
                dialogue_id=batch.get("dialogue_id"),
                speaker_id=batch.get("speaker_id"),
                start_time=batch.get("start_time"),
                end_time=batch.get("end_time"),
            )
            loss = output["loss"]
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        logits = output["logits"].detach()
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        batch_predictions = np.argmax(probabilities, axis=1).tolist()
        batch_targets = labels.detach().cpu().tolist()
        losses.append(float(loss.detach().cpu().item()))
        predictions.extend(int(value) for value in batch_predictions)
        targets.extend(int(value) for value in batch_targets)

        for index, pred_id in enumerate(batch_predictions):
            row = {
                "dialogue_id": batch["dialogue_id"][index],
                "utterance_id": batch["utterance_id"][index],
                "speaker_id": batch["speaker_id"][index],
                "start_time": float(batch["start_time"][index].detach().cpu().item()),
                "end_time": float(batch["end_time"][index].detach().cpu().item()),
                "gold_label": ID2LABEL[int(batch_targets[index])],
                "pred_label": ID2LABEL[int(pred_id)],
            }
            for label_idx, label_name in ID2LABEL.items():
                row[f"prob_{label_name}"] = float(probabilities[index][label_idx])
            prediction_rows.append(row)

        if progress:
            iterator.set_postfix(loss=f"{np.mean(losses):.4f}")

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "targets": targets,
        "predictions": predictions,
        "prediction_rows": prediction_rows,
    }


def save_checkpoint(path: Path, model: WavLMSERBaseline, config: Mapping[str, Any], epoch: int, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_name": config["experiment_name"],
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "epoch": epoch,
            "metrics": dict(metrics),
            "labels": LABEL_NAMES,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WavLM SER baseline without MAL/TIM.")
    parser.add_argument("--config", default="configs/wavlm_baseline_no_mal_no_tim.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    save_json(output_dir / "config.json", config)

    train_loader, val_loader, test_loader, splits = make_dataloaders(config)
    device = resolve_device(str(config["training"].get("device", "auto")))
    model = build_wavlm_ser_baseline(config["model"]).to(device)
    counts = parameter_counts(model)
    append_log(log_path, f"experiment={config['experiment_name']}")
    append_log(log_path, f"splits train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}")
    append_log(log_path, f"parameters total={counts['total']:,} trainable={counts['trainable']:,}")
    append_log(
        log_path,
        (
            f"freeze_wavlm={config['model']['freeze_wavlm']} "
            f"unfreeze_last_n_layers={config['model']['unfreeze_last_n_layers']}"
        ),
    )

    optimizer = create_optimizer(model, config)
    total_steps = max(1, len(train_loader) * int(config["training"].get("max_epochs", 10)))
    scheduler = create_scheduler(optimizer, config, total_steps)
    wandb_run = init_wandb(config, output_dir, log_path)

    best_ua = -1.0
    best_epoch = 0
    best_validation_metrics: Dict[str, Any] = {}
    max_epochs = int(config["training"].get("max_epochs", 10))
    progress = bool(config["training"].get("progress_bar", True))
    max_grad_norm = float(config["training"].get("gradient_clip", 1.0))

    for epoch in range(1, max_epochs + 1):
        train_output = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            progress=progress,
            description=f"{config['experiment_name']} epoch {epoch}/{max_epochs} train",
        )
        val_output = run_epoch(
            model,
            val_loader,
            device,
            progress=progress,
            description=f"{config['experiment_name']} epoch {epoch}/{max_epochs} validation",
        )
        train_metrics = compute_ser_metrics(train_output["targets"], train_output["predictions"], LABEL_NAMES, train_output["loss"], epoch)
        val_metrics = compute_ser_metrics(val_output["targets"], val_output["predictions"], LABEL_NAMES, val_output["loss"], epoch)
        append_log(
            log_path,
            (
                f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_WA={val_metrics['WA']:.6f} "
                f"val_UA={val_metrics['UA']:.6f} val_Macro-F1={val_metrics['Macro-F1']:.6f} "
                f"val_WF1={val_metrics['WF1']:.6f}"
            ),
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "validation/loss": val_metrics["loss"],
                    "validation/WA": val_metrics["WA"],
                    "validation/UA": val_metrics["UA"],
                    "validation/Macro-F1": val_metrics["Macro-F1"],
                    "validation/WF1": val_metrics["WF1"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        save_checkpoint(output_dir / "last.pth", model, config, epoch, val_metrics)
        if float(val_metrics["UA"]) > best_ua:
            best_ua = float(val_metrics["UA"])
            best_epoch = epoch
            best_validation_metrics = val_metrics
            save_checkpoint(output_dir / "best.pth", model, config, epoch, val_metrics)

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_output = run_epoch(
        model,
        test_loader,
        device,
        progress=progress,
        description=f"{config['experiment_name']} test",
    )
    test_metrics = compute_ser_metrics(test_output["targets"], test_output["predictions"], LABEL_NAMES, test_output["loss"], best_epoch)
    metrics_payload = {
        **test_metrics,
        "best_epoch": best_epoch,
        "best_validation": best_validation_metrics,
        "test": test_metrics,
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    save_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"], LABEL_NAMES)
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(output_dir / "confusion_matrix.png", test_metrics["confusion_matrix"], LABEL_NAMES)
    append_log(log_path, f"test_WA={test_metrics['WA']:.6f} test_UA={test_metrics['UA']:.6f}")
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_validation_UA"] = best_ua
        wandb_run.summary["test_UA"] = test_metrics["UA"]
        wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
