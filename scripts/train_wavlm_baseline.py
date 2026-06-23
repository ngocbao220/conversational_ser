from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from models.wavlm_baseline import MeanEmbeddingBaseline, build_mean_embedding_baseline
from utils.dialogue_embeddings import (
    DialogueEmbedding,
    build_dialogue_embeddings,
    load_embedding_cache,
    precompute_wavlm_mean_embeddings,
    save_embedding_cache,
)
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


def create_optimizer(model: MeanEmbeddingBaseline, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    classifier_lr = float(training_cfg.get("learning_rate_classifier", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(trainable_params, lr=classifier_lr, weight_decay=weight_decay)


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


def cache_is_compatible(cache: Mapping[str, Any], config: Mapping[str, Any], expected_utterances: int) -> bool:
    metadata = cache.get("metadata", {})
    return (
        metadata.get("wavlm_model_name") == config["model"].get("wavlm_model_name")
        and int(metadata.get("sampling_rate", -1)) == int(config["dataset"].get("sampling_rate", 16000))
        and int(metadata.get("num_utterances", -1)) == int(expected_utterances)
        and metadata.get("pooling") == "mean"
    )


def prepare_dialogues(config: Mapping[str, Any], device: torch.device, log_path: Path):
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    embedding_cfg = config["precompute"]
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
    all_split_samples = [sample for split_samples in splits.values() for sample in split_samples]
    cache_path = Path(str(embedding_cfg.get("cache_path", Path(config["output_dir"]) / "cache" / "wavlm_mean_embeddings.pt")))
    force_recompute = bool(embedding_cfg.get("force_recompute", False))
    cache = None
    if cache_path.exists() and not force_recompute:
        cache = load_embedding_cache(cache_path)
        if not cache_is_compatible(cache, config, expected_utterances=len(all_split_samples)):
            append_log(log_path, f"embedding cache incompatible, recomputing: {cache_path}")
            cache = None

    if cache is None:
        append_log(log_path, "precompute_mode=fixed_mean_pooled_wavlm")
        rows_by_utterance = precompute_wavlm_mean_embeddings(
            all_split_samples,
            wavlm_model_name=str(model_cfg["wavlm_model_name"]),
            sampling_rate=int(dataset_cfg.get("sampling_rate", 16000)),
            batch_size=int(embedding_cfg.get("batch_size", config["training"].get("eval_batch_size", 16))),
            num_workers=int(config["training"].get("num_workers", 0)),
            device=device,
            max_duration_seconds=dataset_cfg.get("max_duration_seconds"),
            progress=bool(config["training"].get("progress_bar", True)),
        )
        save_embedding_cache(
            cache_path,
            rows_by_utterance,
            {
                "wavlm_model_name": str(model_cfg["wavlm_model_name"]),
                "sampling_rate": int(dataset_cfg.get("sampling_rate", 16000)),
                "num_utterances": len(all_split_samples),
                "pooling": "mean",
                "frozen_wavlm": True,
            },
        )
    else:
        append_log(log_path, f"loaded_embedding_cache={cache_path}")
        rows_by_utterance = cache["rows_by_utterance"]

    dialogue_splits = {
        split_name: build_dialogue_embeddings(split_samples, rows_by_utterance)
        for split_name, split_samples in splits.items()
    }
    return dialogue_splits, splits


def run_epoch(
    model: MeanEmbeddingBaseline,
    dialogues: Sequence[DialogueEmbedding],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    max_grad_norm: float = 1.0,
    progress: bool = True,
    description: str = "",
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    dialogue_order = list(dialogues)
    if is_train:
        random.shuffle(dialogue_order)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    prediction_rows: list[Dict[str, Any]] = []
    iterator = tqdm(dialogue_order, desc=description, disable=not progress, dynamic_ncols=True)

    for dialogue in iterator:
        embeddings = dialogue.embeddings.to(device)
        labels = dialogue.labels.to(device)
        with torch.set_grad_enabled(is_train):
            output = model(embeddings=embeddings, labels=labels)
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
            row = dialogue.rows[index]
            row = {
                "dialogue_id": row["dialogue_id"],
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
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


def save_checkpoint(path: Path, model: MeanEmbeddingBaseline, config: Mapping[str, Any], epoch: int, metrics: Mapping[str, Any]) -> None:
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
    parser = argparse.ArgumentParser(description="Train a fixed mean-pooled WavLM embedding baseline without MAL/TIM.")
    parser.add_argument("--config", default="configs/wavlm_baseline_no_mal_no_tim.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if bool(config.get("cross_session", {}).get("enabled", False)):
        from scripts.run_cross_session import run_cross_session

        summary_path = run_cross_session("scripts.train_wavlm_baseline", args.config)
        print(f"cross_session_summary={summary_path}")
        return
    if str(config["model"].get("pooling", "mean")) != "mean":
        raise ValueError("Cached baseline requires model.pooling=mean to match MAL/TIM embeddings.")
    if not bool(config.get("precompute", {}).get("enabled", True)):
        raise ValueError("Cached baseline requires precompute.enabled=true.")
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    save_json(output_dir / "config.json", config)

    device = resolve_device(str(config["training"].get("device", "auto")))
    dialogue_splits, splits = prepare_dialogues(config, device, log_path)
    train_dialogues = dialogue_splits["train"]
    val_dialogues = dialogue_splits["validation"]
    test_dialogues = dialogue_splits["test"]
    embedding_dim = int(train_dialogues[0].embeddings.shape[-1])
    model = build_mean_embedding_baseline(config["model"], embedding_dim=embedding_dim).to(device)
    counts = parameter_counts(model)
    append_log(log_path, f"experiment={config['experiment_name']}")
    append_log(log_path, f"splits train={len(splits['train'])} validation={len(splits['validation'])} test={len(splits['test'])}")
    append_log(log_path, f"embedding_dim={embedding_dim} pooling=mean frozen_wavlm=true")
    append_log(log_path, f"parameters total={counts['total']:,} trainable={counts['trainable']:,}")

    optimizer = create_optimizer(model, config)
    total_steps = max(1, len(train_dialogues) * int(config["training"].get("max_epochs", 10)))
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
            train_dialogues,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            progress=progress,
            description=f"{config['experiment_name']} epoch {epoch}/{max_epochs} train",
        )
        val_output = run_epoch(
            model,
            val_dialogues,
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
        test_dialogues,
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
