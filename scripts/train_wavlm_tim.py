from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.wavlm_tim import WavLMTIMSerModel, build_wavlm_tim_ser_model
from scripts.evaluate_temporal_subsets import save_temporal_subset_metrics
from utils.dialogue_embeddings import (
    DialogueEmbedding,
    TrainableWavLMMeanExtractor,
    build_audio_dialogues,
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
)
from utils.iemocap_kaggle import (
    LABEL_MAPPING_VERSION,
    ID2LABEL,
    LABEL_NAMES,
    add_dataset_override_args,
    apply_dataset_overrides,
    discover_ser_samples,
    split_samples_for_config,
)
from utils.temporal_features import (
    TEMPORAL_FEATURE_NAMES,
    TemporalInputPolicy,
    TemporalInteractionFeatureBuilder,
    attach_temporal_features_to_dialogues,
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


def trainable_parameters(*modules: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in modules:
        if module is None:
            continue
        params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    return params


def create_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any], total_steps: int):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

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
        append_log(log_path, f"wandb disabled: {exc}. Set wandb.use_wandb=false in config to silence this message.")
        return None


def class_weights_from_dialogues(dialogues: Sequence[DialogueEmbedding], num_labels: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_labels, dtype=torch.float32)
    for dialogue in dialogues:
        counts += torch.bincount(dialogue.labels, minlength=num_labels).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    return weights.to(device)


def cache_is_compatible(cache: Mapping[str, Any], config: Mapping[str, Any], expected_utterances: int) -> bool:
    metadata = cache.get("metadata", {})
    return (
        metadata.get("wavlm_model_name") == config["model"].get("wavlm_model_name")
        and int(metadata.get("sampling_rate", -1)) == int(config["dataset"].get("sampling_rate", 16000))
        and int(metadata.get("num_utterances", -1)) == int(expected_utterances)
        and metadata.get("label_mapping_version") == LABEL_MAPPING_VERSION
    )


def prepare_dialogues(config: Mapping[str, Any], device: torch.device, log_path: Path):
    dataset_cfg = config["dataset"]
    embedding_cfg = config["precompute"]
    samples = discover_ser_samples(dataset_cfg)
    splits = split_samples_for_config(samples, dataset_cfg, seed=int(config.get("seed", 42)))
    if not bool(embedding_cfg.get("enabled", True)):
        from transformers import AutoConfig

        embedding_dim = int(getattr(AutoConfig.from_pretrained(str(config["model"]["wavlm_model_name"])), "hidden_size"))
        append_log(log_path, "precompute_mode=disabled end_to_end_wavlm=true")
        return {
            split_name: build_audio_dialogues(split_samples, embedding_dim=embedding_dim)
            for split_name, split_samples in splits.items()
        }, splits

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
            wavlm_model_name=str(config["model"]["wavlm_model_name"]),
            sampling_rate=int(dataset_cfg.get("sampling_rate", 16000)),
            batch_size=int(embedding_cfg.get("batch_size", config["training"].get("eval_batch_size", 16))),
            num_workers=int(config["training"].get("num_workers", 0)),
            device=device,
            max_duration_seconds=dataset_cfg.get("max_duration_seconds"),
            progress=bool(config["training"].get("progress_bar", True)),
        )
        metadata = {
            "wavlm_model_name": str(config["model"]["wavlm_model_name"]),
            "sampling_rate": int(dataset_cfg.get("sampling_rate", 16000)),
            "num_utterances": len(all_split_samples),
            "pooling": "mean",
            "frozen_wavlm": True,
            "label_mapping_version": LABEL_MAPPING_VERSION,
        }
        save_embedding_cache(cache_path, rows_by_utterance, metadata)
    else:
        append_log(log_path, f"loaded_embedding_cache={cache_path}")
        rows_by_utterance = cache["rows_by_utterance"]

    dialogue_splits = {
        split_name: build_dialogue_embeddings(split_samples, rows_by_utterance)
        for split_name, split_samples in splits.items()
    }
    return dialogue_splits, splits


def save_tim_predictions_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dialogue_id",
        "utterance_id",
        "speaker_id",
        "start_time",
        "end_time",
        *TEMPORAL_FEATURE_NAMES,
        "gold_label",
        "pred_label",
        *[f"prob_{label}" for label in LABEL_NAMES],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_dialogue_epoch(
    model: WavLMTIMSerModel,
    dialogues: Sequence[DialogueEmbedding],
    temporal_builder: TemporalInteractionFeatureBuilder,
    temporal_policy: TemporalInputPolicy,
    device: torch.device,
    wavlm_extractor: TrainableWavLMMeanExtractor | None = None,
    wavlm_batch_size: int = 4,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    class_weights: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0,
    progress: bool = True,
    description: str = "",
) -> Dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    if wavlm_extractor is not None:
        wavlm_extractor.train(is_train)
    dialogue_order = list(dialogues)
    if is_train:
        random.shuffle(dialogue_order)

    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    prediction_rows: list[Dict[str, Any]] = []
    iterator = tqdm(dialogue_order, desc=description, disable=not progress, dynamic_ncols=True)
    for dialogue in iterator:
        embeddings = (
            wavlm_extractor.encode_rows(dialogue.rows, device=device, batch_size=wavlm_batch_size)
            if wavlm_extractor is not None
            else dialogue.embeddings.to(device)
        )
        labels = dialogue.labels.to(device)
        temporal_features = temporal_policy.apply(
            temporal_builder.transform_dialogue(dialogue), dialogue.dialogue_id
        ).to(device)
        with torch.set_grad_enabled(is_train):
            output = model(
                embeddings=embeddings,
                temporal_features=temporal_features,
                labels=None,
                dialogue_id=dialogue.dialogue_id,
            )
            logits = output["logits"]
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model, wavlm_extractor), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        batch_predictions = np.argmax(probabilities, axis=1).tolist()
        batch_targets = labels.detach().cpu().tolist()
        losses.append(float(loss.detach().cpu().item()))
        predictions.extend(int(value) for value in batch_predictions)
        targets.extend(int(value) for value in batch_targets)
        for index, pred_id in enumerate(batch_predictions):
            row = dialogue.rows[index]
            prediction_row = {
                "dialogue_id": row["dialogue_id"],
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "gold_label": ID2LABEL[int(batch_targets[index])],
                "pred_label": ID2LABEL[int(pred_id)],
            }
            for feature_name in TEMPORAL_FEATURE_NAMES:
                prediction_row[feature_name] = float(row[feature_name])
            for label_idx, label_name in ID2LABEL.items():
                prediction_row[f"prob_{label_name}"] = float(probabilities[index][label_idx])
            prediction_rows.append(prediction_row)
        if progress:
            iterator.set_postfix(loss=f"{np.mean(losses):.4f}")
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "targets": targets,
        "predictions": predictions,
        "prediction_rows": prediction_rows,
    }


def select_debug_dialogue(
    dialogue_splits: Mapping[str, Sequence[DialogueEmbedding]],
    split_name: str,
    dialogue_id: Optional[str] = None,
) -> DialogueEmbedding:
    dialogues = list(dialogue_splits[split_name])
    if not dialogues:
        raise ValueError(f"No dialogues available for debug split={split_name!r}.")
    if dialogue_id is None:
        return dialogues[0]
    for dialogue in dialogues:
        if dialogue.dialogue_id == dialogue_id:
            return dialogue
    raise ValueError(f"dialogue_id={dialogue_id!r} not found in split={split_name!r}.")


def save_memory_trace_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "split",
        "dialogue_id",
        "turn_index",
        "utterance_id",
        "start_time",
        "end_time",
        "memory_state_norm_before",
        "memory_state_norm_after",
        "gold_label",
        "pred_label",
    ]
    append = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def trace_tim_dialogue_memory(
    model: WavLMTIMSerModel,
    dialogue: DialogueEmbedding,
    temporal_builder: TemporalInteractionFeatureBuilder,
    temporal_policy: TemporalInputPolicy,
    device: torch.device,
    experiment_name: str,
    split_name: str,
) -> list[Dict[str, Any]]:
    model.eval()
    trace_rows: list[Dict[str, Any]] = []
    embeddings = dialogue.embeddings.to(device)
    labels = dialogue.labels.to(device)
    temporal_features = temporal_policy.apply(
        temporal_builder.transform_dialogue(dialogue), dialogue.dialogue_id
    ).to(device)
    memory = model.memory
    state = memory.initial_state(device=embeddings.device, dtype=embeddings.dtype)

    with torch.no_grad():
        temporal_embeddings = model.temporal_encoder(temporal_features)
        memory_inputs = torch.cat([embeddings, temporal_embeddings], dim=-1)
        for turn_index, (utterance_embedding, memory_input) in enumerate(zip(embeddings, memory_inputs)):
            memory_state_norm_before = float(torch.linalg.vector_norm(state).detach().cpu().item())
            z_i = memory.input_projection(memory_input)
            memory_read = memory.readout(torch.cat([z_i, state], dim=-1))
            state = memory.memory_cell(z_i.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
            memory_state_norm_after = float(torch.linalg.vector_norm(state).detach().cpu().item())
            fused = utterance_embedding + torch.tanh(model.alpha) * memory_read
            logits = model.classifier(fused.unsqueeze(0))
            pred_id = int(torch.argmax(logits, dim=-1).item())
            gold_id = int(labels[turn_index].detach().cpu().item())
            row = dialogue.rows[turn_index]
            trace_rows.append(
                {
                    "experiment": experiment_name,
                    "split": split_name,
                    "dialogue_id": dialogue.dialogue_id,
                    "turn_index": turn_index,
                    "utterance_id": row["utterance_id"],
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "memory_state_norm_before": memory_state_norm_before,
                    "memory_state_norm_after": memory_state_norm_after,
                    "gold_label": ID2LABEL[gold_id],
                    "pred_label": ID2LABEL[pred_id],
                }
            )
    return trace_rows


def save_checkpoint(
    path: Path,
    model: WavLMTIMSerModel,
    config: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, Any],
    wavlm_extractor: TrainableWavLMMeanExtractor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_name": config["experiment_name"],
            "model_state_dict": model.state_dict(),
            "wavlm_extractor_state_dict": wavlm_extractor.state_dict() if wavlm_extractor is not None else None,
            "config": dict(config),
            "epoch": epoch,
            "metrics": dict(metrics),
            "labels": LABEL_NAMES,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WavLM mean-embedding + TIM SER model.")
    parser.add_argument("--config", default="configs/wavlm_tim.yaml")
    add_dataset_override_args(parser)
    parser.add_argument("--debug_memory_trace", action="store_true", help="Save a dialogue-level memory trace for verification.")
    parser.add_argument("--debug_memory_split", choices=["validation", "test"], default="test")
    parser.add_argument("--debug_memory_dialogue_id", default=None)
    parser.add_argument("--debug_memory_trace_path", default="results/memory_trace_debug.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_overrides(config, args)
    if bool(config.get("cross_session", {}).get("enabled", False)):
        from scripts.run_cross_session import run_cross_session

        summary_path = run_cross_session("scripts.train_wavlm_tim", args.config)
        print(f"cross_session_summary={summary_path}")
        return
    if not bool(config["model"].get("use_temporal_features", False)):
        raise ValueError("TIM ablations require use_temporal_features=true to preserve the TIM architecture.")
    if int(config["model"].get("temporal_feature_dim", 16)) != len(TEMPORAL_FEATURE_NAMES):
        raise ValueError(f"temporal_feature_dim must be {len(TEMPORAL_FEATURE_NAMES)} for TIM.")
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    save_json(output_dir / "config.json", config)

    device = resolve_device(str(config["training"].get("device", "auto")))
    dialogue_splits, utterance_splits = prepare_dialogues(config, device, log_path)
    train_dialogues = dialogue_splits["train"]
    val_dialogues = dialogue_splits["validation"]
    test_dialogues = dialogue_splits["test"]

    temporal_builder = TemporalInteractionFeatureBuilder(
        short_gap_threshold=float(config["model"].get("short_gap_threshold", 0.3)),
        long_gap_threshold=float(config["model"].get("long_gap_threshold", 1.0)),
        overlap_threshold=float(config["model"].get("overlap_threshold", 0.05)),
    )
    temporal_builder.fit(train_dialogues)
    temporal_builder.save_stats(output_dir / "temporal_feature_stats.json")
    for dialogues in dialogue_splits.values():
        attach_temporal_features_to_dialogues(dialogues, temporal_builder)
    temporal_policy = TemporalInputPolicy.from_model_config(config["model"])

    embedding_dim = int(train_dialogues[0].embeddings.shape[-1])
    model = build_wavlm_tim_ser_model(config["model"], embedding_dim=embedding_dim).to(device)
    wavlm_extractor = None
    if not bool(config.get("precompute", {}).get("enabled", True)):
        wavlm_extractor = TrainableWavLMMeanExtractor(
            wavlm_model_name=str(config["model"]["wavlm_model_name"]),
            sampling_rate=int(config["dataset"].get("sampling_rate", 16000)),
            max_duration_seconds=config["dataset"].get("max_duration_seconds"),
            freeze_wavlm=bool(config["model"].get("freeze_wavlm", True)),
            unfreeze_last_n_layers=int(config["model"].get("unfreeze_last_n_layers", 0)),
        ).to(device)
    counts = parameter_counts(model)
    wavlm_counts = parameter_counts(wavlm_extractor) if wavlm_extractor is not None else {"total": 0, "trainable": 0, "frozen": 0}
    append_log(log_path, f"experiment={config['experiment_name']}")
    append_log(
        log_path,
        (
            f"splits train={len(utterance_splits['train'])}/{len(train_dialogues)}dialogues "
            f"validation={len(utterance_splits['validation'])}/{len(val_dialogues)}dialogues "
            f"test={len(utterance_splits['test'])}/{len(test_dialogues)}dialogues"
        ),
    )
    append_log(
        log_path,
        (
            f"embedding_dim={embedding_dim} pooling=mean "
            f"end_to_end_wavlm={wavlm_extractor is not None} "
            f"unfreeze_last_n_layers={config['model'].get('unfreeze_last_n_layers', 0)}"
        ),
    )
    append_log(log_path, f"temporal_feature_mode={config['model']['temporal_feature_mode']} dim={config['model']['temporal_feature_dim']}")
    append_log(
        log_path,
        "temporal_input_policy="
        f"mode={temporal_policy.mode} disabled_groups={list(temporal_policy.disabled_feature_groups)} "
        f"shuffle_seed={temporal_policy.shuffle_seed}",
    )
    append_log(log_path, "temporal_stats_source=train_split_only binary_flags_not_normalized=true")
    append_log(log_path, "memory_order=read_before_write reset=dialogue_boundary")
    append_log(log_path, f"parameters tim total={counts['total']:,} trainable={counts['trainable']:,}")
    if wavlm_extractor is not None:
        append_log(log_path, f"parameters wavlm total={wavlm_counts['total']:,} trainable={wavlm_counts['trainable']:,}")

    optimizer = torch.optim.AdamW(
        trainable_parameters(model, wavlm_extractor),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    total_steps = max(1, len(train_dialogues) * int(config["training"].get("max_epochs", 10)))
    scheduler = create_scheduler(optimizer, config, total_steps)
    wandb_run = init_wandb(config, output_dir, log_path)
    class_weights = (
        class_weights_from_dialogues(train_dialogues, int(config["model"].get("num_labels", 4)), device)
        if bool(config["training"].get("use_class_weights", False))
        else None
    )

    best_ua = -1.0
    best_epoch = 0
    best_validation_metrics: Dict[str, Any] = {}
    max_epochs = int(config["training"].get("max_epochs", 10))
    progress = bool(config["training"].get("progress_bar", True))
    max_grad_norm = float(config["training"].get("gradient_clip", 1.0))
    for epoch in range(1, max_epochs + 1):
        train_output = run_dialogue_epoch(
            model,
            train_dialogues,
            temporal_builder,
            temporal_policy,
            device,
            wavlm_extractor=wavlm_extractor,
            wavlm_batch_size=int(config["training"].get("wavlm_batch_size", 4)),
            optimizer=optimizer,
            scheduler=scheduler,
            class_weights=class_weights,
            max_grad_norm=max_grad_norm,
            progress=progress,
            description=f"{config['experiment_name']} epoch {epoch}/{max_epochs} train",
        )
        val_output = run_dialogue_epoch(
            model,
            val_dialogues,
            temporal_builder,
            temporal_policy,
            device,
            wavlm_extractor=wavlm_extractor,
            wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("wavlm_batch_size", 4))),
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

        save_checkpoint(output_dir / "last.pth", model, config, epoch, val_metrics, wavlm_extractor=wavlm_extractor)
        if float(val_metrics["UA"]) > best_ua:
            best_ua = float(val_metrics["UA"])
            best_epoch = epoch
            best_validation_metrics = val_metrics
            save_checkpoint(output_dir / "best.pth", model, config, epoch, val_metrics, wavlm_extractor=wavlm_extractor)

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if wavlm_extractor is not None and checkpoint.get("wavlm_extractor_state_dict") is not None:
        wavlm_extractor.load_state_dict(checkpoint["wavlm_extractor_state_dict"])
    test_output = run_dialogue_epoch(
        model,
        test_dialogues,
        temporal_builder,
        temporal_policy,
        device,
        wavlm_extractor=wavlm_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("wavlm_batch_size", 4))),
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
    predictions_path = output_dir / "predictions.csv"
    save_tim_predictions_csv(predictions_path, test_output["prediction_rows"])
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(output_dir / "confusion_matrix.png", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_temporal_subset_metrics(
        predictions_path=predictions_path,
        output_path=output_dir / "subset_metrics.json",
        strong_overlap_threshold=float(config["analysis"].get("strong_overlap_threshold", 0.25)),
    )
    append_log(log_path, f"test_WA={test_metrics['WA']:.6f} test_UA={test_metrics['UA']:.6f}")
    if args.debug_memory_trace:
        debug_dialogue = select_debug_dialogue(
            dialogue_splits,
            split_name=args.debug_memory_split,
            dialogue_id=args.debug_memory_dialogue_id,
        )
        trace_rows = trace_tim_dialogue_memory(
            model,
            debug_dialogue,
            temporal_builder,
            temporal_policy,
            device,
            experiment_name=str(config["experiment_name"]),
            split_name=args.debug_memory_split,
        )
        save_memory_trace_csv(args.debug_memory_trace_path, trace_rows)
        append_log(
            log_path,
            (
                f"debug_memory_trace_saved={args.debug_memory_trace_path} "
                f"split={args.debug_memory_split} dialogue_id={debug_dialogue.dialogue_id} "
                f"num_rows={len(trace_rows)}"
            ),
        )
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_validation_UA"] = best_ua
        wandb_run.summary["test_UA"] = test_metrics["UA"]
        wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
