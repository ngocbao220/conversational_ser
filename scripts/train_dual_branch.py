from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from models.wavlm_dual_branch_tim import WavLMDualBranchTIMSerModel, build_wavlm_dual_branch_tim_ser_model
from scripts.train_wavlm_tim import (
    append_log,
    class_weights_from_dialogues,
    create_scheduler,
    init_wandb,
    load_config,
    parameter_counts,
    prepare_dialogues,
    resolve_device,
    set_seed,
    trainable_parameters,
)
from utils.dialogue_embeddings import DialogueEmbedding
from utils.experiment_metrics import (
    compute_ser_metrics,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_json,
)
from utils.iemocap_kaggle import ID2LABEL, LABEL_NAMES, add_dataset_override_args, apply_dataset_overrides
from utils.temporal_features import (
    TEMPORAL_FEATURE_NAMES,
    TemporalInputPolicy,
    TemporalInteractionFeatureBuilder,
    attach_temporal_features_to_dialogues,
)


PREDICTION_TEMPORAL_COLUMNS = [
    "duration",
    "gap_prev",
    "overlap_prev",
    "overlap_ratio",
    "is_overlap",
    "is_interrupting_prev",
    "speaker_switch",
    "short_response",
    "long_pause",
]


def configure_trainable_gates(model: WavLMDualBranchTIMSerModel, model_cfg: Mapping[str, Any]) -> None:
    if bool(model_cfg.get("fix_alpha_zero", False)):
        with torch.no_grad():
            model.alpha.fill_(0.0)
        model.alpha.requires_grad_(False)
    if bool(model_cfg.get("fix_beta_zero", False)):
        with torch.no_grad():
            model.beta.fill_(0.0)
        model.beta.requires_grad_(False)


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def configure_phase_trainability(
    model: WavLMDualBranchTIMSerModel,
    phase_name: str,
    model_cfg: Mapping[str, Any],
) -> None:
    set_requires_grad(model, False)
    phase = str(phase_name)
    if phase == "end_to_end":
        set_requires_grad(model, True)
    elif phase == "phase_1_temporal":
        set_requires_grad(model.temporal_encoder, True)
        set_requires_grad(model.temporal_branch, True)
        set_requires_grad(model.classifier, True)
        with torch.no_grad():
            model.alpha.fill_(0.0)
        model.alpha.requires_grad_(False)
        model.beta.requires_grad_(True)
    elif phase == "phase_1_dialogue":
        set_requires_grad(model.dialogue_branch, True)
        set_requires_grad(model.classifier, True)
        model.alpha.requires_grad_(True)
        with torch.no_grad():
            model.beta.fill_(0.0)
        model.beta.requires_grad_(False)
    elif phase == "phase_2_dialogue":
        set_requires_grad(model.dialogue_branch, True)
        set_requires_grad(model.classifier, True)
        model.alpha.requires_grad_(True)
        model.beta.requires_grad_(False)
    elif phase == "phase_2_temporal":
        set_requires_grad(model.temporal_encoder, True)
        set_requires_grad(model.temporal_branch, True)
        set_requires_grad(model.classifier, True)
        model.beta.requires_grad_(True)
        model.alpha.requires_grad_(False)
    elif phase == "phase_3_fusion":
        set_requires_grad(model, True)
    else:
        raise ValueError(f"Unsupported training phase: {phase_name!r}")
    configure_trainable_gates(model, model_cfg)


def make_optimizer_and_scheduler(
    model: WavLMDualBranchTIMSerModel,
    config: Mapping[str, Any],
    train_dialogue_count: int,
    max_epochs: int,
) -> tuple[torch.optim.Optimizer, Any]:
    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters for the current phase.")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    total_steps = max(1, int(train_dialogue_count) * int(max_epochs))
    scheduler = create_scheduler(optimizer, config, total_steps)
    return optimizer, scheduler


def staged_phase_plan(training_stage: Mapping[str, Any], default_max_epochs: int) -> list[dict[str, Any]]:
    mode = str(training_stage.get("mode", "end_to_end"))
    if mode == "end_to_end":
        return [{"name": "end_to_end", "epochs": int(default_max_epochs), "enabled": True}]
    if mode == "3_phase":
        return [
            {
                "name": "phase_1_dialogue",
                "epochs": int(training_stage.get("stage_1_epochs", default_max_epochs)),
                "enabled": bool(training_stage.get("stage_1_train_dialogue_branch", True)),
            },
            {
                "name": "phase_2_temporal",
                "epochs": int(training_stage.get("stage_2_epochs", default_max_epochs)),
                "enabled": bool(training_stage.get("stage_2_freeze_dialogue_train_temporal", True)),
            },
            {
                "name": "phase_3_fusion",
                "epochs": int(training_stage.get("stage_3_epochs", max(1, default_max_epochs // 2))),
                "enabled": bool(training_stage.get("stage_3_finetune_fusion", True)),
            },
        ]
    if mode == "temporal_first_3_phase":
        return [
            {
                "name": "phase_1_temporal",
                "epochs": int(training_stage.get("stage_1_epochs", default_max_epochs)),
                "enabled": bool(training_stage.get("stage_1_train_temporal_branch", True)),
            },
            {
                "name": "phase_2_dialogue",
                "epochs": int(training_stage.get("stage_2_epochs", default_max_epochs)),
                "enabled": bool(training_stage.get("stage_2_train_dialogue_branch", True)),
            },
            {
                "name": "phase_3_fusion",
                "epochs": int(training_stage.get("stage_3_epochs", max(1, default_max_epochs // 2))),
                "enabled": bool(training_stage.get("stage_3_finetune_fusion", True)),
            },
        ]
    raise ValueError("training_stage.mode must be one of: end_to_end, 3_phase, temporal_first_3_phase")


def save_dual_branch_predictions_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dialogue_id",
        "utterance_id",
        "speaker_id",
        "start_time",
        "end_time",
        "gold_label",
        "pred_label",
        *[f"prob_{label}" for label in LABEL_NAMES],
        *PREDICTION_TEMPORAL_COLUMNS,
        "alpha_value",
        "beta_value",
        "dialogue_residual_norm",
        "temporal_residual_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def rows_for_subset(rows: Sequence[Mapping[str, Any]], subset_name: str, strong_overlap_threshold: float) -> list[Mapping[str, Any]]:
    if subset_name == "all":
        return list(rows)
    if subset_name == "no_overlap":
        return [row for row in rows if _float(row, "overlap_prev") <= 0.0]
    if subset_name == "any_overlap":
        return [row for row in rows if _float(row, "overlap_prev") > 0.0]
    if subset_name == "strong_overlap":
        return [row for row in rows if _float(row, "overlap_ratio") >= strong_overlap_threshold]
    if subset_name == "interrupting_prev":
        return [row for row in rows if _float(row, "is_interrupting_prev") >= 0.5]
    if subset_name == "short_response":
        return [row for row in rows if _float(row, "short_response") >= 0.5]
    if subset_name == "long_pause":
        return [row for row in rows if _float(row, "long_pause") >= 0.5]
    if subset_name == "high_temporal_interaction":
        return [
            row for row in rows
            if _float(row, "overlap_prev") > 0.0
            or _float(row, "is_interrupting_prev") >= 0.5
            or _float(row, "short_response") >= 0.5
            or _float(row, "long_pause") >= 0.5
        ]
    if subset_name == "low_temporal_interaction":
        return [
            row for row in rows
            if _float(row, "overlap_prev") <= 0.0
            and _float(row, "is_interrupting_prev") < 0.5
            and _float(row, "short_response") < 0.5
            and _float(row, "long_pause") < 0.5
        ]
    raise ValueError(f"Unknown temporal subset: {subset_name}")


def metrics_for_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    label_to_id = {label: idx for idx, label in enumerate(LABEL_NAMES)}
    targets = [label_to_id[str(row["gold_label"])] for row in rows]
    predictions = [label_to_id[str(row["pred_label"])] for row in rows]
    metrics = compute_ser_metrics(targets, predictions, LABEL_NAMES)
    metrics["num_samples"] = len(rows)
    return metrics


def save_dual_branch_temporal_subset_metrics(
    output_path: Path,
    prediction_rows: Sequence[Mapping[str, Any]],
    strong_overlap_threshold: float = 0.25,
) -> Dict[str, Any]:
    subset_names = [
        "all",
        "no_overlap",
        "any_overlap",
        "strong_overlap",
        "interrupting_prev",
        "short_response",
        "long_pause",
        "high_temporal_interaction",
        "low_temporal_interaction",
    ]
    metrics = {
        name: metrics_for_prediction_rows(rows_for_subset(prediction_rows, name, strong_overlap_threshold))
        for name in subset_names
    }
    save_json(output_path, metrics)
    return metrics


def run_dual_branch_dialogue_epoch(
    model: WavLMDualBranchTIMSerModel,
    dialogues: Sequence[DialogueEmbedding],
    temporal_builder: TemporalInteractionFeatureBuilder,
    temporal_policy: TemporalInputPolicy,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    class_weights: Optional[torch.Tensor] = None,
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
    residual_rows: list[Dict[str, Any]] = []
    iterator = tqdm(dialogue_order, desc=description, disable=not progress, dynamic_ncols=True)
    for dialogue in iterator:
        embeddings = dialogue.embeddings.to(device)
        labels = dialogue.labels.to(device)
        temporal_features = temporal_policy.apply(
            temporal_builder.transform_dialogue(dialogue), dialogue.dialogue_id
        ).to(device)

        with torch.set_grad_enabled(is_train):
            output = model(embeddings=embeddings, temporal_features=temporal_features, labels=None)
            logits = output["logits"]
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        batch_predictions = np.argmax(probabilities, axis=1).tolist()
        batch_targets = labels.detach().cpu().tolist()
        dialogue_norms = output["dialogue_residuals"].detach().norm(dim=-1).cpu().numpy()
        temporal_norms = output["temporal_residuals"].detach().norm(dim=-1).cpu().numpy()
        alpha_value = float(output["alpha_value"])
        beta_value = float(output["beta_value"])
        losses.append(float(loss.detach().cpu().item()))
        predictions.extend(int(value) for value in batch_predictions)
        targets.extend(int(value) for value in batch_targets)

        for index, pred_id in enumerate(batch_predictions):
            row = dialogue.rows[index]
            gold_id = int(batch_targets[index])
            prediction_row = {
                "dialogue_id": row["dialogue_id"],
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "gold_label": ID2LABEL[gold_id],
                "pred_label": ID2LABEL[int(pred_id)],
                "alpha_value": alpha_value,
                "beta_value": beta_value,
                "dialogue_residual_norm": float(dialogue_norms[index]),
                "temporal_residual_norm": float(temporal_norms[index]),
            }
            for feature_name in PREDICTION_TEMPORAL_COLUMNS:
                prediction_row[feature_name] = float(row.get(feature_name, 0.0))
            for label_idx, label_name in ID2LABEL.items():
                prediction_row[f"prob_{label_name}"] = float(probabilities[index][label_idx])
            prediction_rows.append(prediction_row)
            residual_rows.append(
                {
                    **prediction_row,
                    "correct": bool(pred_id == gold_id),
                }
            )
        if progress:
            iterator.set_postfix(loss=f"{np.mean(losses):.4f}")

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "targets": targets,
        "predictions": predictions,
        "prediction_rows": prediction_rows,
        "residual_rows": residual_rows,
    }


def save_checkpoint(path: Path, model: WavLMDualBranchTIMSerModel, config: Mapping[str, Any], epoch: int, metrics: Mapping[str, Any]) -> None:
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


def residual_subset_name(row: Mapping[str, Any], strong_overlap_threshold: float) -> str:
    if _float(row, "overlap_ratio") >= strong_overlap_threshold:
        return "strong_overlap"
    if _float(row, "overlap_prev") > 0.0 or _float(row, "is_interrupting_prev") >= 0.5:
        return "any_overlap"
    if _float(row, "long_pause") >= 0.5:
        return "long_pause"
    if _float(row, "short_response") >= 0.5:
        return "short_response"
    return "low_temporal_interaction"


def save_branch_gate_stats(
    path: Path,
    model: WavLMDualBranchTIMSerModel,
    residual_rows: Sequence[Mapping[str, Any]],
    strong_overlap_threshold: float,
) -> Dict[str, Any]:
    dialogue_norms = np.asarray([_float(row, "dialogue_residual_norm") for row in residual_rows], dtype=np.float32)
    temporal_norms = np.asarray([_float(row, "temporal_residual_norm") for row in residual_rows], dtype=np.float32)
    alpha_value = float(torch.tanh(model.alpha).detach().cpu().item())
    beta_value = float(torch.tanh(model.beta).detach().cpu().item())

    by_emotion: Dict[str, Any] = {}
    for label in LABEL_NAMES:
        rows = [row for row in residual_rows if row["gold_label"] == label]
        if rows:
            by_emotion[label] = {
                "count": len(rows),
                "mean_dialogue_residual_norm": float(np.mean([_float(row, "dialogue_residual_norm") for row in rows])),
                "mean_temporal_residual_norm": float(np.mean([_float(row, "temporal_residual_norm") for row in rows])),
            }

    by_subset: Dict[str, Any] = {}
    for subset in [
        "strong_overlap",
        "any_overlap",
        "long_pause",
        "short_response",
        "low_temporal_interaction",
    ]:
        rows = [row for row in residual_rows if residual_subset_name(row, strong_overlap_threshold) == subset]
        if rows:
            by_subset[subset] = {
                "count": len(rows),
                "mean_dialogue_residual_norm": float(np.mean([_float(row, "dialogue_residual_norm") for row in rows])),
                "mean_temporal_residual_norm": float(np.mean([_float(row, "temporal_residual_norm") for row in rows])),
            }

    high_rows = [
        row for row in residual_rows
        if row["utterance_id"] and residual_subset_name(row, strong_overlap_threshold) != "low_temporal_interaction"
    ]
    low_rows = [
        row for row in residual_rows
        if residual_subset_name(row, strong_overlap_threshold) == "low_temporal_interaction"
    ]
    high_temporal_norm = float(np.mean([_float(row, "temporal_residual_norm") for row in high_rows])) if high_rows else 0.0
    low_temporal_norm = float(np.mean([_float(row, "temporal_residual_norm") for row in low_rows])) if low_rows else 0.0
    payload = {
        "alpha_value": alpha_value,
        "beta_value": beta_value,
        "mean_dialogue_residual_norm": float(dialogue_norms.mean()) if dialogue_norms.size else 0.0,
        "mean_temporal_residual_norm": float(temporal_norms.mean()) if temporal_norms.size else 0.0,
        "residual_norm_by_emotion": by_emotion,
        "residual_norm_by_subset": by_subset,
        "beta_remains_near_zero": abs(beta_value) < 0.05,
        "temporal_branch_higher_on_high_temporal_interaction": high_temporal_norm > low_temporal_norm,
        "high_temporal_interaction_mean_temporal_residual_norm": high_temporal_norm,
        "low_temporal_interaction_mean_temporal_residual_norm": low_temporal_norm,
    }
    save_json(path, payload)
    return payload


def write_single_run_ablation_metrics(output_dir: Path, config: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    row = {
        "name": str(config.get("run_name", config.get("experiment_name", "dual_branch_full"))),
        "WA": float(metrics["WA"]),
        "UA": float(metrics["UA"]),
        "WF1": float(metrics["WF1"]),
        "Macro-F1": float(metrics["Macro-F1"]),
        "output_dir": str(output_dir),
    }
    csv_path = output_dir / "ablation_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    save_json(output_dir / "ablation_metrics.json", {"runs": [row]})


def train_model_with_phase_plan(
    model: WavLMDualBranchTIMSerModel,
    train_dialogues: Sequence[DialogueEmbedding],
    val_dialogues: Sequence[DialogueEmbedding],
    temporal_builder: TemporalInteractionFeatureBuilder,
    temporal_policy: TemporalInputPolicy,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    log_path: Path,
    wandb_run: Any,
    class_weights: Optional[torch.Tensor],
) -> tuple[int, float, Dict[str, Any], list[Dict[str, Any]]]:
    training_stage = config.get("training_stage", {"mode": "end_to_end"})
    max_epochs = int(config["training"].get("max_epochs", 10))
    phase_plan = [phase for phase in staged_phase_plan(training_stage, max_epochs) if phase["enabled"] and int(phase["epochs"]) > 0]
    if not phase_plan:
        raise ValueError("No enabled training phases. Check training_stage config.")

    best_ua = -1.0
    best_epoch = 0
    best_validation_metrics: Dict[str, Any] = {}
    phase_summaries: list[Dict[str, Any]] = []
    global_epoch = 0
    progress = bool(config["training"].get("progress_bar", True))
    max_grad_norm = float(config["training"].get("gradient_clip", 1.0))

    for phase in phase_plan:
        phase_name = str(phase["name"])
        phase_epochs = int(phase["epochs"])
        configure_phase_trainability(model, phase_name, config["model"])
        optimizer, scheduler = make_optimizer_and_scheduler(
            model,
            config,
            train_dialogue_count=len(train_dialogues),
            max_epochs=phase_epochs,
        )
        counts = parameter_counts(model)
        phase_best_ua = -1.0
        phase_best_epoch = 0
        phase_best_metrics: Dict[str, Any] = {}
        phase_best_path = output_dir / f"{phase_name}_best.pth"
        append_log(
            log_path,
            (
                f"phase_start={phase_name} epochs={phase_epochs} "
                f"trainable={counts['trainable']:,} alpha_trainable={model.alpha.requires_grad} "
                f"beta_trainable={model.beta.requires_grad}"
            ),
        )
        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            train_output = run_dual_branch_dialogue_epoch(
                model,
                train_dialogues,
                temporal_builder,
                temporal_policy,
                device,
                optimizer=optimizer,
                scheduler=scheduler,
                class_weights=class_weights,
                max_grad_norm=max_grad_norm,
                progress=progress,
                description=f"{config['experiment_name']} {phase_name} {phase_epoch}/{phase_epochs} train",
            )
            val_output = run_dual_branch_dialogue_epoch(
                model,
                val_dialogues,
                temporal_builder,
                temporal_policy,
                device,
                progress=progress,
                description=f"{config['experiment_name']} {phase_name} {phase_epoch}/{phase_epochs} validation",
            )
            train_metrics = compute_ser_metrics(
                train_output["targets"], train_output["predictions"], LABEL_NAMES, train_output["loss"], global_epoch
            )
            val_metrics = compute_ser_metrics(
                val_output["targets"], val_output["predictions"], LABEL_NAMES, val_output["loss"], global_epoch
            )
            append_log(
                log_path,
                (
                    f"phase={phase_name} phase_epoch={phase_epoch} global_epoch={global_epoch} "
                    f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                    f"val_WA={val_metrics['WA']:.6f} val_UA={val_metrics['UA']:.6f} "
                    f"val_Macro-F1={val_metrics['Macro-F1']:.6f} "
                    f"alpha={float(torch.tanh(model.alpha).detach().cpu().item()):.6f} "
                    f"beta={float(torch.tanh(model.beta).detach().cpu().item()):.6f}"
                ),
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": global_epoch,
                        "phase": phase_name,
                        "phase_epoch": phase_epoch,
                        "train/loss": train_metrics["loss"],
                        "validation/loss": val_metrics["loss"],
                        "validation/WA": val_metrics["WA"],
                        "validation/UA": val_metrics["UA"],
                        "validation/Macro-F1": val_metrics["Macro-F1"],
                        "validation/WF1": val_metrics["WF1"],
                        "gate/alpha": float(torch.tanh(model.alpha).detach().cpu().item()),
                        "gate/beta": float(torch.tanh(model.beta).detach().cpu().item()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=global_epoch,
                )

            save_checkpoint(output_dir / "last.pth", model, config, global_epoch, val_metrics)
            save_checkpoint(output_dir / f"{phase_name}_last.pth", model, config, global_epoch, val_metrics)
            if float(val_metrics["UA"]) > best_ua:
                best_ua = float(val_metrics["UA"])
                best_epoch = global_epoch
                best_validation_metrics = val_metrics
                save_checkpoint(output_dir / "best.pth", model, config, global_epoch, val_metrics)
            if float(val_metrics["UA"]) > phase_best_ua:
                phase_best_ua = float(val_metrics["UA"])
                phase_best_epoch = global_epoch
                phase_best_metrics = val_metrics
                save_checkpoint(phase_best_path, model, config, global_epoch, val_metrics)
        phase_summary = {
            "phase": phase_name,
            "epochs": phase_epochs,
            "best_epoch": phase_best_epoch,
            "best_validation_UA": phase_best_ua,
            "best_checkpoint": str(phase_best_path),
            "best_validation": phase_best_metrics,
        }
        phase_summaries.append(phase_summary)
        append_log(
            log_path,
            f"phase_end={phase_name} best_epoch={phase_best_epoch} best_val_UA={phase_best_ua:.6f} checkpoint={phase_best_path}",
        )
    save_json(output_dir / "phase_training_summary.json", {"phases": phase_summaries})
    return best_epoch, best_ua, best_validation_metrics, phase_summaries


def evaluate_checkpoint_on_test(
    checkpoint_path: Path,
    model: WavLMDualBranchTIMSerModel,
    test_dialogues: Sequence[DialogueEmbedding],
    temporal_builder: TemporalInteractionFeatureBuilder,
    temporal_policy: TemporalInputPolicy,
    device: torch.device,
    config: Mapping[str, Any],
    output_dir: Path,
    phase_name: str,
    progress: bool,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_output = run_dual_branch_dialogue_epoch(
        model,
        test_dialogues,
        temporal_builder,
        temporal_policy,
        device,
        progress=progress,
        description=f"{config['experiment_name']} {phase_name} test",
    )
    test_metrics = compute_ser_metrics(
        test_output["targets"],
        test_output["predictions"],
        LABEL_NAMES,
        test_output["loss"],
        int(checkpoint.get("epoch", 0)),
    )
    phase_dir = output_dir / "phase_tests" / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **test_metrics,
        "phase": phase_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "validation_at_checkpoint": checkpoint.get("metrics", {}),
    }
    save_json(phase_dir / "metrics.json", metrics_payload)
    save_dual_branch_predictions_csv(phase_dir / "predictions.csv", test_output["prediction_rows"])
    save_confusion_matrix_csv(phase_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(phase_dir / "confusion_matrix.png", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_branch_gate_stats(
        phase_dir / "branch_gate_stats.json",
        model,
        test_output["residual_rows"],
        strong_overlap_threshold=float(config["analysis"].get("strong_overlap_threshold", 0.25)),
    )
    return metrics_payload


def save_phase_test_summary(output_dir: Path, phase_metrics: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for payload in phase_metrics:
        rows.append(
            {
                "phase": payload["phase"],
                "checkpoint_epoch": payload["checkpoint_epoch"],
                "validation_UA": float(payload.get("validation_at_checkpoint", {}).get("UA", 0.0)),
                "test_WA": float(payload["WA"]),
                "test_UA": float(payload["UA"]),
                "test_WF1": float(payload["WF1"]),
                "test_Macro-F1": float(payload["Macro-F1"]),
                "checkpoint": payload["checkpoint"],
            }
        )
    csv_path = output_dir / "phase_test_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["phase"])
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_dir / "phase_test_metrics.json", {"phases": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-branch WavLM + temporal dialogue memory SER model.")
    parser.add_argument("--config", default="configs/dual_branch.yaml")
    add_dataset_override_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_overrides(config, args)
    if bool(config.get("cross_session", {}).get("enabled", False)):
        from scripts.run_cross_session import run_cross_session

        summary_path = run_cross_session("scripts.train_dual_branch", args.config)
        print(f"cross_session_summary={summary_path}")
        return

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
    model = build_wavlm_dual_branch_tim_ser_model(config["model"], embedding_dim=embedding_dim).to(device)
    configure_trainable_gates(model, config["model"])
    counts = parameter_counts(model)
    append_log(log_path, f"experiment={config['experiment_name']}")
    append_log(log_path, f"run_name={config.get('run_name', '')}")
    append_log(
        log_path,
        (
            f"splits train={len(utterance_splits['train'])}/{len(train_dialogues)}dialogues "
            f"validation={len(utterance_splits['validation'])}/{len(val_dialogues)}dialogues "
            f"test={len(utterance_splits['test'])}/{len(test_dialogues)}dialogues"
        ),
    )
    append_log(log_path, f"embedding_dim={embedding_dim} dual_branch=true precomputed_wavlm={bool(config['precompute'].get('enabled', True))}")
    append_log(log_path, f"parameters total={counts['total']:,} trainable={counts['trainable']:,}")
    append_log(
        log_path,
        "temporal_input_policy="
        f"mode={temporal_policy.mode} disabled_groups={list(temporal_policy.disabled_feature_groups)} "
        f"shuffle_seed={temporal_policy.shuffle_seed}",
    )
    append_log(log_path, f"initial_alpha={float(torch.tanh(model.alpha).detach().cpu().item()):.6f} initial_beta={float(torch.tanh(model.beta).detach().cpu().item()):.6f}")
    training_stage = config.get("training_stage", {"mode": "end_to_end"})
    append_log(log_path, f"training_stage={training_stage.get('mode', 'end_to_end')}")

    wandb_run = init_wandb(config, output_dir, log_path)
    class_weights = (
        class_weights_from_dialogues(train_dialogues, int(config["model"].get("num_labels", 4)), device)
        if bool(config["training"].get("use_class_weights", False))
        else None
    )
    best_epoch, best_ua, best_validation_metrics, phase_summaries = train_model_with_phase_plan(
        model=model,
        train_dialogues=train_dialogues,
        val_dialogues=val_dialogues,
        temporal_builder=temporal_builder,
        temporal_policy=temporal_policy,
        config=config,
        device=device,
        output_dir=output_dir,
        log_path=log_path,
        wandb_run=wandb_run,
        class_weights=class_weights,
    )

    progress = bool(config["training"].get("progress_bar", True))
    phase_test_metrics = []
    for phase_summary in phase_summaries:
        checkpoint_path = Path(str(phase_summary["best_checkpoint"]))
        if not checkpoint_path.exists():
            append_log(log_path, f"phase_test_skip={phase_summary['phase']} missing_checkpoint={checkpoint_path}")
            continue
        metrics = evaluate_checkpoint_on_test(
            checkpoint_path=checkpoint_path,
            model=model,
            test_dialogues=test_dialogues,
            temporal_builder=temporal_builder,
            temporal_policy=temporal_policy,
            device=device,
            config=config,
            output_dir=output_dir,
            phase_name=str(phase_summary["phase"]),
            progress=progress,
        )
        phase_test_metrics.append(metrics)
        append_log(
            log_path,
            (
                f"phase_test={phase_summary['phase']} "
                f"checkpoint_epoch={metrics['checkpoint_epoch']} "
                f"test_WA={metrics['WA']:.6f} test_UA={metrics['UA']:.6f} "
                f"test_Macro-F1={metrics['Macro-F1']:.6f}"
            ),
        )
    if phase_test_metrics:
        save_phase_test_summary(output_dir, phase_test_metrics)

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_output = run_dual_branch_dialogue_epoch(
        model,
        test_dialogues,
        temporal_builder,
        temporal_policy,
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
    predictions_path = output_dir / "predictions.csv"
    save_dual_branch_predictions_csv(predictions_path, test_output["prediction_rows"])
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(output_dir / "confusion_matrix.png", test_metrics["confusion_matrix"], LABEL_NAMES)
    subset_metrics = save_dual_branch_temporal_subset_metrics(
        output_dir / "temporal_subset_metrics.json",
        test_output["prediction_rows"],
        strong_overlap_threshold=float(config["analysis"].get("strong_overlap_threshold", 0.25)),
    )
    save_branch_gate_stats(
        output_dir / "branch_gate_stats.json",
        model,
        test_output["residual_rows"],
        strong_overlap_threshold=float(config["analysis"].get("strong_overlap_threshold", 0.25)),
    )
    write_single_run_ablation_metrics(output_dir, config, test_metrics)
    append_log(log_path, f"test_WA={test_metrics['WA']:.6f} test_UA={test_metrics['UA']:.6f}")
    append_log(log_path, f"subset_metrics_saved={output_dir / 'temporal_subset_metrics.json'} subsets={','.join(subset_metrics.keys())}")

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_validation_UA"] = best_ua
        wandb_run.summary["test_UA"] = test_metrics["UA"]
        wandb_run.summary["test_WA"] = test_metrics["WA"]
        wandb_run.summary["alpha"] = float(torch.tanh(model.alpha).detach().cpu().item())
        wandb_run.summary["beta"] = float(torch.tanh(model.beta).detach().cpu().item())
        wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
