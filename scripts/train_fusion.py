from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import yaml

from models.wavlm_cdm import build_wavlm_cdm_ser_model
from models.wavlm_cim import build_wavlm_cim_ser_model
from scripts.train_cdm import make_temporal_features
from scripts.train_cim import (
    append_log,
    create_scheduler,
    init_wandb,
    prepare_dialogues,
    resolve_device,
    set_seed,
    trainable_parameters,
)
from utils.dialogue_embeddings import DialogueEmbedding, TrainableWavLMMeanExtractor
from utils.experiment_metrics import compute_ser_metrics, save_confusion_matrix_csv, save_confusion_matrix_png, save_json
from utils.iemocap_kaggle import ID2LABEL, LABEL_NAMES, add_dataset_override_args, apply_dataset_overrides
from utils.temporal_features import TemporalInputPolicy, TemporalInteractionFeatureBuilder, attach_temporal_features_to_dialogues


class LogitFusionHead(nn.Module):
    def __init__(self, num_labels: int = 4, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_labels * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, cdm_logits: torch.Tensor, cim_logits: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cdm_logits, cim_logits], dim=-1))


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def checkpoint_for_session(root: str | Path, test_session: int, filename: str = "best.pth") -> Path:
    root = Path(root)
    if root.is_file():
        return root
    return root / f"test_Ses{test_session:02d}" / filename


def load_frozen_branches(config: Mapping[str, Any], embedding_dim: int, device: torch.device, log_path: Path):
    fusion_cfg = config.get("fusion", {})
    test_session = int(config.get("dataset", {}).get("test_session", 5))
    cdm_path = checkpoint_for_session(fusion_cfg["cdm_checkpoint_root"], test_session, str(fusion_cfg.get("checkpoint_name", "best.pth")))
    cim_path = checkpoint_for_session(fusion_cfg["cim_checkpoint_root"], test_session, str(fusion_cfg.get("checkpoint_name", "best.pth")))
    if not cdm_path.exists():
        raise FileNotFoundError(f"Missing CDM checkpoint: {cdm_path}")
    if not cim_path.exists():
        raise FileNotFoundError(f"Missing CIM checkpoint: {cim_path}")

    cdm_payload = torch.load(cdm_path, map_location=device)
    cim_payload = torch.load(cim_path, map_location=device)
    cdm_cfg = cdm_payload.get("config", config).get("model", config["model"])
    cim_cfg = cim_payload.get("config", config).get("model", config["model"])

    cdm_model = build_wavlm_cdm_ser_model(cdm_cfg, embedding_dim=embedding_dim).to(device)
    cim_model = build_wavlm_cim_ser_model(cim_cfg, embedding_dim=embedding_dim).to(device)
    cdm_model.load_state_dict(cdm_payload["model_state_dict"])
    cim_model.load_state_dict(cim_payload["model_state_dict"])
    cdm_model.eval()
    cim_model.eval()
    for model in (cdm_model, cim_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    append_log(log_path, f"loaded_frozen_cdm={cdm_path}")
    append_log(log_path, f"loaded_frozen_cim={cim_path}")
    return cdm_model, cim_model


def save_predictions_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dialogue_id",
        "utterance_id",
        "speaker_id",
        "start_time",
        "end_time",
        "raw_label",
        "is_target_label",
        "gold_label",
        "pred_label",
        *[f"prob_{label}" for label in LABEL_NAMES],
        *[f"cdm_prob_{label}" for label in LABEL_NAMES],
        *[f"cim_prob_{label}" for label in LABEL_NAMES],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_epoch(
    fusion_head: LogitFusionHead,
    cdm_model: nn.Module,
    cim_model: nn.Module,
    dialogues: Sequence[DialogueEmbedding],
    temporal_builder: TemporalInteractionFeatureBuilder,
    temporal_policy: TemporalInputPolicy,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    max_grad_norm: float = 1.0,
    shuffle: bool = False,
) -> dict[str, Any]:
    is_train = optimizer is not None
    fusion_head.train(is_train)
    dialogue_order = list(dialogues)
    if shuffle:
        random.shuffle(dialogue_order)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    prediction_rows: list[dict[str, Any]] = []

    for dialogue in dialogue_order:
        embeddings = dialogue.embeddings.to(device)
        labels = dialogue.labels.to(device)
        valid_mask = labels >= 0
        has_valid = bool(valid_mask.any().item())
        cim_temporal_features = temporal_policy.apply(
            temporal_builder.transform_dialogue(dialogue), dialogue.dialogue_id
        ).to(device)
        cdm_temporal_features = make_temporal_features(
            embeddings.shape[0], int(cdm_model.config.temporal_feature_dim), device
        )

        with torch.no_grad():
            cdm_logits = cdm_model(embeddings=embeddings, temporal_features=cdm_temporal_features, labels=None)["logits"]
            cim_logits = cim_model(embeddings=embeddings, temporal_features=cim_temporal_features, labels=None)["logits"]
        with torch.set_grad_enabled(is_train):
            logits = fusion_head(cdm_logits.detach(), cim_logits.detach())
            loss = (
                torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
                if has_valid
                else logits.sum() * 0.0
            )
            if is_train and has_valid:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(fusion_head), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if has_valid:
            losses.append(float(loss.detach().cpu().item()))
        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        cdm_probabilities = torch.softmax(cdm_logits.detach(), dim=-1).cpu().numpy()
        cim_probabilities = torch.softmax(cim_logits.detach(), dim=-1).cpu().numpy()
        batch_predictions = np.argmax(probabilities, axis=1).tolist()
        batch_targets = labels.detach().cpu().tolist()
        for index, pred_id in enumerate(batch_predictions):
            row = dialogue.rows[index]
            gold_id = int(batch_targets[index])
            is_target_label = gold_id >= 0
            if is_target_label:
                predictions.append(int(pred_id))
                targets.append(gold_id)
            prediction_row = {
                "dialogue_id": row["dialogue_id"],
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "raw_label": row.get("raw_label", ""),
                "is_target_label": is_target_label,
                "gold_label": ID2LABEL.get(gold_id, "context"),
                "pred_label": ID2LABEL[int(pred_id)],
            }
            for label_idx, label_name in enumerate(LABEL_NAMES):
                prediction_row[f"prob_{label_name}"] = float(probabilities[index][label_idx])
                prediction_row[f"cdm_prob_{label_name}"] = float(cdm_probabilities[index][label_idx])
                prediction_row[f"cim_prob_{label_name}"] = float(cim_probabilities[index][label_idx])
            prediction_rows.append(prediction_row)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "targets": targets,
        "predictions": predictions,
        "prediction_rows": prediction_rows,
    }


def save_checkpoint(path: Path, fusion_head: LogitFusionHead, config: Mapping[str, Any], epoch: int, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_name": config["experiment_name"],
            "fusion_head_state_dict": fusion_head.state_dict(),
            "config": dict(config),
            "epoch": epoch,
            "metrics": dict(metrics),
            "labels": LABEL_NAMES,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logit-level fusion over frozen CDM and CIM checkpoints.")
    parser.add_argument("--config", required=True)
    add_dataset_override_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_overrides(config, args)
    if bool(config.get("cross_session", {}).get("enabled", False)):
        from scripts.run_cross_session import run_cross_session

        summary_path = run_cross_session("scripts.train_fusion", args.config)
        print(f"cross_session_summary={summary_path}")
        return

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config["training"].get("device", "auto")))
    save_json(output_dir / "config.json", config)
    append_log(log_path, f"experiment={config['experiment_name']}")
    append_log(log_path, "fusion=logit_level frozen_cdm=true frozen_cim=true")

    dialogue_splits, utterance_splits = prepare_dialogues(config, device, log_path)
    train_dialogues = dialogue_splits["train"]
    val_dialogues = dialogue_splits["validation"]
    test_dialogues = dialogue_splits["test"]

    temporal_feature_set = str(config["model"].get("temporal_feature_set", "v1"))
    temporal_builder = TemporalInteractionFeatureBuilder(
        short_gap_threshold=float(config["model"].get("short_gap_threshold", 0.3)),
        long_gap_threshold=float(config["model"].get("long_gap_threshold", 1.0)),
        overlap_threshold=float(config["model"].get("overlap_threshold", 0.05)),
        feature_set=temporal_feature_set,
        strong_overlap_ratio_threshold=float(config["model"].get("strong_overlap_ratio_threshold", 0.30)),
        immediate_gap_threshold=float(config["model"].get("immediate_gap_threshold", 0.10)),
        density_window_seconds=float(config["model"].get("density_window_seconds", 10.0)),
    )
    temporal_builder.fit(train_dialogues)
    for dialogues in dialogue_splits.values():
        attach_temporal_features_to_dialogues(dialogues, temporal_builder)
    temporal_builder.save_stats(output_dir / "temporal_feature_stats.json")
    temporal_policy = TemporalInputPolicy.from_model_config(config["model"])

    embedding_dim = int(train_dialogues[0].embeddings.shape[-1])
    cdm_model, cim_model = load_frozen_branches(config, embedding_dim, device, log_path)
    fusion_cfg = config.get("fusion", {})
    fusion_head = LogitFusionHead(
        num_labels=int(config["model"].get("num_labels", 4)),
        hidden_dim=int(fusion_cfg.get("hidden_dim", 32)),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)
    append_log(log_path, f"fusion_trainable={sum(p.numel() for p in fusion_head.parameters() if p.requires_grad):,}")

    wandb_run = init_wandb(config, output_dir, log_path)
    max_epochs = int(config["training"].get("max_epochs", 50))
    optimizer = torch.optim.AdamW(
        trainable_parameters(fusion_head),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler = create_scheduler(optimizer, config, total_steps=max(1, len(train_dialogues) * max_epochs))
    max_grad_norm = float(config["training"].get("gradient_clip", 1.0))
    best_metric = -1.0
    best_epoch = 0
    best_validation_metrics: Dict[str, Any] = {}
    selection_metric = str(config["training"].get("selection_metric", "UA"))

    for epoch in range(1, max_epochs + 1):
        train_output = run_epoch(
            fusion_head,
            cdm_model,
            cim_model,
            train_dialogues,
            temporal_builder,
            temporal_policy,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            shuffle=True,
        )
        val_output = run_epoch(fusion_head, cdm_model, cim_model, val_dialogues, temporal_builder, temporal_policy, device)
        train_metrics = compute_ser_metrics(train_output["targets"], train_output["predictions"], LABEL_NAMES, train_output["loss"], epoch)
        val_metrics = compute_ser_metrics(val_output["targets"], val_output["predictions"], LABEL_NAMES, val_output["loss"], epoch)
        append_log(
            log_path,
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_WA={val_metrics['WA']:.6f} val_UA={val_metrics['UA']:.6f} val_Macro-F1={val_metrics['Macro-F1']:.6f}",
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
        save_checkpoint(output_dir / "last.pth", fusion_head, config, epoch, val_metrics)
        metric_value = float(val_metrics.get(selection_metric, val_metrics["UA"]))
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_validation_metrics = val_metrics
            save_checkpoint(output_dir / "best.pth", fusion_head, config, epoch, val_metrics)

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    fusion_head.load_state_dict(checkpoint["fusion_head_state_dict"])
    test_output = run_epoch(fusion_head, cdm_model, cim_model, test_dialogues, temporal_builder, temporal_policy, device)
    test_metrics = compute_ser_metrics(test_output["targets"], test_output["predictions"], LABEL_NAMES, test_output["loss"], best_epoch)
    payload = {
        "best_epoch": best_epoch,
        "best_validation": best_validation_metrics,
        "test": test_metrics,
        "fusion": {
            "mode": "logit_level",
            "cdm_checkpoint_root": config.get("fusion", {}).get("cdm_checkpoint_root"),
            "cim_checkpoint_root": config.get("fusion", {}).get("cim_checkpoint_root"),
        },
    }
    save_json(output_dir / "metrics.json", payload)
    save_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"])
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(output_dir / "confusion_matrix.png", test_metrics["confusion_matrix"], LABEL_NAMES)
    append_log(log_path, f"test_WA={test_metrics['WA']:.6f} test_UA={test_metrics['UA']:.6f} test_Macro-F1={test_metrics['Macro-F1']:.6f}")
    if wandb_run is not None:
        wandb_run.summary.update({f"test/{key}": value for key, value in test_metrics.items() if isinstance(value, (int, float))})
        wandb_run.finish()


if __name__ == "__main__":
    main()
