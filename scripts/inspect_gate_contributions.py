from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wavlm_dual_branch_cim import build_wavlm_dual_branch_cim_ser_model
from scripts.train_dual_branch import configure_trainable_gates
from scripts.train_wavlm_cim import append_log, prepare_dialogues, resolve_device, set_seed
from utils.dialogue_embeddings import DialogueEmbedding, TrainableWavLMMeanExtractor
from utils.iemocap_kaggle import ID2LABEL, LABEL_NAMES
from utils.temporal_features import (
    TEMPORAL_FEATURE_SETS,
    TemporalInputPolicy,
    TemporalInteractionFeatureBuilder,
    attach_temporal_features_to_dialogues,
)


def load_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint_and_config(checkpoint_path: Path, config_path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path is not None:
        config = load_config_file(config_path)
    elif isinstance(checkpoint.get("config"), Mapping):
        config = dict(checkpoint["config"])
    else:
        sibling = checkpoint_path.parent / "config.json"
        if sibling.exists():
            config = load_config_file(sibling)
        else:
            raise ValueError("No config provided and checkpoint does not contain a config payload.")
    return checkpoint, config


def build_temporal_builder(config: Mapping[str, Any], train_dialogues: Sequence[DialogueEmbedding]) -> TemporalInteractionFeatureBuilder:
    model_cfg = config["model"]
    temporal_feature_set = str(model_cfg.get("temporal_feature_set", "v1"))
    if temporal_feature_set not in TEMPORAL_FEATURE_SETS:
        raise ValueError(
            f"Unknown temporal_feature_set={temporal_feature_set!r}. "
            f"Expected one of {sorted(TEMPORAL_FEATURE_SETS)}."
        )
    expected_temporal_dim = len(TEMPORAL_FEATURE_SETS[temporal_feature_set])
    configured_dim = int(model_cfg.get("temporal_feature_dim", expected_temporal_dim))
    if configured_dim != expected_temporal_dim:
        raise ValueError(
            f"temporal_feature_dim must be {expected_temporal_dim} "
            f"for temporal_feature_set={temporal_feature_set!r}, got {configured_dim}."
        )

    temporal_builder = TemporalInteractionFeatureBuilder(
        short_gap_threshold=float(model_cfg.get("short_gap_threshold", 0.3)),
        long_gap_threshold=float(model_cfg.get("long_gap_threshold", 1.0)),
        overlap_threshold=float(model_cfg.get("overlap_threshold", 0.05)),
        feature_set=temporal_feature_set,
        strong_overlap_ratio_threshold=float(model_cfg.get("strong_overlap_ratio_threshold", 0.30)),
        immediate_gap_threshold=float(model_cfg.get("immediate_gap_threshold", 0.10)),
        density_window_seconds=float(model_cfg.get("density_window_seconds", 10.0)),
    )
    temporal_builder.fit(train_dialogues)
    return temporal_builder


def infer_session_id(dialogue_id: str, utterance_id: str) -> str:
    for value in (dialogue_id, utterance_id):
        if isinstance(value, str) and value.startswith("Ses") and len(value) >= 5:
            return value[:5]
    return "unknown"


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.clip(denominator, 1e-8, None)


def run_inspection(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    split: str,
    device: torch.device,
) -> pd.DataFrame:
    set_seed(int(config.get("seed", 42)))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "gate_contribution_inspector.log"
    log_path.write_text("", encoding="utf-8")

    dialogue_splits, _ = prepare_dialogues(config, device, log_path)
    if split not in dialogue_splits:
        raise ValueError(f"split={split!r} not found. Available splits: {sorted(dialogue_splits)}")
    train_dialogues = dialogue_splits["train"]
    target_dialogues = dialogue_splits[split]

    temporal_builder = build_temporal_builder(config, train_dialogues)
    for dialogues in dialogue_splits.values():
        attach_temporal_features_to_dialogues(dialogues, temporal_builder)
    temporal_policy = TemporalInputPolicy.from_model_config(config["model"])

    if not target_dialogues:
        raise ValueError(f"No dialogues found for split={split!r}.")
    embedding_dim = int(target_dialogues[0].embeddings.shape[-1])
    model = build_wavlm_dual_branch_cim_ser_model(config["model"], embedding_dim=embedding_dim).to(device)
    configure_trainable_gates(model, config["model"])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wavlm_extractor = None
    if not bool(config.get("precompute", {}).get("enabled", True)):
        wavlm_extractor = TrainableWavLMMeanExtractor(
            wavlm_model_name=str(config["model"]["wavlm_model_name"]),
            sampling_rate=int(config["dataset"].get("sampling_rate", 16000)),
            max_duration_seconds=config["dataset"].get("max_duration_seconds"),
            freeze_wavlm=bool(config["model"].get("freeze_wavlm", True)),
            unfreeze_last_n_layers=int(config["model"].get("unfreeze_last_n_layers", 0)),
        ).to(device)
        state = checkpoint.get("wavlm_extractor_state_dict")
        if state is not None:
            wavlm_extractor.load_state_dict(state)
        wavlm_extractor.eval()

    wavlm_batch_size = int(config["training"].get("eval_wavlm_batch_size", config["training"].get("wavlm_batch_size", 4)))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for dialogue in target_dialogues:
            embeddings = (
                wavlm_extractor.encode_rows(dialogue.rows, device=device, batch_size=wavlm_batch_size)
                if wavlm_extractor is not None
                else dialogue.embeddings.to(device)
            )
            labels = dialogue.labels.to(device)
            temporal_features = temporal_policy.apply(
                temporal_builder.transform_dialogue(dialogue), dialogue.dialogue_id
            ).to(device)
            output = model(embeddings=embeddings, temporal_features=temporal_features, labels=None)
            logits = output["logits"]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

            acoustic_norm = embeddings.detach().norm(dim=-1).cpu().numpy()
            dialogue_residual_norm = output["dialogue_residuals"].detach().norm(dim=-1).cpu().numpy()
            temporal_residual_norm = output["temporal_residuals"].detach().norm(dim=-1).cpu().numpy()
            alpha_value = float(output["alpha_value"])
            beta_value = float(output["beta_value"])
            cdm_contribution_norm = np.abs(alpha_value) * dialogue_residual_norm
            cim_contribution_norm = np.abs(beta_value) * temporal_residual_norm
            fused_norm = output["fused_embeddings"].detach().norm(dim=-1).cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            for index, row in enumerate(dialogue.rows):
                utterance_id = str(row.get("utterance_id", ""))
                dialogue_id = str(row.get("dialogue_id", dialogue.dialogue_id))
                record = {
                    "split": split,
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "session_id": infer_session_id(dialogue_id, utterance_id),
                    "speaker_id": row.get("speaker_id", ""),
                    "start_time": float(row.get("start_time", 0.0)),
                    "end_time": float(row.get("end_time", 0.0)),
                    "gold_label": ID2LABEL[int(labels_cpu[index])],
                    "pred_label": ID2LABEL[int(predictions[index])],
                    "correct": bool(int(labels_cpu[index]) == int(predictions[index])),
                    "fusion_mode": str(output["fusion_mode"]),
                    "alpha_value": alpha_value,
                    "beta_value": beta_value,
                    "acoustic_norm": float(acoustic_norm[index]),
                    "cdm_residual_norm": float(dialogue_residual_norm[index]),
                    "cim_residual_norm": float(temporal_residual_norm[index]),
                    "alpha_cdm_contribution_norm": float(cdm_contribution_norm[index]),
                    "beta_cim_contribution_norm": float(cim_contribution_norm[index]),
                    "fused_embedding_norm": float(fused_norm[index]),
                    "alpha_cdm_to_acoustic": float(safe_ratio(cdm_contribution_norm, acoustic_norm)[index]),
                    "beta_cim_to_acoustic": float(safe_ratio(cim_contribution_norm, acoustic_norm)[index]),
                }
                for feature_name in temporal_builder.stats.feature_names:
                    record[feature_name] = float(row.get(feature_name, 0.0))
                for label_idx, label_name in ID2LABEL.items():
                    record[f"prob_{label_name}"] = float(probabilities[index][label_idx])
                rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "gate_contributions_by_utterance.csv", index=False)
    append_log(log_path, f"wrote={output_dir / 'gate_contributions_by_utterance.csv'} rows={len(df)}")
    return df


def summarize(df: pd.DataFrame, output_dir: Path, checkpoint_path: Path, config: Mapping[str, Any]) -> None:
    metric_cols = [
        "acoustic_norm",
        "cdm_residual_norm",
        "cim_residual_norm",
        "alpha_cdm_contribution_norm",
        "beta_cim_contribution_norm",
        "alpha_cdm_to_acoustic",
        "beta_cim_to_acoustic",
        "fused_embedding_norm",
    ]
    summary = {
        "checkpoint": str(checkpoint_path),
        "experiment_name": str(config.get("experiment_name", "")),
        "run_name": str(config.get("run_name", "")),
        "fusion_mode": str(config.get("model", {}).get("fusion_mode", "")),
        "num_utterances": int(len(df)),
        "alpha_value": float(df["alpha_value"].iloc[0]) if len(df) else 0.0,
        "beta_value": float(df["beta_value"].iloc[0]) if len(df) else 0.0,
    }
    for col in metric_cols:
        if col in df:
            summary[f"mean_{col}"] = float(df[col].mean())
            summary[f"median_{col}"] = float(df[col].median())
            summary[f"std_{col}"] = float(df[col].std(ddof=0))
    pd.DataFrame([summary]).to_csv(output_dir / "gate_contributions_summary.csv", index=False)
    (output_dir / "gate_contributions_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    agg = {col: ["mean", "median", "std"] for col in metric_cols if col in df}
    if agg:
        for group_name, columns in {
            "session": ["session_id"],
            "emotion": ["gold_label"],
            "session_emotion": ["session_id", "gold_label"],
            "correctness": ["correct"],
        }.items():
            grouped = df.groupby(columns, dropna=False).agg(agg)
            grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
            grouped = grouped.reset_index()
            grouped["count"] = df.groupby(columns, dropna=False).size().values
            grouped.to_csv(output_dir / f"gate_contributions_by_{group_name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect alpha/beta-scaled CDM/CIM contribution norms for a dual-branch checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth, phase checkpoint, or any dual-branch checkpoint.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config. If omitted, use checkpoint['config'].")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", default=None, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def default_output_dir(checkpoint_path: Path) -> Path:
    try:
        relative_parent = checkpoint_path.parent.relative_to(ROOT / "results")
    except ValueError:
        relative_parent = Path(checkpoint_path.parent.name)
    return ROOT / "reports" / "gate_contributions" / relative_parent


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    config_path = Path(args.config) if args.config else None
    checkpoint, config = load_checkpoint_and_config(checkpoint_path, config_path)
    del checkpoint
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(checkpoint_path)
    )
    device = resolve_device(args.device)
    df = run_inspection(
        checkpoint_path=checkpoint_path,
        config=config,
        output_dir=output_dir,
        split=str(args.split),
        device=device,
    )
    summarize(df, output_dir, checkpoint_path, config)
    print(f"gate_contributions_dir={output_dir}")


if __name__ == "__main__":
    main()
