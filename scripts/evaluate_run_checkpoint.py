#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wavlm_baseline import build_mean_embedding_baseline
from models.wavlm_cdm import build_wavlm_cdm_ser_model
from models.wavlm_dual_branch_cim import build_wavlm_dual_branch_cim_ser_model
from scripts.train_dual_branch import (
    configure_trainable_gates,
    run_dual_branch_dialogue_epoch,
    save_branch_gate_stats,
    save_dual_branch_predictions_csv,
    save_dual_branch_temporal_subset_metrics,
)
from scripts.train_baseline import (
    load_config,
    prepare_dialogues as prepare_baseline_dialogues,
    resolve_device,
    run_epoch as run_baseline_epoch,
    set_seed,
)
from scripts.train_cdm import (
    prepare_dialogues as prepare_cdm_dialogues,
    run_dialogue_epoch as run_cdm_epoch,
)
from utils.dialogue_embeddings import TrainableWavLMMeanExtractor
from utils.experiment_metrics import (
    compute_ser_metrics,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_json,
    save_predictions_csv,
)
from utils.iemocap_kaggle import LABEL_NAMES
from utils.temporal_features import (
    TemporalInputPolicy,
    TemporalInteractionFeatureBuilder,
    attach_temporal_features_to_dialogues,
)


def infer_architecture(config: Mapping[str, Any]) -> str:
    model_cfg = config["model"]
    if "temporal_feature_set" in model_cfg or str(model_cfg.get("fusion_mode", "")):
        return "dual_branch"
    if "memory_dim" in model_cfg:
        return "cdm"
    return "baseline"


def make_log_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluate.log"
    log_path.write_text("", encoding="utf-8")
    return log_path


def build_wavlm_extractor(config: Mapping[str, Any], device: torch.device) -> TrainableWavLMMeanExtractor | None:
    if bool(config.get("precompute", {}).get("enabled", True)):
        return None
    return TrainableWavLMMeanExtractor(
        wavlm_model_name=str(config["model"]["wavlm_model_name"]),
        sampling_rate=int(config["dataset"].get("sampling_rate", 16000)),
        max_duration_seconds=config["dataset"].get("max_duration_seconds"),
        freeze_wavlm=bool(config["model"].get("freeze_wavlm", True)),
        unfreeze_last_n_layers=int(config["model"].get("unfreeze_last_n_layers", 0)),
    ).to(device)


def load_model_and_extractor(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    architecture: str,
    embedding_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, TrainableWavLMMeanExtractor | None]:
    if architecture == "baseline":
        model = build_mean_embedding_baseline(config["model"], embedding_dim=embedding_dim).to(device)
    elif architecture == "cdm":
        model = build_wavlm_cdm_ser_model(config["model"], embedding_dim=embedding_dim).to(device)
    elif architecture == "dual_branch":
        model = build_wavlm_dual_branch_cim_ser_model(config["model"], embedding_dim=embedding_dim).to(device)
        configure_trainable_gates(model, config["model"])
    else:
        raise ValueError(f"Unsupported architecture={architecture!r}.")

    model.load_state_dict(checkpoint["model_state_dict"])
    wavlm_extractor = build_wavlm_extractor(config, device)
    if wavlm_extractor is not None:
        state = checkpoint.get("wavlm_extractor_state_dict")
        if state is None:
            raise ValueError("Checkpoint does not contain wavlm_extractor_state_dict for end-to-end WavLM evaluation.")
        wavlm_extractor.load_state_dict(state)
    return model, wavlm_extractor


def evaluate_baseline(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    dialogue_splits, _ = prepare_baseline_dialogues(config, device, make_log_path(output_dir))
    embedding_dim = int(dialogue_splits["train"][0].embeddings.shape[-1])
    model, wavlm_extractor = load_model_and_extractor(config, checkpoint, "baseline", embedding_dim, device)
    test_output = run_baseline_epoch(
        model,
        dialogue_splits["test"],
        device,
        wavlm_extractor=wavlm_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("eval_batch_size", 4))),
        progress=bool(config["training"].get("progress_bar", True)),
        description=f"{config['experiment_name']} {checkpoint_path.stem} test",
    )
    save_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"], LABEL_NAMES)
    return make_payload(config, checkpoint, checkpoint_path, test_output)


def evaluate_cdm(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    dialogue_splits, _ = prepare_cdm_dialogues(config, device, make_log_path(output_dir))
    embedding_dim = int(dialogue_splits["train"][0].embeddings.shape[-1])
    model, wavlm_extractor = load_model_and_extractor(config, checkpoint, "cdm", embedding_dim, device)
    test_output = run_cdm_epoch(
        model,
        dialogue_splits["test"],
        device,
        temporal_feature_dim=int(config["model"].get("temporal_feature_dim", 16)),
        wavlm_extractor=wavlm_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("wavlm_batch_size", 4))),
        progress=bool(config["training"].get("progress_bar", True)),
        description=f"{config['experiment_name']} {checkpoint_path.stem} test",
    )
    save_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"], LABEL_NAMES)
    return make_payload(config, checkpoint, checkpoint_path, test_output)


def evaluate_dual_branch(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    dialogue_splits, _ = prepare_baseline_dialogues(config, device, make_log_path(output_dir))
    train_dialogues = dialogue_splits["train"]
    temporal_feature_set = str(config["model"].get("temporal_feature_set", "interaction_4"))
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
    temporal_builder.save_stats(output_dir / "temporal_feature_stats.json")
    for dialogues in dialogue_splits.values():
        attach_temporal_features_to_dialogues(dialogues, temporal_builder)
    temporal_policy = TemporalInputPolicy.from_model_config(config["model"])

    embedding_dim = int(train_dialogues[0].embeddings.shape[-1])
    model, wavlm_extractor = load_model_and_extractor(config, checkpoint, "dual_branch", embedding_dim, device)
    test_output = run_dual_branch_dialogue_epoch(
        model,
        dialogue_splits["test"],
        temporal_builder,
        temporal_policy,
        device,
        wavlm_extractor=wavlm_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("wavlm_batch_size", 4))),
        progress=bool(config["training"].get("progress_bar", True)),
        description=f"{config['experiment_name']} {checkpoint_path.stem} test",
    )
    save_dual_branch_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"])
    save_branch_gate_stats(
        output_dir / "branch_gate_stats.json",
        model,
        test_output["residual_rows"],
        strong_overlap_threshold=float(config.get("analysis", {}).get("strong_overlap_threshold", 0.25)),
    )
    save_dual_branch_temporal_subset_metrics(
        output_dir / "temporal_subset_metrics.json",
        test_output["prediction_rows"],
        strong_overlap_threshold=float(config.get("analysis", {}).get("strong_overlap_threshold", 0.25)),
    )
    return make_payload(config, checkpoint, checkpoint_path, test_output)


def make_payload(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    test_output: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    test_metrics = compute_ser_metrics(
        test_output["targets"],
        test_output["predictions"],
        LABEL_NAMES,
        test_output["loss"],
        checkpoint_epoch,
    )
    return {
        **test_metrics,
        "run_name": config.get("run_name", config.get("experiment_name", "")),
        "experiment_name": config.get("experiment_name", ""),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "validation_at_checkpoint": checkpoint.get("metrics", {}),
        "test": test_metrics,
    }


def save_common_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    save_json(output_dir / "metrics.json", payload)
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", payload["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(output_dir / "confusion_matrix.png", payload["confusion_matrix"], LABEL_NAMES)


def default_output_dir(config: Mapping[str, Any], checkpoint_path: Path) -> Path:
    return Path(str(config["output_dir"])) / "eval_checkpoints" / checkpoint_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run checkpoint on the test split using architecture inferred from config.")
    parser.add_argument("--config", required=True, help="Run config YAML. Architecture and temporal features are inferred from this.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <config.output_dir>/last.pth.")
    parser.add_argument("--checkpoint-name", default="last", help="Checkpoint stem used when --checkpoint is omitted.")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory. Defaults to <output_dir>/eval_checkpoints/<checkpoint_stem>.")
    parser.add_argument("--device", default=None, help="Override device. Defaults to training.device from config.")
    parser.add_argument("--architecture", choices=["auto", "baseline", "cdm", "dual_branch"], default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(str(config["output_dir"])) / f"{args.checkpoint_name}.pth"
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device or config["training"].get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = infer_architecture(config) if args.architecture == "auto" else args.architecture

    if architecture == "baseline":
        payload = evaluate_baseline(config, checkpoint, checkpoint_path, output_dir, device)
    elif architecture == "cdm":
        payload = evaluate_cdm(config, checkpoint, checkpoint_path, output_dir, device)
    elif architecture == "dual_branch":
        payload = evaluate_dual_branch(config, checkpoint, checkpoint_path, output_dir, device)
    else:
        raise ValueError(f"Unsupported architecture={architecture!r}.")

    payload["architecture"] = architecture
    save_common_outputs(output_dir, payload)
    print(
        (
            f"{config.get('run_name', config.get('experiment_name', 'run'))} "
            f"{checkpoint_path.name} architecture={architecture} "
            f"epoch={payload['checkpoint_epoch']} WA={payload['WA']:.6f} "
            f"UA={payload['UA']:.6f} Macro-F1={payload['Macro-F1']:.6f} "
            f"WF1={payload['WF1']:.6f} loss={payload['loss']:.6f} "
            f"output={output_dir}"
        ),
        flush=True,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
