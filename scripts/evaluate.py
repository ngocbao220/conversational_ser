#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUNTIME_READY = False


def load_runtime_dependencies() -> None:
    global _RUNTIME_READY
    global LABEL_NAMES, TemporalInputPolicy, TemporalInteractionFeatureBuilder, TrainableWavLMMeanExtractor
    global attach_temporal_features_to_dialogues, build_cdim_ser_model, build_mean_embedding_baseline
    global compute_ser_metrics, load_config, prepare_baseline_dialogues, resolve_device, run_baseline_epoch
    global run_cdim_epoch, save_cdim_predictions_csv, save_confusion_matrix_csv, save_confusion_matrix_png
    global save_json, save_predictions_csv, set_seed

    if _RUNTIME_READY:
        return

    from models.baseline import build_mean_embedding_baseline
    from models.cdim import build_cdim_ser_model
    from scripts.train_baseline import (
        load_config,
        prepare_dialogues as prepare_baseline_dialogues,
        resolve_device,
        run_epoch as run_baseline_epoch,
        set_seed,
    )
    from scripts.train_cdim import (
        run_dialogue_epoch as run_cdim_epoch,
        save_cdim_predictions_csv,
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

    _RUNTIME_READY = True


def infer_architecture(config: Mapping[str, Any]) -> str:
    model_cfg = config["model"]
    if "temporal_feature_set" in model_cfg or "temporal_feature_dim" in model_cfg:
        return "cdim"
    return "baseline"


def make_log_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluate.log"
    log_path.write_text("", encoding="utf-8")
    return log_path


def build_encoder_extractor(config: Mapping[str, Any], device: torch.device):
    if bool(config.get("precompute", {}).get("enabled", True)):
        return None
    return TrainableWavLMMeanExtractor(
        wavlm_model_name=str(config["model"].get("encoder_model_name", config["model"].get("wavlm_model_name"))),
        sampling_rate=int(config["dataset"].get("sampling_rate", 16000)),
        max_duration_seconds=config["dataset"].get("max_duration_seconds"),
        freeze_wavlm=bool(config["model"].get("freeze_encoder", config["model"].get("freeze_wavlm", True))),
        unfreeze_last_n_layers=int(config["model"].get("unfreeze_last_n_layers", 0)),
    ).to(device)


def load_model_and_extractor(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    architecture: str,
    embedding_dim: int,
    device: torch.device,
):
    if architecture == "baseline":
        model = build_mean_embedding_baseline(config["model"], embedding_dim=embedding_dim).to(device)
    elif architecture == "cdim":
        model = build_cdim_ser_model(config["model"], embedding_dim=embedding_dim).to(device)
    else:
        raise ValueError(f"Unsupported architecture={architecture!r}.")

    model.load_state_dict(checkpoint["model_state_dict"])
    encoder_extractor = build_encoder_extractor(config, device)
    if encoder_extractor is not None:
        state = checkpoint.get("wavlm_extractor_state_dict")
        if state is None:
            raise ValueError("Checkpoint does not contain encoder extractor weights for end-to-end evaluation.")
        encoder_extractor.load_state_dict(state)
    return model, encoder_extractor


def evaluate_baseline(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    dialogue_splits, _ = prepare_baseline_dialogues(config, device, make_log_path(output_dir))
    embedding_dim = int(dialogue_splits["train"][0].embeddings.shape[-1])
    model, encoder_extractor = load_model_and_extractor(config, checkpoint, "baseline", embedding_dim, device)
    test_output = run_baseline_epoch(
        model,
        dialogue_splits["test"],
        device,
        wavlm_extractor=encoder_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("eval_batch_size", 4))),
        progress=bool(config["training"].get("progress_bar", True)),
        description=f"{config['experiment_name']} {checkpoint_path.stem} test",
    )
    save_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"], LABEL_NAMES)
    return make_payload(config, checkpoint, checkpoint_path, test_output)


def evaluate_cdim(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    dialogue_splits, _ = prepare_baseline_dialogues(config, device, make_log_path(output_dir))
    train_dialogues = dialogue_splits["train"]
    temporal_builder = TemporalInteractionFeatureBuilder(
        short_gap_threshold=float(config["model"].get("short_gap_threshold", 0.3)),
        long_gap_threshold=float(config["model"].get("long_gap_threshold", 1.0)),
        overlap_threshold=float(config["model"].get("overlap_threshold", 0.05)),
        feature_set=str(config["model"].get("temporal_feature_set", "interaction_4")),
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
    model, encoder_extractor = load_model_and_extractor(config, checkpoint, "cdim", embedding_dim, device)
    test_output = run_cdim_epoch(
        model,
        dialogue_splits["test"],
        temporal_builder,
        temporal_policy,
        device,
        wavlm_extractor=encoder_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("wavlm_batch_size", 4))),
        progress=bool(config["training"].get("progress_bar", True)),
        description=f"{config['experiment_name']} {checkpoint_path.stem} test",
    )
    save_cdim_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"], temporal_builder.stats.feature_names)
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


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    checkpoint_name: str = "last",
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    architecture: str = "auto",
) -> dict[str, Any]:
    load_runtime_dependencies()
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    checkpoint_path = Path(checkpoint) if checkpoint else Path(str(config["output_dir"])) / f"{checkpoint_name}.pth"
    resolved_output_dir = Path(output_dir) if output_dir else default_output_dir(config, checkpoint_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(device_name or config["training"].get("device", "auto")))
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    resolved_architecture = infer_architecture(config) if architecture == "auto" else architecture

    if resolved_architecture == "baseline":
        payload = evaluate_baseline(config, checkpoint_payload, checkpoint_path, resolved_output_dir, device)
    elif resolved_architecture == "cdim":
        payload = evaluate_cdim(config, checkpoint_payload, checkpoint_path, resolved_output_dir, device)
    else:
        raise ValueError(f"Unsupported architecture={resolved_architecture!r}.")

    payload["architecture"] = resolved_architecture
    save_common_outputs(resolved_output_dir, payload)
    print(
        (
            f"{config.get('run_name', config.get('experiment_name', 'run'))} "
            f"{checkpoint_path.name} architecture={resolved_architecture} "
            f"epoch={payload['checkpoint_epoch']} WA={payload['WA']:.6f} "
            f"UA={payload['UA']:.6f} Macro-F1={payload['Macro-F1']:.6f} "
            f"WF1={payload['WF1']:.6f} loss={payload['loss']:.6f} "
            f"output={resolved_output_dir}"
        ),
        flush=True,
    )
    return payload


def discover_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("last.pth") if (path.parent / "config.json").exists())


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "root",
        "run_dir",
        "run_name",
        "architecture",
        "checkpoint_epoch",
        "WA",
        "UA",
        "WF1",
        "Macro-F1",
        "loss",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_roots(
    roots: list[str],
    device_name: str | None = None,
    skip_existing: bool = False,
    summary: str | Path = "results/_last_checkpoint_eval_summary.csv",
) -> None:
    run_dirs: list[tuple[Path, Path]] = []
    for root_arg in roots:
        root = Path(root_arg)
        for run_dir in discover_runs(root):
            run_dirs.append((root, run_dir))

    rows: list[dict[str, Any]] = []
    total = len(run_dirs)
    for index, (root, run_dir) in enumerate(run_dirs, start=1):
        config_path = run_dir / "config.json"
        checkpoint_path = run_dir / "last.pth"
        metrics_path = run_dir / "eval_checkpoints" / "last" / "metrics.json"
        if skip_existing and metrics_path.exists():
            print(f"[{index}/{total}] skip existing {run_dir}", flush=True)
        else:
            print(f"[{index}/{total}] evaluate {run_dir}", flush=True)
            evaluate_checkpoint(
                config_path=config_path,
                checkpoint=checkpoint_path,
                checkpoint_name="last",
                output_dir=None,
                device_name=device_name,
                architecture="auto",
            )

        if metrics_path.exists():
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "root": str(root),
                    "run_dir": str(run_dir),
                    "run_name": metrics.get("run_name", ""),
                    "architecture": metrics.get("architecture", ""),
                    "checkpoint_epoch": metrics.get("checkpoint_epoch", ""),
                    "WA": metrics.get("WA", ""),
                    "UA": metrics.get("UA", ""),
                    "WF1": metrics.get("WF1", ""),
                    "Macro-F1": metrics.get("Macro-F1", ""),
                    "loss": metrics.get("loss", ""),
                    "metrics_path": str(metrics_path),
                }
            )

    write_summary(rows, Path(summary))
    print(f"summary={summary} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints for baseline and CDIM runs.")
    parser.add_argument("--config", help="Run config YAML. Architecture and temporal features are inferred from this.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <config.output_dir>/last.pth.")
    parser.add_argument("--checkpoint-name", default="last", help="Checkpoint stem used when --checkpoint is omitted.")
    parser.add_argument("--output-dir", default=None, help="Evaluation output directory. Defaults to <output_dir>/eval_checkpoints/<checkpoint_stem>.")
    parser.add_argument("--device", default=None, help="Override device. Defaults to training.device from config.")
    parser.add_argument("--architecture", choices=["auto", "baseline", "cdim"], default="auto")
    parser.add_argument("--roots", nargs="+", help="Evaluate last.pth for every run directory under these result roots, e.g. results/main.")
    parser.add_argument("--skip-existing", action="store_true", help="With --roots, skip runs with eval_checkpoints/last/metrics.json.")
    parser.add_argument("--summary", default="results/_last_checkpoint_eval_summary.csv", help="With --roots, CSV summary path.")
    args = parser.parse_args()

    if args.roots:
        evaluate_roots(
            roots=args.roots,
            device_name=args.device,
            skip_existing=args.skip_existing,
            summary=args.summary,
        )
        return

    if not args.config:
        parser.error("either --config or --roots is required")

    evaluate_checkpoint(
        config_path=args.config,
        checkpoint=args.checkpoint,
        checkpoint_name=args.checkpoint_name,
        output_dir=args.output_dir,
        device_name=args.device,
        architecture=args.architecture,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
