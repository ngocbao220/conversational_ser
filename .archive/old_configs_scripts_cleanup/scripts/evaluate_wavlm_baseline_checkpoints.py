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
from scripts.train_baseline import (
    prepare_dialogues,
    resolve_device,
    run_epoch,
    set_seed,
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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def evaluate_checkpoint(checkpoint_path: Path, output_dir: Path, device_name: str) -> dict[str, Any]:
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config: Mapping[str, Any] = checkpoint["config"]
    set_seed(int(config.get("seed", 42)))

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluate.log"
    log_path.write_text("", encoding="utf-8")

    dialogue_splits, _ = prepare_dialogues(config, device, log_path)
    reference_dialogues = dialogue_splits["train"] or dialogue_splits["validation"] or dialogue_splits["test"]
    embedding_dim = int(reference_dialogues[0].embeddings.shape[-1])

    model = build_mean_embedding_baseline(config["model"], embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    wavlm_extractor = None
    if not bool(config.get("precompute", {}).get("enabled", True)):
        wavlm_extractor = TrainableWavLMMeanExtractor(
            wavlm_model_name=str(config["model"]["wavlm_model_name"]),
            sampling_rate=int(config["dataset"].get("sampling_rate", 16000)),
            max_duration_seconds=config["dataset"].get("max_duration_seconds"),
            freeze_wavlm=bool(config["model"].get("freeze_wavlm", True)),
            unfreeze_last_n_layers=int(config["model"].get("unfreeze_last_n_layers", 0)),
        ).to(device)
        wavlm_state = checkpoint.get("wavlm_extractor_state_dict")
        if wavlm_state is None:
            raise ValueError(f"Checkpoint has no wavlm_extractor_state_dict: {checkpoint_path}")
        wavlm_extractor.load_state_dict(wavlm_state)

    test_output = run_epoch(
        model,
        dialogue_splits["test"],
        device,
        wavlm_extractor=wavlm_extractor,
        wavlm_batch_size=int(config["training"].get("eval_wavlm_batch_size", config["training"].get("eval_batch_size", 4))),
        progress=bool(config["training"].get("progress_bar", True)),
        description=f"{checkpoint_path.name} test",
    )
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    test_metrics = compute_ser_metrics(
        test_output["targets"],
        test_output["predictions"],
        LABEL_NAMES,
        test_output["loss"],
        checkpoint_epoch,
    )
    payload = {
        **test_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "validation_at_checkpoint": checkpoint.get("metrics", {}),
        "test": test_metrics,
    }
    save_json(output_dir / "metrics.json", payload)
    save_predictions_csv(output_dir / "predictions.csv", test_output["prediction_rows"], LABEL_NAMES)
    save_confusion_matrix_csv(output_dir / "confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_confusion_matrix_png(output_dir / "confusion_matrix.png", test_metrics["confusion_matrix"], LABEL_NAMES)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WavLM baseline checkpoints on the configured test split.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Path to a baseline .pth checkpoint.")
    parser.add_argument("--output-root", required=True, help="Directory for per-checkpoint evaluation outputs.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    for checkpoint_arg in args.checkpoint:
        checkpoint_path = Path(checkpoint_arg)
        name = checkpoint_path.stem
        metrics = evaluate_checkpoint(checkpoint_path, output_root / name, args.device)
        print(
            (
                f"{name}: epoch={metrics['checkpoint_epoch']} "
                f"WA={metrics['WA']:.6f} UA={metrics['UA']:.6f} "
                f"Macro-F1={metrics['Macro-F1']:.6f} WF1={metrics['WF1']:.6f} "
                f"loss={metrics['loss']:.6f}"
            ),
            flush=True,
        )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
