from __future__ import annotations

import argparse
from typing import Any, Dict


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset-name", default="AbstractTTS/IEMOCAP")
    group.add_argument("--validation-size", type=float, default=0.1)
    group.add_argument("--test-size", type=float, default=0.1)
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--num-proc", type=int, default=1)
    group.add_argument("--max-train-samples", type=int, default=None)
    group.add_argument("--max-validation-samples", type=int, default=None)
    group.add_argument("--max-test-samples", type=int, default=None)
    group.add_argument("--sampling-rate", type=int, default=16000)
    group.add_argument("--max-duration-seconds", type=float, default=12.0)


def add_b0_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("--encoder-name", default="microsoft/wavlm-base")
    group.add_argument("--pooling", choices=["mean", "attention"], default="mean")
    group.add_argument("--freeze-encoder", type=str_to_bool, default=True)
    group.add_argument("--dropout", type=float, default=0.2)
    group.add_argument("--hidden-dim", type=int, default=256)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training")
    group.add_argument("--output-dir", default="outputs/b0_utterance")
    group.add_argument("--batch-size", type=int, default=4)
    group.add_argument("--eval-batch-size", type=int, default=8)
    group.add_argument("--learning-rate", type=float, default=1e-4)
    group.add_argument("--weight-decay", type=float, default=0.01)
    group.add_argument("--epochs", type=int, default=5)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--max-grad-norm", type=float, default=1.0)
    group.add_argument("--num-workers", type=int, default=4)
    group.add_argument("--device", default="auto")


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("logging")
    group.add_argument("--progress-bar", type=str_to_bool, default=True)
    group.add_argument("--progress-ncols", type=int, default=100)
    group.add_argument("--progress-mininterval", type=float, default=2.0)
    group.add_argument("--log-every-steps", type=int, default=50)
    group.add_argument("--log-file", default="train.log")
    group.add_argument("--use-wandb", type=str_to_bool, default=False)
    group.add_argument("--wandb-project", default="conversational-SER")
    group.add_argument("--wandb-run-name", default="baseline")
    group.add_argument("--wandb-entity", default=None)
    group.add_argument("--wandb-mode", default="online")


def build_b0_config(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_cfg: Dict[str, Any] = {
        "name": args.dataset_name,
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "num_proc": args.num_proc,
    }
    optional_limits = {
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "max_test_samples": args.max_test_samples,
    }
    dataset_cfg.update({key: value for key, value in optional_limits.items() if value is not None})

    output_dir = args.output_dir
    return {
        "dataset": dataset_cfg,
        "audio": {
            "sampling_rate": args.sampling_rate,
            "max_duration_seconds": args.max_duration_seconds,
        },
        "baselines": {
            "b0": {
                "name": "B0_utterance",
                "description": "utterance audio -> frozen SSL encoder -> pooling -> classifier -> emotion",
                "checkpoint_path": f"{output_dir}/best.pt",
                "metrics_path": f"{output_dir}/test_metrics.json",
                "model": {
                    "encoder_name": args.encoder_name,
                    "pooling": args.pooling,
                    "freeze_encoder": args.freeze_encoder,
                    "dropout": args.dropout,
                    "hidden_dim": args.hidden_dim,
                },
                "training": {
                    "output_dir": output_dir,
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "max_grad_norm": args.max_grad_norm,
                    "num_workers": args.num_workers,
                    "device": args.device,
                },
            }
        },
        "logging": {
            "progress_bar": args.progress_bar,
            "progress_ncols": args.progress_ncols,
            "progress_mininterval": args.progress_mininterval,
            "log_every_steps": args.log_every_steps,
            "log_file": args.log_file,
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "wandb_entity": args.wandb_entity,
            "wandb_mode": args.wandb_mode,
        },
        "inference": {
            "checkpoint_path": f"{output_dir}/best.pt",
        },
    }
