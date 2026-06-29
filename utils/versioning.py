from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


VERSION_REGISTRY: dict[str, dict[str, Any]] = {
    "1": {
        "name": "v1_tim_concat",
        "trainer_module": "scripts.train_wavlm_tim",
        "experiment_name": "v1_tim_concat",
        "model_overrides": {
            "temporal_feature_set": "v1",
            "temporal_feature_dim": 16,
            "temporal_hidden_dim": 64,
        },
        "training_stage": None,
    },
    "2.1": {
        "name": "v2_1_dual_end2end",
        "trainer_module": "scripts.train_dual_branch",
        "experiment_name": "v2_1_dual_branch_end2end",
        "model_overrides": {
            "temporal_feature_set": "v1",
            "temporal_feature_dim": 16,
            "temporal_hidden_dim": 64,
        },
        "training_stage": {"mode": "end_to_end"},
    },
    "2.2.1": {
        "name": "v2_2_1_dual_dialogue_temporal_fuse",
        "trainer_module": "scripts.train_dual_branch",
        "experiment_name": "v2_2_1_dual_dialogue_temporal_fuse",
        "model_overrides": {
            "temporal_feature_set": "v1",
            "temporal_feature_dim": 16,
            "temporal_hidden_dim": 64,
        },
        "training_stage": {
            "mode": "3_phase",
            "stage_1_train_dialogue_branch": True,
            "stage_1_epochs": 100,
            "stage_2_freeze_dialogue_train_temporal": True,
            "stage_2_epochs": 100,
            "stage_3_finetune_fusion": True,
            "stage_3_epochs": 50,
        },
    },
    "2.2.2": {
        "name": "v2_2_2_dual_temporal_dialogue_fuse",
        "trainer_module": "scripts.train_dual_branch",
        "experiment_name": "v2_2_2_dual_temporal_dialogue_fuse",
        "model_overrides": {
            "temporal_feature_set": "v1",
            "temporal_feature_dim": 16,
            "temporal_hidden_dim": 64,
        },
        "training_stage": {
            "mode": "temporal_first_3_phase",
            "stage_1_train_temporal_branch": True,
            "stage_1_epochs": 100,
            "stage_2_train_dialogue_branch": True,
            "stage_2_epochs": 100,
            "stage_3_finetune_fusion": True,
            "stage_3_epochs": 50,
        },
    },
    "3.1": {
        "name": "v3_1_tim_recommended_v2",
        "trainer_module": "scripts.train_wavlm_tim",
        "experiment_name": "v3_1_tim_recommended_v2",
        "model_overrides": {
            "temporal_feature_set": "recommended_v2",
            "temporal_feature_dim": 36,
            "temporal_hidden_dim": 64,
        },
        "training_stage": None,
    },
    "3.2": {
        "name": "v3_2_tim_compact_primitives",
        "trainer_module": "scripts.train_wavlm_tim",
        "experiment_name": "v3_2_tim_compact_primitives",
        "model_overrides": {
            "temporal_feature_set": "selected_primitives",
            "temporal_feature_dim": 12,
            "temporal_hidden_dim": 32,
        },
        "training_stage": None,
    },
}


SETTING_REGISTRY: dict[str, dict[str, Any]] = {
    "A": {
        "description": "Fair idea-proof setting: frozen WavLM, precomputed embeddings.",
        "model": {
            "freeze_wavlm": True,
            "unfreeze_last_n_layers": 0,
            "memory_dim": 128,
            "temporal_emb_dim": 64,
            "dropout": 0.2,
        },
        "precompute": {
            "enabled": True,
            "cache_path": "results/wavlm_shared/cache/wavlm_mean_embeddings.pt",
            "force_recompute": False,
            "batch_size": 32,
        },
        "training": {
            "batch_size": 16,
            "eval_batch_size": 16,
            "max_epochs": 250,
            "wavlm_batch_size": 4,
            "eval_wavlm_batch_size": 4,
            "learning_rate": 1e-4,
            "learning_rate_classifier": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "gradient_clip": 1.0,
            "use_class_weights": False,
            "num_workers": 2,
            "progress_bar": False,
        },
    },
    "B": {
        "description": "Strong GPU setting: WavLM last-4 fine-tuning.",
        "model": {
            # In this codebase, last-4 fine-tuning is freeze_wavlm=true plus unfreeze_last_n_layers=4.
            "freeze_wavlm": True,
            "unfreeze_last_n_layers": 4,
            "memory_dim": 256,
            "temporal_emb_dim": 128,
            "dropout": 0.25,
        },
        "precompute": {
            "enabled": False,
            "cache_path": "results/wavlm_shared/cache/wavlm_mean_embeddings.pt",
            "force_recompute": False,
            "batch_size": 16,
        },
        "training": {
            "batch_size": 8,
            "eval_batch_size": 8,
            "max_epochs": 250,
            "wavlm_batch_size": 4,
            "eval_wavlm_batch_size": 8,
            "learning_rate": 2e-5,
            "learning_rate_classifier": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "gradient_clip": 1.0,
            "use_class_weights": False,
            "num_workers": 2,
            "progress_bar": False,
        },
    },
}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def base_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "dataset": {
            "name": "iemocap",
            "iemocap_root": "data/iemocap",
            "auto_download": True,
            "kaggle_dataset": "sangayb/iemocap",
            "test_session": 5,
            "validation_ratio": 0.1,
            "sampling_rate": 16000,
            "max_duration_seconds": None,
        },
        "model": {
            "wavlm_model_name": "microsoft/wavlm-base",
            "num_labels": 4,
            "use_mal_memory": True,
            "use_temporal_features": True,
            "temporal_feature_mode": "real",
            "temporal_input_mode": "real",
            "disabled_temporal_feature_groups": [],
            "temporal_shuffle_seed": 0,
            "residual_gate_init": 0.0,
            "alpha_init": 0.0,
            "beta_init": 0.0,
            "fix_alpha_zero": False,
            "fix_beta_zero": False,
            "immediate_gap_threshold": 0.1,
            "short_gap_threshold": 0.3,
            "long_gap_threshold": 1.0,
            "overlap_threshold": 0.05,
            "strong_overlap_ratio_threshold": 0.30,
            "density_window_seconds": 10.0,
        },
        "precompute": {
            "pooling": "mean",
        },
        "cross_session": {
            "enabled": True,
            "test_sessions": [1, 2, 3, 4, 5],
            "run_name": None,
        },
        "training": {
            "device": "auto",
            "batch_mode": "dialogue",
        },
        "analysis": {
            "strong_overlap_threshold": 0.25,
        },
        "wandb": {
            "use_wandb": True,
            "project": "conversational-SER",
            "run_name": None,
            "entity": None,
            "mode": "online",
        },
    }


def resolve_version_config(
    version: str,
    setting: str = "A",
    seed: int = 42,
    output_root: str | Path = "results/versioned_loso",
    max_epochs: int | None = None,
    cross_session: bool | None = None,
    wandb_mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if version not in VERSION_REGISTRY:
        raise ValueError(f"Unknown version={version!r}. Expected one of {sorted(VERSION_REGISTRY)}.")
    setting = setting.upper()
    if setting not in SETTING_REGISTRY:
        raise ValueError(f"Unknown setting={setting!r}. Expected one of {sorted(SETTING_REGISTRY)}.")

    version_spec = VERSION_REGISTRY[version]
    setting_spec = SETTING_REGISTRY[setting]
    output_dir = Path(output_root) / f"setting_{setting}" / str(version_spec["name"])

    config = base_config()
    config = deep_update(config, {"model": setting_spec["model"]})
    config = deep_update(config, {"precompute": setting_spec["precompute"]})
    config = deep_update(config, {"training": setting_spec["training"]})
    config = deep_update(config, {"model": version_spec["model_overrides"]})
    config["version"] = version
    config["setting"] = setting
    config["seed"] = int(seed)
    config["experiment_name"] = f"{version_spec['experiment_name']}_setting_{setting}_seed_{seed}"
    config["run_name"] = config["experiment_name"]
    config["output_dir"] = str(output_dir)
    config["wandb"]["run_name"] = config["experiment_name"]
    if version_spec.get("training_stage") is not None:
        config["training_stage"] = copy.deepcopy(version_spec["training_stage"])
    if max_epochs is not None:
        config["training"]["max_epochs"] = int(max_epochs)
        if config.get("training_stage", {}).get("mode") in {"3_phase", "temporal_first_3_phase"}:
            # Keep 100/100/50 only by default. Explicit max_epochs is for smoke/debug runs.
            config["training_stage"]["stage_1_epochs"] = int(max_epochs)
            config["training_stage"]["stage_2_epochs"] = int(max_epochs)
            config["training_stage"]["stage_3_epochs"] = max(1, int(max_epochs))
    if cross_session is not None:
        config["cross_session"]["enabled"] = bool(cross_session)
    if wandb_mode is not None:
        if wandb_mode == "disabled":
            config["wandb"]["use_wandb"] = False
        else:
            config["wandb"]["use_wandb"] = True
            config["wandb"]["mode"] = wandb_mode

    metadata = {
        "version": version,
        "setting": setting,
        "version_name": version_spec["name"],
        "setting_description": setting_spec["description"],
        "trainer_module": version_spec["trainer_module"],
        "output_dir": str(output_dir),
    }
    return config, metadata


def write_resolved_config(config: dict[str, Any], metadata: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "resolved_config.yaml"
    metadata_path = output_dir / "version_metadata.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path
