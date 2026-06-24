from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from utils.dialogue_embeddings import DialogueEmbedding


TEMPORAL_FEATURE_NAMES = [
    "duration",
    "gap_prev",
    "overlap_prev",
    "overlap_ratio",
    "is_overlap",
    "is_interrupting_prev",
    "speaker_switch",
    "same_speaker",
    "turn_index_norm",
    "prev_gap_abs",
    "short_response",
    "long_pause",
    "speaker_prev_overlap_rate",
    "speaker_prev_mean_gap",
    "speaker_prev_mean_duration",
    "speaker_prev_turn_count_norm",
]

BINARY_TEMPORAL_FEATURES = {
    "is_overlap",
    "is_interrupting_prev",
    "speaker_switch",
    "same_speaker",
    "short_response",
    "long_pause",
}

TEMPORAL_FEATURE_GROUPS = {
    "duration": ("duration", "speaker_prev_mean_duration"),
    "gap": ("gap_prev", "prev_gap_abs", "short_response", "long_pause", "speaker_prev_mean_gap"),
    "overlap": ("overlap_prev", "overlap_ratio", "is_overlap", "is_interrupting_prev", "speaker_prev_overlap_rate"),
    "speaker_switch": ("speaker_switch", "same_speaker"),
    "turn_position": ("turn_index_norm", "speaker_prev_turn_count_norm"),
}

CONTINUOUS_TEMPORAL_FEATURES = [
    name for name in TEMPORAL_FEATURE_NAMES if name not in BINARY_TEMPORAL_FEATURES
]


@dataclass(frozen=True)
class TemporalInputPolicy:
    """Applies a reproducible temporal ablation without changing TIM parameters."""

    mode: str = "real"
    disabled_feature_groups: tuple[str, ...] = ()
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"real", "zero", "shuffled"}:
            raise ValueError("temporal_input_mode must be one of: real, zero, shuffled.")
        unknown_groups = set(self.disabled_feature_groups) - set(TEMPORAL_FEATURE_GROUPS)
        if unknown_groups:
            raise ValueError(f"Unknown temporal feature groups: {sorted(unknown_groups)}.")

    @classmethod
    def from_model_config(cls, model_cfg: Mapping[str, Any]) -> "TemporalInputPolicy":
        raw_groups = model_cfg.get("disabled_temporal_feature_groups", [])
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]
        if not isinstance(raw_groups, Sequence):
            raise ValueError("disabled_temporal_feature_groups must be a list of feature-group names.")
        return cls(
            mode=str(model_cfg.get("temporal_input_mode", "real")),
            disabled_feature_groups=tuple(str(group) for group in raw_groups),
            shuffle_seed=int(model_cfg.get("temporal_shuffle_seed", 0)),
        )

    def apply(self, features: torch.Tensor, dialogue_id: str) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(TEMPORAL_FEATURE_NAMES):
            raise ValueError(
                "Expected temporal feature tensor with shape "
                f"[num_utterances, {len(TEMPORAL_FEATURE_NAMES)}], got {tuple(features.shape)}."
            )
        transformed = features.clone()
        if self.mode == "zero":
            return torch.zeros_like(transformed)

        for group in self.disabled_feature_groups:
            indices = [TEMPORAL_FEATURE_NAMES.index(name) for name in TEMPORAL_FEATURE_GROUPS[group]]
            transformed[:, indices] = 0.0

        if self.mode == "shuffled" and len(transformed) > 1:
            digest = hashlib.blake2b(
                f"{self.shuffle_seed}:{dialogue_id}".encode("utf-8"), digest_size=8
            ).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest, byteorder="big", signed=False))
            transformed = transformed[torch.randperm(len(transformed), generator=generator)]
        return transformed


@dataclass(frozen=True)
class TemporalFeatureStats:
    feature_names: List[str]
    continuous_feature_names: List[str]
    binary_feature_names: List[str]
    mean: Dict[str, float]
    std: Dict[str, float]
    max_train_dialogue_length: int
    short_gap_threshold: float
    long_gap_threshold: float
    overlap_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "continuous_feature_names": self.continuous_feature_names,
            "binary_feature_names": self.binary_feature_names,
            "mean": self.mean,
            "std": self.std,
            "max_train_dialogue_length": self.max_train_dialogue_length,
            "short_gap_threshold": self.short_gap_threshold,
            "long_gap_threshold": self.long_gap_threshold,
            "overlap_threshold": self.overlap_threshold,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TemporalFeatureStats":
        return cls(
            feature_names=list(payload["feature_names"]),
            continuous_feature_names=list(payload["continuous_feature_names"]),
            binary_feature_names=list(payload["binary_feature_names"]),
            mean={str(key): float(value) for key, value in payload["mean"].items()},
            std={str(key): float(value) for key, value in payload["std"].items()},
            max_train_dialogue_length=int(payload["max_train_dialogue_length"]),
            short_gap_threshold=float(payload["short_gap_threshold"]),
            long_gap_threshold=float(payload["long_gap_threshold"]),
            overlap_threshold=float(payload["overlap_threshold"]),
        )


class TemporalInteractionFeatureBuilder:
    def __init__(
        self,
        short_gap_threshold: float = 0.3,
        long_gap_threshold: float = 1.0,
        overlap_threshold: float = 0.05,
        eps: float = 1e-6,
        stats: TemporalFeatureStats | None = None,
    ) -> None:
        self.short_gap_threshold = float(short_gap_threshold)
        self.long_gap_threshold = float(long_gap_threshold)
        self.overlap_threshold = float(overlap_threshold)
        self.eps = float(eps)
        self.stats = stats

    @property
    def feature_dim(self) -> int:
        return len(TEMPORAL_FEATURE_NAMES)

    def fit(self, train_dialogues: Sequence[DialogueEmbedding]) -> TemporalFeatureStats:
        max_train_dialogue_length = max((len(dialogue.rows) for dialogue in train_dialogues), default=1)
        raw_rows: List[Dict[str, float]] = []
        for dialogue in train_dialogues:
            raw_rows.extend(self._raw_dialogue_features(dialogue.rows, max_train_dialogue_length))

        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        for name in CONTINUOUS_TEMPORAL_FEATURES:
            values = np.asarray([row[name] for row in raw_rows], dtype=np.float32)
            mean[name] = float(values.mean()) if values.size else 0.0
            value_std = float(values.std()) if values.size else 1.0
            std[name] = value_std if value_std > self.eps else 1.0

        self.stats = TemporalFeatureStats(
            feature_names=list(TEMPORAL_FEATURE_NAMES),
            continuous_feature_names=list(CONTINUOUS_TEMPORAL_FEATURES),
            binary_feature_names=sorted(BINARY_TEMPORAL_FEATURES),
            mean=mean,
            std=std,
            max_train_dialogue_length=max_train_dialogue_length,
            short_gap_threshold=self.short_gap_threshold,
            long_gap_threshold=self.long_gap_threshold,
            overlap_threshold=self.overlap_threshold,
        )
        return self.stats

    def save_stats(self, path: str | Path) -> None:
        if self.stats is None:
            raise RuntimeError("Cannot save temporal feature stats before fit/load.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_stats_file(cls, path: str | Path) -> "TemporalInteractionFeatureBuilder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = TemporalFeatureStats.from_dict(payload)
        return cls(
            short_gap_threshold=stats.short_gap_threshold,
            long_gap_threshold=stats.long_gap_threshold,
            overlap_threshold=stats.overlap_threshold,
            stats=stats,
        )

    def transform_dialogue(self, dialogue: DialogueEmbedding) -> torch.Tensor:
        if self.stats is None:
            raise RuntimeError("TemporalInteractionFeatureBuilder must be fit before transform.")
        raw_rows = self._raw_dialogue_features(dialogue.rows, self.stats.max_train_dialogue_length)
        features = []
        for row in raw_rows:
            values = []
            for name in TEMPORAL_FEATURE_NAMES:
                value = float(row[name])
                if name in CONTINUOUS_TEMPORAL_FEATURES:
                    value = (value - self.stats.mean[name]) / self.stats.std[name]
                values.append(value)
            features.append(values)
        return torch.tensor(features, dtype=torch.float32)

    def attach_raw_features(self, dialogue: DialogueEmbedding) -> None:
        if self.stats is None:
            raise RuntimeError("TemporalInteractionFeatureBuilder must be fit before attach_raw_features.")
        raw_rows = self._raw_dialogue_features(dialogue.rows, self.stats.max_train_dialogue_length)
        for row, feature_row in zip(dialogue.rows, raw_rows):
            row.update(feature_row)

    def _raw_dialogue_features(self, rows: Sequence[Mapping[str, Any]], max_train_dialogue_length: int) -> List[Dict[str, float]]:
        features: List[Dict[str, float]] = []
        speaker_history: Dict[str, Dict[str, float]] = {}
        previous_row: Mapping[str, Any] | None = None
        previous_gap = 0.0
        max_length = max(1, int(max_train_dialogue_length))
        for turn_index, row in enumerate(rows):
            start_time = float(row["start_time"])
            end_time = float(row["end_time"])
            duration = max(0.0, end_time - start_time)
            speaker_id = str(row["speaker_id"])

            if previous_row is None:
                gap_prev = 0.0
                overlap_prev = 0.0
                previous_speaker = None
                has_previous = False
            else:
                previous_end = float(previous_row["end_time"])
                gap_prev = start_time - previous_end
                overlap_prev = max(0.0, previous_end - start_time)
                previous_speaker = str(previous_row["speaker_id"])
                has_previous = True

            history = speaker_history.get(
                speaker_id,
                {"turn_count": 0.0, "overlap_count": 0.0, "gap_sum": 0.0, "duration_sum": 0.0},
            )
            previous_turns = history["turn_count"]
            speaker_prev_overlap_rate = history["overlap_count"] / max(previous_turns, 1.0)
            speaker_prev_mean_gap = history["gap_sum"] / max(previous_turns, 1.0)
            speaker_prev_mean_duration = history["duration_sum"] / max(previous_turns, 1.0)
            speaker_prev_turn_count_norm = previous_turns / max_length

            speaker_switch = 1.0 if has_previous and speaker_id != previous_speaker else 0.0
            same_speaker = 1.0 if has_previous and speaker_id == previous_speaker else 0.0
            is_overlap = 1.0 if has_previous and overlap_prev > self.overlap_threshold else 0.0
            is_interrupting_prev = 1.0 if has_previous and speaker_switch and start_time < float(previous_row["end_time"]) else 0.0
            row_features = {
                "duration": duration,
                "gap_prev": gap_prev,
                "overlap_prev": overlap_prev,
                "overlap_ratio": overlap_prev / max(duration, self.eps),
                "is_overlap": is_overlap,
                "is_interrupting_prev": is_interrupting_prev,
                "speaker_switch": speaker_switch,
                "same_speaker": same_speaker,
                "turn_index_norm": turn_index / max_length,
                "prev_gap_abs": abs(gap_prev),
                "short_response": 1.0 if has_previous and 0.0 <= gap_prev < self.short_gap_threshold else 0.0,
                "long_pause": 1.0 if has_previous and gap_prev > self.long_gap_threshold else 0.0,
                "speaker_prev_overlap_rate": speaker_prev_overlap_rate,
                "speaker_prev_mean_gap": speaker_prev_mean_gap,
                "speaker_prev_mean_duration": speaker_prev_mean_duration,
                "speaker_prev_turn_count_norm": speaker_prev_turn_count_norm,
            }
            features.append(row_features)

            history["turn_count"] += 1.0
            history["overlap_count"] += is_overlap
            history["gap_sum"] += gap_prev
            history["duration_sum"] += duration
            speaker_history[speaker_id] = history
            previous_row = row
        return features


def attach_temporal_features_to_dialogues(
    dialogues: Sequence[DialogueEmbedding],
    builder: TemporalInteractionFeatureBuilder,
) -> None:
    for dialogue in dialogues:
        builder.attach_raw_features(dialogue)
