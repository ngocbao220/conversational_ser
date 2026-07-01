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

TIM_V2_RECOMMENDED_FEATURE_NAMES = [
    "duration",
    "gap_prev",
    "prev_gap_abs",
    "turn_index_norm",
    "previous_mean_gap",
    "window3_average_gap",
    "window5_average_gap",
    "window3_gap_variance",
    "window5_gap_variance",
    "immediate_response",
    "short_response",
    "long_pause",
    "relative_gap_to_speaker_mean",
    "silence_density_10s",
    "interaction_density_10s",
    "rhythm_variance_window5",
    "rapid_exchange_state",
    "conflict_like_state",
    "hesitation_state",
    "speaker_prev_mean_gap",
    "speaker_prev_mean_duration",
    "speaker_dominance_time_so_far",
    "speaker_prev_overlap_rate",
    "speaker_prev_interruption_rate",
    "speaker_persistence_so_far",
    "speaker_switch",
    "same_speaker",
    "speaker_switch_frequency_window3",
    "speaker_switch_frequency_window5",
    "overlap_prev",
    "overlap_ratio",
    "is_overlap",
    "strong_overlap",
    "overlap_frequency_window3",
    "overlap_frequency_window5",
    "consecutive_overlap_count",
]

TIM_SELECTED_PRIMITIVE_FEATURE_NAMES = [
    "previous_mean_gap",
    "silence_density_10s",
    "speaker_prev_mean_duration",
    "window5_average_gap",
    "speaker_dominance_time_so_far",
    "overlap_frequency_window5",
    "speaker_prev_overlap_rate",
    "speaker_prev_mean_gap",
    "speaker_switch_frequency_window5",
    "window5_gap_variance",
    "conflict_like_state",
    "window3_average_gap",
]

TIM_INTERACTION_4_FEATURE_NAMES = [
    "relative_gap_to_speaker_mean",
    "overlap_ratio",
    "speaker_switch",
    "speaker_prev_overlap_rate",
]

TEMPORAL_FEATURE_SETS = {
    "v1": TEMPORAL_FEATURE_NAMES,
    "recommended_v2": TIM_V2_RECOMMENDED_FEATURE_NAMES,
    "selected_primitives": TIM_SELECTED_PRIMITIVE_FEATURE_NAMES,
    "interaction_4": TIM_INTERACTION_4_FEATURE_NAMES,
}

BINARY_TEMPORAL_FEATURES = {
    "is_overlap",
    "is_interrupting_prev",
    "speaker_switch",
    "same_speaker",
    "short_response",
    "long_pause",
    "immediate_response",
    "strong_overlap",
    "rapid_exchange_state",
    "conflict_like_state",
    "hesitation_state",
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
    disabled_features: tuple[str, ...] = ()
    shuffle_seed: int = 0
    feature_names: tuple[str, ...] = tuple(TEMPORAL_FEATURE_NAMES)

    def __post_init__(self) -> None:
        if self.mode not in {"real", "zero", "shuffled"}:
            raise ValueError("temporal_input_mode must be one of: real, zero, shuffled.")
        unknown_groups = set(self.disabled_feature_groups) - set(TEMPORAL_FEATURE_GROUPS)
        if unknown_groups:
            raise ValueError(f"Unknown temporal feature groups: {sorted(unknown_groups)}.")
        unknown_features = set(self.disabled_features) - set(self.feature_names)
        if unknown_features:
            raise ValueError(f"Unknown temporal features for current feature set: {sorted(unknown_features)}.")

    @classmethod
    def from_model_config(cls, model_cfg: Mapping[str, Any]) -> "TemporalInputPolicy":
        raw_groups = model_cfg.get("disabled_temporal_feature_groups", [])
        if isinstance(raw_groups, str):
            raw_groups = [raw_groups]
        if not isinstance(raw_groups, Sequence):
            raise ValueError("disabled_temporal_feature_groups must be a list of feature-group names.")
        raw_features = model_cfg.get("disabled_temporal_features", [])
        if isinstance(raw_features, str):
            raw_features = [raw_features]
        if not isinstance(raw_features, Sequence):
            raise ValueError("disabled_temporal_features must be a list of feature names.")
        return cls(
            mode=str(model_cfg.get("temporal_input_mode", "real")),
            disabled_feature_groups=tuple(str(group) for group in raw_groups),
            disabled_features=tuple(str(feature) for feature in raw_features),
            shuffle_seed=int(model_cfg.get("temporal_shuffle_seed", 0)),
            feature_names=tuple(TEMPORAL_FEATURE_SETS.get(str(model_cfg.get("temporal_feature_set", "v1")), TEMPORAL_FEATURE_NAMES)),
        )

    def apply(self, features: torch.Tensor, dialogue_id: str) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(
                "Expected temporal feature tensor with shape "
                f"[num_utterances, {len(self.feature_names)}], got {tuple(features.shape)}."
            )
        transformed = features.clone()
        if self.mode == "zero":
            return torch.zeros_like(transformed)

        for group in self.disabled_feature_groups:
            indices = [self.feature_names.index(name) for name in TEMPORAL_FEATURE_GROUPS[group] if name in self.feature_names]
            transformed[:, indices] = 0.0
        if self.disabled_features:
            indices = [self.feature_names.index(name) for name in self.disabled_features]
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
    strong_overlap_ratio_threshold: float = 0.30
    immediate_gap_threshold: float = 0.10
    density_window_seconds: float = 10.0

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
            "strong_overlap_ratio_threshold": self.strong_overlap_ratio_threshold,
            "immediate_gap_threshold": self.immediate_gap_threshold,
            "density_window_seconds": self.density_window_seconds,
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
            strong_overlap_ratio_threshold=float(payload.get("strong_overlap_ratio_threshold", 0.30)),
            immediate_gap_threshold=float(payload.get("immediate_gap_threshold", 0.10)),
            density_window_seconds=float(payload.get("density_window_seconds", 10.0)),
        )


class TemporalInteractionFeatureBuilder:
    def __init__(
        self,
        short_gap_threshold: float = 0.3,
        long_gap_threshold: float = 1.0,
        overlap_threshold: float = 0.05,
        feature_set: str = "v1",
        strong_overlap_ratio_threshold: float = 0.30,
        immediate_gap_threshold: float = 0.10,
        density_window_seconds: float = 10.0,
        eps: float = 1e-6,
        stats: TemporalFeatureStats | None = None,
    ) -> None:
        self.short_gap_threshold = float(short_gap_threshold)
        self.long_gap_threshold = float(long_gap_threshold)
        self.overlap_threshold = float(overlap_threshold)
        self.strong_overlap_ratio_threshold = float(strong_overlap_ratio_threshold)
        self.immediate_gap_threshold = float(immediate_gap_threshold)
        self.density_window_seconds = float(density_window_seconds)
        self.eps = float(eps)
        self.stats = stats
        self.feature_set = feature_set
        if stats is not None:
            self.feature_names = list(stats.feature_names)
        else:
            if feature_set not in TEMPORAL_FEATURE_SETS:
                raise ValueError(f"Unknown temporal feature_set={feature_set!r}. Expected one of {sorted(TEMPORAL_FEATURE_SETS)}.")
            self.feature_names = list(TEMPORAL_FEATURE_SETS[feature_set])

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def fit(self, train_dialogues: Sequence[DialogueEmbedding]) -> TemporalFeatureStats:
        max_train_dialogue_length = max((len(dialogue.rows) for dialogue in train_dialogues), default=1)
        raw_rows: List[Dict[str, float]] = []
        for dialogue in train_dialogues:
            raw_rows.extend(self._raw_dialogue_features(dialogue.rows, max_train_dialogue_length))

        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        continuous_feature_names = [name for name in self.feature_names if name not in BINARY_TEMPORAL_FEATURES]
        binary_feature_names = [name for name in self.feature_names if name in BINARY_TEMPORAL_FEATURES]
        for name in continuous_feature_names:
            values = np.asarray([row[name] for row in raw_rows], dtype=np.float32)
            mean[name] = float(values.mean()) if values.size else 0.0
            value_std = float(values.std()) if values.size else 1.0
            std[name] = value_std if value_std > self.eps else 1.0

        self.stats = TemporalFeatureStats(
            feature_names=list(self.feature_names),
            continuous_feature_names=list(continuous_feature_names),
            binary_feature_names=list(binary_feature_names),
            mean=mean,
            std=std,
            max_train_dialogue_length=max_train_dialogue_length,
            short_gap_threshold=self.short_gap_threshold,
            long_gap_threshold=self.long_gap_threshold,
            overlap_threshold=self.overlap_threshold,
            strong_overlap_ratio_threshold=self.strong_overlap_ratio_threshold,
            immediate_gap_threshold=self.immediate_gap_threshold,
            density_window_seconds=self.density_window_seconds,
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
            strong_overlap_ratio_threshold=stats.strong_overlap_ratio_threshold,
            immediate_gap_threshold=stats.immediate_gap_threshold,
            density_window_seconds=stats.density_window_seconds,
            stats=stats,
        )

    def transform_dialogue(self, dialogue: DialogueEmbedding) -> torch.Tensor:
        if self.stats is None:
            raise RuntimeError("TemporalInteractionFeatureBuilder must be fit before transform.")
        raw_rows = self._raw_dialogue_features(dialogue.rows, self.stats.max_train_dialogue_length)
        features = []
        for row in raw_rows:
            values = []
            for name in self.stats.feature_names:
                value = float(row[name])
                if name in self.stats.continuous_feature_names:
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
        gaps: List[float] = []
        overlaps: List[float] = []
        interruptions: List[float] = []
        switches: List[float] = []
        previous_raw_rows: List[Dict[str, float]] = []
        total_turns_so_far = 0.0
        total_time_so_far = 0.0
        consecutive_overlap = 0.0
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
                {
                    "turn_count": 0.0,
                    "overlap_count": 0.0,
                    "interruption_count": 0.0,
                    "gap_sum": 0.0,
                    "duration_sum": 0.0,
                    "same_speaker_count": 0.0,
                },
            )
            previous_turns = history["turn_count"]
            speaker_prev_overlap_rate = history["overlap_count"] / max(previous_turns, 1.0)
            speaker_prev_interruption_rate = history["interruption_count"] / max(previous_turns, 1.0)
            speaker_prev_mean_gap = history["gap_sum"] / max(previous_turns, 1.0)
            speaker_prev_mean_duration = history["duration_sum"] / max(previous_turns, 1.0)
            speaker_prev_turn_count_norm = previous_turns / max_length

            speaker_switch = 1.0 if has_previous and speaker_id != previous_speaker else 0.0
            same_speaker = 1.0 if has_previous and speaker_id == previous_speaker else 0.0
            is_overlap = 1.0 if has_previous and overlap_prev > self.overlap_threshold else 0.0
            is_interrupting_prev = 1.0 if has_previous and speaker_switch and start_time < float(previous_row["end_time"]) else 0.0
            strong_overlap = 1.0 if overlap_prev / max(duration, self.eps) >= self.strong_overlap_ratio_threshold else 0.0
            consecutive_overlap = consecutive_overlap + 1.0 if is_overlap else 0.0
            gaps.append(gap_prev)
            overlaps.append(is_overlap)
            interruptions.append(is_interrupting_prev)
            switches.append(speaker_switch)

            def window_mean(values: Sequence[float], window: int) -> float:
                subset = list(values)[-window:]
                return float(np.mean(subset)) if subset else 0.0

            def window_var(values: Sequence[float], window: int) -> float:
                subset = list(values)[-window:]
                return float(np.var(subset)) if len(subset) > 1 else 0.0

            previous_mean_gap = float(np.mean(gaps[:-1])) if len(gaps) > 1 else 0.0
            relative_gap_to_speaker_mean = gap_prev - speaker_prev_mean_gap if previous_turns > 0 else 0.0
            window3_average_gap = window_mean(gaps, 3)
            window5_average_gap = window_mean(gaps, 5)
            window3_gap_variance = window_var(gaps, 3)
            window5_gap_variance = window_var(gaps, 5)
            speaker_switch_frequency_window3 = window_mean(switches, 3)
            speaker_switch_frequency_window5 = window_mean(switches, 5)
            overlap_frequency_window3 = window_mean(overlaps, 3)
            overlap_frequency_window5 = window_mean(overlaps, 5)
            interaction_window_rows = [
                raw for raw in previous_raw_rows
                if start_time - raw["start_time"] <= self.density_window_seconds
            ]
            interaction_density_10s = (len(interaction_window_rows) + 1.0) / self.density_window_seconds
            silence_density_10s = sum(max(0.0, value) for value in gaps[-5:]) / self.density_window_seconds
            rapid_exchange_state = (
                1.0 if window5_average_gap < self.short_gap_threshold and speaker_switch_frequency_window5 >= 0.5 else 0.0
            )
            conflict_like_state = 1.0 if overlap_frequency_window5 >= 0.4 or window_mean(interruptions, 5) >= 0.3 else 0.0
            hesitation_state = 1.0 if gap_prev > self.long_gap_threshold or window5_average_gap > self.long_gap_threshold else 0.0
            row_features = {
                "start_time": start_time,
                "duration": duration,
                "gap_prev": gap_prev,
                "overlap_prev": overlap_prev,
                "overlap_ratio": overlap_prev / max(duration, self.eps),
                "is_overlap": is_overlap,
                "strong_overlap": strong_overlap,
                "is_interrupting_prev": is_interrupting_prev,
                "speaker_switch": speaker_switch,
                "same_speaker": same_speaker,
                "turn_index_norm": turn_index / max_length,
                "prev_gap_abs": abs(gap_prev),
                "short_response": 1.0 if has_previous and 0.0 <= gap_prev < self.short_gap_threshold else 0.0,
                "immediate_response": 1.0 if has_previous and 0.0 <= gap_prev < self.immediate_gap_threshold else 0.0,
                "long_pause": 1.0 if has_previous and gap_prev > self.long_gap_threshold else 0.0,
                "previous_mean_gap": previous_mean_gap,
                "window3_average_gap": window3_average_gap,
                "window5_average_gap": window5_average_gap,
                "window3_gap_variance": window3_gap_variance,
                "window5_gap_variance": window5_gap_variance,
                "relative_gap_to_speaker_mean": relative_gap_to_speaker_mean,
                "silence_density_10s": silence_density_10s,
                "interaction_density_10s": interaction_density_10s,
                "rhythm_variance_window5": window5_gap_variance,
                "rapid_exchange_state": rapid_exchange_state,
                "conflict_like_state": conflict_like_state,
                "hesitation_state": hesitation_state,
                "speaker_switch_frequency_window3": speaker_switch_frequency_window3,
                "speaker_switch_frequency_window5": speaker_switch_frequency_window5,
                "overlap_frequency_window3": overlap_frequency_window3,
                "overlap_frequency_window5": overlap_frequency_window5,
                "consecutive_overlap_count": consecutive_overlap,
                "speaker_prev_overlap_rate": speaker_prev_overlap_rate,
                "speaker_prev_interruption_rate": speaker_prev_interruption_rate,
                "speaker_prev_mean_gap": speaker_prev_mean_gap,
                "speaker_prev_mean_duration": speaker_prev_mean_duration,
                "speaker_prev_turn_count_norm": speaker_prev_turn_count_norm,
                "speaker_dominance_time_so_far": history["duration_sum"] / max(total_time_so_far, self.eps),
                "speaker_persistence_so_far": history["same_speaker_count"] / max(previous_turns, 1.0),
            }
            features.append(row_features)

            history["turn_count"] += 1.0
            history["overlap_count"] += is_overlap
            history["interruption_count"] += is_interrupting_prev
            history["gap_sum"] += gap_prev
            history["duration_sum"] += duration
            history["same_speaker_count"] += same_speaker
            speaker_history[speaker_id] = history
            total_turns_so_far += 1.0
            total_time_so_far += duration
            previous_row = row
            previous_raw_rows.append(row_features)
        return features


def attach_temporal_features_to_dialogues(
    dialogues: Sequence[DialogueEmbedding],
    builder: TemporalInteractionFeatureBuilder,
) -> None:
    for dialogue in dialogues:
        builder.attach_raw_features(dialogue)
