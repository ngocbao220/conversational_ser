from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def _word_count(transcript: str) -> int:
    return len([token for token in transcript.strip().split() if token])


def extract_acoustic_features(
    waveform: np.ndarray,
    sampling_rate: int,
    transcript: str = "",
    dataset_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Return simple cues for explanation. Dataset-provided values are preferred."""
    row = dataset_row or {}
    waveform = np.asarray(waveform, dtype=np.float32)
    duration = float(len(waveform) / sampling_rate) if sampling_rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    energy = float(np.mean(np.square(waveform))) if waveform.size else 0.0

    speaking_rate = row.get("speaking_rate")
    if speaking_rate is None:
        speaking_rate = _word_count(transcript) / max(duration, 1e-6)

    return {
        "duration": float(row.get("audio_time", duration)),
        "energy": float(row.get("energy", energy)),
        "rms": float(row.get("rms", rms)),
        "relative_db": float(row.get("relative_db", 20.0 * np.log10(max(rms, 1e-8)))),
        "pitch_mean": float(row.get("pitch_mean", 0.0) or 0.0),
        "pitch_std": float(row.get("pitch_std", 0.0) or 0.0),
        "speech_rate": float(speaking_rate or 0.0),
    }


def describe_acoustic_cues(features: Mapping[str, float]) -> str:
    pitch = features.get("pitch_mean", 0.0)
    pitch_std = features.get("pitch_std", 0.0)
    rms = features.get("rms", 0.0)
    rate = features.get("speech_rate", 0.0)
    duration = features.get("duration", 0.0)
    return (
        f"pitch_mean={pitch:.1f}Hz, pitch_std={pitch_std:.1f}, "
        f"rms={rms:.4f}, duration={duration:.2f}s, speech_rate={rate:.2f} words/s"
    )
