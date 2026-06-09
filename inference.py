from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio
import yaml
from transformers import AutoFeatureExtractor

from dataset import CANONICAL_LABELS
from features import describe_acoustic_cues, extract_acoustic_features
from model import SERModel
from train import resolve_device


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def load_audio(path: str, sampling_rate: int) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
    return waveform.squeeze(0).numpy().astype(np.float32)


def make_explanation(
    predicted_emotion: str,
    confidence: float,
    transcript: str,
    acoustic_features: Dict[str, float],
) -> str:
    transcript_part = transcript.strip() if transcript.strip() else "No transcript was provided."
    cues = describe_acoustic_cues(acoustic_features)
    return (
        f"Predicted emotion is {predicted_emotion} with confidence {confidence:.3f}. "
        f"Transcript: {transcript_part} "
        f"Acoustic cues: {cues}. "
        f"These cues are used as supporting evidence for why the utterance may express {predicted_emotion}."
    )


@torch.no_grad()
def predict(
    audio_path: str,
    checkpoint_path: str,
    transcript: str = "",
    config_path: Optional[str] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config")
    if config is None:
        if config_path is None:
            raise ValueError("Checkpoint has no config. Pass --config explicitly.")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    model_cfg = config["model"]
    audio_cfg = config["audio"]
    sampling_rate = int(audio_cfg.get("sampling_rate", 16000))
    waveform = load_audio(audio_path, sampling_rate)
    max_duration = audio_cfg.get("max_duration_seconds")
    if max_duration:
        waveform = waveform[: int(float(max_duration) * sampling_rate)]

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["encoder_name"])
    encoded = feature_extractor(
        [waveform],
        sampling_rate=sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    model = SERModel(
        encoder_name=model_cfg["encoder_name"],
        num_labels=len(CANONICAL_LABELS),
        pooling=model_cfg.get("pooling", "mean"),
        freeze_encoder=False,
        dropout=float(model_cfg.get("dropout", 0.2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits = model(
        input_values=encoded["input_values"].to(device),
        attention_mask=encoded.get("attention_mask", None).to(device) if encoded.get("attention_mask", None) is not None else None,
    )
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    label_id = int(np.argmax(probabilities))
    predicted_emotion = CANONICAL_LABELS[label_id]
    confidence = float(probabilities[label_id])
    acoustic_features = extract_acoustic_features(waveform, sampling_rate, transcript=transcript)
    explanation = make_explanation(predicted_emotion, confidence, transcript, acoustic_features)
    return {
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(CANONICAL_LABELS)},
        "acoustic_features": acoustic_features,
        "explanation": explanation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SER inference and generate a simple explanation.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="outputs/ser_baseline/best.pt")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = predict(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        transcript=args.transcript,
        config_path=args.config,
        device_name=args.device,
    )
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
