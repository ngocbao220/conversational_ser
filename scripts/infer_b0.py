from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor

from models.wavlm import build_b0_model
from scripts.evaluate_b0 import load_checkpoint, resolve_device
from utils.dataset import CANONICAL_LABELS, ID2LABEL


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    waveform, sampling_rate = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    return waveform, int(sampling_rate)


def predict(checkpoint_path: str | Path, audio_path: str | Path, device_name: str = "auto") -> Dict[str, Any]:
    device = resolve_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    b0_cfg = config["baselines"]["b0"]
    model_cfg = b0_cfg["model"]
    audio_cfg = config.get("audio", {})

    waveform, sampling_rate = load_audio(audio_path)
    target_rate = int(audio_cfg.get("sampling_rate", sampling_rate))
    if sampling_rate != target_rate:
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("Install librosa to resample inference audio.") from exc
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=target_rate)
        sampling_rate = target_rate

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["encoder_name"])
    encoded = feature_extractor(
        [waveform],
        sampling_rate=sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    model = build_b0_model(model_cfg, num_labels=len(CANONICAL_LABELS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(
            input_values=encoded["input_values"].to(device),
            attention_mask=encoded.get("attention_mask").to(device) if encoded.get("attention_mask") is not None else None,
        )
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    label_id = int(np.argmax(probabilities))
    return {
        "baseline": "B0_utterance",
        "audio_path": str(audio_path),
        "emotion": ID2LABEL[label_id],
        "confidence": float(probabilities[label_id]),
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(CANONICAL_LABELS)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0 utterance-level SER inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="outputs/b0_utterance/best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.audio, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
