from __future__ import annotations

import random
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_NAMES = ["angry", "happy", "neutral", "sad"]
LABEL2ID = {"angry": 0, "happy": 1, "neutral": 2, "sad": 3}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
LABEL_MAPPING_VERSION = "iemocap_emotion_8_to_4_v1"

# Every emotion label with a semantic 4-class destination is retained.
# `xxx` means annotators did not agree and `oth` has no defensible 4-class
# target, so neither may be used as a supervised gold label.
RAW_LABEL_MAP = {
    "ang": "angry",
    "fru": "angry",
    "dis": "angry",
    "hap": "happy",
    "exc": "happy",
    "sur": "happy",
    "neu": "neutral",
    "sad": "sad",
    "fea": "sad",
}
UNLABELED_RAW_LABELS = {"xxx", "oth"}

EVAL_RE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s+"
    r"(?P<utterance_id>\S+)\s+(?P<label>\S+)"
)
TRANSCRIPTION_RE = re.compile(
    r"^(?P<utterance_id>\S+)\s+\[(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\]:\s*(?P<text>.*)$"
)
SESSION_RE = re.compile(r"Session(?P<session>\d+)")


@dataclass(frozen=True)
class ConversationSERSample:
    audio_path: str
    label: int
    label_name: str
    session_id: int
    dialogue_id: str
    utterance_id: str
    speaker_id: str
    start_time: float
    end_time: float
    transcript: str = ""
    raw_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utterance_to_dialogue_id(utterance_id: str) -> str:
    parts = utterance_id.split("_")
    if len(parts) < 2:
        return utterance_id
    return "_".join(parts[:-1])


def utterance_to_speaker_id(utterance_id: str, session_id: int) -> str:
    last = utterance_id.split("_")[-1]
    speaker_role = last[0] if last else ""
    return f"Ses{session_id:02d}_{speaker_role}" if speaker_role in {"F", "M"} else f"Ses{session_id:02d}_UNK"


def parse_transcription_file(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TRANSCRIPTION_RE.match(line.strip())
            if not match:
                continue
            utterance_id = match.group("utterance_id")
            rows[utterance_id] = {
                "start_time": float(match.group("start")),
                "end_time": float(match.group("end")),
                "transcript": match.group("text"),
            }
    return rows


def parse_emotion_file(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = EVAL_RE.match(line.strip())
            if not match:
                continue
            utterance_id = match.group("utterance_id")
            raw_label = match.group("label")
            label_name = RAW_LABEL_MAP.get(raw_label)
            if label_name is None:
                continue
            rows[utterance_id] = {
                "start_time": float(match.group("start")),
                "end_time": float(match.group("end")),
                "raw_label": raw_label,
                "label_name": label_name,
                "label": LABEL2ID[label_name],
            }
    return rows


def has_iemocap_sessions(path: Path) -> bool:
    return all((path / f"Session{session_id}").is_dir() for session_id in range(1, 6))


def find_iemocap_session_root(search_root: Path) -> Optional[Path]:
    if has_iemocap_sessions(search_root):
        return search_root
    for candidate in search_root.rglob("*"):
        if candidate.is_dir() and has_iemocap_sessions(candidate):
            return candidate
    return None


def ensure_iemocap_root(
    iemocap_root: str | Path,
    auto_download: bool = False,
    kaggle_dataset: str = "sangayb/iemocap",
) -> Path:
    root = Path(iemocap_root)
    if has_iemocap_sessions(root):
        return root
    if root.exists():
        raise FileNotFoundError(f"IEMOCAP root exists but does not contain Session1..Session5: {root}")
    if not auto_download:
        raise FileNotFoundError(
            f"IEMOCAP root not found: {root}. Set dataset.auto_download=true to download from Kaggle."
        )
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `python -m pip install kaggle` and configure "
            "KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json, then rerun."
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    download_dir = root.parent / f".{root.name}_kaggle_download"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    command = [kaggle_bin, "datasets", "download", "-d", kaggle_dataset, "-p", str(download_dir)]
    subprocess.run(command, check=True)

    downloaded_zips = sorted(download_dir.glob("*.zip"))
    if not downloaded_zips:
        raise RuntimeError(f"Kaggle download completed but no zip file was found in {download_dir}.")
    zip_path = downloaded_zips[0]
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(download_dir)

    session_root = find_iemocap_session_root(download_dir)
    if session_root is None:
        raise RuntimeError(f"Could not find Session1..Session5 after extracting Kaggle dataset in {download_dir}.")
    if root.exists():
        raise FileExistsError(f"Target IEMOCAP root already exists after download: {root}")
    shutil.move(str(session_root), str(root))
    shutil.rmtree(download_dir, ignore_errors=True)
    return root


def discover_iemocap_samples(
    iemocap_root: str | Path,
    auto_download: bool = False,
    kaggle_dataset: str = "sangayb/iemocap",
) -> List[ConversationSERSample]:
    root = ensure_iemocap_root(iemocap_root, auto_download=auto_download, kaggle_dataset=kaggle_dataset)

    samples: List[ConversationSERSample] = []
    for session_dir in sorted(root.glob("Session*")):
        session_match = SESSION_RE.search(session_dir.name)
        if session_match is None:
            continue
        session_id = int(session_match.group("session"))
        eval_dir = session_dir / "dialog" / "EmoEvaluation"
        transcript_dir = session_dir / "dialog" / "transcriptions"
        wav_root = session_dir / "sentences" / "wav"

        for eval_path in sorted(eval_dir.glob("*.txt")):
            dialogue_id = eval_path.stem
            transcript_path = transcript_dir / f"{dialogue_id}.txt"
            transcripts = parse_transcription_file(transcript_path) if transcript_path.exists() else {}
            annotations = parse_emotion_file(eval_path)

            for utterance_id, annotation in annotations.items():
                wav_path = wav_root / dialogue_id / f"{utterance_id}.wav"
                if not wav_path.exists():
                    continue
                transcript_row = transcripts.get(utterance_id, {})
                start_time = float(transcript_row.get("start_time", annotation["start_time"]))
                end_time = float(transcript_row.get("end_time", annotation["end_time"]))
                samples.append(
                    ConversationSERSample(
                        audio_path=str(wav_path),
                        label=int(annotation["label"]),
                        label_name=str(annotation["label_name"]),
                        session_id=session_id,
                        dialogue_id=dialogue_id,
                        utterance_id=utterance_id,
                        speaker_id=utterance_to_speaker_id(utterance_id, session_id),
                        start_time=start_time,
                        end_time=end_time,
                        transcript=str(transcript_row.get("transcript", "")),
                        raw_label=str(annotation["raw_label"]),
                    )
                )
    if not samples:
        raise RuntimeError(f"No valid IEMOCAP samples found under {root}.")
    return samples


def split_loso_by_dialogue(
    samples: Sequence[ConversationSERSample],
    test_session: int,
    validation_ratio: float,
    seed: int,
) -> Dict[str, List[ConversationSERSample]]:
    test = [sample for sample in samples if sample.session_id == test_session]
    train_candidates = [sample for sample in samples if sample.session_id != test_session]
    if not test:
        raise ValueError(f"Test split is empty for TEST_SESSION={test_session}.")
    if not train_candidates:
        raise ValueError(f"Train split is empty for TEST_SESSION={test_session}.")

    dialogue_ids = sorted({sample.dialogue_id for sample in train_candidates})
    rng = random.Random(seed)
    rng.shuffle(dialogue_ids)
    validation_count = max(1, int(round(len(dialogue_ids) * validation_ratio))) if validation_ratio > 0 else 0
    validation_dialogues = set(dialogue_ids[:validation_count])

    train = [sample for sample in train_candidates if sample.dialogue_id not in validation_dialogues]
    validation = [sample for sample in train_candidates if sample.dialogue_id in validation_dialogues]
    if validation_ratio > 0 and not validation:
        raise ValueError("Validation split is empty; lower validation_ratio or inspect dialogue parsing.")
    return {"train": train, "validation": validation, "test": test}


def load_audio_mono(path: str | Path, target_sampling_rate: int) -> np.ndarray:
    import soundfile as sf

    waveform, sampling_rate = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if int(sampling_rate) != int(target_sampling_rate):
        try:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=int(sampling_rate), target_sr=int(target_sampling_rate))
        except ImportError:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(int(sampling_rate), int(target_sampling_rate))
            waveform = resample_poly(
                waveform,
                up=int(target_sampling_rate) // divisor,
                down=int(sampling_rate) // divisor,
            )
    return waveform.astype(np.float32)


class ConversationalSERDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ConversationSERSample],
        sampling_rate: int = 16000,
        max_duration_seconds: Optional[float] = None,
    ) -> None:
        self.samples = list(samples)
        self.sampling_rate = int(sampling_rate)
        self.max_audio_length = (
            int(float(max_duration_seconds) * self.sampling_rate)
            if max_duration_seconds is not None and float(max_duration_seconds) > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        waveform = load_audio_mono(sample.audio_path, self.sampling_rate)
        if self.max_audio_length is not None and waveform.shape[0] > self.max_audio_length:
            waveform = waveform[: self.max_audio_length]
        row = sample.to_dict()
        row["waveform"] = waveform
        return row


class ConversationalSERCollator:
    def __init__(self, feature_extractor: Any, sampling_rate: int = 16000) -> None:
        self.feature_extractor = feature_extractor
        self.sampling_rate = int(sampling_rate)

    def __call__(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        rows = list(batch)
        encoded = self.feature_extractor(
            [np.asarray(row["waveform"], dtype=np.float32) for row in rows],
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        encoded["dialogue_id"] = [str(row["dialogue_id"]) for row in rows]
        encoded["utterance_id"] = [str(row["utterance_id"]) for row in rows]
        encoded["speaker_id"] = [str(row["speaker_id"]) for row in rows]
        encoded["start_time"] = torch.tensor([float(row["start_time"]) for row in rows], dtype=torch.float32)
        encoded["end_time"] = torch.tensor([float(row["end_time"]) for row in rows], dtype=torch.float32)
        encoded["label_name"] = [str(row["label_name"]) for row in rows]
        encoded["audio_path"] = [str(row["audio_path"]) for row in rows]
        return encoded
