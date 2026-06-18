from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict, Features, Sequence, Value, load_dataset


CANONICAL_LABELS = ["neutral", "happy", "sad", "angry"]
LABEL2ID = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Keep this mapping explicit so later experiments can change the 8-to-4 policy.
EMOTION_TO_CANONICAL = {
    "neutral": "neutral",
    "happy": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
    "frustrated": "angry",
}

SKIP_LABELS = {
    "",
    "tie",
    "tie_prediction",
    "ambiguous",
    "unknown",
    "other",
    "fear",
    "surprise",
    "disgust",
}

EMOTION_SCORE_COLUMNS = [
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "frustrated",
    "fear",
    "surprise",
    "disgust",
]

TRANSCRIPT_COLUMNS = ["transcription", "transcript", "text", "sentence", "utterance"]
ID_COLUMNS = (
    "utt_id",
    "utterance_id",
    "utterance_name",
    "id",
    "name",
    "file",
    "filename",
    "file_id",
    "audio_file",
)


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in SKIP_LABELS:
        return None
    return EMOTION_TO_CANONICAL.get(raw)


def infer_emotion_from_scores(example: Mapping[str, Any]) -> Optional[str]:
    scores: List[Tuple[str, float]] = []
    for col in EMOTION_SCORE_COLUMNS:
        value = example.get(col)
        if value is None:
            continue
        try:
            scores.append((col, float(value)))
        except (TypeError, ValueError):
            continue

    if not scores:
        return None

    scores.sort(key=lambda item: item[1], reverse=True)
    if len(scores) > 1 and np.isclose(scores[0][1], scores[1][1]):
        return None
    return normalize_label(scores[0][0])


def get_canonical_label(example: Mapping[str, Any]) -> Optional[str]:
    for label_column in ("major_emotion", "emotion", "label"):
        if label_column in example:
            mapped = normalize_label(example[label_column])
            if mapped is not None:
                return mapped
            raw = str(example[label_column]).strip().lower() if example[label_column] is not None else ""
            if raw in SKIP_LABELS:
                return None
    return infer_emotion_from_scores(example)


def get_transcript(example: Mapping[str, Any]) -> str:
    for col in TRANSCRIPT_COLUMNS:
        if col in example and example[col] is not None:
            return str(example[col])
    return ""


def get_utterance_id(example: Mapping[str, Any]) -> str:
    for col in ID_COLUMNS:
        value = example.get(col)
        if value is not None:
            return str(value)
    audio = example.get("audio")
    if isinstance(audio, Mapping):
        for key in ("path", "file", "filename"):
            value = audio.get(key)
            if value is not None:
                return str(value)
    return ""


def get_session_id(example: Mapping[str, Any]) -> str:
    utt_id = get_utterance_id(example)
    match = re.search(r"(Ses\d{2})", utt_id)
    if match:
        return match.group(1)

    session = example.get("session_id") or example.get("session")
    if session is not None:
        session_text = str(session)
        match = re.search(r"(Ses\d{2})", session_text)
        if match:
            return match.group(1)
        if session_text.isdigit():
            return f"Ses{int(session_text):02d}"
    return ""


def _audio_array(example: Mapping[str, Any]) -> np.ndarray:
    audio = example["audio"]
    if isinstance(audio, dict):
        array = audio.get("array")
    elif hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = samples.data
    else:
        array = audio
    return np.asarray(array, dtype=np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim <= 1:
        return audio
    if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        return np.mean(audio, axis=0)
    return np.mean(audio, axis=1)


def prepare_example(example: Dict[str, Any], max_audio_length: Optional[int] = None) -> Optional[Dict[str, Any]]:
    label = get_canonical_label(example)
    if label is None:
        return None

    audio = to_mono(_audio_array(example))
    if max_audio_length is not None and audio.shape[0] > max_audio_length:
        audio = audio[:max_audio_length]

    prepared = {
        "input_values": audio.astype(np.float32).tolist(),
        "labels": LABEL2ID[label],
        "emotion": label,
        "transcript": get_transcript(example),
        "utterance_id": get_utterance_id(example),
        "session_id": get_session_id(example),
        "speaking_rate": 0.0,
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "rms": 0.0,
        "relative_db": 0.0,
    }
    for feature_name in ("speaking_rate", "pitch_mean", "pitch_std", "rms", "relative_db"):
        if feature_name in example and example[feature_name] is not None:
            prepared[feature_name] = float(example[feature_name])
    return prepared


def filter_and_prepare_dataset(
    dataset: Dataset,
    sampling_rate: int,
    max_duration_seconds: Optional[float],
    num_proc: int = 1,
    row_filter: Optional[Callable[[Mapping[str, Any]], bool]] = None,
) -> Dataset:
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    if row_filter is None:
        dataset = dataset.filter(lambda row: get_canonical_label(row) is not None, num_proc=num_proc)
    else:
        dataset = dataset.filter(lambda row: row_filter(row) and get_canonical_label(row) is not None, num_proc=num_proc)
    max_audio_length = None
    if max_duration_seconds is not None and max_duration_seconds > 0:
        max_audio_length = int(max_duration_seconds * sampling_rate)

    def mapper(example: Dict[str, Any]) -> Dict[str, Any]:
        prepared = prepare_example(example, max_audio_length=max_audio_length)
        if prepared is None:
            raise ValueError("Unexpected unmapped label after filtering.")
        return prepared

    feature_schema = Features(
        {
            "input_values": Sequence(Value("float32")),
            "labels": Value("int64"),
            "emotion": Value("string"),
            "transcript": Value("string"),
            "utterance_id": Value("string"),
            "session_id": Value("string"),
            "speaking_rate": Value("float32"),
            "pitch_mean": Value("float32"),
            "pitch_std": Value("float32"),
            "rms": Value("float32"),
            "relative_db": Value("float32"),
        }
    )
    return dataset.map(
        mapper,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        features=feature_schema,
    )


def load_iemocap_splits(config: Mapping[str, Any]) -> DatasetDict:
    dataset_cfg = config.get("dataset", {})
    audio_cfg = config.get("audio", {})

    name = dataset_cfg.get("name", "AbstractTTS/IEMOCAP")
    seed = int(dataset_cfg.get("seed", 42))
    num_proc = int(dataset_cfg.get("num_proc", 1))
    sampling_rate = int(audio_cfg.get("sampling_rate", 16000))
    max_duration_seconds = audio_cfg.get("max_duration_seconds")

    raw = load_dataset(name)
    if isinstance(raw, Dataset):
        raw = DatasetDict({"train": raw})

    if "train" not in raw:
        first_split = next(iter(raw.keys()))
        raw = DatasetDict({"train": raw[first_split]})

    if str(dataset_cfg.get("split_strategy", "random")).lower() == "loso":
        return load_loso_iemocap_splits(raw, dataset_cfg, sampling_rate, max_duration_seconds, num_proc, seed)

    prepared = DatasetDict(
        {
            split: filter_and_prepare_dataset(ds, sampling_rate, max_duration_seconds, num_proc)
            for split, ds in raw.items()
        }
    )

    validation_split = dataset_cfg.get("validation_split", "validation")
    test_split = dataset_cfg.get("test_split", "test")
    if validation_split in prepared and test_split in prepared:
        return limit_split_sizes(prepared, dataset_cfg, seed)

    train = prepared["train"]
    validation_size = float(dataset_cfg.get("validation_size", 0.1))
    test_size = float(dataset_cfg.get("test_size", 0.1))
    holdout_size = validation_size + test_size
    if holdout_size <= 0:
        return DatasetDict({"train": train, "validation": train.select([]), "test": train.select([])})

    first = stratified_or_random_split(train, test_size=holdout_size, seed=seed)
    holdout = first["test"]
    validation_fraction = validation_size / holdout_size
    second = stratified_or_random_split(holdout, test_size=1.0 - validation_fraction, seed=seed)
    return limit_split_sizes(
        DatasetDict({"train": first["train"], "validation": second["train"], "test": second["test"]}),
        dataset_cfg,
        seed,
    )


def load_loso_iemocap_splits(
    raw: DatasetDict,
    dataset_cfg: Mapping[str, Any],
    sampling_rate: int,
    max_duration_seconds: Optional[float],
    num_proc: int,
    seed: int,
) -> DatasetDict:
    test_session = str(dataset_cfg.get("test_session", "Ses05"))
    source = raw["train"]

    prepared_train = filter_and_prepare_dataset(
        source,
        sampling_rate,
        max_duration_seconds,
        num_proc,
        row_filter=lambda row: get_session_id(row) not in {"", test_session},
    )
    prepared_test = filter_and_prepare_dataset(
        source,
        sampling_rate,
        max_duration_seconds,
        num_proc,
        row_filter=lambda row: get_session_id(row) == test_session,
    )

    if len(prepared_test) == 0:
        raise ValueError(f"LOSO test split is empty for test_session={test_session!r}.")
    if len(prepared_train) == 0:
        raise ValueError(f"LOSO train split is empty for test_session={test_session!r}.")

    validation_size = float(dataset_cfg.get("validation_size", 0.1))
    if validation_size > 0:
        train_validation = stratified_or_random_split(prepared_train, test_size=validation_size, seed=seed)
        train = train_validation["train"]
        validation = train_validation["test"]
    else:
        train = prepared_train
        validation = prepared_train.select([])

    return limit_split_sizes(
        DatasetDict({"train": train, "validation": validation, "test": prepared_test}),
        dataset_cfg,
        seed,
    )


def stratified_or_random_split(dataset: Dataset, test_size: float, seed: int) -> DatasetDict:
    try:
        return dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column="labels")
    except ValueError:
        return dataset.train_test_split(test_size=test_size, seed=seed)


def limit_split_sizes(datasets: DatasetDict, dataset_cfg: Mapping[str, Any], seed: int) -> DatasetDict:
    limits = {
        "train": dataset_cfg.get("max_train_samples"),
        "validation": dataset_cfg.get("max_validation_samples"),
        "test": dataset_cfg.get("max_test_samples"),
    }
    limited = DatasetDict()
    for split, dataset in datasets.items():
        limit = limits.get(split)
        if limit is None:
            limited[split] = dataset
            continue
        limit = min(int(limit), len(dataset))
        limited[split] = dataset.shuffle(seed=seed).select(range(limit))
    return limited


@dataclass
class SERDataCollator:
    feature_extractor: Any
    sampling_rate: int

    def __call__(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        rows = list(batch)
        waveforms = [np.asarray(row["input_values"], dtype=np.float32) for row in rows]
        encoded = self.feature_extractor(
            waveforms,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([int(row["labels"]) for row in rows], dtype=torch.long)
        encoded["transcripts"] = [str(row.get("transcript", "")) for row in rows]
        encoded["emotions"] = [str(row.get("emotion", "")) for row in rows]
        return encoded
