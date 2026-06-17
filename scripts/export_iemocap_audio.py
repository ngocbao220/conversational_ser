from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset

from utils.dataset import get_canonical_label, get_transcript


ID_COLUMNS = (
    "utt_id",
    "utterance_id",
    "utterance_name",
    "id",
    "name",
    "file",
    "filename",
    "audio_file",
    "audio_path",
)


def parse_iemocap_id(utt_id: str) -> Dict[str, Any]:
    """
    Example:
    Ses01F_impro01_F000
    Ses02M_script03_1_M012
    """
    m = re.search(r"(Ses\d{2})", utt_id)
    if not m:
        raise ValueError(f"Cannot parse session from: {utt_id}")
    session_id = m.group(1)

    parts = utt_id.split("_")
    last = parts[-1]
    speaker_role = last[0]

    if speaker_role not in ["F", "M"]:
        raise ValueError(f"Cannot parse speaker from: {utt_id}")

    turn_index = int(re.sub(r"\D", "", last))
    dialogue_id = "_".join(parts[:-1])
    speaker_id = f"{session_id}_{speaker_role}"

    return {
        "session_id": session_id,
        "dialogue_id": dialogue_id,
        "speaker_role": speaker_role,
        "speaker_id": speaker_id,
        "turn_index": turn_index,
    }


def safe_filename(value: str) -> str:
    value = Path(value).stem
    value = value.strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def get_audio_payload(example: Mapping[str, Any]) -> tuple[np.ndarray, int, Optional[str]]:
    audio = example["audio"]
    source_path: Optional[str] = None

    if isinstance(audio, Mapping):
        source_path = str(audio.get("path") or "") or None
        array = audio.get("array")
        sampling_rate = int(audio.get("sampling_rate") or 0)
    elif hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = samples.data
        sampling_rate = int(getattr(samples, "sample_rate", 0) or 0)
    else:
        array = audio
        sampling_rate = 0

    waveform = np.asarray(array, dtype=np.float32)
    if waveform.ndim > 1:
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=1)
    if sampling_rate <= 0:
        raise ValueError("Cannot determine audio sampling rate.")
    return waveform, sampling_rate, source_path


def raw_label(example: Mapping[str, Any]) -> str:
    for col in ("major_emotion", "emotion", "label"):
        if col in example and example[col] is not None:
            return str(example[col])
    return ""


def find_utterance_id(example: Mapping[str, Any], split: str, index: int, source_path: Optional[str]) -> str:
    for col in ID_COLUMNS:
        value = example.get(col)
        if value:
            return safe_filename(str(value))
    if source_path:
        return safe_filename(source_path)
    return f"{split}_{index:06d}"


def iter_splits(dataset: Dataset | DatasetDict) -> Iterable[tuple[str, Dataset]]:
    if isinstance(dataset, DatasetDict):
        return dataset.items()
    return [("train", dataset)]


def export_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    audio_root = output_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(args.dataset_name)
    rows: list[Dict[str, Any]] = []

    for split, dataset in iter_splits(raw):
        if args.splits and split not in args.splits:
            continue
        dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
        total = len(dataset)
        limit = min(args.limit, total) if args.limit else total

        for index in range(limit):
            example = dataset[index]
            waveform, sampling_rate, source_path = get_audio_payload(example)
            utt_id = find_utterance_id(example, split, index, source_path)

            try:
                parsed = parse_iemocap_id(utt_id)
            except ValueError as exc:
                parsed = {
                    "session_id": "",
                    "dialogue_id": "",
                    "speaker_role": "",
                    "speaker_id": "",
                    "turn_index": "",
                    "parse_error": str(exc),
                }

            session = parsed.get("session_id") or "unknown_session"
            dialogue = parsed.get("dialogue_id") or "unknown_dialogue"
            audio_dir = audio_root / split / session / dialogue
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{utt_id}.wav"
            sf.write(str(audio_path), waveform, sampling_rate)

            row = {
                "split": split,
                "index": index,
                "utterance_id": utt_id,
                "audio_file": audio_path.name,
                "audio_path": str(audio_path),
                "source_audio_path": source_path or "",
                "sampling_rate": sampling_rate,
                "duration": float(len(waveform) / sampling_rate),
                "transcript": get_transcript(example),
                "original_label": raw_label(example),
                "mapped_label": get_canonical_label(example) or "",
                **parsed,
            }
            rows.append(row)
            print(f"[{split}] {index + 1}/{limit} {utt_id} -> {audio_path}")

    write_metadata(output_dir, rows)
    print(f"Saved {len(rows)} audio files under {audio_root}")
    print(f"Saved metadata: {output_dir / 'metadata.csv'}")
    print(f"Saved metadata: {output_dir / 'metadata.jsonl'}")


def write_metadata(output_dir: Path, rows: list[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metadata.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "metadata.csv"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/export IEMOCAP audio and metadata.")
    parser.add_argument("--dataset-name", default="AbstractTTS/IEMOCAP")
    parser.add_argument("--output-dir", default="iemocap_export")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--splits", nargs="*", default=None, help="Optional split names to export, e.g. train test.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max examples per split for smoke tests.")
    args = parser.parse_args()
    export_dataset(args)


if __name__ == "__main__":
    main()
