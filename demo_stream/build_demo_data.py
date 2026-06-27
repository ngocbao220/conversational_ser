from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

METADATA_PATH = ROOT / "iemocap_export" / "metadata.csv"
IEMOCAP_ROOT = ROOT / "iemocap"
PREDICTION_CANDIDATES = {
    "baseline": (
        ROOT / "results" / "wavlm_no_mal_no_tim" / "predictions.csv",
        ROOT / "results" / "wavlm_baseline_no_mal_no_tim" / "predictions.csv",
        ROOT / "outputs" / "hf_checkpoints" / "wavlm_baseline_no_mal_no_tim" / "predictions.csv",
    ),
    "mal": (
        ROOT / "results" / "wavlm_mal_no_tim" / "predictions.csv",
        ROOT / "outputs" / "hf_checkpoints" / "wavlm_mal_no_tim" / "predictions.csv",
    ),
    "tim": (
        ROOT / "results" / "wavlm_tim" / "predictions.csv",
        ROOT / "outputs" / "hf_checkpoints" / "wavlm_tim" / "predictions.csv",
    ),
}
LABELS = ("angry", "happy", "neutral", "sad")
TARGET_SESSIONS = {"Ses05"}
METADATA_LABEL_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "excited": "happy",
    "surprise": "happy",
    "sad": "sad",
    "fear": "sad",
    "angry": "angry",
    "frustrated": "angry",
    "disgust": "angry",
    "disgusted": "angry",
}
TRANSCRIPT_LINE = re.compile(
    r"^(?P<utterance_id>\S+)\s+\[(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\]:\s*(?P<text>.*)$"
)
EVAL_LINE = re.compile(
    r"^\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\]\s+"
    r"(?P<utterance_id>\S+)\s+(?P<label>\S+)"
)
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


def read_csv_by_utterance(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["utterance_id"]: row for row in csv.DictReader(handle)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def utterance_turn_index(utterance_id: str) -> int | None:
    last = utterance_id.split("_")[-1]
    digits = re.sub(r"\D", "", last)
    return int(digits) if digits else None


def utterance_speaker_role(utterance_id: str) -> str:
    last = utterance_id.split("_")[-1]
    return last[0] if last and last[0] in {"F", "M"} else ""


def read_evaluation_labels(session_ids: set[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for session_id in session_ids:
        session_num = session_id.replace("Ses", "").lstrip("0")
        eval_dir = IEMOCAP_ROOT / f"Session{session_num}" / "dialog" / "EmoEvaluation"
        if not eval_dir.exists():
            continue
        for path in eval_dir.glob("*.txt"):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = EVAL_LINE.match(line.strip())
                    if match:
                        labels[match.group("utterance_id")] = match.group("label")
    return labels


def build_metadata_rows_from_iemocap(session_ids: set[str]) -> list[dict[str, str]]:
    transcript_rows = read_transcription_times(session_ids)
    raw_labels = read_evaluation_labels(session_ids)
    rows = []
    for utterance_id, transcript in sorted(transcript_rows.items()):
        dialogue_id = "_".join(utterance_id.split("_")[:-1])
        session_id = utterance_id[:5]
        if session_id not in session_ids:
            continue
        speaker_role = utterance_speaker_role(utterance_id)
        turn_index = utterance_turn_index(utterance_id)
        audio_path = resolve_iemocap_audio_path(utterance_id)
        duration = float(transcript["end_time"]) - float(transcript["start_time"])
        rows.append(
            {
                "utterance_id": utterance_id,
                "dialogue_id": dialogue_id,
                "session_id": session_id,
                "speaker_id": f"{session_id}_{speaker_role}" if speaker_role else "",
                "speaker_role": speaker_role,
                "turn_index": str(turn_index) if turn_index is not None else "",
                "duration": f"{duration:.4f}",
                "transcript": str(transcript.get("transcript", "")),
                "audio_path": audio_path,
                "original_label": raw_labels.get(utterance_id, ""),
            }
        )
    return rows


def resolve_prediction_files() -> dict[str, Path]:
    resolved = {}
    for model_name, candidates in PREDICTION_CANDIDATES.items():
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is not None:
            resolved[model_name] = path
        else:
            print(f"No predictions found for {model_name}; checked: {', '.join(str(path.relative_to(ROOT)) for path in candidates)}")
    return resolved


def read_transcription_times(session_ids: set[str]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for session_id in session_ids:
        session_num = session_id.replace("Ses", "").lstrip("0")
        transcript_dir = IEMOCAP_ROOT / f"Session{session_num}" / "dialog" / "transcriptions"
        if not transcript_dir.exists():
            continue

        for path in transcript_dir.glob("*.txt"):
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = TRANSCRIPT_LINE.match(line.strip())
                    if not match:
                        continue
                    rows[match.group("utterance_id")] = {
                        "start_time": float(match.group("start")),
                        "end_time": float(match.group("end")),
                        "transcript": match.group("text").strip(),
                    }
    return rows


def as_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def canonical_metadata_label(row: dict[str, str]) -> str:
    raw_label = str(row.get("original_label", "")).strip().lower()
    return RAW_LABEL_MAP.get(raw_label) or METADATA_LABEL_MAP.get(raw_label, "")


def resolve_iemocap_audio_path(utterance_id: str, metadata_path: str = "") -> str:
    dialogue_id = "_".join(str(utterance_id).split("_")[:-1])
    session_match = re.match(r"Ses(?P<session>\d{2})", str(utterance_id))
    if dialogue_id and session_match:
        session_num = str(int(session_match.group("session")))
        direct_path = (
            IEMOCAP_ROOT
            / f"Session{session_num}"
            / "sentences"
            / "wav"
            / dialogue_id
            / f"{utterance_id}.wav"
        )
        if direct_path.exists():
            return str(direct_path.relative_to(ROOT))

    if metadata_path:
        candidate = ROOT / metadata_path
        if candidate.exists():
            return metadata_path
    return metadata_path


def prediction_payload(row: dict[str, str] | None) -> dict[str, object] | None:
    if not row:
        return None

    probs = {label: as_float(row.get(f"prob_{label}")) for label in LABELS}
    confidence = max(probs.values()) if probs else 0.0
    return {
        "label": row.get("pred_label", ""),
        "confidence": round(confidence, 4),
        "probabilities": probs,
    }


def compare_predictions(model_predictions: dict[str, dict[str, object]], gold_label: str) -> dict[str, object]:
    baseline = model_predictions.get("baseline")
    mal = model_predictions.get("mal")
    tim = model_predictions.get("tim")

    baseline_correct = bool(baseline and baseline.get("label") == gold_label)
    mal_correct = bool(mal and mal.get("label") == gold_label)
    tim_correct = bool(tim and tim.get("label") == gold_label)
    has_all_predictions = bool(baseline and mal and tim)
    has_mal_tim_predictions = bool(mal and tim)

    if not has_all_predictions:
        outcome = "missing_prediction"
    elif tim_correct and not baseline_correct and not mal_correct:
        outcome = "tim_correct_baseline_mal_wrong"
    elif tim_correct and not baseline_correct:
        outcome = "tim_correct_baseline_wrong"
    elif tim_correct and not mal_correct:
        outcome = "tim_correct_mal_wrong"
    elif tim_correct:
        outcome = "all_or_tim_correct"
    elif mal_correct and not tim_correct:
        outcome = "mal_correct_tim_wrong"
    elif baseline_correct and not tim_correct:
        outcome = "baseline_correct_tim_wrong"
    else:
        outcome = "all_wrong"

    if not has_mal_tim_predictions:
        mal_tim_outcome = "missing_prediction"
    elif tim_correct and not mal_correct:
        mal_tim_outcome = "tim_only_correct"
    elif mal_correct and not tim_correct:
        mal_tim_outcome = "mal_only_correct"
    elif tim_correct:
        mal_tim_outcome = "both_correct"
    else:
        mal_tim_outcome = "both_wrong"

    return {
        "has_all_predictions": has_all_predictions,
        "has_mal_tim_predictions": has_mal_tim_predictions,
        "baseline_correct": baseline_correct,
        "mal_correct": mal_correct,
        "tim_correct": tim_correct,
        "outcome": outcome,
        "mal_tim_outcome": mal_tim_outcome,
    }


def main() -> None:
    if METADATA_PATH.exists():
        metadata_rows = [
            row for row in read_csv_rows(METADATA_PATH)
            if row.get("session_id") in TARGET_SESSIONS
        ]
        metadata_source = str(METADATA_PATH.relative_to(ROOT))
    else:
        metadata_rows = build_metadata_rows_from_iemocap(TARGET_SESSIONS)
        metadata_source = "iemocap"
    metadata = {row["utterance_id"]: row for row in metadata_rows}
    transcription_times = read_transcription_times(TARGET_SESSIONS)
    prediction_files = resolve_prediction_files()
    predictions = {
        model_name: read_csv_by_utterance(path)
        for model_name, path in prediction_files.items()
    }

    items: list[dict[str, object]] = []
    session_counts: Counter[str] = Counter()
    dialogue_counts: Counter[str] = Counter()
    dialogue_prediction_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    mal_tim_counts: Counter[str] = Counter()

    for meta in metadata_rows:
        utterance_id = meta["utterance_id"]
        pred_rows = {name: rows.get(utterance_id) for name, rows in predictions.items()}
        first_pred = next((row for row in pred_rows.values() if row), {})
        transcript_row = transcription_times.get(utterance_id, {})

        dialogue_id = meta.get("dialogue_id") or first_pred.get("dialogue_id", "")
        session_id = meta.get("session_id") or dialogue_id[:5]
        speaker_id = meta.get("speaker_id") or first_pred.get("speaker_id", "")
        gold_label = first_pred.get("gold_label") or canonical_metadata_label(meta)
        turn_index = meta.get("turn_index", "")

        model_predictions = {}
        for model_name, row in pred_rows.items():
            payload = prediction_payload(row)
            if payload:
                model_predictions[model_name] = payload

        tim_label = model_predictions.get("tim", {}).get("label", "")
        if tim_label:
            label_counts[str(tim_label)] += 1
            dialogue_prediction_counts[dialogue_id] += 1

        comparison = compare_predictions(model_predictions, gold_label)
        evidence_counts[str(comparison["outcome"])] += 1
        mal_tim_counts[str(comparison["mal_tim_outcome"])] += 1

        item = {
            "utterance_id": utterance_id,
            "dialogue_id": dialogue_id,
            "session_id": session_id,
            "speaker_id": speaker_id,
            "speaker_role": meta.get("speaker_role", speaker_id[-1:] if speaker_id else ""),
            "turn_index": int(turn_index) if str(turn_index).isdigit() else None,
            "start_time": round(float(transcript_row["start_time"]), 4)
            if "start_time" in transcript_row else round(as_float(first_pred.get("start_time")), 4)
            if first_pred.get("start_time") else None,
            "end_time": round(float(transcript_row["end_time"]), 4)
            if "end_time" in transcript_row else round(as_float(first_pred.get("end_time")), 4)
            if first_pred.get("end_time") else None,
            "duration": round(as_float(meta.get("duration") or first_pred.get("duration")), 3),
            "transcript": (str(transcript_row.get("transcript") or meta.get("transcript") or "")).strip(),
            "audio_path": resolve_iemocap_audio_path(utterance_id, meta.get("audio_path", "")),
            "gold_label": gold_label,
            "predictions": model_predictions,
            "comparison": comparison,
        }
        items.append(item)
        session_counts[session_id] += 1
        dialogue_counts[dialogue_id] += 1

    items.sort(
        key=lambda row: (
            str(row["session_id"]),
            str(row["dialogue_id"]),
            row["start_time"] if row["start_time"] is not None else 0.0,
            row["turn_index"] if row["turn_index"] is not None else 10_000,
        )
    )

    sessions = [
        {"id": session_id, "count": count}
        for session_id, count in sorted(session_counts.items())
    ]
    dialogues_by_session: dict[str, list[dict[str, object]]] = defaultdict(list)
    for dialogue_id, count in sorted(dialogue_counts.items()):
        dialogues_by_session[dialogue_id[:5]].append({
            "id": dialogue_id,
            "count": count,
            "predicted_count": dialogue_prediction_counts[dialogue_id],
        })

    payload = {
        "generated_from": {
            "metadata": metadata_source,
            "predictions": {
                model: str(path.relative_to(ROOT)) for model, path in prediction_files.items()
            },
        },
        "models": [
            {"id": "tim", "name": "WavLM + TIM"},
            {"id": "mal", "name": "WavLM + MAL"},
            {"id": "baseline", "name": "WavLM baseline"},
        ],
        "labels": list(LABELS),
        "summary": {
            "utterance_count": len(items),
            "predicted_utterance_count": sum(1 for item in items if item["predictions"].get("tim")),
            "fully_compared_count": sum(1 for item in items if item["comparison"]["has_all_predictions"]),
            "dialogue_count": len(dialogue_counts),
            "session_count": len(session_counts),
            "tim_label_counts": dict(sorted(label_counts.items())),
            "evidence_counts": dict(sorted(evidence_counts.items())),
            "mal_tim_paired_counts": dict(sorted(mal_tim_counts.items())),
        },
        "sessions": sessions,
        "dialogues_by_session": dialogues_by_session,
        "utterances": items,
    }

    out_path = OUT_DIR / "demo_data.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)} with {len(items)} utterances")


if __name__ == "__main__":
    main()
