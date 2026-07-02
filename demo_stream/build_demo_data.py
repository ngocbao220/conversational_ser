from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

METADATA_PATH = ROOT / "iemocap_export" / "metadata.csv"
IEMOCAP_ROOT = ROOT / "data" / "iemocap"

PREDICTION_CANDIDATES = {
    "baseline": (
        ROOT / "results" / "versioned_loso" / "baseline_wavlm" / "cross_session" / "run_20260630_091825" / "test_Ses05" / "predictions.csv"
    ),
    "cdm": (
        ROOT / "results" / "versioned_loso" / "cdm_wavlm" / "cross_session" / "run_20260630_092932" / "test_Ses05" / "predictions.csv"
    ),
    "cim_v1": (
        ROOT / "results" / "versioned_loso" / "v1_cim_concat" / "cross_session" / "run_20260629_165854" / "test_Ses05" / "predictions.csv"
    ),
    "cim": ROOT / "results" / "cim_full_loso" / "branch_concat_interaction4" / "cross_session" / "run_20260701_045803" / "test_Ses05" / "predictions.csv",
}
MODEL_DISPLAY_NAMES = {
    "baseline": "WavLM baseline",
    "cdm": "WavLM + CDM",
    "cim_v1": "WavLM + CIM v1",
    "cim": "WavLM + CIM",
}
MODEL_ORDER = ("cim", "cim_v1", "cdm", "baseline")
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
RAW_LABEL_FULL_NAMES = {
    "ang": "angry",
    "fru": "frustrated",
    "dis": "disgusted",
    "hap": "happy",
    "exc": "excited",
    "sur": "surprised",
    "neu": "neutral",
    "sad": "sad",
    "fea": "fearful",
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
        if isinstance(candidates, (str, Path)):
            candidates = (Path(candidates),)
        else:
            candidates = tuple(Path(candidate) for candidate in candidates)
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


def raw_emotion_label(row: dict[str, str]) -> str:
    raw_label = str(row.get("original_label", "")).strip().lower()
    return RAW_LABEL_MAP.get(raw_label, raw_label)


def raw_emotion_full_name(row: dict[str, str]) -> str:
    raw_label = str(row.get("original_label", "")).strip().lower()
    return RAW_LABEL_FULL_NAMES.get(raw_label, raw_label)


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
    cdm = model_predictions.get("cdm")
    cim = model_predictions.get("cim")

    baseline_correct = bool(baseline and baseline.get("label") == gold_label)
    cdm_correct = bool(cdm and cdm.get("label") == gold_label)
    cim_correct = bool(cim and cim.get("label") == gold_label)
    has_all_predictions = bool(baseline and cdm and cim)
    has_cdm_cim_predictions = bool(cdm and cim)

    if not has_all_predictions:
        outcome = "missing_prediction"
    elif cim_correct and not baseline_correct and not cdm_correct:
        outcome = "cim_correct_baseline_cdm_wrong"
    elif cim_correct and not baseline_correct:
        outcome = "cim_correct_baseline_wrong"
    elif cim_correct and not cdm_correct:
        outcome = "cim_correct_cdm_wrong"
    elif cim_correct:
        outcome = "all_or_cim_correct"
    elif cdm_correct and not cim_correct:
        outcome = "cdm_correct_cim_wrong"
    elif baseline_correct and not cim_correct:
        outcome = "baseline_correct_cim_wrong"
    else:
        outcome = "all_wrong"

    if not has_cdm_cim_predictions:
        cdm_cim_outcome = "missing_prediction"
    elif cim_correct and not cdm_correct:
        cdm_cim_outcome = "cim_only_correct"
    elif cdm_correct and not cim_correct:
        cdm_cim_outcome = "cdm_only_correct"
    elif cim_correct:
        cdm_cim_outcome = "both_correct"
    else:
        cdm_cim_outcome = "both_wrong"

    return {
        "has_all_predictions": has_all_predictions,
        "has_cdm_cim_predictions": has_cdm_cim_predictions,
        "baseline_correct": baseline_correct,
        "cdm_correct": cdm_correct,
        "cim_correct": cim_correct,
        "outcome": outcome,
        "cdm_cim_outcome": cdm_cim_outcome,
    }


def add_interaction_features(items: list[dict[str, object]]) -> None:
    by_dialogue: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        by_dialogue[str(item["dialogue_id"])].append(item)

    for dialogue_items in by_dialogue.values():
        dialogue_items.sort(
            key=lambda row: (
                row["start_time"] if row["start_time"] is not None else 0.0,
                row["turn_index"] if row["turn_index"] is not None else 10_000,
            )
        )
        speaker_history: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "turn_count": 0.0,
                "overlap_count": 0.0,
                "gap_sum": 0.0,
                "duration_sum": 0.0,
            }
        )
        previous: dict[str, object] | None = None
        for index, item in enumerate(dialogue_items):
            speaker_id = str(item.get("speaker_id") or "")
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            duration = max(0.0, float(end_time) - float(start_time)) if start_time is not None and end_time is not None else float(item.get("duration") or 0.0)

            if previous and start_time is not None and previous.get("end_time") is not None:
                previous_end = float(previous["end_time"])
                gap_prev = float(start_time) - previous_end
                overlap_prev = max(0.0, previous_end - float(start_time))
                previous_speaker = str(previous.get("speaker_id") or "")
                speaker_switch = speaker_id != previous_speaker
            else:
                gap_prev = 0.0
                overlap_prev = 0.0
                speaker_switch = False

            history = speaker_history[speaker_id]
            previous_turns = history["turn_count"]
            speaker_prev_mean_gap = history["gap_sum"] / max(previous_turns, 1.0)
            speaker_prev_mean_duration = history["duration_sum"] / max(previous_turns, 1.0)
            speaker_prev_overlap_rate = history["overlap_count"] / max(previous_turns, 1.0)
            relative_gap = gap_prev - speaker_prev_mean_gap if previous_turns > 0 else 0.0
            overlap_ratio = overlap_prev / max(duration, 1e-6)

            item["interaction_features"] = {
                "duration": round(duration, 3),
                "gap_prev": round(gap_prev, 3),
                "relative_gap": round(relative_gap, 3),
                "overlap_prev": round(overlap_prev, 3),
                "overlap_ratio": round(overlap_ratio, 3),
                "speaker_switch": bool(previous and speaker_switch),
                "is_interrupting_prev": bool(previous and speaker_switch and overlap_prev > 0.0),
                "speaker_prev_overlap_rate": round(speaker_prev_overlap_rate, 3),
                "speaker_prev_mean_gap": round(speaker_prev_mean_gap, 3),
                "speaker_prev_mean_duration": round(speaker_prev_mean_duration, 3),
                "turn_position": round(index / max(len(dialogue_items) - 1, 1), 3),
            }

            history["turn_count"] += 1.0
            history["overlap_count"] += 1.0 if overlap_prev > 0.05 else 0.0
            history["gap_sum"] += gap_prev
            history["duration_sum"] += duration
            previous = item


def model_display_name(model_id: str) -> str:
    if model_id in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_id]
    return model_id.replace("_", " ").replace("-", " ").title()


def resolved_model_list(prediction_files: dict[str, Path]) -> list[dict[str, str]]:
    ordered_ids = [model_id for model_id in MODEL_ORDER if model_id in prediction_files]
    ordered_ids.extend(model_id for model_id in prediction_files if model_id not in ordered_ids)
    return [{"id": model_id, "name": model_display_name(model_id)} for model_id in ordered_ids]


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
    cdm_cim_counts: Counter[str] = Counter()

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

        cim_label = model_predictions.get("cim", {}).get("label", "")
        if cim_label:
            label_counts[str(cim_label)] += 1
            dialogue_prediction_counts[dialogue_id] += 1

        comparison = compare_predictions(model_predictions, gold_label)
        evidence_counts[str(comparison["outcome"])] += 1
        cdm_cim_counts[str(comparison["cdm_cim_outcome"])] += 1

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
            "raw_emotion": raw_emotion_label(meta),
            "raw_emotion_full": raw_emotion_full_name(meta),
            "raw_label": meta.get("original_label", ""),
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
    add_interaction_features(items)

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
        "models": resolved_model_list(prediction_files),
        "labels": list(LABELS),
        "summary": {
            "utterance_count": len(items),
            "predicted_utterance_count": sum(1 for item in items if item["predictions"].get("cim")),
            "fully_compared_count": sum(1 for item in items if item["comparison"]["has_all_predictions"]),
            "dialogue_count": len(dialogue_counts),
            "session_count": len(session_counts),
            "cim_label_counts": dict(sorted(label_counts.items())),
            "evidence_counts": dict(sorted(evidence_counts.items())),
            "cdm_cim_paired_counts": dict(sorted(cdm_cim_counts.items())),
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
