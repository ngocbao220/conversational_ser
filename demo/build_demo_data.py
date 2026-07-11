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

DEFAULT_BACKBONE = "wavlm"
FINAL_RESULT_ROOT = ROOT / "results" / "final_result"
BACKBONE_DISPLAY_NAMES = {
    "wavlm": "WavLM",
    "wav2vec": "wav2vec 2.0",
    "hubert": "HuBERT",
}
MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "cdm": "CDM",
    "cim": "CDIM",
}
MODEL_ORDER = ("cim", "cdm", "baseline")
LABELS = ("angry", "happy", "neutral", "sad")
TARGET_SESSIONS = {f"Ses{session:02d}" for session in range(1, 6)}
METADATA_LABEL_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
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
    "hap": "happy",
    "exc": "happy",
    "neu": "neutral",
    "sad": "sad",
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
    "xxx": "unknown",
    "oth": "other",
}


def read_csv_by_utterance(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["utterance_id"]: row for row in csv.DictReader(handle)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_cross_session_prediction_paths(backbone: str, model_name: str) -> list[Path]:
    run_dirs = sorted((FINAL_RESULT_ROOT / backbone / model_name / "cross_session").glob("run_*"))
    if not run_dirs:
        return []
    run_dir = run_dirs[-1]
    return [
        run_dir / f"test_Ses{session:02d}" / "predictions.csv"
        for session in range(1, 6)
    ]


def prediction_candidates_by_backbone() -> dict[str, dict[str, list[Path]]]:
    return {
        backbone: {
            model_name: latest_cross_session_prediction_paths(backbone, model_name)
            for model_name in MODEL_ORDER
        }
        for backbone in BACKBONE_DISPLAY_NAMES
    }


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


def resolve_prediction_files(candidates_by_model: dict[str, object], prefix: str = "") -> dict[str, list[Path]]:
    resolved = {}
    for model_name, candidates in candidates_by_model.items():
        if isinstance(candidates, (str, Path)):
            candidates = (Path(candidates),)
        else:
            candidates = tuple(Path(candidate) for candidate in candidates)
        paths = [candidate for candidate in candidates if candidate.exists()]
        if paths:
            resolved[model_name] = paths
        else:
            label = f"{prefix}/{model_name}" if prefix else model_name
            print(f"No predictions found for {label}; checked: {', '.join(str(path.relative_to(ROOT)) for path in candidates)}")
    return resolved


def read_prediction_files_by_utterance(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        rows.update(read_csv_by_utterance(path))
    return rows


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
    return raw_label


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
        cdm_cim_outcome = "cim_correct_cdm_wrong"
    elif cdm_correct and not cim_correct:
        cdm_cim_outcome = "cdm_correct_cim_wrong"
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


def ignored_label_comparison() -> dict[str, object]:
    return {
        "has_all_predictions": False,
        "has_cdm_cim_predictions": False,
        "baseline_correct": False,
        "cdm_correct": False,
        "cim_correct": False,
        "outcome": "ignored_label",
        "cdm_cim_outcome": "ignored_label",
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


def compare_predictions_by_backbone(
    predictions_by_backbone: dict[str, dict[str, dict[str, object]]],
    gold_label: str,
    is_target_label: bool,
) -> dict[str, dict[str, object]]:
    return {
        backbone: compare_predictions(model_predictions, gold_label) if is_target_label else ignored_label_comparison()
        for backbone, model_predictions in predictions_by_backbone.items()
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
    prediction_files_by_backbone = {
        backbone: resolve_prediction_files(candidates, prefix=backbone)
        for backbone, candidates in prediction_candidates_by_backbone().items()
    }
    predictions_by_backbone = {
        backbone: {
            model_name: read_prediction_files_by_utterance(paths)
            for model_name, paths in prediction_files.items()
        }
        for backbone, prediction_files in prediction_files_by_backbone.items()
    }
    prediction_files = prediction_files_by_backbone.get(DEFAULT_BACKBONE, {})

    items: list[dict[str, object]] = []
    session_counts: Counter[str] = Counter()
    dialogue_counts: Counter[str] = Counter()
    dialogue_prediction_counts_by_backbone: dict[str, Counter[str]] = defaultdict(Counter)
    label_counts_by_backbone: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_counts_by_backbone: dict[str, Counter[str]] = defaultdict(Counter)
    cdm_cim_counts_by_backbone: dict[str, Counter[str]] = defaultdict(Counter)

    for meta in metadata_rows:
        utterance_id = meta["utterance_id"]
        pred_rows_by_backbone = {
            backbone: {name: rows.get(utterance_id) for name, rows in predictions.items()}
            for backbone, predictions in predictions_by_backbone.items()
        }
        first_pred = next(
            (
                row
                for pred_rows in pred_rows_by_backbone.values()
                for row in pred_rows.values()
                if row
            ),
            {},
        )
        transcript_row = transcription_times.get(utterance_id, {})

        dialogue_id = meta.get("dialogue_id") or first_pred.get("dialogue_id", "")
        session_id = meta.get("session_id") or dialogue_id[:5]
        speaker_id = meta.get("speaker_id") or first_pred.get("speaker_id", "")
        mapped_label = canonical_metadata_label(meta)
        gold_label = mapped_label if mapped_label in LABELS else ""
        is_target_label = gold_label in LABELS
        is_ignored_label = not is_target_label
        turn_index = meta.get("turn_index", "")

        model_predictions_by_backbone = {}
        for backbone, pred_rows in pred_rows_by_backbone.items():
            model_predictions: dict[str, dict[str, object]] = {}
            for model_name, row in pred_rows.items():
                payload = prediction_payload(row)
                if payload:
                    model_predictions[model_name] = payload
            model_predictions_by_backbone[backbone] = model_predictions

            primary_prediction = model_predictions.get("cim") or next(iter(model_predictions.values()), {})
            primary_label = primary_prediction.get("label", "") if primary_prediction else ""
            if primary_label:
                label_counts_by_backbone[backbone][str(primary_label)] += 1
            if model_predictions:
                dialogue_prediction_counts_by_backbone[backbone][dialogue_id] += 1

        comparison_by_backbone = compare_predictions_by_backbone(
            model_predictions_by_backbone,
            gold_label,
            is_target_label,
        )
        for backbone, comparison in comparison_by_backbone.items():
            evidence_counts_by_backbone[backbone][str(comparison["outcome"])] += 1
            if is_target_label:
                cdm_cim_counts_by_backbone[backbone][str(comparison["cdm_cim_outcome"])] += 1

        model_predictions = model_predictions_by_backbone.get(DEFAULT_BACKBONE, {})
        comparison = comparison_by_backbone.get(DEFAULT_BACKBONE, ignored_label_comparison())

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
            "mapped_label": mapped_label,
            "is_target_label": is_target_label,
            "is_ignored_label": is_ignored_label,
            "gold_label": gold_label,
            "predictions": model_predictions,
            "predictions_by_backbone": model_predictions_by_backbone,
            "comparison": comparison,
            "comparison_by_backbone": comparison_by_backbone,
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
    default_dialogue_prediction_counts = dialogue_prediction_counts_by_backbone.get(DEFAULT_BACKBONE, Counter())
    for dialogue_id, count in sorted(dialogue_counts.items()):
        dialogues_by_session[dialogue_id[:5]].append({
            "id": dialogue_id,
            "count": count,
            "predicted_count": default_dialogue_prediction_counts[dialogue_id],
            "predicted_count_by_backbone": {
                backbone: counts[dialogue_id]
                for backbone, counts in dialogue_prediction_counts_by_backbone.items()
            },
        })

    backbones = [
        {"id": backbone, "name": BACKBONE_DISPLAY_NAMES.get(backbone, backbone)}
        for backbone in prediction_files_by_backbone
        if prediction_files_by_backbone[backbone]
    ]
    models_by_backbone = {
        backbone: resolved_model_list(files)
        for backbone, files in prediction_files_by_backbone.items()
        if files
    }
    summaries_by_backbone = {
        backbone: {
            "predicted_utterance_count": sum(1 for item in items if item["predictions_by_backbone"].get(backbone)),
            "fully_compared_count": sum(1 for item in items if item["comparison_by_backbone"].get(backbone, {}).get("has_all_predictions")),
            "cim_label_counts": dict(sorted(label_counts_by_backbone[backbone].items())),
            "evidence_counts": dict(sorted(evidence_counts_by_backbone[backbone].items())),
            "cdm_cim_paired_counts": dict(sorted(cdm_cim_counts_by_backbone[backbone].items())),
        }
        for backbone in prediction_files_by_backbone
        if prediction_files_by_backbone[backbone]
    }
    default_summary = summaries_by_backbone.get(DEFAULT_BACKBONE, {})

    payload = {
        "generated_from": {
            "metadata": metadata_source,
            "predictions": {
                model: [str(path.relative_to(ROOT)) for path in paths]
                for model, paths in prediction_files.items()
            },
            "predictions_by_backbone": {
                backbone: {
                    model: [str(path.relative_to(ROOT)) for path in paths]
                    for model, paths in files.items()
                }
                for backbone, files in prediction_files_by_backbone.items()
            },
        },
        "backbones": backbones,
        "default_backbone": DEFAULT_BACKBONE,
        "models": resolved_model_list(prediction_files),
        "models_by_backbone": models_by_backbone,
        "labels": list(LABELS),
        "summary": {
            "utterance_count": len(items),
            "target_label_count": sum(1 for item in items if item["is_target_label"]),
            "ignored_label_count": sum(1 for item in items if item["is_ignored_label"]),
            "predicted_utterance_count": default_summary.get("predicted_utterance_count", 0),
            "fully_compared_count": default_summary.get("fully_compared_count", 0),
            "dialogue_count": len(dialogue_counts),
            "session_count": len(session_counts),
            "cim_label_counts": default_summary.get("cim_label_counts", {}),
            "evidence_counts": default_summary.get("evidence_counts", {}),
            "cdm_cim_paired_counts": default_summary.get("cdm_cim_paired_counts", {}),
            "by_backbone": summaries_by_backbone,
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
