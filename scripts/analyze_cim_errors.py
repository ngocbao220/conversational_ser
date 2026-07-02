from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


PAIR_CATEGORIES = (
    "cim_only_correct",
    "cdm_only_correct",
    "both_correct",
    "both_wrong",
)


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"utterance_id", "dialogue_id", "gold_label", "pred_label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    by_id = {row["utterance_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{path} contains duplicate utterance_id values.")
    return by_id


def as_float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def paired_category(cdm_row: Mapping[str, str], cim_row: Mapping[str, str]) -> str:
    gold = cim_row["gold_label"]
    cdm_correct = cdm_row["pred_label"] == gold
    cim_correct = cim_row["pred_label"] == gold
    if cim_correct and not cdm_correct:
        return "cim_only_correct"
    if cdm_correct and not cim_correct:
        return "cdm_only_correct"
    if cim_correct:
        return "both_correct"
    return "both_wrong"


def summarize_groups(groups: Mapping[str, Iterable[str]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for group_name, categories in sorted(groups.items()):
        counts = Counter(categories)
        total = sum(counts.values())
        cdm_correct = counts["cdm_only_correct"] + counts["both_correct"]
        cim_correct = counts["cim_only_correct"] + counts["both_correct"]
        summary[group_name] = {
            "n": total,
            "paired_counts": {category: counts[category] for category in PAIR_CATEGORIES},
            "cdm_accuracy": cdm_correct / total if total else 0.0,
            "cim_accuracy": cim_correct / total if total else 0.0,
            "cim_minus_cdm_accuracy": (cim_correct - cdm_correct) / total if total else 0.0,
        }
    return summary


def gap_bucket(row: Mapping[str, str], turn_index: int, short_gap: float, long_gap: float) -> str:
    if turn_index == 0:
        return "first_turn"
    gap = as_float(row, "gap_prev")
    if gap < 0.0:
        return "overlap"
    if gap < short_gap:
        return "short_gap"
    if gap < long_gap:
        return "medium_gap"
    return "long_gap"


def turn_bucket(turn_index: int, dialogue_size: int) -> str:
    if dialogue_size <= 1:
        return "single_turn"
    position = turn_index / (dialogue_size - 1)
    if position <= 1.0 / 3.0:
        return "start"
    if position <= 2.0 / 3.0:
        return "middle"
    return "end"


def analyze(cdm_path: Path, cim_path: Path, output_dir: Path, short_gap: float, long_gap: float) -> dict[str, Any]:
    cdm_rows = load_predictions(cdm_path)
    cim_rows = load_predictions(cim_path)
    common_ids = set(cdm_rows) & set(cim_rows)
    if not common_ids:
        raise ValueError("CDM and CIM predictions have no utterance IDs in common.")
    if set(cdm_rows) != set(cim_rows):
        raise ValueError("CDM and CIM prediction files must contain the same utterance IDs.")
    if any(cdm_rows[row_id]["gold_label"] != cim_rows[row_id]["gold_label"] for row_id in common_ids):
        raise ValueError("CDM and CIM prediction files disagree on gold labels.")

    ordered_by_dialogue: dict[str, list[str]] = defaultdict(list)
    for utterance_id, row in cim_rows.items():
        ordered_by_dialogue[row["dialogue_id"]].append(utterance_id)
    for utterance_ids in ordered_by_dialogue.values():
        utterance_ids.sort(
            key=lambda utterance_id: (
                as_float(cim_rows[utterance_id], "start_time"),
                as_float(cim_rows[utterance_id], "end_time"),
                utterance_id,
            )
        )

    duration_threshold = median(as_float(row, "end_time") - as_float(row, "start_time") for row in cim_rows.values())
    groups: dict[str, dict[str, list[str]]] = {
        "overlap": defaultdict(list),
        "gap": defaultdict(list),
        "duration": defaultdict(list),
        "speaker_transition": defaultdict(list),
        "turn_position": defaultdict(list),
        "emotion_transition": defaultdict(list),
    }
    paired_rows: list[dict[str, Any]] = []

    for dialogue_id, utterance_ids in sorted(ordered_by_dialogue.items()):
        previous_gold: str | None = None
        for turn_index, utterance_id in enumerate(utterance_ids):
            cim_row = cim_rows[utterance_id]
            cdm_row = cdm_rows[utterance_id]
            category = paired_category(cdm_row, cim_row)
            duration = max(0.0, as_float(cim_row, "end_time") - as_float(cim_row, "start_time"))
            overlap = "overlap" if as_float(cim_row, "is_overlap") >= 0.5 else "no_overlap"
            speaker_transition = (
                "first_turn" if turn_index == 0 else "speaker_switch" if as_float(cim_row, "speaker_switch") >= 0.5 else "same_speaker"
            )
            transition = "first_turn" if previous_gold is None else f"{previous_gold}->{cim_row['gold_label']}"

            groups["overlap"][overlap].append(category)
            groups["gap"][gap_bucket(cim_row, turn_index, short_gap, long_gap)].append(category)
            groups["duration"]["short_or_equal_median" if duration <= duration_threshold else "longer_than_median"].append(category)
            groups["speaker_transition"][speaker_transition].append(category)
            groups["turn_position"][turn_bucket(turn_index, len(utterance_ids))].append(category)
            groups["emotion_transition"][transition].append(category)

            paired_rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "turn_index": turn_index,
                    "gold_label": cim_row["gold_label"],
                    "cdm_pred_label": cdm_row["pred_label"],
                    "cim_pred_label": cim_row["pred_label"],
                    "paired_category": category,
                    "duration": duration,
                    "gap_prev": as_float(cim_row, "gap_prev"),
                    "is_overlap": overlap,
                    "speaker_transition": speaker_transition,
                    "turn_position": turn_bucket(turn_index, len(utterance_ids)),
                    "emotion_transition": transition,
                }
            )
            previous_gold = cim_row["gold_label"]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    overall = Counter(row["paired_category"] for row in paired_rows)
    payload = {
        "inputs": {"cdm_predictions": str(cdm_path), "cim_predictions": str(cim_path)},
        "n": len(paired_rows),
        "thresholds": {"short_gap_seconds": short_gap, "long_gap_seconds": long_gap, "duration_median_seconds": duration_threshold},
        "overall_paired_counts": {category: overall[category] for category in PAIR_CATEGORIES},
        "groups": {name: summarize_groups(group) for name, group in groups.items()},
    }
    (output_dir / "paired_error_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired CDM versus CIM errors by temporal dialogue conditions.")
    parser.add_argument("--cdm-predictions", type=Path, default=Path("results/wavlm_cdm_no_cim/predictions.csv"))
    parser.add_argument("--cim-predictions", type=Path, default=Path("results/wavlm_cim/predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/paired_error_analysis"))
    parser.add_argument("--short-gap", type=float, default=0.3)
    parser.add_argument("--long-gap", type=float, default=1.0)
    args = parser.parse_args()
    payload = analyze(args.cdm_predictions, args.cim_predictions, args.output_dir, args.short_gap, args.long_gap)
    print(f"paired_error_analysis={args.output_dir / 'paired_error_analysis.json'} n={payload['n']}")


if __name__ == "__main__":
    main()
