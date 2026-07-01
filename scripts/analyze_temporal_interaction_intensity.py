from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "results" / "feature_engineering" / "features_all.csv"
VERSIONED_ROOT = ROOT / "results" / "versioned_loso"
OUT_DIR = ROOT / "reports" / "temporal_interaction_intensity"

LABELS = ["angry", "happy", "neutral", "sad"]

MODEL_DIRS = {
    "Baseline": "baseline_wavlm",
    "MAL": "mal_wavlm",
    "TIM v1 concat": "v1_tim_concat",
    "TIM v3.1 recommended": "v3_1_tim_recommended_v2",
    "TIM v3.2 compact": "v3_2_tim_compact_primitives",
    "Dual v2.1 end2end": "v2_1_dual_end2end",
    "Dual v2.2.1 3phase": "v2_2_1_dual_dialogue_temporal_fuse",
    "Dual v2.2.2 temporal-first": "v2_2_2_dual_temporal_dialogue_fuse",
}


def safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())


def safe_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    median = values.median()
    iqr = values.quantile(0.75) - values.quantile(0.25)
    if not np.isfinite(iqr) or iqr <= 1e-9:
        std = values.std(ddof=0)
        return (values - values.mean()) / std if std > 1e-9 else values * 0.0
    return (values - median) / iqr


def macro_f1_from_predictions(group: pd.DataFrame) -> float:
    gold = group["gold_label"].astype(str)
    pred = group["pred_label"].astype(str)
    scores = []
    for label in LABELS:
        tp = int(((gold == label) & (pred == label)).sum())
        fp = int(((gold != label) & (pred == label)).sum())
        fn = int(((gold == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def accuracy_from_predictions(group: pd.DataFrame) -> float:
    return float((group["gold_label"].astype(str) == group["pred_label"].astype(str)).mean())


def read_predictions(model_dir: str) -> pd.DataFrame:
    rows = []
    root = VERSIONED_ROOT / model_dir
    for path in sorted(root.glob("cross_session/*/test_Ses*/predictions.csv")):
        session_text = path.parent.name.replace("test_Ses", "")
        try:
            test_session = int(session_text)
        except ValueError:
            continue
        frame = pd.read_csv(path)
        if not {"dialogue_id", "utterance_id", "gold_label", "pred_label"}.issubset(frame.columns):
            continue
        frame = frame[["dialogue_id", "utterance_id", "gold_label", "pred_label"]].copy()
        frame["test_session"] = test_session
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_dialogue_temporal_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, dialogue_id), group in features.groupby(["dataset", "dialogue_id"], sort=False):
        if str(dataset) != "IEMOCAP":
            continue
        session_match = pd.Series([dialogue_id]).str.extract(r"Ses(\d\d)")[0].iloc[0]
        session = int(session_match) if pd.notna(session_match) else -1
        n = len(group)
        total_duration = float(group["end_time"].max() - group["start_time"].min()) if n else 0.0
        mean_abs_gap = safe_mean(group, "abs_gap")
        mean_positive_gap = float(np.maximum(pd.to_numeric(group.get("gap_prev", 0.0), errors="coerce").fillna(0.0), 0.0).mean())
        mean_negative_gap_abs = float(np.maximum(-pd.to_numeric(group.get("gap_prev", 0.0), errors="coerce").fillna(0.0), 0.0).mean())
        overlap_ratio = clip01(safe_mean(group, "overlap_ratio"))
        overlap_rate = clip01(safe_mean(group, "overlap_flag"))
        interruption_rate = clip01(safe_mean(group, "interruption_flag"))
        strong_overlap_rate = clip01(safe_mean(group, "strong_overlap"))
        speaker_switch_rate = clip01(safe_mean(group, "speaker_switch"))
        speaker_switch_w5 = clip01(safe_mean(group, "speaker_switch_frequency_window5"))
        short_response_rate = clip01(safe_mean(group, "short_response"))
        immediate_response_rate = clip01(safe_mean(group, "immediate_response"))
        rapid_exchange_rate = clip01(safe_mean(group, "rapid_exchange_state"))
        conflict_like_rate = clip01(safe_mean(group, "conflict_like_state"))
        floor_competition_rate = clip01(safe_mean(group, "floor_competition_state"))
        interaction_density = safe_mean(group, "interaction_density_10s")
        silence_density = safe_mean(group, "silence_density_10s")
        long_pause_rate = clip01(safe_mean(group, "long_pause"))
        hesitation_rate = clip01(safe_mean(group, "hesitation_state"))
        gap_variance_w5 = safe_mean(group, "window5_gap_variance")
        burstiness_w5 = safe_mean(group, "burstiness_window5")
        consecutive_overlap_mean = safe_mean(group, "consecutive_overlap_count")
        consecutive_overlap_norm = clip01(consecutive_overlap_mean / 5.0)

        rows.append(
            {
                "dataset": dataset,
                "dialogue_id": dialogue_id,
                "session": session,
                "n_utterances": n,
                "total_duration": total_duration,
                "overlap_rate": overlap_rate,
                "mean_overlap_ratio": overlap_ratio,
                "interruption_rate": interruption_rate,
                "strong_overlap_rate": strong_overlap_rate,
                "floor_competition_rate": floor_competition_rate,
                "consecutive_overlap_norm": consecutive_overlap_norm,
                "speaker_switch_rate": speaker_switch_rate,
                "speaker_switch_frequency_w5": speaker_switch_w5,
                "short_response_rate": short_response_rate,
                "immediate_response_rate": immediate_response_rate,
                "rapid_exchange_rate": rapid_exchange_rate,
                "conflict_like_rate": conflict_like_rate,
                "interaction_density_10s": interaction_density,
                "long_pause_rate": long_pause_rate,
                "hesitation_rate": hesitation_rate,
                "silence_density_10s": silence_density,
                "mean_abs_gap": mean_abs_gap,
                "mean_positive_gap": mean_positive_gap,
                "mean_negative_gap_abs": mean_negative_gap_abs,
                "window5_gap_variance": gap_variance_w5,
                "burstiness_window5": burstiness_w5,
            }
        )
    return pd.DataFrame(rows)


def add_tii(dialogues: pd.DataFrame) -> pd.DataFrame:
    result = dialogues.copy()
    result["floor_competition"] = (
        0.30 * result["overlap_rate"]
        + 0.20 * result["mean_overlap_ratio"]
        + 0.20 * result["interruption_rate"]
        + 0.15 * result["strong_overlap_rate"]
        + 0.10 * result["floor_competition_rate"]
        + 0.05 * result["consecutive_overlap_norm"]
    )
    result["turn_exchange"] = (
        0.60 * result["speaker_switch_rate"]
        + 0.40 * result["speaker_switch_frequency_w5"]
    )
    result["response_tempo"] = (
        0.35 * result["short_response_rate"]
        + 0.25 * result["immediate_response_rate"]
        + 0.25 * result["rapid_exchange_rate"]
        + 0.15 * clip01_series(result["mean_negative_gap_abs"] / 2.0)
    )
    result["interaction_density"] = clip01_series(result["interaction_density_10s"] / 1.0)
    result["rhythm_instability"] = clip01_series(result["window5_gap_variance"] / result["window5_gap_variance"].quantile(0.90))
    result["silence_penalty"] = (
        0.40 * result["long_pause_rate"]
        + 0.30 * result["hesitation_rate"]
        + 0.20 * clip01_series(result["silence_density_10s"])
        + 0.10 * clip01_series(result["mean_positive_gap"] / 3.0)
    )

    component_weights = {
        "floor_competition": 0.30,
        "turn_exchange": 0.20,
        "response_tempo": 0.25,
        "interaction_density": 0.15,
        "rhythm_instability": 0.10,
    }
    positive = sum(weight * robust_z(result[col]) for col, weight in component_weights.items())
    penalty = robust_z(result["silence_penalty"])
    result["TII_z"] = positive - 0.25 * penalty
    result["TII_percentile"] = result["TII_z"].rank(pct=True) * 100.0
    result["TII_level"] = pd.cut(
        result["TII_percentile"],
        bins=[-0.01, 33.33, 66.67, 100.0],
        labels=["low", "medium", "high"],
    )
    return result


def clip01_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def build_gain_tables(dialogues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dialogue_rows = []
    for model, model_dir in MODEL_DIRS.items():
        pred = read_predictions(model_dir)
        if pred.empty:
            continue
        for dialogue_id, group in pred.groupby("dialogue_id"):
            model_dialogue_rows.append(
                {
                    "model": model,
                    "dialogue_id": dialogue_id,
                    "test_session": int(group["test_session"].iloc[0]),
                    "n_pred": len(group),
                    "accuracy": accuracy_from_predictions(group),
                    "macro_f1": macro_f1_from_predictions(group),
                }
            )
    model_dialogues = pd.DataFrame(model_dialogue_rows)
    if model_dialogues.empty:
        return model_dialogues, pd.DataFrame()

    wide_acc = model_dialogues.pivot_table(index="dialogue_id", columns="model", values="accuracy", aggfunc="first")
    wide_f1 = model_dialogues.pivot_table(index="dialogue_id", columns="model", values="macro_f1", aggfunc="first")
    joined = dialogues.merge(wide_acc.add_suffix("_accuracy"), left_on="dialogue_id", right_index=True, how="left")
    joined = joined.merge(wide_f1.add_suffix("_macro_f1"), left_on="dialogue_id", right_index=True, how="left")
    for model in MODEL_DIRS:
        if model == "MAL":
            continue
        if f"{model}_accuracy" in joined and "MAL_accuracy" in joined:
            joined[f"{model}_accuracy_gain_vs_MAL"] = joined[f"{model}_accuracy"] - joined["MAL_accuracy"]
        if f"{model}_macro_f1" in joined and "MAL_macro_f1" in joined:
            joined[f"{model}_macro_f1_gain_vs_MAL"] = joined[f"{model}_macro_f1"] - joined["MAL_macro_f1"]
    return model_dialogues, joined


def summarize_gain_by_tii(gain_dialogues: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_DIRS:
        if model == "MAL":
            continue
        for metric in ["accuracy", "macro_f1"]:
            gain_col = f"{model}_{metric}_gain_vs_MAL"
            if gain_col not in gain_dialogues:
                continue
            valid = gain_dialogues.dropna(subset=[gain_col, "TII_z", "TII_level"])
            if valid.empty:
                continue
            corr = valid[["TII_z", gain_col]].corr(method="spearman").iloc[0, 1]
            for level, group in valid.groupby("TII_level", observed=False):
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "TII_level": str(level),
                        "n_dialogues": len(group),
                        "mean_gain": float(group[gain_col].mean()),
                        "median_gain": float(group[gain_col].median()),
                        "win_rate_vs_MAL": float((group[gain_col] > 0).mean()),
                        "spearman_tii_gain": float(corr) if np.isfinite(corr) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write_report(session_tii: pd.DataFrame, gain_summary: pd.DataFrame, corr_rows: pd.DataFrame) -> None:
    feature_table = pd.DataFrame(
        [
            ["Floor competition", "overlap_rate, overlap_ratio, interruption_rate, strong_overlap_rate, floor_competition_rate, consecutive_overlap_norm", "Ai tranh sàn nói, nói chồng, chen ngang."],
            ["Turn exchange", "speaker_switch_rate, speaker_switch_frequency_w5", "Hai speaker đổi lượt nhanh hay một người giữ lượt."],
            ["Response tempo", "short_response_rate, immediate_response_rate, rapid_exchange_rate, mean_negative_gap_abs", "Phản hồi nhanh, gap ngắn, hoặc bắt đầu trước khi người trước kết thúc."],
            ["Interaction density", "interaction_density_10s", "Mật độ lượt nói trong cửa sổ 10 giây."],
            ["Rhythm instability", "window5_gap_variance", "Nhịp hội thoại biến động, không đều."],
            ["Silence penalty", "long_pause_rate, hesitation_rate, silence_density_10s, mean_positive_gap", "Khoảng lặng làm giảm cường độ tương tác thời gian."],
        ],
        columns=["component", "features", "meaning"],
    )
    report = [
        "# Temporal Interaction Intensity",
        "",
        "TII is a dataset-level score. It is computed from timestamps and speaker turns only, not from model predictions or emotion labels.",
        "",
        "## Feature Design",
        "",
        feature_table.to_markdown(index=False),
        "",
        "## Formula",
        "",
        "```text",
        "TII_z = 0.30*z(floor_competition)",
        "      + 0.20*z(turn_exchange)",
        "      + 0.25*z(response_tempo)",
        "      + 0.15*z(interaction_density)",
        "      + 0.10*z(rhythm_instability)",
        "      - 0.25*z(silence_penalty)",
        "```",
        "",
        "`TII_percentile` is the percentile rank of `TII_z` across IEMOCAP dialogues.",
        "",
        "## Session-Level TII",
        "",
        session_tii.to_markdown(index=False),
        "",
        "## TIM Gain by TII Level",
        "",
        gain_summary.to_markdown(index=False) if not gain_summary.empty else "No gain summary available.",
        "",
        "## Correlation: Session TII vs TIM Gain",
        "",
        corr_rows.to_markdown(index=False) if not corr_rows.empty else "No correlation table available.",
    ]
    (OUT_DIR / "temporal_interaction_intensity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURE_PATH, low_memory=False)
    dialogues = add_tii(build_dialogue_temporal_table(features))
    dialogues.to_csv(OUT_DIR / "dialogue_tii.csv", index=False)

    session_tii = (
        dialogues.groupby("session")
        .agg(
            n_dialogues=("dialogue_id", "count"),
            n_utterances=("n_utterances", "sum"),
            TII_z_mean=("TII_z", "mean"),
            TII_percentile_mean=("TII_percentile", "mean"),
            floor_competition=("floor_competition", "mean"),
            turn_exchange=("turn_exchange", "mean"),
            response_tempo=("response_tempo", "mean"),
            interaction_density=("interaction_density", "mean"),
            rhythm_instability=("rhythm_instability", "mean"),
            silence_penalty=("silence_penalty", "mean"),
        )
        .reset_index()
        .sort_values("TII_z_mean", ascending=False)
    )
    session_tii.to_csv(OUT_DIR / "session_tii.csv", index=False)

    model_dialogues, gain_dialogues = build_gain_tables(dialogues)
    model_dialogues.to_csv(OUT_DIR / "dialogue_model_metrics.csv", index=False)
    gain_dialogues.to_csv(OUT_DIR / "dialogue_tii_model_gain.csv", index=False)
    gain_summary = summarize_gain_by_tii(gain_dialogues)
    gain_summary.to_csv(OUT_DIR / "gain_by_tii_level.csv", index=False)

    corr_rows = []
    if not gain_dialogues.empty:
        session_gain = gain_dialogues.groupby("session").mean(numeric_only=True).reset_index()
        session_gain = session_gain.merge(session_tii[["session", "TII_z_mean", "TII_percentile_mean"]], on="session", how="left")
        session_gain.to_csv(OUT_DIR / "session_tii_gain.csv", index=False)
        for model in MODEL_DIRS:
            if model == "MAL":
                continue
            for metric in ["accuracy", "macro_f1"]:
                col = f"{model}_{metric}_gain_vs_MAL"
                if col in session_gain:
                    corr = session_gain[["TII_z_mean", col]].corr(method="spearman").iloc[0, 1]
                    corr_rows.append({"model": model, "metric": metric, "spearman_session_tii_gain": corr})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "session_tii_gain_correlation.csv", index=False)
    write_report(session_tii, gain_summary, corr_df)

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print(session_tii.to_string(index=False))
    if not gain_summary.empty:
        print("\nGain by TII level")
        print(gain_summary.to_string(index=False))


if __name__ == "__main__":
    main()
