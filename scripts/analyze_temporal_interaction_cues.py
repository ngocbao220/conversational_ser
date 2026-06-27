from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:  # pragma: no cover - optional dependency path
    variance_inflation_factor = None

from utils.iemocap_kaggle import LABEL_NAMES, discover_iemocap_samples


CANONICAL_LABELS = ["angry", "happy", "neutral", "sad"]
MELD_LABEL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "joy": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
}
NUMERIC_FEATURES = [
    "duration",
    "gap_to_previous",
    "gap_to_next",
    "overlap_duration",
    "overlap_ratio",
    "overlap_flag",
    "interruption_flag",
    "speaker_switch",
    "same_speaker_continuation",
    "turn_index",
    "normalized_turn_position",
    "immediate_response",
    "short_response",
    "long_pause",
    "speaker_previous_mean_gap",
    "speaker_previous_mean_duration",
    "speaker_previous_overlap_rate",
    "speaker_previous_turn_count",
    "previous_overlap_rate",
    "previous_speaker_switch_rate",
    "previous_mean_gap",
    "previous_mean_duration",
    "window3_overlap_frequency",
    "window3_interruption_frequency",
    "window3_average_gap",
    "window3_average_duration",
    "window3_speaker_switch_frequency",
    "window5_overlap_frequency",
    "window5_interruption_frequency",
    "window5_average_gap",
    "window5_average_duration",
    "window5_speaker_switch_frequency",
    "relative_duration",
    "relative_gap_to_speaker_mean",
]
BINARY_FEATURES = {
    "overlap_flag",
    "interruption_flag",
    "speaker_switch",
    "same_speaker_continuation",
    "immediate_response",
    "short_response",
    "long_pause",
}


@dataclass(frozen=True)
class DatasetLoadResult:
    name: str
    frame: pd.DataFrame
    status: str


def parse_time_to_seconds(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    if text.count(":") >= 1:
        parts = text.split(":")
        try:
            parts = [float(part) for part in parts]
        except ValueError:
            return float("nan")
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_iemocap(iemocap_root: Path) -> DatasetLoadResult:
    samples = discover_iemocap_samples(iemocap_root, auto_download=False)
    rows = []
    for sample in samples:
        rows.append(
            {
                "dataset": "IEMOCAP",
                "dialogue_id": sample.dialogue_id,
                "utterance_id": sample.utterance_id,
                "speaker_id": sample.speaker_id,
                "session_id": sample.session_id,
                "start_time": sample.start_time,
                "end_time": sample.end_time,
                "emotion": sample.label_name,
                "raw_emotion": sample.raw_label,
                "text": sample.transcript,
            }
        )
    return DatasetLoadResult("IEMOCAP", pd.DataFrame(rows), f"loaded {len(rows)} labelled utterances")


def candidate_meld_roots(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    return [
        Path("meld"),
        Path("MELD"),
        Path("data/meld"),
        Path("data/MELD"),
        Path("datasets/meld"),
        Path("datasets/MELD"),
        Path("data/meld-dataset"),
        Path("datasets/meld-dataset"),
    ]


def find_meld_csvs(meld_root: Path | None) -> list[Path]:
    csv_paths: list[Path] = []
    for root in candidate_meld_roots(meld_root):
        if not root.exists():
            continue
        csv_paths.extend(sorted(root.rglob("*sent_emo*.csv")))
        if not csv_paths:
            csv_paths.extend(sorted(root.rglob("*.csv")))
    return [path for path in csv_paths if path.is_file()]


def ensure_meld_root(
    meld_root: Path,
    auto_download: bool = False,
    kaggle_dataset: str = "zaber666/meld-dataset",
) -> tuple[Path, str]:
    if find_meld_csvs(meld_root):
        return meld_root, f"using local MELD root: {meld_root}"
    if meld_root.exists() and not auto_download:
        return meld_root, f"MELD root exists but no compatible CSV files were found: {meld_root}"
    if not auto_download:
        return meld_root, "missing local MELD CSV files"

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        return (
            meld_root,
            "missing local MELD CSV files; Kaggle CLI not found for auto-download "
            f"of {kaggle_dataset}",
        )

    meld_root.parent.mkdir(parents=True, exist_ok=True)
    download_dir = meld_root.parent / f".{meld_root.name}_kaggle_download"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    command = [kaggle_bin, "datasets", "download", "-d", kaggle_dataset, "-p", str(download_dir)]
    subprocess.run(command, check=True)
    downloaded_zips = sorted(download_dir.glob("*.zip"))
    if not downloaded_zips:
        return meld_root, f"Kaggle download for {kaggle_dataset} completed but no zip file was found"

    meld_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(downloaded_zips[0], "r") as archive:
        archive.extractall(meld_root)
    shutil.rmtree(download_dir, ignore_errors=True)
    if not find_meld_csvs(meld_root):
        return meld_root, f"downloaded {kaggle_dataset} but no compatible CSV files were found under {meld_root}"
    return meld_root, f"downloaded {kaggle_dataset} to {meld_root}"


def load_meld(
    meld_root: Path | None,
    auto_download: bool = False,
    kaggle_dataset: str = "zaber666/meld-dataset",
) -> DatasetLoadResult:
    resolved_root = meld_root or Path("data/meld-dataset")
    root, root_status = ensure_meld_root(
        resolved_root,
        auto_download=auto_download and meld_root is not None or auto_download,
        kaggle_dataset=kaggle_dataset,
    )
    csv_paths = find_meld_csvs(root if root.exists() else meld_root)
    if not csv_paths:
        return DatasetLoadResult("MELD", pd.DataFrame(), root_status)

    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        lower = {column.lower(): column for column in frame.columns}
        if "emotion" not in lower:
            continue
        dialogue_col = lower.get("dialogue_id") or lower.get("dialogueid") or lower.get("old_dialogue_id")
        utterance_col = lower.get("utterance_id") or lower.get("utteranceid") or lower.get("old_utterance_id")
        speaker_col = lower.get("speaker")
        start_col = lower.get("starttime") or lower.get("start_time")
        end_col = lower.get("endtime") or lower.get("end_time")
        if dialogue_col is None or utterance_col is None or speaker_col is None or start_col is None or end_col is None:
            continue
        subset = pd.DataFrame(
            {
                "dataset": "MELD",
                "dialogue_id": frame[dialogue_col].map(lambda value: f"{path.stem}_{value}"),
                "utterance_id": frame[utterance_col].map(lambda value: f"{path.stem}_{value}"),
                "speaker_id": frame[speaker_col].astype(str),
                "session_id": path.stem,
                "start_time": frame[start_col].map(parse_time_to_seconds),
                "end_time": frame[end_col].map(parse_time_to_seconds),
                "raw_emotion": frame[lower["emotion"]].astype(str).str.lower(),
                "text": frame[lower.get("utterance", lower["emotion"])].astype(str),
            }
        )
        subset["emotion"] = subset["raw_emotion"].map(MELD_LABEL_MAP)
        frames.append(subset)

    if not frames:
        return DatasetLoadResult("MELD", pd.DataFrame(), f"found CSVs but no compatible MELD schema: {csv_paths}")
    meld = pd.concat(frames, ignore_index=True)
    meld = meld[meld["emotion"].isin(CANONICAL_LABELS)].copy()
    meld = meld.dropna(subset=["start_time", "end_time"])
    meld = meld[meld["end_time"] >= meld["start_time"]].copy()
    if meld.empty:
        return DatasetLoadResult("MELD", pd.DataFrame(), "MELD CSVs found but no 4-class rows with valid timestamps")
    return DatasetLoadResult(
        "MELD",
        meld,
        f"{root_status}; loaded {len(meld)} 4-class timestamped utterances from {len(csv_paths)} CSVs",
    )


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.sort_values(["dialogue_id", "start_time", "end_time", "utterance_id"]).reset_index(drop=True)
    feature_rows: list[dict[str, float]] = []
    for _, dialogue in frame.groupby("dialogue_id", sort=False):
        dialogue = dialogue.reset_index(drop=True)
        speaker_history: dict[str, dict[str, float]] = {}
        previous_overlap_count = 0.0
        previous_switch_count = 0.0
        previous_gap_sum = 0.0
        previous_duration_sum = 0.0
        max_index = max(len(dialogue) - 1, 1)
        for i, row in dialogue.iterrows():
            start = float(row["start_time"])
            end = float(row["end_time"])
            duration = max(0.0, end - start)
            speaker = str(row["speaker_id"])

            if i == 0:
                gap_prev = np.nan
                overlap = 0.0
                speaker_switch = 0.0
                same_speaker = 0.0
                interruption = 0.0
            else:
                prev = dialogue.iloc[i - 1]
                gap_prev = start - float(prev["end_time"])
                overlap = max(0.0, float(prev["end_time"]) - start)
                speaker_switch = 1.0 if speaker != str(prev["speaker_id"]) else 0.0
                same_speaker = 1.0 - speaker_switch
                interruption = 1.0 if speaker_switch and overlap > 0.0 else 0.0

            if i == len(dialogue) - 1:
                gap_next = np.nan
            else:
                nxt = dialogue.iloc[i + 1]
                gap_next = float(nxt["start_time"]) - end

            history = speaker_history.get(
                speaker,
                {"turn_count": 0.0, "overlap_count": 0.0, "gap_sum": 0.0, "duration_sum": 0.0},
            )
            previous_turns = float(i)
            previous_overlap_rate = previous_overlap_count / max(previous_turns, 1.0)
            previous_speaker_switch_rate = previous_switch_count / max(previous_turns, 1.0)
            previous_mean_gap = previous_gap_sum / max(previous_turns, 1.0)
            previous_mean_duration = previous_duration_sum / max(previous_turns, 1.0)
            speaker_turns = history["turn_count"]
            speaker_mean_gap = history["gap_sum"] / max(speaker_turns, 1.0)
            speaker_mean_duration = history["duration_sum"] / max(speaker_turns, 1.0)
            speaker_overlap_rate = history["overlap_count"] / max(speaker_turns, 1.0)
            window_features: dict[str, float] = {}
            for window in (3, 5):
                start_idx = max(0, i - window)
                prior_features = feature_rows[-(i - start_idx) :] if i - start_idx > 0 else []
                if prior_features:
                    window_features[f"window{window}_overlap_frequency"] = float(
                        np.mean([item["overlap_flag"] for item in prior_features])
                    )
                    window_features[f"window{window}_interruption_frequency"] = float(
                        np.mean([item["interruption_flag"] for item in prior_features])
                    )
                    window_features[f"window{window}_average_gap"] = float(
                        np.nanmean([item["gap_to_previous"] for item in prior_features])
                    )
                    window_features[f"window{window}_average_duration"] = float(
                        np.mean([item["duration"] for item in prior_features])
                    )
                    window_features[f"window{window}_speaker_switch_frequency"] = float(
                        np.mean([item["speaker_switch"] for item in prior_features])
                    )
                else:
                    window_features[f"window{window}_overlap_frequency"] = 0.0
                    window_features[f"window{window}_interruption_frequency"] = 0.0
                    window_features[f"window{window}_average_gap"] = 0.0
                    window_features[f"window{window}_average_duration"] = 0.0
                    window_features[f"window{window}_speaker_switch_frequency"] = 0.0

            feature = {
                "duration": duration,
                "gap_to_previous": gap_prev,
                "gap_to_next": gap_next,
                "overlap_duration": overlap,
                "overlap_ratio": overlap / max(duration, 1e-6),
                "overlap_flag": 1.0 if overlap > 0.05 else 0.0,
                "interruption_flag": interruption,
                "speaker_switch": speaker_switch,
                "same_speaker_continuation": same_speaker,
                "turn_index": float(i),
                "normalized_turn_position": float(i / max_index),
                "immediate_response": 1.0 if np.isfinite(gap_prev) and 0.0 <= gap_prev < 0.2 else 0.0,
                "short_response": 1.0 if np.isfinite(gap_prev) and 0.0 <= gap_prev < 0.5 else 0.0,
                "long_pause": 1.0 if np.isfinite(gap_prev) and gap_prev > 1.0 else 0.0,
                "speaker_previous_mean_gap": speaker_mean_gap,
                "speaker_previous_mean_duration": speaker_mean_duration,
                "speaker_previous_overlap_rate": speaker_overlap_rate,
                "speaker_previous_turn_count": speaker_turns,
                "previous_overlap_rate": previous_overlap_rate,
                "previous_speaker_switch_rate": previous_speaker_switch_rate,
                "previous_mean_gap": previous_mean_gap,
                "previous_mean_duration": previous_mean_duration,
                **window_features,
                "relative_duration": duration - speaker_mean_duration,
                "relative_gap_to_speaker_mean": (gap_prev if np.isfinite(gap_prev) else 0.0) - speaker_mean_gap,
            }
            feature_rows.append(feature)

            history["turn_count"] += 1.0
            history["overlap_count"] += feature["overlap_flag"]
            history["gap_sum"] += gap_prev if np.isfinite(gap_prev) else 0.0
            history["duration_sum"] += duration
            speaker_history[speaker] = history
            previous_overlap_count += feature["overlap_flag"]
            previous_switch_count += speaker_switch
            previous_gap_sum += gap_prev if np.isfinite(gap_prev) else 0.0
            previous_duration_sum += duration
    features = pd.DataFrame(feature_rows)
    return pd.concat([frame, features], axis=1)


def describe_feature(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "q025": np.nan, "q975": np.nan}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "median": float(clean.median()),
        "q025": float(clean.quantile(0.025)),
        "q975": float(clean.quantile(0.975)),
    }


def cramers_eta_squared(groups: list[np.ndarray], values: np.ndarray, labels: np.ndarray) -> float:
    grand_mean = np.nanmean(values)
    ss_between = 0.0
    ss_total = float(np.nansum((values - grand_mean) ** 2))
    for group in groups:
        if len(group):
            ss_between += len(group) * float((np.nanmean(group) - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else np.nan


def epsilon_squared_kruskal(h: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return max(0.0, (h - k + 1.0) / (n - k))


def cliff_delta(x: np.ndarray, y: np.ndarray, max_pairs: int = 200_000) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    if x.size * y.size > max_pairs:
        rng = np.random.default_rng(42)
        x = rng.choice(x, size=min(x.size, int(math.sqrt(max_pairs))), replace=False)
        y = rng.choice(y, size=min(y.size, int(math.sqrt(max_pairs))), replace=False)
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def analyze_dataset(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    dataset = str(frame["dataset"].iloc[0])
    dataset_dir = output_dir / dataset.lower()
    plots_dir = output_dir / "plots" / dataset.lower()
    histogram_dir = output_dir / "plots" / "histograms" / dataset.lower()
    boxplot_dir = output_dir / "plots" / "boxplots" / dataset.lower()
    violin_dir = output_dir / "plots" / "violin_plots" / dataset.lower()
    heatmap_dir = output_dir / "plots" / "correlation_heatmaps"
    subset_dir = output_dir / "plots" / "subset_comparison_figures"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for directory in (histogram_dir, boxplot_dir, violin_dir, heatmap_dir, subset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    y = LabelEncoder().fit_transform(frame["emotion"])
    feature_rows = []
    emotion_rows = []
    importance_rows = []

    for feature in NUMERIC_FEATURES:
        stats_row = {"dataset": dataset, "feature": feature, **describe_feature(frame[feature])}
        feature_rows.append(stats_row)
        for emotion, group in frame.groupby("emotion"):
            emotion_rows.append(
                {"dataset": dataset, "emotion": emotion, "feature": feature, **describe_feature(group[feature])}
            )

        clean = frame[[feature, "emotion"]].replace([np.inf, -np.inf], np.nan).dropna()
        groups = [group[feature].astype(float).to_numpy() for _, group in clean.groupby("emotion")]
        values = clean[feature].astype(float).to_numpy()
        labels = LabelEncoder().fit_transform(clean["emotion"]) if not clean.empty else np.array([])
        if len(groups) >= 2 and all(len(group) >= 2 for group in groups):
            f_stat, anova_p = stats.f_oneway(*groups)
            h_stat, kruskal_p = stats.kruskal(*groups)
            eta2 = cramers_eta_squared(groups, values, labels)
            eps2 = epsilon_squared_kruskal(float(h_stat), len(values), len(groups))
            spearman = stats.spearmanr(values, labels).correlation
            pearson = stats.pearsonr(values, labels).statistic if len(np.unique(values)) > 1 else np.nan
            mi = mutual_info_classif(values.reshape(-1, 1), labels, random_state=42, discrete_features=False)[0]
            mw_p_values = []
            deltas = []
            for emotion in sorted(clean["emotion"].unique()):
                one = clean.loc[clean["emotion"] == emotion, feature].astype(float).to_numpy()
                rest = clean.loc[clean["emotion"] != emotion, feature].astype(float).to_numpy()
                if len(one) and len(rest):
                    mw_p_values.append(stats.mannwhitneyu(one, rest, alternative="two-sided").pvalue)
                    deltas.append(abs(cliff_delta(one, rest)))
            mw_min_p = float(np.min(mw_p_values)) if mw_p_values else np.nan
            max_abs_cliff_delta = float(np.nanmax(deltas)) if deltas else np.nan
        else:
            anova_p = kruskal_p = eta2 = eps2 = spearman = pearson = mi = mw_min_p = max_abs_cliff_delta = np.nan
        importance_rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "pearson_label_code": pearson,
                "spearman_label_code": spearman,
                "mutual_information": mi,
                "distance_correlation": np.nan,
                "anova_p": anova_p,
                "kruskal_p": kruskal_p,
                "min_mannwhitney_p_one_vs_rest": mw_min_p,
                "eta_squared": eta2,
                "epsilon_squared": eps2,
                "max_abs_cliff_delta_one_vs_rest": max_abs_cliff_delta,
            }
        )

    plot_features = [
        "duration",
        "gap_to_previous",
        "overlap_duration",
        "overlap_ratio",
        "speaker_switch",
        "normalized_turn_position",
        "previous_overlap_rate",
        "window5_speaker_switch_frequency",
    ]
    for feature in plot_features:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        sns.histplot(data=frame, x=feature, hue="emotion", element="step", stat="density", common_norm=False, ax=axes[0])
        sns.boxplot(data=frame, x="emotion", y=feature, ax=axes[1])
        sns.violinplot(data=frame, x="emotion", y=feature, ax=axes[2], inner="quartile", cut=0)
        for axis in axes:
            axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{feature}_distribution.png", dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7, 4))
        sns.histplot(data=frame, x=feature, hue="emotion", element="step", stat="density", common_norm=False, ax=axis)
        fig.tight_layout()
        fig.savefig(histogram_dir / f"{feature}.png", dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=frame, x="emotion", y=feature, ax=axis)
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(boxplot_dir / f"{feature}.png", dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7, 4))
        sns.violinplot(data=frame, x="emotion", y=feature, ax=axis, inner="quartile", cut=0)
        axis.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(violin_dir / f"{feature}.png", dpi=160)
        plt.close(fig)

    feature_frame = frame[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = feature_frame.corr(method="spearman")
    fig, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="vlag", center=0, ax=axis)
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_heatmap.png", dpi=180)
    fig.savefig(heatmap_dir / f"{dataset.lower()}_correlation_heatmap.png", dpi=180)
    plt.close(fig)
    corr.to_csv(dataset_dir / "correlation_matrix.csv")

    redundancy = corr.abs()
    redundancy.to_csv(dataset_dir / "redundancy_matrix.csv")
    vif_rows = []
    if variance_inflation_factor is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = StandardScaler().fit_transform(feature_frame)
            for idx, feature in enumerate(NUMERIC_FEATURES):
                try:
                    vif = float(variance_inflation_factor(scaled, idx))
                except Exception:
                    vif = np.nan
                vif_rows.append({"dataset": dataset, "feature": feature, "vif": vif})
    pd.DataFrame(vif_rows).to_csv(dataset_dir / "vif.csv", index=False)

    subset_summary = subset_analysis(frame)
    subset_summary.to_csv(dataset_dir / "subset_emotion_distribution.csv", index=False)
    plot_subset_summary(subset_summary, plots_dir)
    plot_subset_summary(subset_summary, subset_dir, stem=f"{dataset.lower()}_subset_emotion_distribution")

    progression = dialogue_progression(frame)
    progression.to_csv(dataset_dir / "dialogue_progression.csv", index=False)
    speaker = speaker_behavior(frame)
    speaker.to_csv(dataset_dir / "speaker_behavior.csv", index=False)

    return {
        "dataset": dataset,
        "feature_statistics": pd.DataFrame(feature_rows),
        "emotion_statistics": pd.DataFrame(emotion_rows),
        "feature_importance": pd.DataFrame(importance_rows),
        "correlation_matrix": corr,
        "redundancy_matrix": redundancy,
        "subset_summary": subset_summary,
        "progression": progression,
        "speaker_behavior": speaker,
    }


def subset_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    dialogue_stats = frame.groupby("dialogue_id").agg(
        overlap_rate=("overlap_flag", "mean"),
        interruption_rate=("interruption_flag", "mean"),
        mean_gap=("gap_to_previous", "mean"),
        speaker_switch_rate=("speaker_switch", "mean"),
    )
    subsets = {}
    for name, column in [
        ("overlap", "overlap_rate"),
        ("interruption", "interruption_rate"),
        ("slow_response", "mean_gap"),
        ("speaker_switch", "speaker_switch_rate"),
    ]:
        median = dialogue_stats[column].median()
        subsets[f"high_{name}"] = set(dialogue_stats.index[dialogue_stats[column] >= median])
        subsets[f"low_{name}"] = set(dialogue_stats.index[dialogue_stats[column] < median])
    rows = []
    for subset_name, dialogue_ids in subsets.items():
        subset = frame[frame["dialogue_id"].isin(dialogue_ids)]
        counts = subset["emotion"].value_counts(normalize=False)
        rates = subset["emotion"].value_counts(normalize=True)
        for emotion in CANONICAL_LABELS:
            rows.append(
                {
                    "dataset": frame["dataset"].iloc[0],
                    "subset": subset_name,
                    "emotion": emotion,
                    "count": int(counts.get(emotion, 0)),
                    "rate": float(rates.get(emotion, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def plot_subset_summary(
    subset_summary: pd.DataFrame,
    plots_dir: Path,
    stem: str = "subset_emotion_distribution",
) -> None:
    if subset_summary.empty:
        return
    fig, axis = plt.subplots(figsize=(14, 5))
    sns.barplot(data=subset_summary, x="subset", y="rate", hue="emotion", ax=axis)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("emotion rate")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{stem}.png", dpi=160)
    plt.close(fig)


def dialogue_progression(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.001, 1 / 3, 2 / 3, 1.001]
    labels = ["beginning", "middle", "end"]
    temp = frame.copy()
    temp["dialogue_stage"] = pd.cut(temp["normalized_turn_position"], bins=bins, labels=labels)
    rows = []
    for (stage, emotion), group in temp.groupby(["dialogue_stage", "emotion"], observed=False):
        rows.append(
            {
                "dataset": frame["dataset"].iloc[0],
                "stage": str(stage),
                "emotion": emotion,
                "count": int(len(group)),
                "duration_mean": float(group["duration"].mean()),
                "gap_mean": float(group["gap_to_previous"].mean()),
                "overlap_rate": float(group["overlap_flag"].mean()),
                "speaker_switch_rate": float(group["speaker_switch"].mean()),
            }
        )
    return pd.DataFrame(rows)


def speaker_behavior(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["dataset", "speaker_id", "emotion"])
        .agg(
            count=("utterance_id", "count"),
            average_response_latency=("gap_to_previous", "mean"),
            average_overlap=("overlap_duration", "mean"),
            average_interruption=("interruption_flag", "mean"),
            average_duration=("duration", "mean"),
            average_relative_duration=("relative_duration", "mean"),
        )
        .reset_index()
    )


def recommendation_table(feature_importance: pd.DataFrame, redundancy_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in feature_importance.groupby("feature"):
        median_mi = float(group["mutual_information"].median())
        min_p = float(group["kruskal_p"].min())
        median_effect = float(group["epsilon_squared"].median())
        if feature in redundancy_matrix.index:
            max_redundancy = float(redundancy_matrix.loc[feature].drop(index=feature, errors="ignore").max())
        else:
            max_redundancy = np.nan
        if max_redundancy >= 0.9:
            tier = "Redundant"
        elif min_p < 0.001 and median_effect >= 0.01 and median_mi >= 0.01:
            tier = "Highly Recommended"
        elif min_p < 0.01 and (median_effect >= 0.005 or median_mi >= 0.005):
            tier = "Useful"
        elif min_p < 0.05:
            tier = "Weak"
        else:
            tier = "Noisy"
        rows.append(
            {
                "feature": feature,
                "recommendation": tier,
                "median_mutual_information": median_mi,
                "best_kruskal_p": min_p,
                "median_epsilon_squared": median_effect,
                "max_abs_spearman_redundancy": max_redundancy,
                "justification": (
                    f"MI={median_mi:.4f}, best Kruskal p={min_p:.2e}, "
                    f"epsilon^2={median_effect:.4f}, max redundancy={max_redundancy:.3f}"
                ),
            }
        )
    tier_order = {"Highly Recommended": 0, "Useful": 1, "Weak": 2, "Redundant": 3, "Noisy": 4}
    return pd.DataFrame(rows).sort_values(["recommendation", "median_mutual_information"], key=lambda col: col.map(tier_order) if col.name == "recommendation" else -col)


def cross_dataset_comparison(results: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    importance = pd.concat([result["feature_importance"] for result in results], ignore_index=True)
    pivot = importance.pivot_table(index="feature", columns="dataset", values="mutual_information", aggfunc="mean")
    rows = []
    for feature, row in pivot.iterrows():
        available = row.dropna()
        if len(available) >= 2:
            spread = float(available.max() - available.min())
            consistency = "consistent" if spread < 0.01 else "dataset-specific"
        else:
            spread = np.nan
            consistency = "insufficient-data"
        rows.append({"feature": feature, "mi_spread": spread, "cross_dataset_status": consistency, **row.to_dict()})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "cross_dataset_feature_comparison.csv", index=False)
    return comparison


def load_iemocap_embedding_cache(cache_path: Path, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame] | None:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu")
    rows = payload.get("rows_by_utterance", {})
    embeddings = []
    kept_indices = []
    for idx, row in frame.iterrows():
        item = rows.get(str(row["utterance_id"]))
        if item is None:
            continue
        embeddings.append(torch.as_tensor(item["embedding"]).numpy())
        kept_indices.append(idx)
    if not embeddings:
        return None
    return np.stack(embeddings), frame.loc[kept_indices].reset_index(drop=True)


def complementarity_analysis(frame: pd.DataFrame, output_dir: Path, cache_path: Path) -> pd.DataFrame:
    loaded = load_iemocap_embedding_cache(cache_path, frame)
    rows = []
    if loaded is None:
        rows.append({"dataset": "IEMOCAP", "analysis": "wavlm_complementarity", "status": f"missing cache: {cache_path}"})
        return pd.DataFrame(rows)
    embeddings, matched = loaded
    temporal = matched[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    labels = LabelEncoder().fit_transform(matched["emotion"])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        "WavLM embedding only": embeddings,
        "Temporal features only": temporal,
        "WavLM + Temporal features": np.concatenate([embeddings, temporal], axis=1),
    }
    for name, matrix in classifiers.items():
        scores = []
        for train_idx, test_idx in splitter.split(matrix, labels):
            clf = make_pipeline(
                StandardScaler(),
                SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=42),
            )
            clf.fit(matrix[train_idx], labels[train_idx])
            pred = clf.predict(matrix[test_idx])
            scores.append(f1_score(labels[test_idx], pred, average="macro"))
        rows.append(
            {
                "dataset": "IEMOCAP",
                "analysis": "classifier_macro_f1",
                "feature_set": name,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
                "n": len(scores),
                "status": "ok",
            }
        )

    max_samples = min(2500, len(labels))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(labels), size=max_samples, replace=False)
    emb_sample = StandardScaler().fit_transform(embeddings[sample_idx])
    temporal_sample = StandardScaler().fit_transform(temporal[sample_idx])
    cca_components = min(5, temporal_sample.shape[1], emb_sample.shape[1])
    cca = CCA(n_components=cca_components, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u, v = cca.fit_transform(emb_sample, temporal_sample)
    cca_corrs = [float(np.corrcoef(u[:, i], v[:, i])[0, 1]) for i in range(cca_components)]
    rows.append(
        {
            "dataset": "IEMOCAP",
            "analysis": "cca_embedding_temporal",
            "mean": float(np.nanmean(cca_corrs)),
            "max": float(np.nanmax(cca_corrs)),
            "n": cca_components,
            "status": "ok",
        }
    )

    predictability_n = min(3000, len(labels))
    predictability_idx = rng.choice(len(labels), size=predictability_n, replace=False)
    predictability_embeddings = embeddings[predictability_idx]
    predictability_temporal = temporal[predictability_idx]
    predictability_labels = labels[predictability_idx]
    predictability_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for feature_idx, feature in enumerate(NUMERIC_FEATURES):
        y = predictability_temporal[:, feature_idx]
        if np.nanstd(y) < 1e-8:
            r2 = np.nan
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr"))
            scores = []
            for train_idx, test_idx in predictability_splitter.split(predictability_embeddings, predictability_labels):
                model.fit(predictability_embeddings[train_idx], y[train_idx])
                scores.append(model.score(predictability_embeddings[test_idx], y[test_idx]))
            r2 = float(np.mean(scores))
        rows.append(
            {
                "dataset": "IEMOCAP",
                "analysis": "temporal_predictable_from_wavlm",
                "feature": feature,
                "r2_from_wavlm": r2,
                "complementarity_label": "redundant" if np.isfinite(r2) and r2 >= 0.25 else "complementary",
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def write_reports(
    output_dir: Path,
    load_statuses: Sequence[DatasetLoadResult],
    results: list[dict[str, Any]],
    feature_statistics: pd.DataFrame,
    emotion_statistics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    recommendations: pd.DataFrame,
    cross_comparison: pd.DataFrame,
    complementarity: pd.DataFrame,
) -> None:
    status_lines = "\n".join(f"- {item.name}: {item.status}" for item in load_statuses)
    top_features = (
        feature_importance.sort_values("mutual_information", ascending=False)
        .groupby("dataset")
        .head(8)[["dataset", "feature", "mutual_information", "kruskal_p", "epsilon_squared"]]
    )
    report = f"""# Temporal Interaction Cue Study

## Scope

This study tests whether timing and turn-taking cues are associated with emotion labels in conversational SER.
It does not train the TIM architecture. It computes temporal interaction features from dialogue timestamps and
speaker turns, then evaluates distributional differences, correlation, statistical significance, redundancy,
subset behavior, dialogue progression, and complementarity with frozen WavLM embeddings when a cache is available.

## Dataset Status

{status_lines}

## Strongest Feature Associations

{top_features.to_markdown(index=False)}

## Feature Recommendation Summary

{recommendations['recommendation'].value_counts().to_markdown()}

## Interpretation

- Temporal cues should be treated as grouped interaction signals rather than a flat concatenated vector.
- Features with significant Kruskal/ANOVA results and non-trivial mutual information are the strongest candidates.
- Features with high redundancy should be represented once per group or handled by gating/attention.
- Dataset-specific features should not be trusted as universal SER cues until they replicate across MELD and IEMOCAP.
- If WavLM can linearly predict a temporal feature with high R2, that feature is likely redundant with acoustics.
  Otherwise, it is a stronger candidate for complementary TIM conditioning.

## Complementarity

{complementarity.head(40).to_markdown(index=False) if not complementarity.empty else 'No complementarity results.'}
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")

    cross_report = f"""# Cross-Dataset Temporal Cue Comparison

The table below compares mutual information by feature across available datasets. `insufficient-data` means one
of the required datasets was not available locally or did not contain usable timestamps.

{cross_comparison.to_markdown(index=False)}
"""
    (output_dir / "cross_dataset_comparison.md").write_text(cross_report, encoding="utf-8")

    design = f"""# TIM Design Recommendation

## Recommended Direction

Do not concatenate all temporal features directly. The statistical analysis supports a grouped, gated temporal module:

1. Encode feature groups separately: duration, response gap, overlap/interruption, speaker continuity, dialogue position,
   causal speaker statistics, causal dialogue statistics, and short-window dynamics.
2. Use feature-group gates conditioned on the acoustic embedding and memory state.
3. Use a sparse or low-rank temporal attention layer so noisy groups can be suppressed per utterance.
4. Normalize temporal cues causally using speaker/dialogue history rather than global future information.
5. Include a zero/shuffled temporal control during experiments to verify that gains come from aligned timing.

## Feature Tiers

{recommendations.to_markdown(index=False)}

## Practical Architecture

- Acoustic branch: frozen or lightly fine-tuned WavLM utterance embedding.
- Memory branch: causal dialogue memory as in MAL.
- Temporal branch: group encoders with small MLPs, one per temporal cue group.
- Gate: `gate_g = sigmoid(W_g [wavlm_i, memory_state_i])` for each group.
- Fusion: `temporal_context = sum_g gate_g * encoder_g(features_g)`.
- Regularization: group dropout and entropy penalty on gates to avoid relying on one noisy cue such as overlap alone.
- Ablations: full, zero, shuffled, no-overlap, no-duration, no-gap, no-speaker, no-position.
"""
    (output_dir / "tim_design_recommendation.md").write_text(design, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical study of temporal interaction cues for conversational SER.")
    parser.add_argument("--iemocap-root", default="iemocap")
    parser.add_argument("--meld-root", default=None, help="Local MELD root. Defaults to data/meld-dataset.")
    parser.add_argument(
        "--meld-auto-download",
        action="store_true",
        help="Download MELD with Kaggle CLI if no local MELD CSV files are found.",
    )
    parser.add_argument("--meld-kaggle-dataset", default="zaber666/meld-dataset")
    parser.add_argument("--output-dir", default="reports/temporal_interaction_study")
    parser.add_argument("--embedding-cache", default="results/wavlm_tim/cache/wavlm_mean_embeddings.pt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    load_statuses = [
        load_iemocap(Path(args.iemocap_root)),
        load_meld(
            Path(args.meld_root) if args.meld_root else None,
            auto_download=bool(args.meld_auto_download),
            kaggle_dataset=str(args.meld_kaggle_dataset),
        ),
    ]
    (output_dir / "dataset_status.json").write_text(
        json.dumps({item.name: item.status for item in load_statuses}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    frames = []
    for item in load_statuses:
        if item.frame.empty:
            continue
        enriched = add_temporal_features(item.frame)
        enriched.to_csv(output_dir / f"{item.name.lower()}_temporal_features.csv", index=False)
        frames.append(enriched)
    if not frames:
        raise RuntimeError("No datasets with valid temporal rows were available.")

    results = [analyze_dataset(frame, output_dir) for frame in frames]
    feature_statistics = pd.concat([result["feature_statistics"] for result in results], ignore_index=True)
    emotion_statistics = pd.concat([result["emotion_statistics"] for result in results], ignore_index=True)
    feature_importance = pd.concat([result["feature_importance"] for result in results], ignore_index=True)

    feature_statistics.to_csv(output_dir / "feature_statistics.csv", index=False)
    emotion_statistics.to_csv(output_dir / "emotion_statistics.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    combined_corr = pd.concat(
        {result["dataset"]: result["correlation_matrix"] for result in results}, names=["dataset", "feature"]
    )
    combined_redundancy = pd.concat(
        {result["dataset"]: result["redundancy_matrix"] for result in results}, names=["dataset", "feature"]
    )
    combined_corr.to_csv(output_dir / "correlation_matrix.csv")
    combined_redundancy.to_csv(output_dir / "redundancy_matrix.csv")

    recommendations = recommendation_table(feature_importance, results[0]["redundancy_matrix"])
    recommendations.to_csv(output_dir / "recommended_temporal_features.csv", index=False)
    cross_comparison = cross_dataset_comparison(results, output_dir)

    iemocap_frame = next((frame for frame in frames if frame["dataset"].iloc[0] == "IEMOCAP"), None)
    complementarity = (
        complementarity_analysis(iemocap_frame, output_dir, Path(args.embedding_cache))
        if iemocap_frame is not None
        else pd.DataFrame()
    )
    complementarity.to_csv(output_dir / "complementarity_analysis.csv", index=False)
    write_reports(
        output_dir,
        load_statuses,
        results,
        feature_statistics,
        emotion_statistics,
        feature_importance,
        recommendations,
        cross_comparison,
        complementarity,
    )
    print(f"analysis_report={output_dir / 'analysis_report.md'}")
    print(f"tim_design_recommendation={output_dir / 'tim_design_recommendation.md'}")


if __name__ == "__main__":
    main()
