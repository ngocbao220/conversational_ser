from __future__ import annotations

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.iemocap_kaggle import LABEL_NAMES, RAW_LABEL_MAP, parse_emotion_file


FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_RESULT_DIR = PROJECT_ROOT / "results" / "final_result"


STYLE = [
    {"color": "#f6b0ad", "hatch": "o"},
    {"color": "#b7d7e8", "hatch": "||"},
    {"color": "#bfe8bf", "hatch": "-"},
    {"color": "#d9c3df", "hatch": "+"},
    {"color": "#ffd59b", "hatch": "x"},
    {"color": "#fff9b8", "hatch": "/"},
    {"color": "#e8dcc5", "hatch": "\\"},
]


def plot_ablation(labels: list[str], values: list[float], title: str, output_name: str, ylabel: str = "UA (%)") -> None:
    fig, ax = plt.subplots(figsize=(6.2, 2.95))
    x = np.arange(len(labels))
    display_labels = [
        label.replace("CIDM full", "CIDM\nfull")
        .replace("without interaction", "without\ninteraction")
        .replace("without acoustic", "without\nacoustic")
        for label in labels
    ]

    for idx, (label, value) in enumerate(zip(labels, values)):
        style = STYLE[idx % len(STYLE)]
        ax.bar(
            x[idx],
            value,
            width=0.68,
            label=label,
            color=style["color"],
            edgecolor="#222222",
            linewidth=0.8,
            hatch=style["hatch"],
            zorder=3,
        )
        ax.text(
            x[idx],
            value + 1.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_ylabel(ylabel, fontsize=8)
    ymin = max(0.0, min(values) - 8.0)
    ymax = min(100.0, max(values) + 7.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=7, fontstyle="normal")
    for tick in ax.get_xticklabels():
        tick.set_multialignment("center")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=9, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def latest_summary(backbone: str, model: str) -> dict:
    paths = sorted((FINAL_RESULT_DIR / backbone / model / "cross_session").glob("run_*/cross_session_summary.json"))
    if not paths:
        raise FileNotFoundError(f"No cross-session summary for {backbone}/{model}")
    with paths[-1].open(encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_metric(backbone: str, model: str, metric: str = "UA") -> float:
    summary = latest_summary(backbone, model)
    return float(summary["aggregate"][metric]["mean"]) * 100.0


def plot_cim_ablation_by_backbone() -> None:
    backbones = {
        "wavlm": "WavLM",
        "wav2vec": "wav2vec 2.0",
        "hubert": "HuBERT",
    }
    ablation_models = [
        ("baseline", "Baseline"),
        ("cim", "CIDM full"),
        ("cim_acoustic_only", "without interaction"),
        ("cim_temporal_only", "without acoustic"),
        ("cim_shuffled_temporal", "shuffled"),
    ]

    for backbone, display_name in backbones.items():
        labels = [label for _, label in ablation_models]
        values = [aggregate_metric(backbone, model, metric="UA") for model, _ in ablation_models]
        plot_ablation(
            labels=labels,
            values=values,
            title=f"{display_name}: CIDM ablation",
            output_name=f"cim_ablation_{backbone}.png",
            ylabel="UA (%)",
        )


def emotion_counts_after_mapping(iemocap_root: Path) -> tuple[dict[str, int], int, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    ignored = 0
    total = 0
    for eval_path in sorted(iemocap_root.glob("Session*/dialog/EmoEvaluation/*.txt")):
        for row in parse_emotion_file(eval_path).values():
            total += 1
            mapped = RAW_LABEL_MAP.get(str(row["raw_label"]).lower())
            if mapped in counts:
                counts[mapped] += 1
            else:
                ignored += 1
    return counts, ignored, total


def plot_emotion_distribution() -> None:
    counts, _, _ = emotion_counts_after_mapping(PROJECT_ROOT / "data" / "iemocap")
    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = ["#d85c5c", "#f2b84b", "#7a8aa0", "#5f80d8"]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.72,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontsize": 8, "color": "#151922", "weight": "bold"},
    )
    for text in autotexts:
        text.set_fontsize(8)

    legend_labels = [f"{label}: {count:,}" for label, count in counts.items()]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=False,
        fontsize=8,
    )
    target_total = sum(counts.values())
    ax.text(0, -1.18, f"Total: {target_total:,} utterances", ha="center", va="center", fontsize=8)
    ax.set(aspect="equal")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "emotion_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_emotion_distribution()
    plot_cim_ablation_by_backbone()


if __name__ == "__main__":
    main()
