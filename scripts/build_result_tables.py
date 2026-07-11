from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


METRICS = ("WA", "UA", "WF1", "Macro-F1")

BACKBONE_LABELS = {
    "wavlm": "WavLM",
    "hubert": "HuBERT",
    "wav2vec": "Wav2Vec2",
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "cdim": "CDIM",
}

MODEL_ORDER = {
    "baseline": 0,
    "cdim": 1,
}


@dataclass(frozen=True)
class ResultRow:
    category: str
    backbone: str
    model: str
    folds: int
    sessions: tuple[int, ...]
    metrics: dict[str, float]


def metric_value(data: dict, name: str) -> float | None:
    names = (name, f"test_{name}", f"test/{name}", name.replace("-", "_"), f"test_{name.replace('-', '_')}")
    for candidate in names:
        value = data.get(candidate)
        if value is not None:
            return float(value)
    return None


def parse_metrics(root: Path) -> list[ResultRow]:
    grouped: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)

    for metrics_path in root.rglob("metrics.json"):
        if "phase_tests" in metrics_path.parts:
            continue
        try:
            rel = metrics_path.relative_to(root).parts
        except ValueError:
            continue
        if len(rel) < 4 or rel[0] != "main":
            continue

        category, backbone, model = "main", rel[1], rel[2]
        if model not in MODEL_LABELS:
            continue

        session = None
        for part in rel:
            match = re.fullmatch(r"test_Ses(\d+)", part)
            if match:
                session = int(match.group(1))
                break
        if session is None:
            continue

        try:
            raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data = raw.get("test") if isinstance(raw.get("test"), dict) else raw
        values = {name: metric_value(data, name) for name in METRICS}
        if any(value is None for value in values.values()):
            continue
        grouped[(category, backbone, model)][session] = {name: float(values[name]) for name in METRICS}

    rows: list[ResultRow] = []
    for (category, backbone, model), by_session in grouped.items():
        sessions = tuple(sorted(by_session))
        mean_metrics = {
            name: sum(by_session[session][name] for session in sessions) / len(sessions) * 100.0
            for name in METRICS
        }
        rows.append(ResultRow(category, backbone, model, len(sessions), sessions, mean_metrics))

    return sorted(rows, key=lambda row: (row.backbone, MODEL_ORDER.get(row.model, 99), row.model))


def row_dict(row: ResultRow) -> dict[str, str]:
    return {
        "Backbone": BACKBONE_LABELS.get(row.backbone, row.backbone),
        "Model": MODEL_LABELS.get(row.model, row.model),
        "Folds": f"{row.folds}/5",
        "Sessions": ",".join(f"Ses{session:02d}" for session in row.sessions),
        **{name: f"{row.metrics[name]:.2f}" for name in METRICS},
    }


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    fields = ("Backbone", "Model", "Folds", "Sessions", *METRICS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_dict(row))


def best_by_backbone(rows: list[ResultRow]) -> dict[str, dict[str, float]]:
    best: dict[str, dict[str, float]] = {}
    for backbone in sorted({row.backbone for row in rows}):
        candidates = [row for row in rows if row.backbone == backbone and row.folds == 5]
        if candidates:
            best[backbone] = {metric: max(row.metrics[metric] for row in candidates) for metric in METRICS}
    return best


def marked_value(row: ResultRow, metric: str, best: dict[str, dict[str, float]]) -> str:
    value = f"{row.metrics[metric]:.2f}"
    if row.folds == 5 and abs(row.metrics[metric] - best.get(row.backbone, {}).get(metric, float('nan'))) < 1e-9:
        return f"**{value}**"
    return value


def write_markdown(path: Path, title: str, rows: list[ResultRow]) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo results found.\n", encoding="utf-8")
        return

    best = best_by_backbone(rows)
    headers = ["Backbone", "Model", "Folds", *METRICS]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    previous_backbone = None
    for row in rows:
        if previous_backbone is not None and row.backbone != previous_backbone:
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        item = row_dict(row)
        values = [item["Backbone"], item["Model"], item["Folds"]]
        values.extend(marked_value(row, metric, best) for metric in METRICS)
        lines.append("| " + " | ".join(values) + " |")
        previous_backbone = row.backbone
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_value(row: ResultRow, metric: str, best: dict[str, dict[str, float]]) -> str:
    value = f"{row.metrics[metric]:.2f}"
    if row.folds == 5 and abs(row.metrics[metric] - best.get(row.backbone, {}).get(metric, float('nan'))) < 1e-9:
        return rf"\textbf{{{value}}}"
    return value


def write_latex(path: Path, rows: list[ResultRow], caption: str, label: str) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    best = best_by_backbone(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Backbone & Model & WA & UA & WF1 & Macro-F1 \\",
        r"\midrule",
    ]
    previous_backbone = None
    for row in rows:
        if previous_backbone is not None and row.backbone != previous_backbone:
            lines.append(r"\midrule")
        metric_text = " & ".join(latex_value(row, metric, best) for metric in METRICS)
        fold_note = "" if row.folds == 5 else rf" ({row.folds}/5)"
        lines.append(rf"{BACKBONE_LABELS.get(row.backbone, row.backbone)} & {MODEL_LABELS.get(row.model, row.model)}{fold_note} & {metric_text} \\")
        previous_backbone = row.backbone
    lines.extend([r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_png(path: Path, rows: list[ResultRow], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {path.name}: matplotlib unavailable ({exc})")
        return
    if not rows:
        return

    body = [
        [
            BACKBONE_LABELS.get(row.backbone, row.backbone),
            MODEL_LABELS.get(row.model, row.model) + ("" if row.folds == 5 else f" ({row.folds}/5)"),
            *[f"{row.metrics[metric]:.2f}" for metric in METRICS],
        ]
        for row in rows
    ]
    best = best_by_backbone(rows)

    fig_height = max(1.8, 0.34 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    table = ax.table(
        cellText=body,
        colLabels=["Backbone", "Model", *METRICS],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18, 0.31, 0.12, 0.12, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f2f2f2")
        elif col_idx >= 2:
            result = rows[row_idx - 1]
            metric = METRICS[col_idx - 2]
            if result.folds == 5 and abs(result.metrics[metric] - best.get(result.backbone, {}).get(metric, float('nan'))) < 1e-9:
                cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def emit_table(out_dir: Path, stem: str, title: str, rows: list[ResultRow], caption: str, label: str) -> None:
    write_csv(out_dir / f"{stem}.csv", rows)
    write_markdown(out_dir / f"{stem}.md", title, rows)
    write_latex(out_dir / f"{stem}.tex", rows, caption, label)
    write_png(out_dir / f"{stem}.png", rows, title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSV, Markdown, LaTeX, and PNG result tables.")
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_metrics(args.root)
    emit_table(
        args.out_dir,
        "main_results",
        "Main Results",
        rows,
        "Cross-session test results.",
        "tab:main-results",
    )
    print(f"Wrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()
