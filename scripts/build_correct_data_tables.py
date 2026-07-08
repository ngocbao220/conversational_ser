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
    "wav2vec": "Wav2Vec",
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "cdm": "CDM",
    "cim": "CIM",
    "cdm_cim": "CDM+CIM",
    "cdm_cim_fusion_head": "Fusion Head",
    "cdm_cim_logit_fusion": "Logit Fusion",
    "cim_acoustic_only": "CIM Acoustic-only",
    "cim_zero_temporal": "CIM Zero Temporal",
    "cim_temporal_only": "CIM Temporal-only",
    "cim_shuffled_temporal": "CIM Shuffled Temporal",
    "zero_dialogue_memory": "Zero Dialogue Memory",
    "zero_interaction_memory": "Zero Interaction Memory",
    "zero_both_memory": "Zero Both Memory",
    "shuffled_dialogue_memory": "Shuffled Dialogue Memory",
    "shuffled_interaction_memory": "Shuffled Interaction Memory",
}

MODEL_ORDER = {
    "baseline": 0,
    "cdm": 1,
    "cim": 2,
    "cdm_cim": 3,
    "cdm_cim_fusion_head": 4,
    "cdm_cim_logit_fusion": 5,
    "cim_acoustic_only": 10,
    "cim_zero_temporal": 11,
    "cim_shuffled_temporal": 12,
    "cim_temporal_only": 13,
    "zero_both_memory": 20,
    "zero_dialogue_memory": 21,
    "zero_interaction_memory": 22,
    "shuffled_dialogue_memory": 23,
    "shuffled_interaction_memory": 24,
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
    grouped: dict[tuple[str, str, str], dict[int, tuple[Path, dict[str, float]]]] = defaultdict(dict)

    for metrics_path in root.rglob("metrics.json"):
        if "phase_tests" in metrics_path.parts:
            continue

        parts = metrics_path.parts
        try:
            idx = parts.index("correct_data")
        except ValueError:
            continue
        rel = parts[idx + 1 :]

        if len(rel) >= 6 and rel[0] == "features" and rel[1] == "4new":
            category, backbone, model = "features/4new", rel[2], rel[3]
        elif len(rel) >= 5 and rel[0] == "main":
            category, backbone, model = "main", rel[1], rel[2]
        elif len(rel) >= 5 and rel[0] == "cim_cdm_ablation":
            category, backbone, model = "cim_cdm_ablation", rel[1], rel[2]
        else:
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
            raw = json.loads(metrics_path.read_text())
        except Exception:
            continue
        data = raw.get("test") if isinstance(raw.get("test"), dict) else raw

        values = {name: metric_value(data, name) for name in METRICS}
        if any(value is None for value in values.values()):
            continue

        grouped[(category, backbone, model)][session] = (metrics_path, values)  # latest path wins per session

    rows = []
    for (category, backbone, model), by_session in grouped.items():
        sessions = tuple(sorted(by_session))
        mean_metrics = {
            name: sum(by_session[session][1][name] for session in sessions) / len(sessions) * 100.0
            for name in METRICS
        }
        rows.append(ResultRow(category, backbone, model, len(sessions), sessions, mean_metrics))

    return sorted(rows, key=lambda row: (row.category, row.backbone, MODEL_ORDER.get(row.model, 99), row.model))


def row_dict(row: ResultRow) -> dict[str, str]:
    return {
        "Backbone": BACKBONE_LABELS.get(row.backbone, row.backbone),
        "Model": MODEL_LABELS.get(row.model, row.model),
        "Folds": f"{row.folds}/5" if row.folds != 5 else "5/5",
        "Sessions": ",".join(f"Ses{session:02d}" for session in row.sessions),
        **{name: f"{row.metrics[name]:.2f}" for name in METRICS},
    }


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    fields = ("Backbone", "Model", "Folds", "Sessions", *METRICS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_dict(row))


def markdown_value(row: ResultRow, metric: str, best: dict[str, float]) -> str:
    value = f"{row.metrics[metric]:.2f}"
    if abs(row.metrics[metric] - best[metric]) < 1e-9:
        return f"**{value}**"
    return value


def write_markdown(path: Path, title: str, rows: list[ResultRow], include_sessions: bool = False) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo results found.\n")
        return

    best = {metric: max(row.metrics[metric] for row in rows if row.folds == 5) for metric in METRICS}
    headers = ["Backbone", "Model", "Folds"]
    if include_sessions:
        headers.append("Sessions")
    headers.extend(METRICS)

    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        item = row_dict(row)
        values = [item["Backbone"], item["Model"], item["Folds"]]
        if include_sessions:
            values.append(item["Sessions"])
        values.extend(markdown_value(row, metric, best) for metric in METRICS)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def latex_value(row: ResultRow, metric: str, best: dict[str, float]) -> str:
    value = f"{row.metrics[metric]:.2f}"
    if abs(row.metrics[metric] - best[metric]) < 1e-9:
        return rf"\textbf{{{value}}}"
    return value


def write_latex(path: Path, rows: list[ResultRow], caption: str, label: str) -> None:
    if not rows:
        path.write_text("")
        return

    best = {metric: max(row.metrics[metric] for row in rows if row.folds == 5) for metric in METRICS}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Backbone & Model & WA & UA & WF1 & Macro-F1 \\",
        r"\midrule",
    ]
    for row in rows:
        backbone = BACKBONE_LABELS.get(row.backbone, row.backbone)
        model = MODEL_LABELS.get(row.model, row.model)
        metric_text = " & ".join(latex_value(row, metric, best) for metric in METRICS)
        fold_note = "" if row.folds == 5 else rf" ({row.folds}/5)"
        lines.append(rf"{backbone} & {model}{fold_note} & {metric_text} \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))



def write_publication_png(path: Path, rows: list[ResultRow], title: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {path.name}: matplotlib unavailable ({exc})")
        return

    if not rows:
        return

    headers = ["Backbone", "Model", *METRICS]
    body = [
        [
            BACKBONE_LABELS.get(row.backbone, row.backbone),
            MODEL_LABELS.get(row.model, row.model) + ("" if row.folds == 5 else f" ({row.folds}/5)"),
            *[f"{row.metrics[metric]:.2f}" for metric in METRICS],
        ]
        for row in rows
    ]
    best = {metric: max(row.metrics[metric] for row in rows if row.folds == 5) for metric in METRICS}

    fig_height = max(1.8, 0.34 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    table = ax.table(
        cellText=body,
        colLabels=headers,
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
            cell.set_linewidth(1.0)
        elif col_idx >= 2:
            result = rows[row_idx - 1]
            metric = METRICS[col_idx - 2]
            if result.folds == 5 and abs(result.metrics[metric] - best[metric]) < 1e-9:
                cell.set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)



def ranks_by_backbone(rows: list[ResultRow]) -> dict[str, dict[str, dict[str, float | None]]]:
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for backbone in sorted({row.backbone for row in rows}):
        candidates = [row for row in rows if row.backbone == backbone and row.folds == 5]
        if not candidates:
            continue
        result[backbone] = {}
        for metric in METRICS:
            values = sorted({row.metrics[metric] for row in candidates}, reverse=True)
            result[backbone][metric] = {
                "best": values[0] if values else None,
                "second": values[1] if len(values) > 1 else None,
            }
    return result


def rank_for(row: ResultRow, metric: str, ranks: dict[str, dict[str, dict[str, float | None]]]) -> str | None:
    if row.folds != 5:
        return None
    metric_ranks = ranks.get(row.backbone, {}).get(metric, {})
    value = row.metrics[metric]
    best = metric_ranks.get("best")
    second = metric_ranks.get("second")
    if best is not None and abs(value - best) < 1e-9:
        return "best"
    if second is not None and abs(value - second) < 1e-9:
        return "second"
    return None


def grouped_markdown_value(row: ResultRow, metric: str, ranks: dict[str, dict[str, dict[str, float | None]]]) -> str:
    value = f"{row.metrics[metric]:.2f}"
    rank = rank_for(row, metric, ranks)
    if rank == "best":
        return f"**{value}**"
    if rank == "second":
        return f"<u>{value}</u>"
    return value


def write_grouped_markdown(path: Path, title: str, rows: list[ResultRow]) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo results found.\n")
        return

    ranks = ranks_by_backbone(rows)
    headers = ["Backbone", "Model", "Folds", *METRICS]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    previous_backbone = None
    for row in rows:
        if previous_backbone is not None and row.backbone != previous_backbone:
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        item = row_dict(row)
        values = [item["Backbone"], item["Model"], item["Folds"]]
        values.extend(grouped_markdown_value(row, metric, ranks) for metric in METRICS)
        lines.append("| " + " | ".join(values) + " |")
        previous_backbone = row.backbone
    path.write_text("\n".join(lines) + "\n")


def grouped_latex_value(row: ResultRow, metric: str, ranks: dict[str, dict[str, dict[str, float | None]]]) -> str:
    value = f"{row.metrics[metric]:.2f}"
    rank = rank_for(row, metric, ranks)
    if rank == "best":
        return rf"\textbf{{{value}}}"
    if rank == "second":
        return rf"\underline{{{value}}}"
    return value


def write_grouped_latex(path: Path, rows: list[ResultRow], caption: str, label: str) -> None:
    if not rows:
        path.write_text("")
        return

    ranks = ranks_by_backbone(rows)
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
        backbone = BACKBONE_LABELS.get(row.backbone, row.backbone)
        model = MODEL_LABELS.get(row.model, row.model)
        metric_text = " & ".join(grouped_latex_value(row, metric, ranks) for metric in METRICS)
        fold_note = "" if row.folds == 5 else rf" ({row.folds}/5)"
        lines.append(rf"{backbone} & {model}{fold_note} & {metric_text} \\")
        previous_backbone = row.backbone
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_grouped_publication_png(path: Path, rows: list[ResultRow], title: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping {path.name}: matplotlib unavailable ({exc})")
        return

    if not rows:
        return

    ranks = ranks_by_backbone(rows)
    headers = ["Backbone", "Model", *METRICS]
    body = []
    for row in rows:
        body.append([
            BACKBONE_LABELS.get(row.backbone, row.backbone),
            MODEL_LABELS.get(row.model, row.model) + ("" if row.folds == 5 else f" ({row.folds}/5)"),
            *[f"{row.metrics[metric]:.2f}" for metric in METRICS],
        ])

    fig_height = max(1.8, 0.34 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    table = ax.table(
        cellText=body,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18, 0.31, 0.12, 0.12, 0.12, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.35)

    group_starts = {1}
    for idx in range(1, len(rows)):
        if rows[idx].backbone != rows[idx - 1].backbone:
            group_starts.add(idx + 1)

    second_cells = []
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2 if row_idx in group_starts or row_idx == 0 else 0.6)
        if row_idx == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f2f2f2")
        elif col_idx >= 2:
            result = rows[row_idx - 1]
            metric = METRICS[col_idx - 2]
            rank = rank_for(result, metric, ranks)
            if rank == "best":
                cell.set_text_props(fontweight="bold")
            elif rank == "second":
                second_cells.append(cell)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    for cell in second_cells:
        bbox = cell.get_window_extent(renderer)
        (x0, y0) = inv.transform((bbox.x0, bbox.y0))
        (x1, y1) = inv.transform((bbox.x1, bbox.y1))
        y = y0 + (y1 - y0) * 0.28
        pad = (x1 - x0) * 0.30
        ax.add_line(plt.Line2D([x0 + pad, x1 - pad], [y, y], transform=ax.transAxes, color="black", linewidth=1.1))

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def emit_grouped_table(out_dir: Path, stem: str, title: str, rows: list[ResultRow], caption: str, label: str) -> None:
    write_csv(out_dir / f"{stem}.csv", rows)
    write_grouped_markdown(out_dir / f"{stem}.md", title, rows)
    write_grouped_latex(out_dir / f"{stem}.tex", rows, caption, label)
    write_grouped_publication_png(out_dir / f"{stem}.png", rows, title)

def emit_table(out_dir: Path, stem: str, title: str, rows: list[ResultRow], caption: str, label: str) -> None:
    write_csv(out_dir / f"{stem}.csv", rows)
    write_markdown(out_dir / f"{stem}.md", title, rows, include_sessions=any(row.folds != 5 for row in rows))
    write_latex(out_dir / f"{stem}.tex", rows, caption, label)
    write_publication_png(out_dir / f"{stem}.png", rows, title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/correct_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/correct_data/tables"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_metrics(args.root)

    main_rows = [row for row in rows if row.category == "main"]
    new_rows = [row for row in rows if row.category == "features/4new" and not row.model.startswith("cim_")]
    cim_ablation_rows = [row for row in rows if row.category == "features/4new" and row.model in {
        "baseline",
        "cdm",
        "cim",
        "cim_acoustic_only",
        "cim_zero_temporal",
        "cim_shuffled_temporal",
        "cim_temporal_only",
    }]
    memory_ablation_rows = [
        row
        for row in rows
        if (row.category == "main" and row.backbone == "wavlm" and row.model in {"baseline", "cdm_cim"})
        or row.category == "cim_cdm_ablation"
    ]
    memory_order = {
        "baseline": 0,
        "cdm_cim": 1,
        "zero_both_memory": 2,
        "zero_dialogue_memory": 3,
        "zero_interaction_memory": 4,
        "shuffled_dialogue_memory": 5,
        "shuffled_interaction_memory": 6,
    }
    memory_ablation_rows = sorted(memory_ablation_rows, key=lambda row: memory_order.get(row.model, 99))

    emit_table(
        args.out_dir,
        "old_feature_results",
        "Old Feature Results",
        main_rows,
        "Cross-session test results using the old temporal feature set.",
        "tab:old-feature-results",
    )
    emit_table(
        args.out_dir,
        "new_4feature_results",
        "New 4-Feature Results",
        new_rows,
        "Cross-session test results using the new four temporal features.",
        "tab:new-4feature-results",
    )
    emit_grouped_table(
        args.out_dir,
        "cim_4feature_ablation",
        "CIM 4-Feature Ablation",
        cim_ablation_rows,
        "CIM ablation results using the new four temporal features.",
        "tab:cim-4feature-ablation",
    )
    emit_table(
        args.out_dir,
        "cdm_cim_memory_ablation",
        "CDM+CIM Memory Ablation",
        memory_ablation_rows,
        "Memory ablation results for the CDM+CIM model.",
        "tab:cdm-cim-memory-ablation",
    )

    wavlm_old_new = [
        row
        for row in rows
        if row.backbone == "wavlm"
        and row.model in {"baseline", "cdm", "cim", "cdm_cim", "cdm_cim_logit_fusion"}
        and row.category in {"main", "features/4new"}
    ]
    write_csv(args.out_dir / "old_vs_new_wavlm.csv", wavlm_old_new)

    print(f"Wrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()
