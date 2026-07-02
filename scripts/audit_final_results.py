from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_DIR = ROOT / "reports" / "final_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExpectedRun:
    section: str
    name: str
    result_dir: str
    config: str
    trainer: str
    criteria: dict[str, Any]
    command: str


def nested_get(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def latest_summary(result_dir: Path) -> Path | None:
    summaries = sorted(result_dir.glob("cross_session/run_*/cross_session_summary.json"))
    return summaries[-1] if summaries else None


def read_metrics(summary_path: Path | None) -> dict[str, str]:
    if summary_path is None:
        return {"WA": "", "UA": "", "WF1": "", "Macro-F1": ""}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    output = {}
    for metric in ["WA", "UA", "WF1", "Macro-F1"]:
        item = aggregate.get(metric, {})
        if "mean" in item and "std" in item:
            output[metric] = f"{100 * item['mean']:.2f} ± {100 * item['std']:.2f}"
        else:
            output[metric] = ""
    return output


def read_run_config(result_dir: Path) -> dict[str, Any]:
    candidates = [
        result_dir / "resolved_config.yaml",
        result_dir / "config.yaml",
        result_dir / "config.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def criteria_status(result_dir: Path, expected: ExpectedRun) -> tuple[str, list[str]]:
    summary = latest_summary(result_dir)
    if summary is None:
        return "missing", ["no cross_session_summary.json"]

    cfg = read_run_config(result_dir)
    if not cfg:
        return "available_unverified", ["missing resolved_config.yaml/config.json for fairness check"]

    problems: list[str] = []
    for key, wanted in expected.criteria.items():
        got = nested_get(cfg, key)
        if key == "model.memory_ablation_mode" and wanted == "normal" and got is None:
            continue
        if got != wanted:
            problems.append(f"{key}: expected {wanted!r}, got {got!r}")
    if problems:
        return "needs_rerun", problems
    return "available_fair", []


def expected_runs() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []

    def add(section: str, name: str, result_dir: str, config: str, trainer: str, criteria: dict[str, Any]) -> None:
        runs.append(
            ExpectedRun(
                section=section,
                name=name,
                result_dir=result_dir,
                config=config,
                trainer=trainer,
                criteria=criteria,
                command=f"python -m {trainer} --config {config}",
            )
        )

    common = {
        "cross_session.enabled": True,
        "cross_session.test_sessions": [1, 2, 3, 4, 5],
        "model.freeze_wavlm": True,
        "model.unfreeze_last_n_layers": 0,
        "precompute.enabled": True,
    }
    cdm_cim4 = {
        **common,
        "model.temporal_feature_set": "interaction_4",
        "model.temporal_feature_dim": 4,
        "model.fusion_mode": "branch_sum",
        "model.temporal_input_mode": "real",
    }
    cim4 = {
        **common,
        "model.temporal_feature_set": "interaction_4",
        "model.temporal_feature_dim": 4,
        "model.fusion_mode": "temporal_residual_sum",
        "model.temporal_input_mode": "real",
    }

    for backbone, model_name in [
        ("wavlm", "microsoft/wavlm-base"),
        ("wav2vec", "facebook/wav2vec2-base"),
        ("hubert", "facebook/hubert-base-ls960"),
    ]:
        add("main", f"{backbone} / Baseline", f"results/main/{backbone}/baseline", f"configs/main_{backbone}_baseline.yaml", "scripts.train_wavlm_baseline", {**common, "model.wavlm_model_name": model_name})
        add("main", f"{backbone} / CDM", f"results/main/{backbone}/cdm", f"configs/main_{backbone}_cdm.yaml", "scripts.train_wavlm_cdm", {**common, "model.wavlm_model_name": model_name, "model.memory_ablation_mode": "normal"})
        add("main", f"{backbone} / CIM", f"results/main/{backbone}/cim", f"configs/main_{backbone}_cim.yaml", "scripts.train_dual_branch", {**cim4, "model.wavlm_model_name": model_name})
        add("main", f"{backbone} / CDM + CIM", f"results/main/{backbone}/cdm_cim", f"configs/main_{backbone}_cdm_cim.yaml", "scripts.train_dual_branch", {**cdm_cim4, "model.wavlm_model_name": model_name})

    add("cdm_ablation", "CDM zero", "results/cdm_ablation/zero", "configs/cdm_ablation_zero.yaml", "scripts.train_wavlm_cdm", {**common, "model.memory_ablation_mode": "zero_state"})
    add("cdm_ablation", "CDM shuffled", "results/cdm_ablation/shuffled", "configs/cdm_ablation_shuffled.yaml", "scripts.train_wavlm_cdm", {**common, "model.memory_ablation_mode": "shuffled_order"})
    add("cdm_ablation", "CDM full", "results/cdm_ablation/full", "configs/cdm_ablation_full.yaml", "scripts.train_wavlm_cdm", {**common, "model.memory_ablation_mode": "normal"})

    add("cim_ablation", "CIM zero", "results/cim_ablation/zero", "configs/cim_ablation_zero.yaml", "scripts.train_dual_branch", {**cim4, "model.temporal_input_mode": "zero"})
    add("cim_ablation", "CIM shuffled", "results/cim_ablation/shuffled", "configs/cim_ablation_shuffled.yaml", "scripts.train_dual_branch", {**cim4, "model.temporal_input_mode": "shuffled"})
    for slug, feature in [
        ("no_overlap_ratio", "overlap_ratio"),
        ("no_relative_gap", "relative_gap_to_speaker_mean"),
        ("no_speaker_switch", "speaker_switch"),
        ("no_speaker_overlap_rate", "speaker_prev_overlap_rate"),
    ]:
        add("cim_ablation", f"CIM w/o {feature}", f"results/cim_ablation/{slug}", f"configs/cim_ablation_{slug}.yaml", "scripts.train_dual_branch", {**cim4, "model.disabled_temporal_features": [feature]})

    add("architecture_ablation", "Residual + gate", "results/architecture_ablation/residual_gate", "configs/architecture_residual_gate.yaml", "scripts.train_dual_branch", {**cdm_cim4, "model.fusion_mode": "residual_gated"})
    add("architecture_ablation", "Residual + sum", "results/architecture_ablation/residual_sum", "configs/architecture_residual_sum.yaml", "scripts.train_dual_branch", {**cdm_cim4, "model.fusion_mode": "residual_sum"})
    add("architecture_ablation", "Branch + sum", "results/architecture_ablation/branch_sum", "configs/architecture_branch_sum.yaml", "scripts.train_dual_branch", cdm_cim4)
    add("architecture_ablation", "Branch + concat", "results/architecture_ablation/branch_concat", "configs/architecture_branch_concat.yaml", "scripts.train_dual_branch", {**cdm_cim4, "model.fusion_mode": "branch_concat"})

    add("training_strategy_ablation", "CDM -> CIM", "results/training_strategy_ablation/cdm_then_cim", "configs/training_strategy_cdm_then_cim.yaml", "scripts.train_dual_branch", {**cdm_cim4, "training_stage.mode": "3_phase"})
    add("training_strategy_ablation", "CIM -> CDM", "results/training_strategy_ablation/cim_then_cdm", "configs/training_strategy_cim_then_cdm.yaml", "scripts.train_dual_branch", {**cdm_cim4, "training_stage.mode": "temporal_first_3_phase"})

    for slug, feature_set, dim in [
        ("feature_4", "interaction_4", 4),
        ("feature_12", "selected_primitives", 12),
        ("feature_16", "v1", 16),
        ("feature_36", "recommended_v2", 36),
    ]:
        add("feature_ablation", slug, f"results/feature_ablation/{slug}", f"configs/feature_ablation_{slug}.yaml", "scripts.train_dual_branch", {**common, "model.fusion_mode": "branch_sum", "model.temporal_feature_set": feature_set, "model.temporal_feature_dim": dim, "model.temporal_input_mode": "real"})

    return runs


def make_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Final Result Inventory", ""]
    for section in ["main", "cdm_ablation", "cim_ablation", "architecture_ablation", "training_strategy_ablation", "feature_ablation"]:
        lines.extend([f"## {section}", "", "| Run | Status | WA | UA | WF1 | Macro-F1 | Reason | Command |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for row in rows:
            if row["section"] != section:
                continue
            command = f"`{row['command']}`" if row["status"] != "available_fair" else ""
            reason = row.get("problems", "")
            lines.append(
                f"| {row['name']} | {row['status']} | {row['WA']} | {row['UA']} | {row['WF1']} | {row['Macro-F1']} | {reason} | {command} |"
            )
        lines.append("")

    needs = [row for row in rows if row["status"] != "available_fair"]
    lines.extend(["## Commands To Rerun", ""])
    for row in needs:
        lines.append(f"- {row['name']}: `{row['command']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for expected in expected_runs():
        result_dir = ROOT / expected.result_dir
        summary = latest_summary(result_dir)
        status, problems = criteria_status(result_dir, expected)
        row = {
            "section": expected.section,
            "name": expected.name,
            "result_dir": expected.result_dir,
            "config": expected.config,
            "trainer": expected.trainer,
            "command": expected.command,
            "status": status,
            "problems": "; ".join(problems),
            **read_metrics(summary),
        }
        rows.append(row)

    (REPORT_DIR / "inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "inventory.md").write_text(make_markdown(rows), encoding="utf-8")
    print(f"Wrote {REPORT_DIR / 'inventory.md'}")


if __name__ == "__main__":
    main()
