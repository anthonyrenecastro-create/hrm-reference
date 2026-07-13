"""Matched configuration-level ablation suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hrm.config import HRMConfig, apply_ablation
from hrm.experiments.runner import run_task
from hrm.experiments.statistics import benjamini_hochberg, holm, paired_statistics
from hrm.types import jsonable

CANONICAL_ABLATIONS = [
    "no_guidance",
    "no_hrm_adaptation",
    "fixed_topology",
    "fixed_geometry",
    "fixed_graph",
    "no_hierarchy",
    "memory_only",
    "fixed_allocation",
    "no_memory",
    "no_cognition_to_field",
    "no_field_to_cognition",
    "no_regional_coupling",
    "no_topology_repair",
    "no_adaptive_timestep",
]


def _configuration_diff(full: HRMConfig, ablated: HRMConfig) -> dict[str, dict[str, Any]]:
    before = full.model_dump(mode="json")
    after = ablated.model_dump(mode="json")
    result: dict[str, dict[str, Any]] = {}

    def visit(path: str, left: Any, right: Any) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                visit(f"{path}.{key}" if path else key, left.get(key), right.get(key))
        elif left != right:
            result[path] = {"full": left, "ablated": right}

    visit("", before, after)
    return result


def _outcome(summary: dict[str, Any]) -> float:
    metrics = summary["final_metrics"]
    reconstruction = metrics.get("reconstruction_error")
    if reconstruction is not None and str(reconstruction) != "nan":
        return float(reconstruction)
    return float(metrics["prediction_error"])


def run_ablation_suite(
    config: HRMConfig,
    task: str,
    seeds: list[int],
    output_dir: str | Path,
    ablations: list[str] | None = None,
    *,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs = Path(run_root) if run_root is not None else root
    ablations = ablations or CANONICAL_ABLATIONS
    full_results = {
        seed: run_task(
            config, task, seed, runs / "full" / f"seed-{seed}", run_id=f"{task}-full-{seed}"
        )
        for seed in seeds
    }
    comparisons: list[dict[str, Any]] = []
    stats_records: list[dict[str, Any]] = []
    configuration_differences: dict[str, dict[str, dict[str, Any]]] = {}
    for ablation in ablations:
        ablated_config = apply_ablation(config, ablation)
        configuration_differences[ablation] = _configuration_diff(config, ablated_config)
        full_values: list[float] = []
        ablated_values: list[float] = []
        for seed in seeds:
            result = run_task(
                ablated_config,
                task,
                seed,
                runs / ablation / f"seed-{seed}",
                run_id=f"{task}-{ablation}-{seed}",
            )
            full = full_results[seed]
            full_outcome, ablated_outcome = _outcome(full.summary), _outcome(result.summary)
            full_cost, ablated_cost = (
                full.summary["total_measured_cost"],
                result.summary["total_measured_cost"],
            )
            cost_ratio = ablated_cost / (full_cost + 1e-12)
            diverged = (
                full.summary["final_numeric_state_digest"]
                != result.summary["final_numeric_state_digest"]
            )
            comparisons.append(
                {
                    "seed": seed,
                    "ablation": ablation,
                    "full_outcome": full_outcome,
                    "ablated_outcome": ablated_outcome,
                    "utility_effect": ablated_outcome - full_outcome,
                    "full_cost": full_cost,
                    "ablated_cost": ablated_cost,
                    "cost_ratio": cost_ratio,
                    "cost_equal_within_5pct": abs(cost_ratio - 1.0) <= 0.05,
                    "state_diverged": diverged,
                    "full_ledger_valid": full.summary["ledger_valid"],
                    "ablated_ledger_valid": result.summary["ledger_valid"],
                }
            )
            full_values.append(-full_outcome)
            ablated_values.append(-ablated_outcome)
        stats = paired_statistics(
            pd.Series(full_values).to_numpy(),
            pd.Series(ablated_values).to_numpy(),
            seed=config.seed,
        )
        stats_records.append({"ablation": ablation, **stats})
    p_values = [item["wilcoxon_p"] for item in stats_records]
    holm_values, bh_values = holm(p_values), benjamini_hochberg(p_values)
    for item, h, b in zip(stats_records, holm_values, bh_values, strict=True):
        item["holm_p"] = h
        item["benjamini_hochberg_p"] = b
    frame = pd.DataFrame(comparisons)
    frame.to_csv(root / "results.csv", index=False)
    payload = {
        "task": task,
        "seeds": seeds,
        "ablations": ablations,
        "comparisons": comparisons,
        "statistics": stats_records,
        "configuration_differences": configuration_differences,
        "claims_boundary": "Synthetic matched comparisons only; utility requires activation, divergence, valid ledgers, and comparable cost.",
    }
    (root / "results.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "config_diffs.json").write_text(
        json.dumps(jsonable(configuration_differences), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        f"# Canonical HRM ablations — {task}",
        "",
        f"Seeds: {len(seeds)}",
        "",
        "| Ablation | Mean utility effect | 95% CI | Wilcoxon p | Holm p | Diverged seeds | Cost-equal seeds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in stats_records:
        rows = frame[frame.ablation == item["ablation"]]
        equal = int(rows.cost_equal_within_5pct.sum())
        diverged = int(rows.state_diverged.sum())
        lines.append(
            f"| {item['ablation']} | {item['paired_mean_difference']:.6g} | [{item['bootstrap_ci_95'][0]:.6g}, {item['bootstrap_ci_95'][1]:.6g}] | {item['wilcoxon_p']:.4g} | {item['holm_p']:.4g} | {diverged}/{len(rows)} | {equal}/{len(rows)} |"
        )
    lines.extend(
        [
            "",
            "Positive utility means the full system had lower error. Statistical results do not imply support when activation, divergence, cost, or ledger gates fail.",
            "",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
