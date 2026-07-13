"""Run summaries and optional diagnostic plots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def write_summary_markdown(summary: dict[str, Any], path: str | Path) -> None:
    gates = summary.get("mechanism_gates", {})
    lines = [
        f"# HRM run: {summary['run_id']}",
        "",
        f"- Task: `{summary['task']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Steps: `{summary['steps']}`",
        f"- Maximum relative ledger residual: `{summary['maximum_relative_residual']:.3e}`",
        f"- Final state digest: `{summary['final_state_digest']}`",
        "",
        "## Claims boundary",
        "",
        "This run demonstrates implementation behavior on a synthetic task. It is not external validation or evidence of general intelligence.",
        "",
        "## Mechanism gates",
        "",
        "| Mechanism | Executed | Activated | Divergence | Utility |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, gate in gates.items():
        lines.append(
            f"| {name} | {gate.get('G1_execution')} | {gate.get('G2_activation')} | {gate.get('G3_divergence')} | {gate.get('G4_utility')} |"
        )
    lines.extend(
        [
            "",
            "## Final metrics",
            "",
            "```json",
            json.dumps(summary.get("final_metrics", {}), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_metric_plot(metrics: pd.DataFrame, path: str | Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".mplconfig"))
    import matplotlib.pyplot as plt

    columns = [
        name
        for name in ("reconstruction_error", "prediction_error", "field_variance", "current_cost")
        if name in metrics and metrics[name].notna().any()
    ]
    if not columns:
        return
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, 2.4 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns, strict=True):
        axis.plot(metrics["step"], metrics[column], marker="o", linewidth=1.2)
        axis.set_ylabel(column)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
