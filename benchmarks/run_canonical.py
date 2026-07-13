"""Run a practical canonical task matrix without bypassing production transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hrm.cli import TASKS
from hrm.config import load_config
from hrm.experiments.runner import run_task
from hrm.types import jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="artifacts/benchmark-canonical")
    parser.add_argument("--work-root")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    work_root = Path(args.work_root) if args.work_root else output / "runs"
    summaries = []
    for task in TASKS:
        for seed in range(args.seeds):
            result = run_task(config, task, seed, work_root / task / f"seed-{seed}")
            summaries.append(result.summary)
    rows = [
        {
            "task": summary["task"],
            "seed": summary["seed"],
            "ledger_valid": summary["ledger_valid"],
            "maximum_relative_residual": summary["maximum_relative_residual"],
            "reconstruction_error": summary["final_metrics"].get("reconstruction_error"),
            "prediction_error": summary["final_metrics"].get("prediction_error"),
            "measured_cost": summary["total_measured_cost"],
            "accepted_events": summary["accepted_events"],
            "final_state_digest": summary["final_state_digest"],
            "final_numeric_state_digest": summary["final_numeric_state_digest"],
        }
        for summary in summaries
    ]
    payload = {
        "runs": summaries,
        "all_ledgers_valid": all(summary["ledger_valid"] for summary in summaries),
        "claims_boundary": "short synthetic benchmark; not a validation claim",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    lines = [
        "# Canonical micro-suite",
        "",
        f"Tasks: {len(TASKS)}; seeds per task: {args.seeds}; runs: {len(summaries)}.",
        "",
        f"All ledgers valid: {payload['all_ledgers_valid']}.",
        "",
        "This is a short synthetic calibration run. It establishes execution, activation, attribution, and replay prerequisites; it does not establish external support or broad validation.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "index.json")


if __name__ == "__main__":
    main()
