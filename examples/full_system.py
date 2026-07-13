"""Run all seven tasks with the full reference configuration and one seed."""

from __future__ import annotations

import json
from pathlib import Path

from hrm.cli import TASKS
from hrm.config import load_config
from hrm.experiments.runner import run_task


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/full_hrm.yaml")
    summaries = {}
    for task in TASKS:
        result = run_task(
            config,
            task,
            config.seed,
            root / "artifacts/examples/full-system" / task,
        )
        summaries[task] = result.summary
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
