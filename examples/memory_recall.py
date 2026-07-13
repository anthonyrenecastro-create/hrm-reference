"""Run the canonical associative recall task through the authoritative transition."""

from __future__ import annotations

import json
from pathlib import Path

from hrm.config import load_config
from hrm.experiments.runner import run_task


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/minimal.yaml")
    result = run_task(config, "associative-recall", config.seed, root / "artifacts/examples/recall")
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
