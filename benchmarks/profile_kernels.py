"""Profile the authoritative transition on the minimal reconstruction task."""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path

from hrm.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--output", default="artifacts/profile/hrm.prof")
    args = parser.parse_args()
    config = load_config(args.config)
    profile_path = Path(args.output)
    run_path = profile_path.parent / "run"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    cProfile.runctx(
        "run_task(config, 'spatial-reconstruction', config.seed, run_path)",
        globals(),
        locals(),
        str(profile_path),
    )
    print(profile_path)


if __name__ == "__main__":
    main()
