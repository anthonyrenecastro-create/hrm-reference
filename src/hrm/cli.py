"""Command-line interface for runs, experiments, ledgers, replay, and reports."""

from __future__ import annotations

import argparse
import cProfile
import json
import sys
from pathlib import Path

import pandas as pd

from hrm.config import HRMConfig, load_config
from hrm.core.ledger import verify_ledger_file
from hrm.experiments.ablation import run_ablation_suite
from hrm.experiments.replay import replay_run
from hrm.experiments.runner import run_task
from hrm.types import jsonable

TASKS = [
    "spatial-reconstruction",
    "sequence-prediction",
    "associative-recall",
    "perturbation-recovery",
    "resource-adaptation",
    "topology-usefulness",
    "topology-irrelevance",
]


def _config(path: str | None) -> HRMConfig:
    return load_config(path) if path else HRMConfig()


def _run(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = (
        Path(args.output)
        if args.output
        else Path(config.output_root) / (args.run_id or f"{args.task}-seed-{config.seed}")
    )
    result = run_task(config, args.task, config.seed, output, run_id=args.run_id)
    print(json.dumps(jsonable(result.summary), indent=2, sort_keys=True))
    return 0


def _experiment(args: argparse.Namespace) -> int:
    config = _config(args.config)
    root = Path(args.output or config.output_root) / f"experiment-{args.task}"
    summaries = []
    for seed in range(args.seeds):
        result = run_task(config, args.task, seed, root / f"seed-{seed}")
        summaries.append(result.summary)
    rows = [
        {
            "seed": item["seed"],
            "outcome": item["final_metrics"].get(
                "reconstruction_error", item["final_metrics"].get("prediction_error")
            ),
            "cost": item["total_measured_cost"],
            "ledger_valid": item["ledger_valid"],
            "final_state_digest": item["final_state_digest"],
        }
        for item in summaries
    ]
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    payload = {
        "task": args.task,
        "seeds": args.seeds,
        "runs": summaries,
        "claims_boundary": "synthetic experiment; no external validation",
    }
    (root / "results.json").write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "summary.md").write_text(
        f"# {args.task} experiment\n\nSeeds: {args.seeds}\n\nAll ledger-valid: {all(x['ledger_valid'] for x in summaries)}\n",
        encoding="utf-8",
    )
    print(root)
    return 0


def _ablate(args: argparse.Namespace) -> int:
    config = _config(args.config)
    root = Path(args.output or config.output_root) / f"ablations-{args.task}"
    result = run_ablation_suite(config, args.task, list(range(args.seeds)), root)
    print(json.dumps({"output": str(root), "comparisons": len(result["comparisons"])}, indent=2))
    return 0


def _ledger(args: argparse.Namespace) -> int:
    root = Path(args.path)
    path = root / "ledger.jsonl" if root.is_dir() else root
    result = verify_ledger_file(path, args.tolerance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _replay(args: argparse.Namespace) -> int:
    print(json.dumps(replay_run(args.path), indent=2, sort_keys=True))
    return 0


def _report(args: argparse.Namespace) -> int:
    root = Path(args.path)
    path = root / "summary.json" if root.is_dir() else root
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


def _profile(args: argparse.Namespace) -> int:
    config = _config(args.config)
    output = Path(args.output or config.output_root) / "profile-smoke"
    profile = output.parent / "hrm.prof"
    profile.parent.mkdir(parents=True, exist_ok=True)
    cProfile.runctx(
        "run_task(config, 'spatial-reconstruction', config.seed, output)",
        globals(),
        locals(),
        str(profile),
    )
    print(profile)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrm", description="Helical Recursive Manifold reference implementation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config")
    validate.add_argument("path")
    validate.set_defaults(
        func=lambda a: (
            print(json.dumps(load_config(a.path).model_dump(mode="json"), indent=2)),
            0,
        )[1]
    )
    run = sub.add_parser("run")
    run.add_argument("--config")
    run.add_argument("--task", choices=TASKS, default="spatial-reconstruction")
    run.add_argument("--output")
    run.add_argument("--run-id")
    run.set_defaults(func=_run)
    experiment = sub.add_parser("experiment")
    experiment.add_argument("task", choices=TASKS)
    experiment.add_argument("--seeds", type=int, default=5)
    experiment.add_argument("--config")
    experiment.add_argument("--output")
    experiment.set_defaults(func=_experiment)
    ablate = sub.add_parser("ablate")
    ablate.add_argument("--config")
    ablate.add_argument("--suite", choices=["canonical"], default="canonical")
    ablate.add_argument("--task", choices=TASKS, default="spatial-reconstruction")
    ablate.add_argument("--seeds", type=int, default=5)
    ablate.add_argument("--output")
    ablate.set_defaults(func=_ablate)
    ledger = sub.add_parser("ledger")
    ledger_sub = ledger.add_subparsers(dest="ledger_command", required=True)
    verify = ledger_sub.add_parser("verify")
    verify.add_argument("path")
    verify.add_argument("--tolerance", type=float)
    verify.set_defaults(func=_ledger)
    replay = sub.add_parser("replay")
    replay.add_argument("path")
    replay.set_defaults(func=_replay)
    report = sub.add_parser("report")
    report.add_argument("path")
    report.set_defaults(func=_report)
    profile = sub.add_parser("profile")
    profile.add_argument("--config")
    profile.add_argument("--output")
    profile.set_defaults(func=_profile)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"hrm: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
