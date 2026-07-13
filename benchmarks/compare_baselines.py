"""Small capacity-disclosed reconstruction baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hrm.baselines.graph_diffusion import graph_diffusion
from hrm.baselines.persistence import persistence
from hrm.baselines.reaction_diffusion import reaction_diffusion
from hrm.config import load_config
from hrm.experiments.runner import run_task
from hrm.tasks import build_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--output", default="artifacts/baseline-comparison.json")
    parser.add_argument("--work-root")
    args = parser.parse_args()
    config = load_config(args.config)
    bundle = build_task("spatial-reconstruction", config, config.seed)
    target = bundle.inputs[-1].target_field
    if target is None:
        raise RuntimeError("spatial reconstruction did not provide a target")
    observation = bundle.initial_state.field.values
    persisted, persistence_cost = persistence(observation)
    diffused, diffusion_cost = graph_diffusion(
        bundle.initial_state,
        observation,
        steps=config.steps,
        dt=config.dynamics.dt,
        coefficient=config.dynamics.diffusion,
    )
    reacted, reaction_cost = reaction_diffusion(
        bundle.initial_state,
        observation,
        steps=config.steps,
        dt=config.dynamics.dt,
        diffusion=config.dynamics.diffusion,
        a=config.dynamics.reaction_a,
        b=config.dynamics.reaction_b,
    )
    work_root = Path(args.work_root) if args.work_root else Path(args.output).parent / "hrm"
    hrm = run_task(config, "spatial-reconstruction", config.seed, work_root)
    payload = {
        "capacity": {
            "nodes": config.dimensions.n_nodes,
            "channels": config.dimensions.channels,
            "steps": config.steps,
        },
        "results": {
            "persistence": {
                "mse": float(np.mean((persisted - target) ** 2)),
                "cost": persistence_cost,
            },
            "graph_diffusion": {
                "mse": float(np.mean((diffused - target) ** 2)),
                "cost": diffusion_cost,
            },
            "reaction_diffusion": {
                "mse": float(np.mean((reacted - target) ** 2)),
                "cost": reaction_cost,
            },
            "hrm": {
                "mse": hrm.summary["final_metrics"]["reconstruction_error"],
                "cost": hrm.summary["total_measured_cost"],
            },
        },
        "claims_boundary": "single-seed smoke comparison; unequal operation models are disclosed and no superiority is inferred",
    }
    target_path = Path(args.output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(target_path)


if __name__ == "__main__":
    main()
