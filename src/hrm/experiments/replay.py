"""Deterministic replay of a recorded run bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hrm.config import load_config
from hrm.core.engine_selection import build_transition_engine
from hrm.core.ledger import load_ledger
from hrm.core.proposal import ExternalInput
from hrm.experiments.runner import apply_configured_perturbation
from hrm.state.digest import state_digest
from hrm.state.serialization import load_state
from hrm.tasks import build_task


def replay_run(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    config = load_config(root / "config.resolved.yaml")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    schedule = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    originals = load_ledger(root / "ledger.jsonl")
    state = load_state(root / "initial_state.npz", config.dtype)
    task = build_task(summary["task"], config, summary["seed"])
    engine = build_transition_engine(config)
    mismatches: list[dict[str, Any]] = []
    for step, raw in enumerate(schedule["inputs"]):
        perturbation_identities: list[str] = []
        if step in task.perturbations:
            identity, perturb = task.perturbations[step]
            state = perturb(state)
            perturbation_identities.append(identity)
        for index, spec in enumerate(config.perturbation_schedule):
            if spec.step != step:
                continue
            identity = spec.label or f"configured:{spec.kind}:{index}"
            state = apply_configured_perturbation(
                state,
                spec,
                seed=summary["seed"],
                run_id=summary["run_id"],
                index=index,
            )
            perturbation_identities.append(identity)
        perturbation_identity = (
            "+".join(perturbation_identities)
            if perturbation_identities
            else raw.get("perturbation_id")
        )
        if perturbation_identities:
            state.run_metadata["last_perturbation"] = perturbation_identity
        external = ExternalInput(
            np.asarray(raw["values"], dtype=config.numpy_dtype),
            None
            if raw["target_field"] is None
            else np.asarray(raw["target_field"], dtype=config.numpy_dtype),
            None
            if raw["task_target"] is None
            else np.asarray(raw["task_target"], dtype=config.numpy_dtype),
            raw["input_id"],
            perturbation_identity,
            bool(raw.get("supervised_guidance", False)),
            raw.get("metadata", {}),
        )
        result = engine.step(state, external, run_id=summary["run_id"])
        expected = originals[step]
        if (
            result.ledger.pre_state_digest != expected["pre_state_digest"]
            or result.ledger.post_state_digest != expected["post_state_digest"]
        ):
            mismatches.append(
                {
                    "step": step,
                    "expected_pre": expected["pre_state_digest"],
                    "actual_pre": result.ledger.pre_state_digest,
                    "expected_post": expected["post_state_digest"],
                    "actual_post": result.ledger.post_state_digest,
                }
            )
        state = result.state
    final_digest = state_digest(state)
    if final_digest != summary["final_state_digest"]:
        mismatches.append(
            {"step": "final", "expected": summary["final_state_digest"], "actual": final_digest}
        )
    if mismatches:
        raise ValueError(f"deterministic replay failed: {mismatches[:3]}")
    return {"steps": len(originals), "final_state_digest": final_digest, "matched": True}
