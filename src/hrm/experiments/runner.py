"""Canonical run execution and required artifact generation."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hrm.config import HRMConfig, PerturbationConfig, save_config
from hrm.core.determinism import DeterministicContext
from hrm.core.engine_selection import build_transition_engine
from hrm.core.invariants import refresh_topology
from hrm.core.ledger import LedgerRecord, write_ledger
from hrm.experiments.claims import mechanism_gate_report
from hrm.experiments.reporting import write_metric_plot, write_summary_markdown
from hrm.experiments.schedule import serialize_schedule
from hrm.state.digest import numeric_state_digest, state_digest
from hrm.state.serialization import save_state
from hrm.state.unified import HRMState
from hrm.tasks import build_task
from hrm.types import jsonable


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    output_dir: Path
    summary: dict[str, Any]
    ledgers: tuple[LedgerRecord, ...]


def apply_configured_perturbation(
    state: HRMState,
    spec: PerturbationConfig,
    *,
    seed: int,
    run_id: str,
    index: int,
) -> HRMState:
    result = state.clone()
    stream = f"configured_perturbation:{index}:{spec.kind}:{spec.label or ''}"
    generator = DeterministicContext(seed, spec.step, run_id).generator(stream)
    if spec.kind == "field_noise":
        noise = generator.normal(0.0, spec.magnitude, result.field.values.shape)
        result.field.values += np.where(result.field.active_channels, noise, 0.0)
    elif spec.kind == "field_scale":
        result.field.values *= spec.magnitude
    elif spec.kind == "geometry_jitter":
        result.geometry.coordinates += generator.normal(
            0.0, spec.magnitude, result.geometry.coordinates.shape
        )
    elif spec.kind == "memory_attenuation":
        result.memory.strengths[result.memory.occupied] *= spec.magnitude
    refresh_topology(result)
    return result


def _environment(config: HRMConfig) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "pydantic", "PyYAML", "pandas", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "uncommitted"
    blas: dict[str, Any] = {}
    try:
        info = np.__config__.CONFIG
        blas = jsonable(info.get("Build Dependencies", {}).get("blas", {}))  # type: ignore[assignment]
    except Exception:
        blas = {"status": "unavailable"}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": packages,
        "numeric_dtype": config.dtype,
        "blas": blas,
        "determinism": "counter-derived streams; runtime excluded from identity",
        "source_commit": commit,
    }


def run_task(
    config: HRMConfig,
    task_name: str,
    seed: int,
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> RunResult:
    resolved = HRMConfig.model_validate(config.model_dump(mode="python") | {"seed": seed})
    bundle = build_task(task_name, resolved, seed)
    run_id = run_id or f"{bundle.name}-seed-{seed}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    save_config(resolved, out / "config.resolved.yaml")
    default = HRMConfig()
    diff = {
        key: {
            "default": default.model_dump(mode="json").get(key),
            "resolved": resolved.model_dump(mode="json").get(key),
        }
        for key in type(resolved).model_fields
        if default.model_dump(mode="json").get(key) != resolved.model_dump(mode="json").get(key)
    }
    (out / "config.diff.json").write_text(
        json.dumps(jsonable(diff), indent=2, sort_keys=True), encoding="utf-8"
    )
    save_state(bundle.initial_state, out / "initial_state.npz")
    schedule = serialize_schedule(bundle, resolved)
    (out / "schedule.json").write_text(
        json.dumps(schedule, indent=2, sort_keys=True), encoding="utf-8"
    )

    state = bundle.initial_state.clone()
    engine = build_transition_engine(resolved)
    ledgers: list[LedgerRecord] = []
    metrics: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    digests: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    recovery_reference: np.ndarray | None = None
    recovery_initial_distance: float | None = None
    perturbation_step: int | None = None
    recovered_at: int | None = None
    for step, external_input in enumerate(bundle.inputs):
        perturbation_identities: list[str] = []
        pre_perturbation_field: np.ndarray | None = None
        if step in bundle.perturbations:
            identity, perturb = bundle.perturbations[step]
            pre_perturbation_field = state.field.values.copy()
            state = perturb(state)
            perturbation_identities.append(identity)
        for index, spec in enumerate(resolved.perturbation_schedule):
            if spec.step != step:
                continue
            if pre_perturbation_field is None:
                pre_perturbation_field = state.field.values.copy()
            identity = spec.label or f"configured:{spec.kind}:{index}"
            state = apply_configured_perturbation(
                state, spec, seed=seed, run_id=run_id, index=index
            )
            perturbation_identities.append(identity)
        if perturbation_identities and pre_perturbation_field is not None:
            recovery_reference = pre_perturbation_field
            recovery_initial_distance = float(
                np.linalg.norm(state.field.values - recovery_reference)
            )
            perturbation_step = step
            recovered_at = None
            combined_identity = "+".join(perturbation_identities)
            state.run_metadata["last_perturbation"] = combined_identity
            external_input = replace(external_input, perturbation_id=combined_identity)
        pre = state_digest(state)
        result = engine.step(state, external_input, run_id=run_id)
        state = result.state
        ledgers.append(result.ledger)
        step_metrics = {"step": step, **result.ledger.metrics_after}
        if recovery_reference is not None and recovery_initial_distance is not None:
            recovery_distance = float(np.linalg.norm(state.field.values - recovery_reference))
            reference_norm = float(np.linalg.norm(recovery_reference))
            fidelity = float(
                np.sum(state.field.values * recovery_reference)
                / ((np.linalg.norm(state.field.values) * reference_norm) + 1e-12)
            )
            robustness = 1.0 - recovery_distance / (recovery_initial_distance + 1e-12)
            if recovered_at is None and recovery_distance <= 0.1 * recovery_initial_distance:
                recovered_at = step
            step_metrics.update(
                {
                    "perturbation_robustness": robustness,
                    "recovery_fidelity": fidelity,
                    "recovery_distance": recovery_distance,
                    "recovery_time": float("nan")
                    if recovered_at is None or perturbation_step is None
                    else recovered_at - perturbation_step,
                }
            )
        else:
            step_metrics.update(
                {
                    "perturbation_robustness": float("nan"),
                    "recovery_fidelity": float("nan"),
                    "recovery_distance": float("nan"),
                    "recovery_time": float("nan"),
                }
            )
        metrics.append(step_metrics)
        costs.append(
            {
                "step": step,
                "measured_cost": result.ledger.measured_cost,
                "runtime_seconds": result.ledger.runtime_seconds,
                "correction_cost": result.ledger.metrics_after.get("correction_cost", 0.0),
            }
        )
        digests.append({"step": step, "pre": pre, "post": result.ledger.post_state_digest})
        events.extend(
            {"step": step, "status": "accepted", **event} for event in result.ledger.accepted_events
        )
        events.extend(
            {"step": step, "status": "rejected", **event} for event in result.ledger.rejected_events
        )
    save_state(state, out / "final_state.npz")
    write_ledger(ledgers, out / "ledger.jsonl")
    for filename, rows in (("state_digests.jsonl", digests), ("events.jsonl", events)):
        with (out / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(jsonable(row), sort_keys=True, separators=(",", ":")) + "\n"
                )
    metrics_frame = pd.DataFrame(metrics)
    costs_frame = pd.DataFrame(costs)
    metrics_frame.to_csv(out / "metrics.csv", index=False)
    costs_frame.to_csv(out / "costs.csv", index=False)
    gates = mechanism_gate_report(ledgers)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "task": bundle.name,
        "seed": seed,
        "steps": len(ledgers),
        "configuration_digest": resolved.digest,
        "initial_state_digest": state_digest(bundle.initial_state),
        "final_state_digest": state_digest(state),
        "initial_numeric_state_digest": numeric_state_digest(bundle.initial_state),
        "final_numeric_state_digest": numeric_state_digest(state),
        "maximum_absolute_residual": max((item.absolute_residual for item in ledgers), default=0.0),
        "maximum_relative_residual": max((item.relative_residual for item in ledgers), default=0.0),
        "ledger_valid": all(
            item.relative_residual <= resolved.ledger.tolerance
            or item.absolute_residual <= resolved.ledger.tolerance
            for item in ledgers
        ),
        "mechanism_gates": gates,
        "final_metrics": metrics[-1] if metrics else {},
        "total_measured_cost": float(costs_frame["measured_cost"].sum())
        if not costs_frame.empty
        else 0.0,
        "total_runtime_seconds": float(costs_frame["runtime_seconds"].sum())
        if not costs_frame.empty
        else 0.0,
        "accepted_events": sum(len(item.accepted_events) for item in ledgers),
        "rejected_events": sum(len(item.rejected_events) for item in ledgers),
        "claims": {
            "implementation": True,
            "activation": "per mechanism gate report",
            "synthetic_support": "not inferred from a single run",
            "external_support": False,
            "validated": False,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary_markdown(summary, out / "summary.md")
    write_metric_plot(metrics_frame, out / "plots" / "metrics.png")
    (out / "environment.json").write_text(
        json.dumps(_environment(resolved), indent=2, sort_keys=True), encoding="utf-8"
    )
    return RunResult(run_id, out, summary, tuple(ledgers))
