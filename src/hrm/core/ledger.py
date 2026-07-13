"""Per-transition attribution ledger and independent reconstruction verifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.events import StructuralEvent
from hrm.core.safety import CorrectionRecord
from hrm.core.transport import TransportRecord
from hrm.types import jsonable


def _arrays(data: dict[str, Any]) -> dict[str, np.ndarray]:
    return {path: np.asarray(value, dtype=float) for path, value in data.items()}


def _json_arrays(data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {path: np.asarray(value).tolist() for path, value in data.items()}


def _sum_into(total: dict[str, np.ndarray], deltas: dict[str, np.ndarray]) -> None:
    for path, value in deltas.items():
        if path not in total:
            total[path] = np.zeros_like(np.asarray(value, dtype=float))
        total[path] = total[path] + np.asarray(value, dtype=float)


def delta_norm(delta: dict[str, np.ndarray]) -> float:
    return float(
        np.sqrt(sum(float(np.sum(np.asarray(value, dtype=float) ** 2)) for value in delta.values()))
    )


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    schema_version: str
    run_id: str
    seed: int
    step: int
    numeric_precision: str
    solver: str
    configuration_digest: str
    generator_identity: dict[str, Any]
    input_identity: str
    perturbation_identity: str | None
    pre_state_digest: str
    post_state_digest: str
    state_block_versions: dict[str, int]
    timestep: float
    adaptive_timestep_decision: str | None
    main_contributions: tuple[dict[str, Any], ...]
    interaction_contributions: tuple[dict[str, Any], ...]
    stochastic_record: dict[str, Any]
    proposed_events: tuple[dict[str, Any], ...]
    accepted_events: tuple[dict[str, Any], ...]
    rejected_events: tuple[dict[str, Any], ...]
    transport_metadata: dict[str, Any]
    event_effects: dict[str, Any]
    corrections: tuple[dict[str, Any], ...]
    correction_effects: dict[str, Any]
    metrics_before: dict[str, Any]
    metrics_after: dict[str, Any]
    measured_cost: float
    runtime_seconds: float
    reconstructed_delta: dict[str, Any]
    observed_delta: dict[str, Any]
    absolute_residual: float
    relative_residual: float

    def to_dict(self) -> dict[str, Any]:
        return jsonable({name: getattr(self, name) for name in self.__dataclass_fields__})  # type: ignore[return-value]


def proposal_record(proposal: MechanismProposal, store_arrays: bool) -> dict[str, Any]:
    return {
        "mechanism": proposal.mechanism,
        "activation": proposal.activation,
        "reads": sorted(str(x) for x in proposal.reads),
        "writes": sorted(str(x) for x in proposal.writes),
        "proposed_norm": proposal.norm,
        "applied_norm": proposal.norm,
        "contributions": _json_arrays(proposal.contributions) if store_arrays else {},
        "contribution_shapes": {
            key: list(np.asarray(value).shape) for key, value in proposal.contributions.items()
        },
        "diagnostics": jsonable(proposal.diagnostics),
        "proposed_cost": proposal.proposed_cost,
        "required_invariants": list(proposal.required_invariants),
        "dtype": proposal.dtype,
        "skipped_reason": proposal.skipped_reason,
        "interaction_metadata": jsonable(proposal.interaction_metadata),
    }


def reconstruct_from_components(
    main: list[MechanismProposal],
    interactions: list[MechanismProposal],
    stochastic: dict[str, np.ndarray],
    event_effects: dict[str, np.ndarray],
    correction_effects: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    total: dict[str, np.ndarray] = {}
    for proposal in [*main, *interactions]:
        _sum_into(total, proposal.contributions)
    _sum_into(total, stochastic)
    _sum_into(total, event_effects)
    _sum_into(total, correction_effects)
    return total


def calculate_residual(
    observed: dict[str, np.ndarray], reconstructed: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float, float]:
    residual: dict[str, np.ndarray] = {}
    for path, value in observed.items():
        residual[path] = np.asarray(value, dtype=float) - np.asarray(
            reconstructed.get(path, np.zeros_like(value)), dtype=float
        )
    absolute = delta_norm(residual)
    relative = absolute / (delta_norm(observed) + 1e-12)
    return residual, absolute, relative


def build_ledger_record(
    *,
    run_id: str,
    seed: int,
    step: int,
    dtype: str,
    solver: str,
    config_digest: str,
    generator_identity: dict[str, Any],
    input_identity: str,
    perturbation_identity: str | None,
    pre_digest: str,
    post_digest: str,
    versions: dict[str, int],
    timestep: float,
    adaptive_reason: str | None,
    main: list[MechanismProposal],
    interactions: list[MechanismProposal],
    stochastic: dict[str, np.ndarray],
    proposed_events: list[StructuralEvent],
    accepted_events: list[StructuralEvent],
    rejected_events: list[StructuralEvent],
    transport: TransportRecord,
    event_effects: dict[str, np.ndarray],
    corrections: list[CorrectionRecord],
    correction_effects: dict[str, np.ndarray],
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    measured_cost: float,
    runtime_seconds: float,
    observed_delta: dict[str, np.ndarray],
    tolerance: float,
    store_arrays: bool,
    schema_version: str,
) -> LedgerRecord:
    reconstructed = reconstruct_from_components(
        main, interactions, stochastic, event_effects, correction_effects
    )
    _, absolute, relative = calculate_residual(observed_delta, reconstructed)
    if relative > tolerance and absolute > tolerance:
        raise RuntimeError(
            f"ledger reconstruction failed: absolute={absolute:.3e}, relative={relative:.3e}, tolerance={tolerance:.3e}"
        )
    return LedgerRecord(
        schema_version,
        run_id,
        seed,
        step,
        dtype,
        solver,
        config_digest,
        generator_identity,
        input_identity,
        perturbation_identity,
        pre_digest,
        post_digest,
        versions,
        timestep,
        adaptive_reason,
        tuple(proposal_record(p, store_arrays) for p in main),
        tuple(proposal_record(p, store_arrays) for p in interactions),
        {
            "kind": "intentional_seeded",
            "contributions": _json_arrays(stochastic),
            "norm": delta_norm(stochastic),
        },
        tuple(e.to_dict() for e in proposed_events),
        tuple(e.to_dict() for e in accepted_events),
        tuple(e.to_dict() for e in rejected_events),
        transport.to_dict(),
        _json_arrays(event_effects),
        tuple(c.to_dict() for c in corrections),
        _json_arrays(correction_effects),
        cast(dict[str, Any], jsonable(metrics_before)),
        cast(dict[str, Any], jsonable(metrics_after)),
        measured_cost,
        runtime_seconds,
        _json_arrays(reconstructed),
        _json_arrays(observed_delta),
        absolute,
        relative,
    )


def verify_record(record: dict[str, Any], tolerance: float | None = None) -> tuple[float, float]:
    total: dict[str, np.ndarray] = {}
    for group in ("main_contributions", "interaction_contributions"):
        for proposal in record[group]:
            contributions = proposal.get("contributions", {})
            if not contributions and proposal.get("applied_norm", 0.0) > 0:
                raise ValueError(
                    "ledger does not store arrays required for independent verification"
                )
            _sum_into(total, _arrays(contributions))
    _sum_into(total, _arrays(record["stochastic_record"].get("contributions", {})))
    _sum_into(total, _arrays(record.get("event_effects", {})))
    _sum_into(total, _arrays(record.get("correction_effects", {})))
    observed = _arrays(record["observed_delta"])
    _, absolute, relative = calculate_residual(observed, total)
    limit = float(
        tolerance
        if tolerance is not None
        else max(1e-12, record.get("relative_residual", 0.0) + 1e-12)
    )
    if absolute > limit and relative > limit:
        raise ValueError(
            f"ledger record {record.get('step')} failed: absolute={absolute:.3e}, relative={relative:.3e}"
        )
    return absolute, relative


def write_ledger(records: list[LedgerRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def load_ledger(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_ledger_file(path: str | Path, tolerance: float | None = None) -> dict[str, float | int]:
    records = load_ledger(path)
    maxima = [verify_record(record, tolerance) for record in records]
    return {
        "records": len(records),
        "maximum_absolute_residual": max((x[0] for x in maxima), default=0.0),
        "maximum_relative_residual": max((x[1] for x in maxima), default=0.0),
    }
