"""Safe-transition projection, rollback, and separately attributed repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrm.config import HRMConfig
from hrm.core.invariants import refresh_topology
from hrm.core.repair import component_bridge_candidates
from hrm.state.digest import numeric_view, view_delta
from hrm.state.unified import HRMState
from hrm.types import Diagnostics


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    kind: str
    reason: str
    affected_paths: tuple[str, ...]
    norm: float
    metadata: Diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "affected_paths": list(self.affected_paths),
            "norm": self.norm,
            "metadata": self.metadata,
        }


def safe_transition(
    candidate: HRMState,
    rollback_state: HRMState,
    config: HRMConfig,
    *,
    rollback_effect: dict[str, np.ndarray] | None = None,
) -> tuple[HRMState, list[CorrectionRecord], dict[str, np.ndarray]]:
    result = candidate.clone()
    before = numeric_view(result)
    corrections: list[CorrectionRecord] = []
    all_arrays = list(numeric_view(result).values())
    if not all(np.all(np.isfinite(array)) for array in all_arrays):
        if not config.safety.rollback_on_nonfinite:
            raise FloatingPointError("nonfinite candidate and rollback disabled")
        result = rollback_state.clone()
        exact_ledger_inverse = rollback_effect is not None
        if rollback_effect is None:
            # Standalone callers do not have the proposal ledger available. Use a
            # finite projection of the rejected candidate so diagnostics remain
            # serializable; the transition engine supplies the exact inverse of
            # all already-attributed contributions.
            finite_before = {
                path: np.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=config.safety.field_abs_bound,
                    neginf=-config.safety.field_abs_bound,
                )
                for path, value in before.items()
            }
            rollback_effect = view_delta(numeric_view(result), finite_before)
        compact = {
            path: np.asarray(value, dtype=float)
            for path, value in rollback_effect.items()
            if np.any(np.asarray(value) != 0)
        }
        total_norm = float(np.sqrt(sum(float(np.sum(value**2)) for value in compact.values())))
        corrections.append(
            CorrectionRecord(
                "rollback",
                "nonfinite_candidate",
                tuple(sorted(compact)),
                total_norm,
                {"exact_ledger_inverse": exact_ledger_inverse},
            )
        )
        return result, corrections, compact
    active = result.field.active_channels
    result.field.values[~active] = 0.0
    clipped = np.clip(
        result.field.values, -config.safety.field_abs_bound, config.safety.field_abs_bound
    )
    if not np.array_equal(clipped, result.field.values):
        result.field.values = clipped
        corrections.append(
            CorrectionRecord("projection", "field_absolute_bound", ("field.values",), 0.0, {})
        )
    norm = float(np.linalg.norm(result.field.values[active]))
    if norm > config.safety.field_norm_bound:
        result.field.values *= config.safety.field_norm_bound / (norm + 1e-12)
        corrections.append(
            CorrectionRecord(
                "projection",
                "field_norm_bound",
                ("field.values",),
                norm - config.safety.field_norm_bound,
                {},
            )
        )
    cognitive_norm = float(np.linalg.norm(result.cognition.latent))
    if cognitive_norm > config.safety.cognitive_norm_bound:
        result.cognition.latent *= config.safety.cognitive_norm_bound / (cognitive_norm + 1e-12)
        corrections.append(
            CorrectionRecord(
                "projection",
                "cognitive_norm_bound",
                ("cognition.latent",),
                cognitive_norm - config.safety.cognitive_norm_bound,
                {},
            )
        )
    bounded_coords = np.clip(
        result.geometry.coordinates, result.geometry.bounds[:, 0], result.geometry.bounds[:, 1]
    )
    if not np.array_equal(bounded_coords, result.geometry.coordinates):
        result.geometry.coordinates = bounded_coords
        corrections.append(
            CorrectionRecord(
                "geometry_repair", "coordinate_bounds", ("geometry.coordinates",), 0.0, {}
            )
        )
    if config.topology.repair_enabled and not config.fixed_topology:
        bridges = component_bridge_candidates(result)
        if bridges:
            matrix = result.topology.weights.tolil(copy=True)
            for i, j in bridges:
                matrix[i, j] = matrix[j, i] = 1.0
            result.topology.weights = matrix.tocsr()
            refresh_topology(result)
            corrections.append(
                CorrectionRecord(
                    "topology_repair",
                    "connect_components",
                    ("topology.weights",),
                    float(len(bridges)),
                    {"bridges": [list(e) for e in bridges]},
                )
            )
    result.memory.strengths = np.clip(result.memory.strengths, 0.0, config.memory.max_strength)
    result.memory.ages = np.maximum(result.memory.ages, 0.0)
    if result.hierarchy.feedback_gain > config.hierarchy.max_feedback_gain:
        result.hierarchy.feedback_gain = config.hierarchy.max_feedback_gain
        corrections.append(
            CorrectionRecord(
                "hierarchy_gain_reduction", "feedback_bound", ("hierarchy.feedback_gain",), 0.0, {}
            )
        )
    if result.resources.current_use > result.resources.total_budget:
        scale = result.resources.total_budget / (result.resources.current_use + 1e-12)
        result.resources.node_allocation *= scale
        result.resources.region_allocation *= scale
        result.resources.current_use = result.resources.total_budget
        corrections.append(
            CorrectionRecord(
                "budget_correction",
                "total_budget",
                (
                    "resources.current_use",
                    "resources.node_allocation",
                    "resources.region_allocation",
                ),
                0.0,
                {},
            )
        )
    after = numeric_view(result)
    delta = view_delta(after, before)
    compact = {path: value for path, value in delta.items() if np.any(value != 0)}
    total_norm = float(np.sqrt(sum(np.sum(value**2) for value in compact.values())))
    if corrections:
        corrections = [
            CorrectionRecord(c.kind, c.reason, c.affected_paths, c.norm or total_norm, c.metadata)
            for c in corrections
        ]
    if len(corrections) > config.safety.max_corrections_per_step:
        raise RuntimeError("safe transition exceeded correction limit")
    return result, corrections, compact
