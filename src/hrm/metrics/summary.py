"""Metric projection used by transitions and experiment reports."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.csgraph import connected_components

from hrm.core.proposal import ExternalInput
from hrm.state.unified import HRMState


def compute_metrics(state: HRMState, external_input: ExternalInput | None = None) -> dict[str, Any]:
    values = np.where(state.field.active_channels, state.field.values, 0.0)
    active = values[state.field.active_channels]
    target = None if external_input is None else external_input.target_field
    reconstruction = float(np.mean((values - target) ** 2)) if target is not None else float("nan")
    prediction = float(np.mean(state.cognition.residual**2))
    active_nodes = np.flatnonzero(state.field.active_nodes)
    if active_nodes.size:
        components, _ = connected_components(
            state.topology.weights[active_nodes][:, active_nodes], directed=False
        )
    else:
        components = 0
    occupied = np.flatnonzero(state.memory.occupied)
    retention = (
        float(np.mean(state.memory.strengths[occupied] / (1.0 + state.memory.ages[occupied])))
        if occupied.size
        else 0.0
    )
    uncertainty = float(np.mean(state.cognition.uncertainty))
    uncertainty_calibration = abs(uncertainty - float(np.sqrt(max(0.0, prediction))))
    return {
        "prediction_error": prediction,
        "reconstruction_error": reconstruction,
        "field_coherence": float(np.mean(state.field.local_coherence[state.field.active_nodes]))
        if active_nodes.size
        else 0.0,
        "field_variance": float(np.var(active)) if active.size else 0.0,
        "field_energy": float(np.sum(state.field.energy)),
        "geometric_stress": float(np.mean(state.geometry.stress[state.field.active_nodes]))
        if active_nodes.size
        else 0.0,
        "structural_validity": float(components <= 1),
        "connected_components": int(components),
        "edge_count": len(state.topology.edges),
        "memory_retention": retention,
        "memory_interference": float(state.memory.interference),
        "memory_occupancy": state.memory.occupancy,
        "memory_use": state.memory.occupancy,
        "uncertainty": uncertainty,
        "uncertainty_calibration": uncertainty_calibration,
        "collapse_risk": float(state.field.collapse_indicator),
        "active_nodes": int(np.count_nonzero(state.field.active_nodes)),
        "active_channels": int(np.count_nonzero(state.field.active_channels)),
        "active_operations": int(state.resources.cumulative_operations),
        "current_cost": float(state.resources.current_use),
        "measured_cost": float(state.resources.measured_cost),
        "switching_cost": float(state.resources.switching_cost),
    }
