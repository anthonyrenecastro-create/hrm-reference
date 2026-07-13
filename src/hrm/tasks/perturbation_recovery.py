"""Controlled field, memory, geometry, and topology damage with replayable repair."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.invariants import refresh_topology
from hrm.core.proposal import ExternalInput
from hrm.state.unified import HRMState, create_initial_state
from hrm.tasks.base import TaskBundle
from hrm.tasks.spatial_reconstruction import target_pattern


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    target = target_pattern(config, seed)
    state.field.values = target.copy()
    state.field.target = target.copy()
    step = max(1, config.steps // 3)

    def perturb(source: HRMState) -> HRMState:
        rng = np.random.default_rng(seed + 53)
        damaged = source.clone()
        nodes = np.arange(max(1, damaged.field.values.shape[0] // 5))
        damaged.field.values[nodes] += rng.normal(0, 1.0, damaged.field.values[nodes].shape)
        damaged.geometry.coordinates[nodes] += rng.normal(
            0, 0.1, damaged.geometry.coordinates[nodes].shape
        )
        if damaged.memory.occupancy:
            occupied = np.flatnonzero(damaged.memory.occupied)
            damaged.memory.strengths[occupied] *= 0.5
        if len(damaged.topology.edges) > 3:
            i, j = damaged.topology.edges[0]
            matrix = damaged.topology.weights.tolil(copy=True)
            matrix[i, j] = matrix[j, i] = 0.0
            damaged.topology.weights = matrix.tocsr()
            refresh_topology(damaged)
        return damaged

    inputs = [
        ExternalInput(
            0.15 * target,
            target_field=target,
            input_id=f"recovery:{seed}:{t}",
            perturbation_id="mixed-damage-v1" if t == step else None,
        )
        for t in range(config.steps)
    ]
    return TaskBundle(
        "perturbation-recovery",
        state,
        inputs,
        perturbations={step: ("mixed-damage-v1", perturb)},
        metadata={
            "perturbation_step": step,
            "damage": ["field", "memory", "geometry", "topology"],
            "target": target.tolist(),
        },
    )
