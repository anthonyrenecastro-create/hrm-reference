"""Path-sensitive propagation task with a deliberately weak bottleneck."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.invariants import refresh_topology
from hrm.core.proposal import ExternalInput
from hrm.state.unified import create_initial_state
from hrm.tasks.base import TaskBundle


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    n, d = state.field.values.shape
    half = n // 2
    # Keep one bridge between halves so the graph remains valid but path-limited.
    matrix = state.topology.weights.tolil(copy=True)
    for i, j in state.topology.edges:
        if (i < half) != (j < half) and {i, j} != {half - 1, half}:
            matrix[i, j] = matrix[j, i] = 0.0
    matrix[half - 1, half] = matrix[half, half - 1] = 0.05
    state.topology.weights = matrix.tocsr()
    refresh_topology(state)
    source = np.zeros((n, d), dtype=config.numpy_dtype)
    source[: max(1, half // 3), 0] = 1.0
    target = np.zeros_like(source)
    target[half:, 0] = 1.0
    inputs = [
        ExternalInput(source, target_field=target, input_id=f"topology-useful:{seed}:{step}")
        for step in range(config.steps)
    ]
    return TaskBundle(
        "topology-usefulness",
        state,
        inputs,
        metadata={"bottleneck": [half - 1, half], "purpose": "path_sensitive"},
    )
