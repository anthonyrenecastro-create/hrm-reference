"""Synthetic delayed sequence prediction."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.state.unified import create_initial_state
from hrm.tasks.base import TaskBundle


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    rng = np.random.default_rng(seed + 29)
    d = config.dimensions.channels
    total = config.steps + 3
    sequence = np.zeros((total, d), dtype=config.numpy_dtype)
    sequence[0] = rng.normal(0, 0.3, d)
    sequence[1] = rng.normal(0, 0.3, d)
    for t in range(2, total):
        sequence[t] = np.tanh(
            0.55 * sequence[t - 1] - 0.25 * sequence[t - 2] + 0.15 * np.roll(sequence[t - 1], 1)
        )
    inputs = []
    for step in range(config.steps):
        values = np.broadcast_to(sequence[step], state.field.values.shape).copy()
        inputs.append(
            ExternalInput(
                values,
                task_target=sequence[step + 1],
                input_id=f"sequence:{seed}:{step}",
                metadata={"held_out_transition": step + 1},
            )
        )
    return TaskBundle(
        "sequence-prediction",
        state,
        inputs,
        metadata={"sequence": sequence.tolist(), "dependency_lags": [1, 2]},
    )
