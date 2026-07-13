"""Heterogeneous reconstruction load for equal-budget allocation studies."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.state.unified import create_initial_state
from hrm.tasks.base import TaskBundle
from hrm.tasks.spatial_reconstruction import target_pattern


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    target = target_pattern(config, seed)
    rng = np.random.default_rng(seed + 67)
    noise_scale = np.linspace(0.05, 0.8, config.dimensions.n_nodes)[:, None]
    observation = target + noise_scale * rng.standard_normal(target.shape)
    state.field.values = (0.1 * observation).astype(config.numpy_dtype)
    inputs = [
        ExternalInput(observation, target_field=target, input_id=f"resource:{seed}:{step}")
        for step in range(config.steps)
    ]
    return TaskBundle(
        "resource-adaptation",
        state,
        inputs,
        metadata={"heterogeneous_noise": noise_scale.ravel().tolist(), "target": target.tolist()},
    )
