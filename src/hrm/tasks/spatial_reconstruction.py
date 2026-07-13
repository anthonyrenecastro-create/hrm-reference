"""Corrupted multidimensional pattern reconstruction."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.state.unified import create_initial_state
from hrm.tasks.base import TaskBundle


def target_pattern(config: HRMConfig, seed: int = 0) -> np.ndarray:
    n, d = config.dimensions.n_nodes, config.dimensions.channels
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    target = np.column_stack(
        [np.sin((channel // 2 + 1) * x + (channel % 2) * np.pi / 2) for channel in range(d)]
    )
    return target.astype(config.numpy_dtype)


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    rng = np.random.default_rng(seed + 11)
    target = target_pattern(config, seed)
    corrupted = target + 0.35 * rng.standard_normal(target.shape)
    missing = rng.random(target.shape) < 0.2
    corrupted[missing] = 0.0
    state.field.values = (0.2 * corrupted).astype(config.numpy_dtype)
    state.field.target = target.copy()
    inputs = [
        ExternalInput(
            corrupted.astype(config.numpy_dtype),
            target_field=target,
            input_id=f"spatial:{seed}:{step}",
            supervised_guidance=False,
            metadata={"target_access": "metrics_only"},
        )
        for step in range(config.steps)
    ]
    return TaskBundle(
        "spatial-reconstruction",
        state,
        inputs,
        metadata={
            "corruption_std": 0.35,
            "missing_fraction": 0.2,
            "target": target.tolist(),
            "observation": corrupted.tolist(),
        },
    )
