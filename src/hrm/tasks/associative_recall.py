"""Capacity-controlled associative recall from corrupted cues."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.operators.pooling import fit_vector
from hrm.state.unified import create_initial_state
from hrm.tasks.base import TaskBundle


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    rng = np.random.default_rng(seed + 41)
    count = min(4, config.dimensions.memory_capacity)
    patterns = rng.choice((-1.0, 1.0), size=(count, config.dimensions.channels)).astype(
        config.numpy_dtype
    )
    for slot, pattern in enumerate(patterns):
        state.memory.keys[slot] = fit_vector(pattern, config.dimensions.memory_key_dim)
        state.memory.values[slot] = fit_vector(pattern, config.dimensions.memory_value_dim)
        state.memory.strengths[slot] = 1.0
        state.memory.occupied[slot] = True
    state.memory.provenance = tuple(
        f"seed-pattern:{i}" if i < count else "" for i in range(config.dimensions.memory_capacity)
    )
    target = patterns[seed % count]
    cue = target.copy()
    cue[rng.random(cue.shape) < 0.45] = 0.0
    values = np.broadcast_to(cue, state.field.values.shape).copy()
    target_field = np.broadcast_to(target, state.field.values.shape).copy()
    inputs = [
        ExternalInput(
            values,
            target_field=target_field,
            task_target=target,
            input_id=f"recall:{seed}:{step}",
            metadata={"cue_fraction": float(np.mean(cue != 0))},
        )
        for step in range(config.steps)
    ]
    return TaskBundle(
        "associative-recall",
        state,
        inputs,
        metadata={
            "patterns": patterns.tolist(),
            "cue": cue.tolist(),
            "target": target.tolist(),
            "capacity": count,
        },
    )
