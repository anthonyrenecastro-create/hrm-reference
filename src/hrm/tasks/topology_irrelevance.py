"""Fixed-graph-solvable workload used to penalize needless topology churn."""

from __future__ import annotations

import numpy as np

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.state.unified import create_initial_state
from hrm.tasks.base import TaskBundle


def build(config: HRMConfig, seed: int) -> TaskBundle:
    state = create_initial_state(config, seed)
    target = np.full_like(state.field.values, 0.2)
    state.field.values = target.copy()
    inputs = [
        ExternalInput(
            np.zeros_like(target),
            target_field=target,
            input_id=f"topology-irrelevant:{seed}:{step}",
        )
        for step in range(config.steps)
    ]
    return TaskBundle(
        "topology-irrelevance", state, inputs, metadata={"purpose": "avoid_unnecessary_mutation"}
    )
