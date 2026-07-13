from __future__ import annotations

import numpy as np
import pytest

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.state.unified import create_initial_state


@pytest.fixture
def config() -> HRMConfig:
    return HRMConfig.model_validate(
        {
            "steps": 4,
            "dimensions": {
                "n_nodes": 10,
                "channels": 4,
                "spatial_dim": 2,
                "latent_dim": 4,
                "memory_capacity": 6,
                "memory_key_dim": 4,
                "memory_value_dim": 4,
                "regions": 2,
            },
            "resources": {
                "total_budget": 5000,
                "min_width": 1,
                "max_width": 4,
                "event_budget": 3,
                "memory_allowance": 6,
            },
        }
    )


@pytest.fixture
def state(config: HRMConfig):
    return create_initial_state(config)


@pytest.fixture
def external(state):
    return ExternalInput(np.zeros_like(state.field.values), input_id="test")
