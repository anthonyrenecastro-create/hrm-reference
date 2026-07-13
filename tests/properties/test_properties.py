from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.core.transition import TransitionEngine
from hrm.state.unified import create_initial_state


@given(
    seed=st.integers(min_value=0, max_value=10000),
    scale=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=12, deadline=None)
def test_ledger_identity_over_seeded_inputs(seed, scale):
    config = HRMConfig.model_validate(
        {
            "seed": seed,
            "dimensions": {
                "n_nodes": 8,
                "channels": 3,
                "spatial_dim": 2,
                "latent_dim": 3,
                "memory_capacity": 4,
                "memory_key_dim": 3,
                "memory_value_dim": 3,
                "regions": 2,
            },
            "resources": {"total_budget": 5000, "max_width": 3, "event_budget": 2},
        }
    )
    state = create_initial_state(config)
    external = ExternalInput(np.full_like(state.field.values, scale))
    result = TransitionEngine(config).step(state, external, run_id="property")
    assert result.ledger.relative_residual <= config.ledger.tolerance
