from __future__ import annotations

import pytest

from hrm.core.transition import TransitionEngine
from hrm.tasks import build_task


@pytest.mark.parametrize(
    "name",
    [
        "spatial-reconstruction",
        "sequence-prediction",
        "associative-recall",
        "perturbation-recovery",
        "resource-adaptation",
        "topology-usefulness",
        "topology-irrelevance",
    ],
)
def test_every_canonical_task_runs_one_valid_step(config, name):
    bundle = build_task(name, config, config.seed)
    result = TransitionEngine(config).step(bundle.initial_state, bundle.inputs[0], run_id=name)
    assert result.ledger.relative_residual <= config.ledger.tolerance
