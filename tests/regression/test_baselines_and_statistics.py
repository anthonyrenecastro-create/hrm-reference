from __future__ import annotations

import numpy as np

from hrm.baselines.associative_memory import associative_recall
from hrm.baselines.fixed_allocation import fixed_width
from hrm.baselines.graph_diffusion import graph_diffusion
from hrm.baselines.persistence import persistence
from hrm.baselines.reaction_diffusion import reaction_diffusion
from hrm.baselines.recurrent import recurrent_predict
from hrm.experiments.statistics import benjamini_hochberg, holm, paired_statistics


def test_capacity_disclosed_baselines_return_finite_outputs(state):
    observation = state.field.values
    persisted, persistence_cost = persistence(observation)
    diffused, diffusion_cost = graph_diffusion(
        state, observation, steps=2, dt=0.01, coefficient=0.1
    )
    reacted, reaction_cost = reaction_diffusion(
        state, observation, steps=2, dt=0.01, diffusion=0.1, a=0.2, b=0.3
    )
    prediction, recurrent_cost = recurrent_predict(np.vstack([observation[:4, 0]] * 3))
    recalled, memory_cost = associative_recall(
        np.eye(3), np.arange(6, dtype=float).reshape(3, 2), np.array([0.0, 1.0, 0.0])
    )
    widths = fixed_width(4, 3, 7)
    for value in (persisted, diffused, reacted, prediction, recalled, widths):
        assert np.all(np.isfinite(value))
    assert all(
        cost > 0
        for cost in (
            persistence_cost,
            diffusion_cost,
            reaction_cost,
            recurrent_cost,
            memory_cost,
        )
    )
    assert widths.sum() == 7


def test_paired_statistics_preserve_per_seed_results_and_corrections():
    result = paired_statistics(np.array([3.0, 4.0, 5.0]), np.array([2.0, 4.5, 4.0]), seed=9)
    assert result["n"] == 3
    assert len(result["per_seed_difference"]) == 3
    assert len(result["bootstrap_ci_95"]) == 2
    for adjusted in (holm([0.01, 0.2, 0.03]), benjamini_hochberg([0.01, 0.2, 0.03])):
        assert len(adjusted) == 3
        assert all(0 <= value <= 1 for value in adjusted)
