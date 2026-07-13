from __future__ import annotations

import numpy as np

from hrm.operators.graph import effective_weights
from hrm.operators.laplacian import build_laplacian
from hrm.operators.stability import largest_eigenvalue, stable_timestep


def test_laplacian_sign_and_spectrum(config, state):
    weights = effective_weights(state, config)
    for kind in ("combinatorial", "symmetric", "random_walk"):
        laplacian = build_laplacian(weights, kind)
        assert laplacian.shape == weights.shape
        assert largest_eigenvalue(laplacian) >= 0


def test_stability_bound_can_reduce_requested_step(config, state):
    laplacian = build_laplacian(state.topology.weights, "symmetric")
    step, eigenvalue = stable_timestep(laplacian, 10.0, 10.0, 10.0, 1.0)
    assert step < 1.0
    assert eigenvalue > 0


def test_geometry_changes_derived_weights(config, state):
    before = effective_weights(state, config).toarray()
    state.geometry.coordinates *= 1.5
    after = effective_weights(state, config).toarray()
    assert not np.array_equal(before, after)
