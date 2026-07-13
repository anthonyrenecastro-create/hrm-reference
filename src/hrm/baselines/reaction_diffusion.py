"""Fixed-graph reaction–diffusion baseline."""

from __future__ import annotations

import numpy as np

from hrm.operators.laplacian import build_laplacian
from hrm.state.unified import HRMState


def reaction_diffusion(
    state: HRMState,
    observation: np.ndarray,
    *,
    steps: int,
    dt: float,
    diffusion: float,
    a: float,
    b: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(observation, dtype=float).copy()
    laplacian = build_laplacian(state.topology.weights, "symmetric")
    cost = 0.0
    for _ in range(steps):
        values += dt * (-diffusion * (laplacian @ values) + a * values - b * values**3)
        cost += float(2 * laplacian.nnz * values.shape[1] + 4 * values.size)
    return values, cost
