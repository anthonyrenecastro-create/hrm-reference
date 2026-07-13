"""Fixed-operator graph diffusion baseline."""

from __future__ import annotations

import numpy as np

from hrm.operators.laplacian import build_laplacian
from hrm.state.unified import HRMState


def graph_diffusion(
    state: HRMState,
    observation: np.ndarray,
    *,
    steps: int,
    dt: float,
    coefficient: float,
    laplacian_kind: str = "symmetric",
) -> tuple[np.ndarray, float]:
    values = np.asarray(observation, dtype=float).copy()
    laplacian = build_laplacian(state.topology.weights, laplacian_kind)
    cost = 0.0
    for _ in range(steps):
        values -= dt * coefficient * (laplacian @ values)
        cost += float(2 * laplacian.nnz * values.shape[1])
    return values, cost
