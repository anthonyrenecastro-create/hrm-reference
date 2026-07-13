"""Conservative explicit-step stability estimates."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def largest_eigenvalue(matrix: csr_matrix) -> float:
    if matrix.shape[0] <= 256:
        dense = matrix.toarray()
        if np.allclose(dense, dense.T, rtol=0.0, atol=0.0):
            return float(np.max(np.abs(np.linalg.eigvalsh(dense))))
        return float(np.max(np.abs(np.linalg.eigvals(dense))))
    try:
        start = np.linspace(1.0, 2.0, matrix.shape[0])
        start /= np.linalg.norm(start)
        return float(abs(eigsh(matrix, k=1, which="LM", return_eigenvectors=False, v0=start)[0]))
    except Exception:
        return float(np.linalg.norm(matrix.toarray(), ord=2))


def stable_timestep(
    laplacian: csr_matrix,
    diffusion: float,
    reaction_bound: float,
    coupling_bound: float,
    requested: float,
) -> tuple[float, float]:
    eigenvalue = largest_eigenvalue(laplacian)
    denominator = diffusion * eigenvalue + reaction_bound + coupling_bound + 1e-12
    bound = 2.0 / denominator if denominator > 0 else requested
    return min(requested, 0.95 * bound), eigenvalue
