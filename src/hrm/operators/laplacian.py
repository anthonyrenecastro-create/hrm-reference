"""Sparse Laplacians with explicit sign conventions."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, eye

from hrm.types import LaplacianKind


def build_laplacian(weights: csr_matrix, kind: LaplacianKind | str) -> csr_matrix:
    weights = weights.tocsr()
    degree = np.asarray(weights.sum(axis=1)).ravel()
    if str(kind) == LaplacianKind.COMBINATORIAL:
        return (diags(degree) - weights).tocsr()
    if str(kind) == LaplacianKind.SYMMETRIC:
        inv_sqrt = np.zeros_like(degree, dtype=float)
        positive = degree > 0
        inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        return (
            eye(weights.shape[0], format="csr") - diags(inv_sqrt) @ weights @ diags(inv_sqrt)
        ).tocsr()
    if str(kind) == LaplacianKind.RANDOM_WALK:
        inv = np.zeros_like(degree, dtype=float)
        positive = degree > 0
        inv[positive] = 1.0 / degree[positive]
        return (eye(weights.shape[0], format="csr") - diags(inv) @ weights).tocsr()
    raise ValueError(f"unknown Laplacian kind: {kind}")
