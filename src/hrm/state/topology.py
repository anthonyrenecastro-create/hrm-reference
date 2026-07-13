"""Sparse topology state and structural metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


@dataclass(slots=True)
class TopologyState:
    weights: csr_matrix
    edges: tuple[tuple[int, int], ...]
    simplices: tuple[tuple[int, ...], ...]
    degree: NDArray[np.int_]
    components: NDArray[np.int_]
    edge_ages: dict[str, int]
    topology_version: int = 0
    operator_version: int = 0
    event_history: tuple[str, ...] = ()
    structurally_valid: bool = True

    def validate(self, active_nodes: NDArray[np.bool_] | None = None) -> None:
        if self.weights.ndim != 2 or self.weights.shape[0] != self.weights.shape[1]:
            raise ValueError("topology weights must be square")
        n = self.weights.shape[0]
        if self.degree.shape != (n,) or self.components.shape != (n,):
            raise ValueError("topology diagnostic shape mismatch")
        if not np.all(np.isfinite(self.weights.data)) or np.any(self.weights.data < 0):
            raise ValueError("topology weights must be finite and nonnegative")
        if (self.weights - self.weights.T).nnz:
            raise ValueError("reference topology must be symmetric")
        canonical = tuple(sorted((min(i, j), max(i, j)) for i, j in self.edges))
        if canonical != self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("edge set is not canonical")
        if any(i == j or i < 0 or j >= n for i, j in self.edges):
            raise ValueError("invalid edge")
        support = {
            tuple(sorted((int(i), int(j))))
            for i, j in zip(*self.weights.nonzero(), strict=False)
            if i != j
        }
        if support != set(self.edges):
            raise ValueError("weight support does not match accepted edges")
        if active_nodes is not None and active_nodes.shape != (n,):
            raise ValueError("active node mask shape mismatch")


def edge_key(i: int, j: int) -> str:
    return f"{min(i, j)}:{max(i, j)}"
