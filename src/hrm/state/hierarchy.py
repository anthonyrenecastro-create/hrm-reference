"""One-level hierarchy state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


@dataclass(slots=True)
class HierarchyState:
    region_assignments: NDArray[np.int_]
    coarse_state: NDArray[np.floating]
    restriction: csr_matrix
    prolongation: csr_matrix
    routing_weights: NDArray[np.floating]
    aggregation_loss: float
    feedback_gain: float
    version: int = 0

    def validate(self, n_nodes: int) -> None:
        regions = self.coarse_state.shape[0]
        if self.region_assignments.shape != (n_nodes,):
            raise ValueError("region assignment shape mismatch")
        if self.restriction.shape != (regions, n_nodes) or self.prolongation.shape != (
            n_nodes,
            regions,
        ):
            raise ValueError("hierarchy operator shape mismatch")
        if self.routing_weights.shape != (n_nodes, regions):
            raise ValueError("routing weight shape mismatch")
        if np.any(self.region_assignments < 0) or np.any(self.region_assignments >= regions):
            raise ValueError("invalid region assignment")
        if not np.all(np.isfinite(self.coarse_state)) or not np.all(
            np.isfinite(self.routing_weights)
        ):
            raise ValueError("hierarchy contains nonfinite values")
