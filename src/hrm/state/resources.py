"""Finite computational resource state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ResourceState:
    total_budget: float
    current_use: float
    node_allocation: NDArray[np.floating]
    region_allocation: NDArray[np.floating]
    active_width: NDArray[np.int_]
    update_priority: NDArray[np.floating]
    memory_allowance: int
    structural_event_allowance: int
    cumulative_operations: int
    estimated_cost: float
    measured_cost: float
    switching_cost: float
    version: int = 0

    def validate(self, channels: int) -> None:
        n = self.node_allocation.shape[0]
        if self.active_width.shape != (n,) or self.update_priority.shape != (n,):
            raise ValueError("resource node shape mismatch")
        arrays = (self.node_allocation, self.region_allocation, self.update_priority)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("resource arrays contain nonfinite values")
        if np.any(self.node_allocation < 0) or np.any(self.region_allocation < 0):
            raise ValueError("allocations must be nonnegative")
        if np.any(self.active_width < 0) or np.any(self.active_width > channels):
            raise ValueError("active width outside carrier")
        if self.current_use < -1e-12 or self.current_use > self.total_budget + 1e-9:
            raise ValueError("resource budget violated")
