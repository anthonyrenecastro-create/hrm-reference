"""Distributed field state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class FieldState:
    values: NDArray[np.floating]
    active_nodes: NDArray[np.bool_]
    active_channels: NDArray[np.bool_]
    target: NDArray[np.floating] | None
    energy: NDArray[np.floating]
    local_coherence: NDArray[np.floating]
    variance: float
    collapse_indicator: float
    version: int = 0

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("field values must be [nodes, channels]")
        n, d = self.values.shape
        if self.active_nodes.shape != (n,):
            raise ValueError("active_nodes shape mismatch")
        if self.active_channels.shape != (n, d):
            raise ValueError("active_channels shape mismatch")
        if self.target is not None and self.target.shape != (n, d):
            raise ValueError("field target shape mismatch")
        if self.energy.shape != (n,) or self.local_coherence.shape != (n,):
            raise ValueError("field diagnostic shape mismatch")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("field contains nonfinite values")
