"""Continuous geometry state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class GeometryState:
    coordinates: NDArray[np.floating]
    local_scale: NDArray[np.floating]
    metric_diagonal: NDArray[np.floating]
    curvature: NDArray[np.floating]
    stress: NDArray[np.floating]
    anchor_coordinates: NDArray[np.floating]
    bounds: NDArray[np.floating]
    accumulated_distortion: float
    version: int = 0

    def validate(self) -> None:
        if self.coordinates.ndim != 2:
            raise ValueError("coordinates must be [nodes, spatial_dim]")
        n, p = self.coordinates.shape
        if self.anchor_coordinates.shape != (n, p):
            raise ValueError("anchor coordinate shape mismatch")
        if self.metric_diagonal.shape != (n, p):
            raise ValueError("metric shape mismatch")
        if (
            self.local_scale.shape != (n,)
            or self.curvature.shape != (n,)
            or self.stress.shape != (n,)
        ):
            raise ValueError("geometry diagnostic shape mismatch")
        if self.bounds.shape != (p, 2):
            raise ValueError("geometry bounds shape mismatch")
        if (
            not np.all(np.isfinite(self.coordinates))
            or np.any(self.local_scale <= 0)
            or np.any(self.metric_diagonal <= 0)
        ):
            raise ValueError("invalid geometry values")
