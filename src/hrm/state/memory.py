"""Bounded working and associative memory state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class MemoryState:
    working: NDArray[np.floating]
    keys: NDArray[np.floating]
    values: NDArray[np.floating]
    strengths: NDArray[np.floating]
    ages: NDArray[np.floating]
    provenance: tuple[str, ...]
    occupied: NDArray[np.bool_]
    read_count: int
    write_count: int
    eviction_count: int
    interference: float
    version: int = 0

    @property
    def occupancy(self) -> int:
        return int(np.count_nonzero(self.occupied))

    def validate(self) -> None:
        capacity = self.keys.shape[0]
        if self.keys.ndim != 2 or self.values.ndim != 2 or self.values.shape[0] != capacity:
            raise ValueError("memory keys and values must share capacity")
        if (
            self.strengths.shape != (capacity,)
            or self.ages.shape != (capacity,)
            or self.occupied.shape != (capacity,)
        ):
            raise ValueError("memory metadata shape mismatch")
        if len(self.provenance) != capacity:
            raise ValueError("memory provenance length mismatch")
        arrays = (self.working, self.keys, self.values, self.strengths, self.ages)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("memory contains nonfinite values")
        if np.any(self.strengths < 0) or np.any(self.ages < 0):
            raise ValueError("memory strength and age must be nonnegative")
