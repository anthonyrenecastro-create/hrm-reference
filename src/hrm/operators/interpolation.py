"""Deterministic fixed-carrier initialization and restriction helpers."""

from __future__ import annotations

import numpy as np


def neighbor_interpolation(values: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    if neighbors.size == 0:
        return np.zeros(values.shape[1], dtype=values.dtype)
    result: np.ndarray = np.mean(values[neighbors], axis=0)
    return result
