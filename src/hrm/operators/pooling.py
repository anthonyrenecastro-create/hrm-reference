"""Mask-aware pooling helpers."""

from __future__ import annotations

import numpy as np


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int = 0) -> np.ndarray:
    numerator = np.sum(np.where(mask, values, 0.0), axis=axis)
    denominator = np.maximum(1, np.sum(mask, axis=axis))
    result: np.ndarray = numerator / denominator
    return result


def fit_vector(vector: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=float)
    count = min(size, vector.size)
    out[:count] = np.ravel(vector)[:count]
    return out
