"""Capacity-matched cosine nearest-associate baseline."""

from __future__ import annotations

import numpy as np


def associative_recall(
    keys: np.ndarray, values: np.ndarray, cue: np.ndarray
) -> tuple[np.ndarray, float]:
    cue = np.asarray(cue, dtype=float)
    similarities = (
        keys @ cue / ((np.linalg.norm(keys, axis=1) + 1e-12) * (np.linalg.norm(cue) + 1e-12))
    )
    return values[int(np.argmax(similarities))].copy(), float(keys.size + values.shape[0])
