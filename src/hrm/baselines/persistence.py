"""Identity/persistence baseline."""

from __future__ import annotations

import numpy as np


def persistence(observation: np.ndarray) -> tuple[np.ndarray, float]:
    return np.asarray(observation, dtype=float).copy(), float(np.asarray(observation).size)
