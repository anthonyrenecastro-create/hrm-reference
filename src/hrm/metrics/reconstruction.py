from __future__ import annotations

import numpy as np


def reconstruction_error(
    values: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None
) -> float:
    delta = (values - target) if mask is None else (values - target)[mask]
    return float(np.mean(delta**2))
