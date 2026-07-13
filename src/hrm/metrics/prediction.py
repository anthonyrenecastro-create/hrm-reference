from __future__ import annotations

import numpy as np


def prediction_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))
