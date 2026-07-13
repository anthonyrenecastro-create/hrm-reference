from __future__ import annotations

import numpy as np


def retention(target: np.ndarray, recalled: np.ndarray) -> float:
    return float(
        np.dot(target, recalled) / ((np.linalg.norm(target) * np.linalg.norm(recalled)) + 1e-12)
    )
