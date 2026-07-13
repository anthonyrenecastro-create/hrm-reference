from __future__ import annotations

import numpy as np


def mean_stress(stress: np.ndarray) -> float:
    return float(np.mean(stress))
