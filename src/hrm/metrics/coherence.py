from __future__ import annotations

import numpy as np


def mean_coherence(local: np.ndarray) -> float:
    return float(np.mean(local))
