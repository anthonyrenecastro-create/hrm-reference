from __future__ import annotations

import numpy as np


def collapse_risk(values: np.ndarray, task_error: float) -> float:
    return float((1 / (1 + 100 * np.var(values))) * (1 + task_error))
