"""Small capacity-controlled linear recurrent baseline."""

from __future__ import annotations

import numpy as np


def recurrent_predict(sequence: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, float]:
    if sequence.shape[0] < 3:
        return sequence[-1].copy(), float(sequence.size)
    x, y = sequence[:-1], sequence[1:]
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    prediction = np.tanh(sequence[-1] @ weights)
    cost = float(x.shape[0] * x.shape[1] ** 2 + x.shape[1] ** 3)
    return prediction, cost
