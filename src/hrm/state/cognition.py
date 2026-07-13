"""Bounded predictive and inferential state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class CognitionState:
    latent: NDArray[np.floating]
    prediction: NDArray[np.floating]
    task_prediction: NDArray[np.floating]
    residual: NDArray[np.floating]
    uncertainty: NDArray[np.floating]
    pooled_field_context: NDArray[np.floating]
    retrieved_memory_context: NDArray[np.floating]
    guidance_precursor: NDArray[np.floating]
    version: int = 0

    def validate(self) -> None:
        arrays = (
            self.latent,
            self.prediction,
            self.task_prediction,
            self.residual,
            self.uncertainty,
            self.pooled_field_context,
            self.retrieved_memory_context,
            self.guidance_precursor,
        )
        if any(a.ndim != 1 for a in arrays):
            raise ValueError("cognitive arrays must be vectors")
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("cognition contains nonfinite values")
        if np.any(self.uncertainty < 0):
            raise ValueError("uncertainty must be nonnegative")
