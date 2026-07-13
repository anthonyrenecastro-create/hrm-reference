"""Geometry and stress visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hrm.state.unified import HRMState


def _xy(state: HRMState) -> tuple[np.ndarray, np.ndarray]:
    coordinates = state.geometry.coordinates
    x = coordinates[:, 0]
    y = coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros_like(x)
    return x, y


def plot_geometry(state: HRMState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x, y = _xy(state)
    figure, axis = plt.subplots(figsize=(6, 5))
    points = axis.scatter(x, y, c=state.geometry.stress, cmap="magma", s=45)
    axis.set(
        title=f"Geometry and stress at step {state.step}",
        xlabel="coordinate 0",
        ylabel="coordinate 1",
    )
    axis.set_aspect("equal", adjustable="datalim")
    figure.colorbar(points, ax=axis, label="local stress")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
