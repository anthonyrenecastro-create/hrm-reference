"""Sparse topology visualization without a NetworkX dependency."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hrm.state.unified import HRMState


def plot_topology(state: HRMState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    coordinates = state.geometry.coordinates
    x = coordinates[:, 0]
    y = coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros_like(x)
    figure, axis = plt.subplots(figsize=(6, 5))
    for i, j in state.topology.edges:
        axis.plot((x[i], x[j]), (y[i], y[j]), color="0.65", linewidth=0.8, zorder=1)
    axis.scatter(
        x,
        y,
        c=state.topology.degree,
        cmap="viridis",
        s=45,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )
    axis.set(title=f"Topology at step {state.step}", xlabel="coordinate 0", ylabel="coordinate 1")
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
