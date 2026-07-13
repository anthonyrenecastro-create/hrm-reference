"""Raw metric trajectory visualization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectories(
    metrics: pd.DataFrame,
    path: str | Path,
    columns: Iterable[str] = ("reconstruction_error", "prediction_error", "geometric_stress"),
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    for column in columns:
        if column in metrics and not metrics[column].isna().all():
            axis.plot(metrics["step"], metrics[column], label=column)
    axis.set(xlabel="step", ylabel="raw metric", title="HRM metric trajectories")
    if axis.lines:
        axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
