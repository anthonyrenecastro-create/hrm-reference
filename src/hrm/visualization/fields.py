"""Field heatmap visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from hrm.state.unified import HRMState


def plot_field(state: HRMState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(state.field.values, aspect="auto", cmap="coolwarm")
    axis.set(xlabel="channel", ylabel="node", title=f"Field at step {state.step}")
    figure.colorbar(image, ax=axis, label="value")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
