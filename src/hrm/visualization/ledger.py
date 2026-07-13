"""Per-mechanism ledger norm visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_ledger_norms(records: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        step = int(record["step"])
        for proposal in record["main_contributions"]:
            name = str(proposal["mechanism"])
            series.setdefault(name, []).append((step, float(proposal["applied_norm"])))
    figure, axis = plt.subplots(figsize=(9, 5))
    for name, values in sorted(series.items()):
        axis.plot([item[0] for item in values], [item[1] for item in values], label=name)
    axis.set(xlabel="step", ylabel="applied contribution norm", title="Attribution ledger")
    if len(series) <= 12:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
