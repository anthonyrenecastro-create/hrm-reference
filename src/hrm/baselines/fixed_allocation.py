"""Equal-budget fixed allocation schedule."""

from __future__ import annotations

import numpy as np


def fixed_width(n_nodes: int, channels: int, total_active: int) -> np.ndarray:
    base, remainder = divmod(total_active, n_nodes)
    widths = np.full(n_nodes, min(channels, base), dtype=int)
    widths[:remainder] = np.minimum(channels, widths[:remainder] + 1)
    return widths
