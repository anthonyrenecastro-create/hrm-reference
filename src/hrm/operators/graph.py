"""Derived effective graph weights."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from hrm.config import HRMConfig
from hrm.state.unified import HRMState


def effective_weights(state: HRMState, config: HRMConfig) -> csr_matrix:
    """Combine topology-owned confidence with geometry-owned distance."""
    base = state.topology.weights.tocoo()
    if config.fixed_graph or not config.geometry.adaptive_weights:
        return base.tocsr(copy=True)
    coords = state.geometry.coordinates
    distances = np.linalg.norm(coords[base.row] - coords[base.col], axis=1)
    geometry_factor = np.exp(-(distances**2) / (2.0 * config.graph.distance_scale**2))
    return csr_matrix((base.data * geometry_factor, (base.row, base.col)), shape=base.shape)


def active_edge_count(state: HRMState) -> int:
    active = state.field.active_nodes
    return sum(1 for i, j in state.topology.edges if active[i] and active[j])
