"""Minimal deterministic structural repair helpers."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

from hrm.state.unified import HRMState


def component_bridge_candidates(state: HRMState) -> list[tuple[int, int]]:
    active = np.flatnonzero(state.field.active_nodes)
    if active.size <= 1:
        return []
    _, labels = connected_components(state.topology.weights[active][:, active], directed=False)
    components = sorted(set(int(x) for x in labels))
    if len(components) <= 1:
        return []
    coords = state.geometry.coordinates
    bridges: list[tuple[int, int]] = []
    root = active[labels == components[0]]
    for label in components[1:]:
        other = active[labels == label]
        distances = np.linalg.norm(coords[root][:, None, :] - coords[other][None, :, :], axis=2)
        index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        i, j = sorted((int(root[index[0]]), int(other[index[1]])))
        bridges.append((i, j))
        root = np.concatenate((root, other))
    return bridges
