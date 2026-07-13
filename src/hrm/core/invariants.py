"""Numerical carrier mutation helpers and deterministic derived-state refresh."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from hrm.config import HRMConfig
from hrm.state.topology import edge_key
from hrm.state.unified import HRMState
from hrm.types import StateBlock


def assign_numeric_path(state: HRMState, path: str, value: np.ndarray) -> None:
    block_name, attribute = path.split(".", 1)
    block = getattr(state, block_name)
    current = getattr(block, attribute)
    incoming = np.asarray(value)
    if path == "topology.weights":
        setattr(block, attribute, csr_matrix(incoming, dtype=state.field.values.dtype))
    elif isinstance(current, np.ndarray):
        setattr(block, attribute, np.asarray(incoming, dtype=current.dtype))
    elif isinstance(current, bool):
        setattr(block, attribute, bool(incoming.item()))
    elif isinstance(current, int):
        setattr(block, attribute, int(round(float(incoming.item()))))
    else:
        setattr(block, attribute, float(incoming.item()))


def apply_deltas(state: HRMState, deltas: dict[str, np.ndarray]) -> HRMState:
    result = state.clone()
    for path, delta in deltas.items():
        block_name, attribute = path.split(".", 1)
        current = getattr(getattr(result, block_name), attribute)
        base = current.toarray() if path == "topology.weights" else np.asarray(current)
        assign_numeric_path(
            result, path, np.asarray(base, dtype=float) + np.asarray(delta, dtype=float)
        )
    return result


def refresh_topology(state: HRMState) -> None:
    weights = state.topology.weights.tocsr()
    weights.setdiag(0)
    weights.eliminate_zeros()
    state.topology.weights = weights
    rows, cols = weights.nonzero()
    state.topology.edges = tuple(
        sorted({(int(i), int(j)) for i, j in zip(rows, cols, strict=False) if i < j})
    )
    state.topology.degree = np.asarray((weights > 0).sum(axis=1)).ravel().astype(int)
    _, labels = connected_components(weights, directed=False)
    state.topology.components = labels.astype(int)
    for edge in state.topology.edges:
        state.topology.edge_ages.setdefault(edge_key(*edge), 0)
    state.topology.edge_ages = {
        key: age
        for key, age in state.topology.edge_ages.items()
        if key in {edge_key(*e) for e in state.topology.edges}
    }


def refresh_derived(state: HRMState, config: HRMConfig, changed_paths: Iterable[str]) -> None:
    values = np.where(state.field.active_channels, state.field.values, 0.0)
    weights = state.topology.weights
    energy = np.zeros(values.shape[0], dtype=state.field.values.dtype)
    coherence = np.ones(values.shape[0], dtype=state.field.values.dtype)
    for i in range(values.shape[0]):
        start, end = weights.indptr[i], weights.indptr[i + 1]
        js, ws = weights.indices[start:end], weights.data[start:end]
        if js.size:
            disagreement = np.sum(ws[:, None] * (values[js] - values[i]) ** 2)
            energy[i] = 0.5 * disagreement + 0.25 * np.sum(values[i] ** 4)
            coherence[i] = 1.0 / (1.0 + disagreement / (np.sum(ws) * values.shape[1] + 1e-12))
    state.field.energy = energy
    state.field.local_coherence = coherence
    active_values = values[state.field.active_channels]
    state.field.variance = float(np.var(active_values)) if active_values.size else 0.0
    state.field.collapse_indicator = float(1.0 / (1.0 + 100.0 * state.field.variance))
    refresh_topology(state)
    coordinates = state.geometry.coordinates
    geometry_stress = np.zeros(coordinates.shape[0], dtype=coordinates.dtype)
    curvature = np.zeros(coordinates.shape[0], dtype=coordinates.dtype)
    for i, j in state.topology.edges:
        feature_distance = float(np.linalg.norm(values[i] - values[j]))
        preferred = min(
            config.geometry.max_separation,
            max(config.geometry.min_separation, 0.25 + feature_distance),
        )
        mismatch = abs(float(np.linalg.norm(coordinates[i] - coordinates[j])) - preferred)
        geometry_stress[i] += mismatch
        geometry_stress[j] += mismatch
    for i in range(coordinates.shape[0]):
        start, end = weights.indptr[i], weights.indptr[i + 1]
        js, ws = weights.indices[start:end], weights.data[start:end]
        if js.size:
            geometry_stress[i] /= max(1, js.size)
            neighbor_center = np.average(coordinates[js], axis=0, weights=ws)
            curvature[i] = np.linalg.norm(neighbor_center - coordinates[i])
    state.geometry.stress = geometry_stress
    state.geometry.curvature = curvature
    state.geometry.accumulated_distortion += float(
        np.mean(np.linalg.norm(coordinates - state.geometry.anchor_coordinates, axis=1))
    )
    changed_blocks = {path.split(".", 1)[0] for path in changed_paths}
    for name in changed_blocks:
        block = getattr(state, name)
        if hasattr(block, "version"):
            block.version += 1
    if "topology" in changed_blocks:
        state.topology.topology_version += 1
        state.topology.operator_version += 1
    if "geometry" in changed_blocks and config.geometry.adaptive_weights:
        state.topology.operator_version += 1
    state.step += 1


def block_from_path(path: str) -> StateBlock:
    return StateBlock(path.split(".", 1)[0])
