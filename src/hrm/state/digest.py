"""Deterministic state digest and numerical projection helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.sparse import issparse

from hrm.state.unified import HRMState


def _update_hash(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"none")
    elif isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        hasher.update(str(contiguous.dtype).encode())
        hasher.update(json.dumps(contiguous.shape).encode())
        hasher.update(contiguous.tobytes())
    elif issparse(value):
        matrix = value.tocsr()
        _update_hash(hasher, matrix.data)
        _update_hash(hasher, matrix.indices)
        _update_hash(hasher, matrix.indptr)
        _update_hash(hasher, np.asarray(matrix.shape, dtype=np.int64))
    elif is_dataclass(value):
        for item in fields(value):
            hasher.update(item.name.encode())
            _update_hash(hasher, getattr(value, item.name))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            hasher.update(str(key).encode())
            _update_hash(hasher, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _update_hash(hasher, item)
    elif isinstance(value, Enum):
        hasher.update(str(value.value).encode())
    else:
        hasher.update(repr(value).encode())


def state_digest(state: HRMState) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, state)
    return hasher.hexdigest()


def numeric_view(state: HRMState) -> dict[str, np.ndarray]:
    """Project authoritative dynamics into a fixed numerical carrier.

    Derived diagnostic caches and structural histories are intentionally excluded.
    Boolean masks and integer allocations are projected as float arrays so event
    effects remain exactly reconstructible.
    """
    t = state.topology.weights.toarray()
    return {
        "field.values": np.asarray(state.field.values, dtype=float).copy(),
        "field.active_nodes": state.field.active_nodes.astype(float),
        "field.active_channels": state.field.active_channels.astype(float),
        "geometry.coordinates": np.asarray(state.geometry.coordinates, dtype=float).copy(),
        "geometry.local_scale": np.asarray(state.geometry.local_scale, dtype=float).copy(),
        "geometry.metric_diagonal": np.asarray(state.geometry.metric_diagonal, dtype=float).copy(),
        "topology.weights": np.asarray(t, dtype=float),
        "memory.working": np.asarray(state.memory.working, dtype=float).copy(),
        "memory.keys": np.asarray(state.memory.keys, dtype=float).copy(),
        "memory.values": np.asarray(state.memory.values, dtype=float).copy(),
        "memory.strengths": np.asarray(state.memory.strengths, dtype=float).copy(),
        "memory.ages": np.asarray(state.memory.ages, dtype=float).copy(),
        "memory.occupied": state.memory.occupied.astype(float),
        "memory.read_count": np.asarray(float(state.memory.read_count)),
        "memory.write_count": np.asarray(float(state.memory.write_count)),
        "memory.eviction_count": np.asarray(float(state.memory.eviction_count)),
        "memory.interference": np.asarray(float(state.memory.interference)),
        "cognition.latent": np.asarray(state.cognition.latent, dtype=float).copy(),
        "cognition.prediction": np.asarray(state.cognition.prediction, dtype=float).copy(),
        "cognition.task_prediction": np.asarray(
            state.cognition.task_prediction, dtype=float
        ).copy(),
        "cognition.residual": np.asarray(state.cognition.residual, dtype=float).copy(),
        "cognition.uncertainty": np.asarray(state.cognition.uncertainty, dtype=float).copy(),
        "cognition.pooled_field_context": np.asarray(
            state.cognition.pooled_field_context, dtype=float
        ).copy(),
        "cognition.retrieved_memory_context": np.asarray(
            state.cognition.retrieved_memory_context, dtype=float
        ).copy(),
        "cognition.guidance_precursor": np.asarray(
            state.cognition.guidance_precursor, dtype=float
        ).copy(),
        "hierarchy.coarse_state": np.asarray(state.hierarchy.coarse_state, dtype=float).copy(),
        "hierarchy.routing_weights": np.asarray(
            state.hierarchy.routing_weights, dtype=float
        ).copy(),
        "hierarchy.aggregation_loss": np.asarray(float(state.hierarchy.aggregation_loss)),
        "hierarchy.feedback_gain": np.asarray(float(state.hierarchy.feedback_gain)),
        "resources.current_use": np.asarray(float(state.resources.current_use)),
        "resources.node_allocation": np.asarray(
            state.resources.node_allocation, dtype=float
        ).copy(),
        "resources.region_allocation": np.asarray(
            state.resources.region_allocation, dtype=float
        ).copy(),
        "resources.active_width": np.asarray(state.resources.active_width, dtype=float).copy(),
        "resources.update_priority": np.asarray(
            state.resources.update_priority, dtype=float
        ).copy(),
        "resources.cumulative_operations": np.asarray(float(state.resources.cumulative_operations)),
        "resources.estimated_cost": np.asarray(float(state.resources.estimated_cost)),
        "resources.measured_cost": np.asarray(float(state.resources.measured_cost)),
        "resources.switching_cost": np.asarray(float(state.resources.switching_cost)),
    }


def numeric_state_digest(state: HRMState) -> str:
    """Digest only the ledger-authoritative numerical carrier.

    Use this for matched trajectory divergence. The complete state digest also
    contains configuration/run metadata and would make two configurations appear
    divergent even when their numerical trajectories are identical.
    """
    hasher = hashlib.sha256()
    _update_hash(hasher, numeric_view(state))
    return hasher.hexdigest()


def view_delta(
    after: dict[str, np.ndarray], before: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if after.keys() != before.keys():
        raise ValueError("numeric carrier keys differ")
    return {
        key: np.asarray(after[key], dtype=float) - np.asarray(before[key], dtype=float)
        for key in after
    }
