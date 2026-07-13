"""Shared enums and JSON-compatible type helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
Diagnostics: TypeAlias = dict[str, Any]


class StateBlock(StrEnum):
    FIELD = "field"
    GEOMETRY = "geometry"
    TOPOLOGY = "topology"
    MEMORY = "memory"
    COGNITION = "cognition"
    HIERARCHY = "hierarchy"
    RESOURCES = "resources"


class EventType(StrEnum):
    EDGE_GROW = "edge_grow"
    EDGE_PRUNE = "edge_prune"
    EDGE_REWIRE = "edge_rewire"
    EDGE_FLIP = "edge_flip"
    NODE_ACTIVATE = "node_activate"
    NODE_DEACTIVATE = "node_deactivate"
    CHANNEL_ACTIVATE = "channel_activate"
    CHANNEL_DEACTIVATE = "channel_deactivate"
    MEMORY_WRITE = "memory_write"
    MEMORY_EVICT = "memory_evict"
    GEOMETRY_REPAIR = "geometry_repair"
    TOPOLOGY_REPAIR = "topology_repair"


class EventDisposition(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    APPLIED = "applied"
    REPAIRED = "repaired"


class LaplacianKind(StrEnum):
    COMBINATORIAL = "combinatorial"
    SYMMETRIC = "symmetric"
    RANDOM_WALK = "random_walk"


class ReactionKind(StrEnum):
    CUBIC = "cubic"
    TANH = "tanh"


def jsonable(value: Any) -> JSONValue:
    """Convert common numerical and enum values to stable JSON-compatible data."""
    import numpy as np

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    return str(value)
