"""Authoritative structured HRM state and deterministic initialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from hrm.config import HRMConfig
from hrm.state.cognition import CognitionState
from hrm.state.field import FieldState
from hrm.state.geometry import GeometryState
from hrm.state.hierarchy import HierarchyState
from hrm.state.memory import MemoryState
from hrm.state.resources import ResourceState
from hrm.state.topology import TopologyState, edge_key


@dataclass(slots=True)
class HRMState:
    field: FieldState
    geometry: GeometryState
    topology: TopologyState
    memory: MemoryState
    cognition: CognitionState
    hierarchy: HierarchyState
    resources: ResourceState
    step: int = 0
    run_metadata: dict[str, Any] = dc_field(default_factory=dict)

    def clone(self) -> HRMState:
        cloned = copy.deepcopy(self)

        def make_writable(value: Any) -> None:
            if isinstance(value, np.ndarray):
                value.setflags(write=True)
            elif hasattr(value, "data") and hasattr(value, "indices") and hasattr(value, "indptr"):
                value.data.setflags(write=True)
                value.indices.setflags(write=True)
                value.indptr.setflags(write=True)
            elif hasattr(value, "__dataclass_fields__"):
                for name in value.__dataclass_fields__:
                    make_writable(getattr(value, name))

        make_writable(cloned)
        return cloned

    def validate(self, config: HRMConfig) -> None:
        self.field.validate()
        self.geometry.validate()
        self.topology.validate(self.field.active_nodes)
        self.memory.validate()
        self.cognition.validate()
        self.hierarchy.validate(self.field.values.shape[0])
        self.resources.validate(self.field.values.shape[1])
        n, d = self.field.values.shape
        dims = config.dimensions
        if (n, d) != (dims.n_nodes, dims.channels):
            raise ValueError("field carrier differs from configured dimensions")
        if self.geometry.coordinates.shape != (n, dims.spatial_dim):
            raise ValueError("geometry carrier differs from configured dimensions")
        if self.memory.keys.shape != (dims.memory_capacity, dims.memory_key_dim):
            raise ValueError("memory key carrier differs from configured dimensions")
        if self.memory.values.shape != (dims.memory_capacity, dims.memory_value_dim):
            raise ValueError("memory value carrier differs from configured dimensions")


def _initial_coordinates(config: HRMConfig) -> np.ndarray:
    n = config.dimensions.n_nodes
    p = config.dimensions.spatial_dim
    if p == 1:
        return config.initialization.coordinate_scale * np.linspace(-1.0, 1.0, n)[:, None]
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    coords = np.zeros((n, p), dtype=config.numpy_dtype)
    coords[:, 0] = np.cos(theta)
    coords[:, 1] = np.sin(theta)
    if p == 3:
        coords[:, 2] = np.linspace(-0.25, 0.25, n)
    return config.initialization.coordinate_scale * coords


def _initial_graph(config: HRMConfig, coords: np.ndarray) -> csr_matrix:
    n = config.dimensions.n_nodes
    edges: set[tuple[int, int]] = set()
    if config.graph.kind == "ring":
        for i in range(n):
            for offset in range(1, config.graph.neighbors + 1):
                a, b = sorted((i, (i + offset) % n))
                edges.add((a, b))
    elif config.graph.kind == "grid":
        side = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            row, col = divmod(i, side)
            for dr, dc in ((1, 0), (0, 1)):
                j = (row + dr) * side + col + dc
                if j < n and (dr == 0 or col == (j % side)):
                    edges.add((i, j))
    else:
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        for i in range(n):
            for j in np.argsort(dist[i])[1 : config.graph.neighbors + 1]:
                a, b = sorted((i, int(j)))
                edges.add((a, b))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, j in sorted(edges):
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        weight = (
            float(np.exp(-(distance**2) / (2.0 * config.graph.distance_scale**2)))
            if config.graph.weighted
            else 1.0
        )
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((weight, weight))
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=config.numpy_dtype)


def create_initial_state(config: HRMConfig, seed: int | None = None) -> HRMState:
    dims = config.dimensions
    rng = np.random.default_rng(config.seed if seed is None else seed)
    dtype = np.dtype(config.numpy_dtype)
    coords = _initial_coordinates(config).astype(dtype)
    weights = _initial_graph(config, coords)
    upper_i, upper_j = weights.nonzero()
    edges = tuple(
        sorted({(int(i), int(j)) for i, j in zip(upper_i, upper_j, strict=False) if i < j})
    )
    degree = np.asarray((weights > 0).sum(axis=1)).ravel().astype(int)
    _, labels = connected_components(weights, directed=False)
    n, d = dims.n_nodes, dims.channels
    active_width = np.full(n, config.resources.max_width or d, dtype=int)
    active_channels = np.arange(d)[None, :] < active_width[:, None]
    values = (config.initialization.field_scale * rng.standard_normal((n, d))).astype(dtype)
    regions = dims.regions
    assignments = (np.arange(n) * regions // n).astype(int)
    restriction_dense = np.zeros((regions, n), dtype=dtype)
    for r in range(regions):
        members = np.flatnonzero(assignments == r)
        restriction_dense[r, members] = 1.0 / max(1, len(members))
    prolongation_dense = np.zeros((n, regions), dtype=dtype)
    prolongation_dense[np.arange(n), assignments] = 1.0
    routing = prolongation_dense.copy()
    key_dim, value_dim = dims.memory_key_dim, dims.memory_value_dim
    state = HRMState(
        field=FieldState(
            values=values,
            active_nodes=np.ones(n, dtype=bool),
            active_channels=active_channels,
            target=None,
            energy=np.zeros(n, dtype=dtype),
            local_coherence=np.zeros(n, dtype=dtype),
            variance=float(np.var(values)),
            collapse_indicator=0.0,
        ),
        geometry=GeometryState(
            coordinates=coords,
            local_scale=np.ones(n, dtype=dtype),
            metric_diagonal=np.ones((n, dims.spatial_dim), dtype=dtype),
            curvature=np.zeros(n, dtype=dtype),
            stress=np.zeros(n, dtype=dtype),
            anchor_coordinates=coords.copy(),
            bounds=np.column_stack(
                (
                    np.full(dims.spatial_dim, -config.safety.coordinate_bound),
                    np.full(dims.spatial_dim, config.safety.coordinate_bound),
                )
            ).astype(dtype),
            accumulated_distortion=0.0,
        ),
        topology=TopologyState(
            weights=weights,
            edges=edges,
            simplices=(),
            degree=degree,
            components=labels.astype(int),
            edge_ages={edge_key(i, j): 0 for i, j in edges},
        ),
        memory=MemoryState(
            working=np.zeros(value_dim, dtype=dtype),
            keys=np.zeros((dims.memory_capacity, key_dim), dtype=dtype),
            values=np.zeros((dims.memory_capacity, value_dim), dtype=dtype),
            strengths=np.zeros(dims.memory_capacity, dtype=dtype),
            ages=np.zeros(dims.memory_capacity, dtype=dtype),
            provenance=tuple("" for _ in range(dims.memory_capacity)),
            occupied=np.zeros(dims.memory_capacity, dtype=bool),
            read_count=0,
            write_count=0,
            eviction_count=0,
            interference=0.0,
        ),
        cognition=CognitionState(
            latent=np.zeros(dims.latent_dim, dtype=dtype),
            prediction=np.zeros(d, dtype=dtype),
            task_prediction=np.zeros(d, dtype=dtype),
            residual=np.zeros(d, dtype=dtype),
            uncertainty=np.ones(d, dtype=dtype),
            pooled_field_context=np.zeros(d, dtype=dtype),
            retrieved_memory_context=np.zeros(value_dim, dtype=dtype),
            guidance_precursor=np.zeros(d, dtype=dtype),
        ),
        hierarchy=HierarchyState(
            region_assignments=assignments,
            coarse_state=np.zeros((regions, d), dtype=dtype),
            restriction=csr_matrix(restriction_dense),
            prolongation=csr_matrix(prolongation_dense),
            routing_weights=routing,
            aggregation_loss=0.0,
            feedback_gain=config.hierarchy.top_down_gain,
        ),
        resources=ResourceState(
            total_budget=config.resources.total_budget,
            current_use=0.0,
            node_allocation=np.ones(n, dtype=dtype),
            region_allocation=np.ones(regions, dtype=dtype),
            active_width=active_width,
            update_priority=np.ones(n, dtype=dtype),
            memory_allowance=config.resources.memory_allowance or dims.memory_capacity,
            structural_event_allowance=config.resources.event_budget,
            cumulative_operations=0,
            estimated_cost=0.0,
            measured_cost=0.0,
            switching_cost=0.0,
        ),
        run_metadata={
            "seed": config.seed if seed is None else seed,
            "config_digest": config.digest,
        },
    )
    state.validate(config)
    return state
