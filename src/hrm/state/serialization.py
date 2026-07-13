"""Stable JSON-in-NPZ state serialization for deterministic replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.sparse import csr_matrix

from hrm.state.cognition import CognitionState
from hrm.state.field import FieldState
from hrm.state.geometry import GeometryState
from hrm.state.hierarchy import HierarchyState
from hrm.state.memory import MemoryState
from hrm.state.resources import ResourceState
from hrm.state.topology import TopologyState
from hrm.state.unified import HRMState
from hrm.types import jsonable


def state_to_dict(state: HRMState) -> dict[str, Any]:
    def arr(value: np.ndarray) -> list[Any]:
        return cast(list[Any], value.tolist())

    return {
        "schema": "hrm-state-1.0",
        "step": state.step,
        "run_metadata": jsonable(state.run_metadata),
        "field": {
            "values": arr(state.field.values),
            "active_nodes": arr(state.field.active_nodes),
            "active_channels": arr(state.field.active_channels),
            "target": None if state.field.target is None else arr(state.field.target),
            "energy": arr(state.field.energy),
            "local_coherence": arr(state.field.local_coherence),
            "variance": state.field.variance,
            "collapse_indicator": state.field.collapse_indicator,
            "version": state.field.version,
        },
        "geometry": {
            "coordinates": arr(state.geometry.coordinates),
            "local_scale": arr(state.geometry.local_scale),
            "metric_diagonal": arr(state.geometry.metric_diagonal),
            "curvature": arr(state.geometry.curvature),
            "stress": arr(state.geometry.stress),
            "anchor_coordinates": arr(state.geometry.anchor_coordinates),
            "bounds": arr(state.geometry.bounds),
            "accumulated_distortion": state.geometry.accumulated_distortion,
            "version": state.geometry.version,
        },
        "topology": {
            "weights": state.topology.weights.toarray().tolist(),
            "edges": state.topology.edges,
            "simplices": state.topology.simplices,
            "degree": arr(state.topology.degree),
            "components": arr(state.topology.components),
            "edge_ages": state.topology.edge_ages,
            "topology_version": state.topology.topology_version,
            "operator_version": state.topology.operator_version,
            "event_history": state.topology.event_history,
            "structurally_valid": state.topology.structurally_valid,
        },
        "memory": {
            "working": arr(state.memory.working),
            "keys": arr(state.memory.keys),
            "values": arr(state.memory.values),
            "strengths": arr(state.memory.strengths),
            "ages": arr(state.memory.ages),
            "provenance": state.memory.provenance,
            "occupied": arr(state.memory.occupied),
            "read_count": state.memory.read_count,
            "write_count": state.memory.write_count,
            "eviction_count": state.memory.eviction_count,
            "interference": state.memory.interference,
            "version": state.memory.version,
        },
        "cognition": {
            key: arr(getattr(state.cognition, key))
            for key in (
                "latent",
                "prediction",
                "task_prediction",
                "residual",
                "uncertainty",
                "pooled_field_context",
                "retrieved_memory_context",
                "guidance_precursor",
            )
        }
        | {"version": state.cognition.version},
        "hierarchy": {
            "region_assignments": arr(state.hierarchy.region_assignments),
            "coarse_state": arr(state.hierarchy.coarse_state),
            "restriction": state.hierarchy.restriction.toarray().tolist(),
            "prolongation": state.hierarchy.prolongation.toarray().tolist(),
            "routing_weights": arr(state.hierarchy.routing_weights),
            "aggregation_loss": state.hierarchy.aggregation_loss,
            "feedback_gain": state.hierarchy.feedback_gain,
            "version": state.hierarchy.version,
        },
        "resources": {
            "total_budget": state.resources.total_budget,
            "current_use": state.resources.current_use,
            "node_allocation": arr(state.resources.node_allocation),
            "region_allocation": arr(state.resources.region_allocation),
            "active_width": arr(state.resources.active_width),
            "update_priority": arr(state.resources.update_priority),
            "memory_allowance": state.resources.memory_allowance,
            "structural_event_allowance": state.resources.structural_event_allowance,
            "cumulative_operations": state.resources.cumulative_operations,
            "estimated_cost": state.resources.estimated_cost,
            "measured_cost": state.resources.measured_cost,
            "switching_cost": state.resources.switching_cost,
            "version": state.resources.version,
        },
    }


def state_from_dict(data: dict[str, Any], dtype: str = "float64") -> HRMState:
    dt = np.dtype(dtype)
    f, g, t, m, c, h, b = (
        data[k]
        for k in ("field", "geometry", "topology", "memory", "cognition", "hierarchy", "resources")
    )
    return HRMState(
        field=FieldState(
            np.asarray(f["values"], dtype=dt),
            np.asarray(f["active_nodes"], dtype=bool),
            np.asarray(f["active_channels"], dtype=bool),
            None if f["target"] is None else np.asarray(f["target"], dtype=dt),
            np.asarray(f["energy"], dtype=dt),
            np.asarray(f["local_coherence"], dtype=dt),
            float(f["variance"]),
            float(f["collapse_indicator"]),
            int(f["version"]),
        ),
        geometry=GeometryState(
            np.asarray(g["coordinates"], dtype=dt),
            np.asarray(g["local_scale"], dtype=dt),
            np.asarray(g["metric_diagonal"], dtype=dt),
            np.asarray(g["curvature"], dtype=dt),
            np.asarray(g["stress"], dtype=dt),
            np.asarray(g["anchor_coordinates"], dtype=dt),
            np.asarray(g["bounds"], dtype=dt),
            float(g["accumulated_distortion"]),
            int(g["version"]),
        ),
        topology=TopologyState(
            csr_matrix(np.asarray(t["weights"], dtype=dt)),
            tuple((int(e[0]), int(e[1])) for e in t["edges"]),
            tuple(tuple(int(x) for x in s) for s in t["simplices"]),
            np.asarray(t["degree"], dtype=int),
            np.asarray(t["components"], dtype=int),
            {str(k): int(v) for k, v in t["edge_ages"].items()},
            int(t["topology_version"]),
            int(t["operator_version"]),
            tuple(t["event_history"]),
            bool(t["structurally_valid"]),
        ),
        memory=MemoryState(
            np.asarray(m["working"], dtype=dt),
            np.asarray(m["keys"], dtype=dt),
            np.asarray(m["values"], dtype=dt),
            np.asarray(m["strengths"], dtype=dt),
            np.asarray(m["ages"], dtype=dt),
            tuple(m["provenance"]),
            np.asarray(m["occupied"], dtype=bool),
            int(m["read_count"]),
            int(m["write_count"]),
            int(m["eviction_count"]),
            float(m["interference"]),
            int(m["version"]),
        ),
        cognition=CognitionState(
            latent=np.asarray(c["latent"], dtype=dt),
            prediction=np.asarray(c["prediction"], dtype=dt),
            task_prediction=np.asarray(c["task_prediction"], dtype=dt),
            residual=np.asarray(c["residual"], dtype=dt),
            uncertainty=np.asarray(c["uncertainty"], dtype=dt),
            pooled_field_context=np.asarray(c["pooled_field_context"], dtype=dt),
            retrieved_memory_context=np.asarray(c["retrieved_memory_context"], dtype=dt),
            guidance_precursor=np.asarray(c["guidance_precursor"], dtype=dt),
            version=int(c["version"]),
        ),
        hierarchy=HierarchyState(
            np.asarray(h["region_assignments"], dtype=int),
            np.asarray(h["coarse_state"], dtype=dt),
            csr_matrix(np.asarray(h["restriction"], dtype=dt)),
            csr_matrix(np.asarray(h["prolongation"], dtype=dt)),
            np.asarray(h["routing_weights"], dtype=dt),
            float(h["aggregation_loss"]),
            float(h["feedback_gain"]),
            int(h["version"]),
        ),
        resources=ResourceState(
            float(b["total_budget"]),
            float(b["current_use"]),
            np.asarray(b["node_allocation"], dtype=dt),
            np.asarray(b["region_allocation"], dtype=dt),
            np.asarray(b["active_width"], dtype=int),
            np.asarray(b["update_priority"], dtype=dt),
            int(b["memory_allowance"]),
            int(b["structural_event_allowance"]),
            int(b["cumulative_operations"]),
            float(b["estimated_cost"]),
            float(b["measured_cost"]),
            float(b["switching_cost"]),
            int(b["version"]),
        ),
        step=int(data["step"]),
        run_metadata=dict(data.get("run_metadata", {})),
    )


def save_state(state: HRMState, path: str | Path) -> None:
    payload = json.dumps(state_to_dict(state), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(Path(path), state_json=np.asarray(payload))


def load_state(path: str | Path, dtype: str = "float64") -> HRMState:
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = str(archive["state_json"].item())
    return state_from_dict(json.loads(payload), dtype=dtype)
