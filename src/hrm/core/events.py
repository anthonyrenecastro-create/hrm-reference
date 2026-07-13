"""Typed immutable structural events, validation, and application."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from hrm.config import HRMConfig
from hrm.core.invariants import refresh_topology
from hrm.state.digest import numeric_view, view_delta
from hrm.state.topology import edge_key
from hrm.state.unified import HRMState
from hrm.types import Diagnostics, EventDisposition, EventType, jsonable

TOPOLOGY_EVENT_TYPES = frozenset(
    {
        EventType.EDGE_GROW,
        EventType.EDGE_PRUNE,
        EventType.EDGE_REWIRE,
        EventType.EDGE_FLIP,
        EventType.NODE_ACTIVATE,
        EventType.NODE_DEACTIVATE,
        EventType.TOPOLOGY_REPAIR,
    }
)


@dataclass(frozen=True, slots=True)
class StructuralEvent:
    event_id: str
    event_type: EventType
    candidate_basis: Diagnostics
    affected_entities: tuple[str, ...]
    proposed_score: float
    predicted_utility: float
    estimated_cost: float
    estimated_risk: float
    validation_result: bool | None = None
    accepted: bool = False
    rejection_reason: str | None = None
    deterministic_ordering_key: tuple[float | str, ...] = ()
    induced_transport: Diagnostics = field(
        default_factory=lambda: {"kind": "identity_fixed_carrier"}
    )
    applied_numerical_effect: Diagnostics = field(default_factory=dict)
    disposition: EventDisposition = EventDisposition.PROPOSED

    @classmethod
    def create(
        cls,
        event_type: EventType,
        candidate_basis: Diagnostics,
        affected_entities: tuple[str, ...],
        score: float,
        utility: float,
        cost: float,
        risk: float,
        step: int,
    ) -> StructuralEvent:
        payload = json.dumps(
            [str(event_type), jsonable(candidate_basis), affected_entities, step], sort_keys=True
        )
        event_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
        order: tuple[float | str, ...] = (
            -float(score),
            str(event_type),
            *affected_entities,
            event_id,
        )
        return cls(
            event_id,
            event_type,
            candidate_basis,
            affected_entities,
            float(score),
            float(utility),
            float(cost),
            float(risk),
            deterministic_ordering_key=order,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": str(self.event_type),
            "candidate_basis": jsonable(self.candidate_basis),
            "affected_entities": list(self.affected_entities),
            "proposed_score": self.proposed_score,
            "predicted_utility": self.predicted_utility,
            "estimated_cost": self.estimated_cost,
            "estimated_risk": self.estimated_risk,
            "validation_result": self.validation_result,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "deterministic_ordering_key": jsonable(self.deterministic_ordering_key),
            "induced_transport": jsonable(self.induced_transport),
            "applied_numerical_effect": jsonable(self.applied_numerical_effect),
            "disposition": str(self.disposition),
        }


def _edge_from(event: StructuralEvent, key: str = "edge") -> tuple[int, int]:
    raw = event.candidate_basis[key]
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"event {event.event_id} has invalid {key}")
    i, j = sorted((int(raw[0]), int(raw[1])))
    return i, j


def _connected_after(
    weights: csr_matrix, edge: tuple[int, int], add: bool, active: np.ndarray
) -> bool:
    matrix = weights.tolil(copy=True)
    i, j = edge
    matrix[i, j] = matrix[j, i] = 1.0 if add else 0.0
    nodes = np.flatnonzero(active)
    if nodes.size <= 1:
        return True
    count, _ = connected_components(matrix.tocsr()[nodes][:, nodes], directed=False)
    return bool(count == 1)


def _connected_after_rewire(
    weights: csr_matrix,
    remove_edge: tuple[int, int],
    add_edge: tuple[int, int],
    active: np.ndarray,
) -> bool:
    matrix = weights.tolil(copy=True)
    matrix[remove_edge] = matrix[remove_edge[::-1]] = 0.0
    matrix[add_edge] = matrix[add_edge[::-1]] = 1.0
    nodes = np.flatnonzero(active)
    if nodes.size <= 1:
        return True
    count, _ = connected_components(matrix.tocsr()[nodes][:, nodes], directed=False)
    return bool(count == 1)


def _invalid_basis(event: StructuralEvent, state: HRMState) -> str | None:
    scores = (
        event.proposed_score,
        event.predicted_utility,
        event.estimated_cost,
        event.estimated_risk,
    )
    if not all(np.isfinite(value) for value in scores):
        return "nonfinite_event_metadata"
    if event.estimated_cost < 0 or event.estimated_risk < 0:
        return "negative_event_cost_or_risk"
    n, channels = state.field.values.shape
    try:
        if event.event_type in {
            EventType.EDGE_GROW,
            EventType.EDGE_PRUNE,
            EventType.TOPOLOGY_REPAIR,
        }:
            edge = _edge_from(event)
            if edge[0] < 0 or edge[1] >= n:
                return "edge_out_of_range"
        elif event.event_type == EventType.EDGE_REWIRE:
            for key in ("remove_edge", "add_edge"):
                edge = _edge_from(event, key)
                if edge[0] < 0 or edge[1] >= n:
                    return "edge_out_of_range"
        elif event.event_type in {EventType.NODE_ACTIVATE, EventType.NODE_DEACTIVATE}:
            if not 0 <= int(event.candidate_basis["node"]) < n:
                return "node_out_of_range"
        elif event.event_type in {
            EventType.CHANNEL_ACTIVATE,
            EventType.CHANNEL_DEACTIVATE,
        }:
            node = int(event.candidate_basis["node"])
            channel = int(event.candidate_basis["channel"])
            if not 0 <= node < n or not 0 <= channel < channels:
                return "allocation_index_out_of_range"
        elif event.event_type in {EventType.MEMORY_WRITE, EventType.MEMORY_EVICT}:
            slot = int(event.candidate_basis["slot"])
            if not 0 <= slot < state.memory.occupied.size:
                return "memory_slot_out_of_range"
            if event.event_type == EventType.MEMORY_WRITE:
                memory_key = np.asarray(event.candidate_basis["key"], dtype=float)
                value = np.asarray(event.candidate_basis["value"], dtype=float)
                strength = float(event.candidate_basis["strength"])
                if memory_key.shape != state.memory.keys[slot].shape:
                    return "memory_key_shape"
                if value.shape != state.memory.values[slot].shape:
                    return "memory_value_shape"
                if not np.all(np.isfinite(memory_key)) or not np.all(np.isfinite(value)):
                    return "nonfinite_memory_payload"
                if not np.isfinite(strength) or strength < 0:
                    return "invalid_memory_strength"
    except (KeyError, TypeError, ValueError):
        return "invalid_event_basis"
    weight = event.candidate_basis.get("weight")
    if weight is not None and (not np.isfinite(float(weight)) or float(weight) < 0):
        return "invalid_edge_weight"
    return None


def validate_events(
    events: list[StructuralEvent], state: HRMState, config: HRMConfig
) -> tuple[list[StructuralEvent], list[StructuralEvent]]:
    accepted: list[StructuralEvent] = []
    rejected: list[StructuralEvent] = []
    simulated_edges = set(state.topology.edges)
    budget = state.resources.structural_event_allowance
    topology_changes = 0
    max_topology_changes = max(
        1, int(np.floor(config.topology.churn_limit * max(1, len(simulated_edges))))
    )
    cooldowns = state.run_metadata.get("topology_entity_last_event_step", {})
    if not isinstance(cooldowns, dict):
        cooldowns = {}
    reserved_entities: set[str] = set()
    for event in sorted(events, key=lambda item: item.deterministic_ordering_key):
        reason = _invalid_basis(event, state)
        topology_event = event.event_type in TOPOLOGY_EVENT_TYPES
        allocation_event = event.event_type in {
            EventType.CHANNEL_ACTIVATE,
            EventType.CHANNEL_DEACTIVATE,
        }
        change_count = 2 if event.event_type == EventType.EDGE_REWIRE else int(topology_event)
        cooling_entities = {
            entity
            for entity in event.affected_entities
            if entity in cooldowns
            and state.step - int(cooldowns[entity]) < config.topology.cooldown
        }
        if reason:
            pass
        elif len(accepted) >= budget:
            reason = "event_budget_exhausted"
        elif topology_event and (config.fixed_topology or not config.topology.enabled):
            reason = "topology_disabled"
        elif allocation_event and (config.fixed_allocation or not config.resources.adaptive):
            reason = "allocation_disabled"
        elif topology_event and (
            cooling_entities or reserved_entities.intersection(event.affected_entities)
        ):
            reason = "topology_cooldown"
        elif topology_event and topology_changes + change_count > max_topology_changes:
            reason = "topology_churn_limit"
        elif event.event_type == EventType.EDGE_GROW:
            edge = _edge_from(event)
            if edge[0] == edge[1]:
                reason = "self_loop"
            elif edge in simulated_edges:
                reason = "duplicate_edge"
            elif (
                state.topology.degree[edge[0]] >= config.graph.max_degree
                or state.topology.degree[edge[1]] >= config.graph.max_degree
            ):
                reason = "degree_bound"
            else:
                simulated_edges.add(edge)
                topology_changes += 1
        elif event.event_type == EventType.EDGE_PRUNE:
            edge = _edge_from(event)
            if edge not in simulated_edges:
                reason = "missing_edge"
            elif state.topology.edge_ages.get(edge_key(*edge), 0) < config.topology.min_edge_age:
                reason = "minimum_edge_age"
            elif config.graph.require_connected and not _connected_after(
                state.topology.weights, edge, False, state.field.active_nodes
            ):
                reason = "connectivity_rule"
            else:
                simulated_edges.remove(edge)
                topology_changes += 1
        elif event.event_type == EventType.EDGE_REWIRE:
            remove_edge, add_edge = _edge_from(event, "remove_edge"), _edge_from(event, "add_edge")
            if remove_edge not in simulated_edges:
                reason = "missing_remove_edge"
            elif add_edge in simulated_edges or add_edge[0] == add_edge[1]:
                reason = "invalid_add_edge"
            elif (
                state.topology.edge_ages.get(edge_key(*remove_edge), 0)
                < config.topology.min_edge_age
            ):
                reason = "minimum_edge_age"
            elif any(
                state.topology.degree[node] - int(node in remove_edge) >= config.graph.max_degree
                for node in add_edge
            ):
                reason = "degree_bound"
            elif config.graph.require_connected and not _connected_after_rewire(
                state.topology.weights,
                remove_edge,
                add_edge,
                state.field.active_nodes,
            ):
                reason = "connectivity_rule"
            else:
                simulated_edges.remove(remove_edge)
                simulated_edges.add(add_edge)
                topology_changes += 2
        elif event.event_type == EventType.EDGE_FLIP:
            reason = "unsupported_optional_event"
        elif allocation_event:
            node, channel = (
                int(event.candidate_basis["node"]),
                int(event.candidate_basis["channel"]),
            )
            active = bool(state.field.active_channels[node, channel])
            if event.event_type == EventType.CHANNEL_ACTIVATE and active:
                reason = "already_active"
            elif event.event_type == EventType.CHANNEL_DEACTIVATE and not active:
                reason = "already_inactive"
            elif (
                event.event_type == EventType.CHANNEL_DEACTIVATE
                and state.resources.active_width[node] <= config.resources.min_width
            ):
                reason = "minimum_width"
        elif event.event_type == EventType.MEMORY_WRITE:
            slot = int(event.candidate_basis["slot"])
            if slot >= state.resources.memory_allowance:
                reason = "memory_allowance"
            elif state.memory.occupied[slot] and not any(
                e.event_type == EventType.MEMORY_EVICT and int(e.candidate_basis["slot"]) == slot
                for e in accepted
            ):
                reason = "occupied_without_eviction"
        elif event.event_type == EventType.MEMORY_EVICT:
            slot = int(event.candidate_basis["slot"])
            if not state.memory.occupied[slot]:
                reason = "slot_empty"
        if reason:
            rejected.append(
                replace(
                    event,
                    validation_result=False,
                    rejection_reason=reason,
                    disposition=EventDisposition.REJECTED,
                )
            )
        else:
            accepted.append(
                replace(
                    event,
                    validation_result=True,
                    accepted=True,
                    disposition=EventDisposition.ACCEPTED,
                )
            )
            if topology_event:
                reserved_entities.update(event.affected_entities)
    return accepted, rejected


def apply_events(
    state: HRMState, events: list[StructuralEvent], config: HRMConfig
) -> tuple[HRMState, list[StructuralEvent], dict[str, np.ndarray]]:
    result = state.clone()
    before = numeric_view(result)
    applied: list[StructuralEvent] = []
    for event in events:
        basis = event.candidate_basis
        if event.event_type in {EventType.EDGE_GROW, EventType.TOPOLOGY_REPAIR}:
            i, j = _edge_from(event)
            weight = float(basis.get("weight", 1.0))
            matrix = result.topology.weights.tolil(copy=True)
            matrix[i, j] = matrix[j, i] = weight
            result.topology.weights = matrix.tocsr()
            result.topology.edge_ages[edge_key(i, j)] = 0
        elif event.event_type == EventType.EDGE_PRUNE:
            i, j = _edge_from(event)
            matrix = result.topology.weights.tolil(copy=True)
            matrix[i, j] = matrix[j, i] = 0.0
            result.topology.weights = matrix.tocsr()
            result.topology.edge_ages.pop(edge_key(i, j), None)
        elif event.event_type == EventType.EDGE_REWIRE:
            old = _edge_from(event, "remove_edge")
            new = _edge_from(event, "add_edge")
            matrix = result.topology.weights.tolil(copy=True)
            matrix[old] = 0.0
            matrix[old[::-1]] = 0.0
            matrix[new] = float(basis.get("weight", 1.0))
            matrix[new[::-1]] = float(basis.get("weight", 1.0))
            result.topology.weights = matrix.tocsr()
            result.topology.edge_ages.pop(edge_key(*old), None)
            result.topology.edge_ages[edge_key(*new)] = 0
        elif event.event_type in {EventType.CHANNEL_ACTIVATE, EventType.CHANNEL_DEACTIVATE}:
            node, channel = int(basis["node"]), int(basis["channel"])
            active = event.event_type == EventType.CHANNEL_ACTIVATE
            result.field.active_channels[node, channel] = active
            result.resources.active_width[node] = int(
                np.count_nonzero(result.field.active_channels[node])
            )
            if active:
                result.field.values[node, channel] = float(basis.get("initial_value", 0.0))
            else:
                result.field.values[node, channel] = 0.0
        elif event.event_type == EventType.NODE_ACTIVATE:
            result.field.active_nodes[int(basis["node"])] = True
        elif event.event_type == EventType.NODE_DEACTIVATE:
            node = int(basis["node"])
            result.field.active_nodes[node] = False
            result.field.active_channels[node] = False
            result.field.values[node] = 0.0
        elif event.event_type == EventType.MEMORY_EVICT:
            slot = int(basis["slot"])
            result.memory.keys[slot] = 0
            result.memory.values[slot] = 0
            result.memory.strengths[slot] = 0
            result.memory.ages[slot] = 0
            result.memory.occupied[slot] = False
            provenance = list(result.memory.provenance)
            provenance[slot] = ""
            result.memory.provenance = tuple(provenance)
            result.memory.eviction_count += 1
        elif event.event_type == EventType.MEMORY_WRITE:
            slot = int(basis["slot"])
            result.memory.keys[slot] = np.asarray(basis["key"], dtype=result.memory.keys.dtype)
            result.memory.values[slot] = np.asarray(
                basis["value"], dtype=result.memory.values.dtype
            )
            result.memory.strengths[slot] = float(basis["strength"])
            result.memory.ages[slot] = 0
            result.memory.occupied[slot] = True
            provenance = list(result.memory.provenance)
            provenance[slot] = str(basis.get("provenance", event.event_id))
            result.memory.provenance = tuple(provenance)
            result.memory.write_count += 1
        result.topology.event_history = (*result.topology.event_history, event.event_id)
        if event.event_type in TOPOLOGY_EVENT_TYPES:
            cooldowns = result.run_metadata.setdefault("topology_entity_last_event_step", {})
            if not isinstance(cooldowns, dict):
                cooldowns = {}
                result.run_metadata["topology_entity_last_event_step"] = cooldowns
            for entity in event.affected_entities:
                cooldowns[entity] = result.step
        applied.append(replace(event, disposition=EventDisposition.APPLIED))
    for key in list(result.topology.edge_ages):
        result.topology.edge_ages[key] += 1
    refresh_topology(result)
    effects = view_delta(numeric_view(result), before)
    compact_effect = {path: delta for path, delta in effects.items() if np.any(delta != 0)}
    applied = [
        replace(event, applied_numerical_effect={"batch_effect_paths": sorted(compact_effect)})
        for event in applied
    ]
    return result, applied, compact_effect
