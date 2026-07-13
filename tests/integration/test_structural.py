from __future__ import annotations

import numpy as np

from hrm.config import apply_ablation
from hrm.core.events import StructuralEvent, apply_events, validate_events
from hrm.core.invariants import refresh_derived
from hrm.core.transport import identity_transport
from hrm.types import EventType


def test_self_loop_and_duplicate_edges_rejected(config, state):
    duplicate = StructuralEvent.create(
        EventType.EDGE_GROW, {"edge": list(state.topology.edges[0])}, ("a", "b"), 1, 1, 1, 0, 0
    )
    self_loop = StructuralEvent.create(EventType.EDGE_GROW, {"edge": [0, 0]}, ("a",), 1, 1, 1, 0, 0)
    accepted, rejected = validate_events([duplicate, self_loop], state, config)
    assert not accepted and {event.rejection_reason for event in rejected} == {
        "duplicate_edge",
        "self_loop",
    }


def test_fixed_topology_rejects_topology_events(config, state):
    fixed = apply_ablation(config, "fixed_topology")
    event = StructuralEvent.create(
        EventType.EDGE_GROW, {"edge": [0, 5], "weight": 1.0}, ("0", "5"), 1, 1, 1, 0, 0
    )
    accepted, rejected = validate_events([event], state, fixed)
    assert not accepted and rejected[0].rejection_reason == "topology_disabled"


def test_channel_deactivation_is_exact_event_effect(config, state):
    event = StructuralEvent.create(
        EventType.CHANNEL_DEACTIVATE, {"node": 0, "channel": 3}, ("0", "3"), 1, 1, 1, 0, 0
    )
    accepted, rejected = validate_events([event], state, config)
    assert accepted and not rejected
    transported, _, record = identity_transport(state, [], accepted)
    result, applied, effects = apply_events(transported, accepted, config)
    assert record.kind == "identity_fixed_carrier"
    assert not result.field.active_channels[0, 3]
    assert "field.active_channels" in effects
    assert applied[0].accepted


def test_rewire_is_applied_and_imposes_entity_cooldown(config, state):
    data = config.model_dump(mode="python")
    data["topology"]["min_edge_age"] = 0
    data["topology"]["churn_limit"] = 1.0
    permissive = type(config).model_validate(data)
    remove_edge = state.topology.edges[0]
    existing = set(state.topology.edges)
    add_edge = next(
        (i, j)
        for i in range(state.field.values.shape[0])
        for j in range(i + 1, state.field.values.shape[0])
        if (i, j) not in existing
    )
    affected = tuple(f"node:{node}" for node in sorted(set(remove_edge + add_edge)))
    event = StructuralEvent.create(
        EventType.EDGE_REWIRE,
        {
            "remove_edge": list(remove_edge),
            "add_edge": list(add_edge),
            "weight": 0.75,
        },
        affected,
        2,
        2,
        1,
        0,
        state.step,
    )
    accepted, rejected = validate_events([event], state, permissive)
    assert accepted and not rejected
    rewired, _, _ = apply_events(state, accepted, permissive)
    assert remove_edge not in rewired.topology.edges
    assert add_edge in rewired.topology.edges
    accepted_again, rejected_again = validate_events([event], rewired, permissive)
    assert not accepted_again
    assert rejected_again[0].rejection_reason == "topology_cooldown"


def test_nonfinite_event_payload_is_rejected(config, state):
    existing = set(state.topology.edges)
    edge = next(
        (i, j)
        for i in range(state.field.values.shape[0])
        for j in range(i + 1, state.field.values.shape[0])
        if (i, j) not in existing
    )
    event = StructuralEvent.create(
        EventType.EDGE_GROW,
        {"edge": list(edge), "weight": np.inf},
        tuple(f"node:{node}" for node in edge),
        1,
        1,
        1,
        0,
        state.step,
    )
    accepted, rejected = validate_events([event], state, config)
    assert not accepted
    assert rejected[0].rejection_reason == "invalid_edge_weight"


def test_topology_change_invalidates_operator_and_refreshes_geometry(config, state):
    topology_version = state.topology.topology_version
    operator_version = state.topology.operator_version
    refresh_derived(state, config, {"topology.weights", "geometry.coordinates"})
    assert state.topology.topology_version == topology_version + 1
    assert state.topology.operator_version == operator_version + 2
    assert np.all(np.isfinite(state.geometry.stress))
    assert np.any(state.geometry.curvature > 0)
