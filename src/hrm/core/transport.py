"""Typed fixed-carrier transport layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.events import StructuralEvent
from hrm.state.unified import HRMState


@dataclass(frozen=True, slots=True)
class TransportRecord:
    kind: str
    source_carrier: dict[str, tuple[int, ...]]
    target_carrier: dict[str, tuple[int, ...]]
    event_ids: tuple[str, ...]
    discarded_information_norm: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_carrier": {k: list(v) for k, v in self.source_carrier.items()},
            "target_carrier": {k: list(v) for k, v in self.target_carrier.items()},
            "event_ids": list(self.event_ids),
            "discarded_information_norm": self.discarded_information_norm,
        }


def identity_transport(
    state: HRMState, proposals: list[MechanismProposal], events: list[StructuralEvent]
) -> tuple[HRMState, list[MechanismProposal], TransportRecord]:
    carrier = {
        "field": state.field.values.shape,
        "geometry": state.geometry.coordinates.shape,
        "topology": state.topology.weights.shape,
        "memory_keys": state.memory.keys.shape,
        "memory_values": state.memory.values.shape,
    }
    record = TransportRecord(
        "identity_fixed_carrier", carrier, carrier, tuple(event.event_id for event in events)
    )
    transported = [
        MechanismProposal(
            p.mechanism,
            p.activation,
            {k: np.asarray(v).copy() for k, v in p.contributions.items()},
            p.reads,
            p.writes,
            p.diagnostics,
            p.proposed_cost,
            p.required_invariants,
            p.candidate_events,
            p.interaction_metadata,
            p.dtype,
            p.shape_metadata,
            p.skipped_reason,
        )
        for p in proposals
    ]
    return state.clone(), transported, record
