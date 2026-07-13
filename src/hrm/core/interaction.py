"""Explicit interaction interface and one reference residual–uncertainty interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hrm.core.contribution import MechanismProposal, zero_proposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.types import StateBlock


class Interaction(Protocol):
    name: str
    reads: frozenset[StateBlock]
    writes: frozenset[StateBlock]

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        main_proposals: list[MechanismProposal],
        context: TransitionContext,
    ) -> MechanismProposal: ...


@dataclass(slots=True)
class ResidualUncertaintyInteraction:
    name: str = "residual_uncertainty_interaction"
    reads: frozenset[StateBlock] = frozenset({StateBlock.COGNITION, StateBlock.RESOURCES})
    writes: frozenset[StateBlock] = frozenset({StateBlock.RESOURCES})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        main_proposals: list[MechanismProposal],
        context: TransitionContext,
    ) -> MechanismProposal:
        if not context.cadence_active(self.name):
            return zero_proposal(self.name, self.reads, self.writes, reason="cadence")
        n = snapshot.state.field.values.shape[0]
        signal = observables.node_error * observables.node_uncertainty
        delta = context.dt * 0.01 * (signal - snapshot.state.resources.update_priority)
        activation = float(np.linalg.norm(delta) > 0)
        return MechanismProposal(
            self.name,
            activation,
            {"resources.update_priority": delta.reshape(n)},
            self.reads,
            self.writes,
            diagnostics={"main_proposals_seen": [p.mechanism for p in main_proposals]},
            proposed_cost=float(n),
            interaction_metadata={"kind": "explicit", "factors": ["residual", "uncertainty"]},
        )
