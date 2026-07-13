"""Separately attributable bottom-up restriction and top-down feedback."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import StateBlock


class HierarchyBottomUpMechanism(BaseMechanism):
    name = "hierarchy_bottom_up"
    reads = frozenset(
        {StateBlock.FIELD, StateBlock.COGNITION, StateBlock.MEMORY, StateBlock.HIERARCHY}
    )
    writes = frozenset({StateBlock.HIERARCHY})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state = snapshot.state
        target = state.hierarchy.restriction @ np.where(
            state.field.active_channels, state.field.values, 0.0
        )
        delta = context.config.hierarchy.bottom_up_gain * (target - state.hierarchy.coarse_state)
        reconstructed = state.hierarchy.prolongation @ target
        loss = float(np.mean((reconstructed - state.field.values) ** 2))
        contributions = {
            "hierarchy.coarse_state": delta,
            "hierarchy.aggregation_loss": np.asarray(loss - state.hierarchy.aggregation_loss),
        }
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={"aggregation_loss": loss, "regions": target.shape[0]},
            proposed_cost=float(state.hierarchy.restriction.nnz * state.field.values.shape[1]),
            dtype=context.config.dtype,
        )


class HierarchyTopDownMechanism(BaseMechanism):
    name = "hierarchy_top_down"
    reads = frozenset({StateBlock.HIERARCHY, StateBlock.FIELD})
    writes = frozenset({StateBlock.FIELD})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state = snapshot.state
        gain = min(state.hierarchy.feedback_gain, context.config.hierarchy.max_feedback_gain)
        feedback = state.hierarchy.prolongation @ state.hierarchy.coarse_state
        delta = context.dt * gain * (feedback - state.field.values)
        delta = np.where(state.field.active_channels, delta, 0.0)
        contributions = {"field.values": delta}
        amplification = float(gain * np.linalg.norm(state.hierarchy.prolongation.toarray(), ord=2))
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={"feedback_gain": gain, "empirical_amplification": amplification},
            proposed_cost=float(state.hierarchy.prolongation.nnz * state.field.values.shape[1]),
            dtype=context.config.dtype,
        )
