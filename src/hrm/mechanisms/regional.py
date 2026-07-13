"""Bounded low-rank regional coupling."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import StateBlock


class RegionalMechanism(BaseMechanism):
    name = "regional"
    reads = frozenset({StateBlock.FIELD, StateBlock.HIERARCHY, StateBlock.TOPOLOGY})
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
        delta = np.zeros_like(state.field.values)
        for region in range(state.hierarchy.coarse_state.shape[0]):
            nodes = np.flatnonzero(
                (state.hierarchy.region_assignments == region) & state.field.active_nodes
            )
            if nodes.size:
                prototype = np.sum(
                    np.where(state.field.active_channels[nodes], state.field.values[nodes], 0.0),
                    axis=0,
                ) / np.maximum(1, np.sum(state.field.active_channels[nodes], axis=0))
                delta[nodes] = (
                    context.dt
                    * context.config.dynamics.regional_gain
                    * (prototype - state.field.values[nodes])
                )
        delta = np.where(state.field.active_channels, delta, 0.0)
        risk = float(np.mean(observables.node_coherence > 0.99))
        contributions = {"field.values": delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "effective_rank_bound": state.hierarchy.coarse_state.shape[0],
                "homogenization_risk": risk,
                "regional_norm": float(np.linalg.norm(delta)),
            },
            proposed_cost=float(
                np.count_nonzero(state.field.active_channels) + state.hierarchy.coarse_state.size
            ),
            dtype=context.config.dtype,
        )
