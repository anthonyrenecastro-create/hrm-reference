"""Bounded local nonlinear field reaction."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import ReactionKind, StateBlock


class ReactionMechanism(BaseMechanism):
    name = "reaction"
    reads = frozenset({StateBlock.FIELD, StateBlock.RESOURCES})
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
        state, cfg = snapshot.state, context.config.dynamics
        phi = state.field.values
        if cfg.reaction == ReactionKind.CUBIC:
            raw = cfg.reaction_a * phi - cfg.reaction_b * phi**3
            if cfg.reaction_competition:
                raw -= cfg.reaction_competition * phi * np.sum(phi**2, axis=1, keepdims=True)
            amplification = np.max(np.abs(cfg.reaction_a - 3.0 * cfg.reaction_b * phi**2))
        else:
            raw = np.tanh(cfg.reaction_a * phi) - cfg.reaction_b * phi
            amplification = abs(cfg.reaction_a) + cfg.reaction_b
        delta = context.dt * np.where(state.field.active_channels, raw, 0.0)
        saturation = float(
            np.mean(
                np.abs(phi[state.field.active_channels])
                > 0.9 * context.config.safety.field_abs_bound
            )
        )
        contributions = {"field.values": delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "kind": str(cfg.reaction),
                "saturation_fraction": saturation,
                "local_amplification_bound": float(amplification),
            },
            proposed_cost=float(4 * np.count_nonzero(state.field.active_channels)),
            dtype=context.config.dtype,
        )
