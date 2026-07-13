"""Owned passive decay and memory aging."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import StateBlock


class PassiveMechanism(BaseMechanism):
    name = "passive"
    reads = frozenset({StateBlock.FIELD, StateBlock.MEMORY, StateBlock.COGNITION})
    writes = reads

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state, dt = snapshot.state, context.dt
        mask = state.field.active_channels
        field = np.where(
            mask,
            -dt * context.config.dynamics.passive_decay * state.field.values,
            0.0,
        )
        working = -dt * context.config.memory.decay * state.memory.working
        latent = -dt * context.config.dynamics.passive_decay * state.cognition.latent
        ages = np.where(state.memory.occupied, dt, 0.0)
        contributions = {
            "field.values": field,
            "memory.working": working,
            "memory.ages": ages,
            "cognition.latent": latent,
        }
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={"decay": context.config.dynamics.passive_decay},
            proposed_cost=float(
                np.count_nonzero(mask) + state.memory.occupancy + state.cognition.latent.size
            ),
            dtype=context.config.dtype,
        )
