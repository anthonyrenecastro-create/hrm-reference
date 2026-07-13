"""Typed external input injection."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import StateBlock


class InputMechanism(BaseMechanism):
    name = "input"
    reads = frozenset({StateBlock.FIELD})
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
        raw = np.asarray(external_input.values, dtype=float)
        if raw.ndim == 1:
            if raw.size != state.field.values.shape[1]:
                raise ValueError("global input channel mismatch")
            raw = np.broadcast_to(raw, state.field.values.shape)
        if raw.shape != state.field.values.shape:
            raise ValueError("node-wise input shape mismatch")
        delta = (
            context.dt
            * context.config.dynamics.input_gain
            * np.where(state.field.active_channels, raw, 0.0)
        )
        leakage = (
            external_input.target_field is not None
            and np.array_equal(raw, np.asarray(external_input.target_field))
            and not external_input.supervised_guidance
        )
        contributions = {"field.values": delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "injected_norm": float(np.linalg.norm(delta)),
                "affected_entries": int(np.count_nonzero(delta)),
                "scaling": context.config.dynamics.input_gain,
                "target_leakage_flag": bool(leakage),
                "supervision_authorized": bool(external_input.supervised_guidance),
            },
            proposed_cost=float(np.count_nonzero(state.field.active_channels)),
            dtype=context.config.dtype,
        )
