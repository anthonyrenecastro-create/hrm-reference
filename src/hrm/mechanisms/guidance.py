"""Bounded residual, uncertainty, and hierarchy guidance."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.operators.pooling import fit_vector
from hrm.types import StateBlock


class GuidanceMechanism(BaseMechanism):
    name = "guidance"
    reads = frozenset({StateBlock.COGNITION, StateBlock.HIERARCHY, StateBlock.FIELD})
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
        if external_input.supervised_guidance and not context.config.allow_supervised_guidance:
            raise ValueError("task requested supervised guidance but configuration forbids it")
        state, cfg = snapshot.state, context.config.dynamics
        residual = (
            observables.prediction_residual
            if observables.supervision_authorized
            else state.cognition.residual
        )
        hierarchy_context = np.mean(state.hierarchy.coarse_state, axis=0)
        vector = (
            fit_vector(residual, state.field.values.shape[1])
            + 0.25 * fit_vector(state.cognition.uncertainty, state.field.values.shape[1])
            + 0.25 * fit_vector(hierarchy_context, state.field.values.shape[1])
        )
        bounded = np.clip(cfg.guidance_gain * vector, -cfg.guidance_bound, cfg.guidance_bound)
        scale = (0.5 + observables.importance)[:, None]
        delta = context.dt * scale * bounded[None, :]
        delta = np.where(state.field.active_channels, delta, 0.0)
        contributions = {"field.values": delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "modulation_norm": float(np.linalg.norm(delta)),
                "bound": cfg.guidance_bound,
                "supervised_target_used": bool(observables.supervision_authorized),
                "target_available": observables.task_target_available,
            },
            proposed_cost=float(np.count_nonzero(state.field.active_channels)),
            dtype=context.config.dtype,
        )
