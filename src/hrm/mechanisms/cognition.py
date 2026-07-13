"""Contractive predictive latent-state mechanism."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.operators.pooling import fit_vector
from hrm.types import StateBlock


class CognitionMechanism(BaseMechanism):
    name = "cognition"
    reads = frozenset(
        {StateBlock.COGNITION, StateBlock.FIELD, StateBlock.MEMORY, StateBlock.HIERARCHY}
    )
    writes = frozenset({StateBlock.COGNITION})

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
        drive = fit_vector(
            np.concatenate((observables.pooled_field, observables.retrieved_memory)),
            state.cognition.latent.size,
        )
        candidate = np.tanh(0.65 * state.cognition.latent + 0.35 * drive)
        latent_delta = (
            context.dt
            * context.config.dynamics.cognition_gain
            * (candidate - state.cognition.latent)
        )
        next_latent = state.cognition.latent + latent_delta
        prediction = np.tanh(fit_vector(next_latent, state.cognition.prediction.size))
        task_prediction = prediction.copy()
        residual = observables.prediction_residual
        uncertainty = np.clip(np.abs(residual), 1e-6, 10.0)
        precursor = np.tanh(residual + 0.25 * uncertainty)
        contributions = {
            "cognition.latent": latent_delta,
            "cognition.prediction": prediction - state.cognition.prediction,
            "cognition.task_prediction": task_prediction - state.cognition.task_prediction,
            "cognition.residual": residual - state.cognition.residual,
            "cognition.uncertainty": uncertainty - state.cognition.uncertainty,
            "cognition.pooled_field_context": fit_vector(
                observables.pooled_field, state.cognition.pooled_field_context.size
            )
            - state.cognition.pooled_field_context,
            "cognition.retrieved_memory_context": fit_vector(
                observables.retrieved_memory, state.cognition.retrieved_memory_context.size
            )
            - state.cognition.retrieved_memory_context,
            "cognition.guidance_precursor": precursor - state.cognition.guidance_precursor,
        }
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "prediction_error": float(np.mean(residual**2)),
                "uncertainty": float(np.mean(uncertainty)),
                "contractive_mix": 0.65,
            },
            proposed_cost=float(state.cognition.latent.size * 8),
            dtype=context.config.dtype,
        )
