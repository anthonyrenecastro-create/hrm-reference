"""Named, separately ablatable bidirectional HRM–RCM couplings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.operators.pooling import fit_vector
from hrm.types import StateBlock


@dataclass(slots=True)
class CouplingMechanism(BaseMechanism):
    name: str
    reads: frozenset[StateBlock] = field(init=False)
    writes: frozenset[StateBlock] = field(init=False)

    def __post_init__(self) -> None:
        source = self.name.split("_to_", 1)[0]
        target = self.name.split("_to_", 1)[1]
        self.reads = frozenset({StateBlock(source)})
        self.writes = frozenset({StateBlock(target)})

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
        gain = context.config.dynamics.field_coupling_gain
        contributions: dict[str, np.ndarray]
        if self.name == "field_to_cognition":
            drive = fit_vector(observables.pooled_field, state.cognition.latent.size)
            contributions = {"cognition.latent": dt * gain * drive}
        elif self.name == "cognition_to_field":
            drive = fit_vector(state.cognition.latent, state.field.values.shape[1])
            contributions = {
                "field.values": dt
                * gain
                * np.where(state.field.active_channels, drive[None, :], 0.0)
            }
        elif self.name == "memory_to_cognition":
            drive = fit_vector(observables.retrieved_memory, state.cognition.latent.size)
            contributions = {
                "cognition.latent": dt * context.config.dynamics.memory_coupling_gain * drive
            }
        elif self.name == "memory_to_field":
            drive = fit_vector(observables.retrieved_memory, state.field.values.shape[1])
            contributions = {
                "field.values": dt
                * context.config.dynamics.memory_coupling_gain
                * np.where(state.field.active_channels, drive[None, :], 0.0)
            }
        elif self.name == "hierarchy_to_cognition":
            drive = fit_vector(
                np.mean(state.hierarchy.coarse_state, axis=0), state.cognition.latent.size
            )
            contributions = {"cognition.latent": dt * gain * drive}
        else:
            raise ValueError(f"unknown coupling {self.name}")
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={"coupling": self.name, "gain": gain},
            proposed_cost=float(sum(np.asarray(v).size for v in contributions.values())),
            dtype=context.config.dtype,
        )
