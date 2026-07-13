"""Mask-aware sparse graph diffusion."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.operators.graph import effective_weights
from hrm.operators.laplacian import build_laplacian
from hrm.types import StateBlock


class DiffusionMechanism(BaseMechanism):
    name = "diffusion"
    reads = frozenset(
        {StateBlock.FIELD, StateBlock.GEOMETRY, StateBlock.TOPOLOGY, StateBlock.RESOURCES}
    )
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
        weights = effective_weights(state, context.config)
        laplacian = build_laplacian(weights, context.config.graph.laplacian)
        delta = np.zeros_like(state.field.values)
        operations = 0
        for channel in range(state.field.values.shape[1]):
            active = np.flatnonzero(
                state.field.active_nodes & state.field.active_channels[:, channel]
            )
            if active.size:
                sub = laplacian[active][:, active]
                delta[active, channel] = (
                    -context.dt
                    * context.config.dynamics.diffusion
                    * (sub @ state.field.values[active, channel])
                )
                operations += int(sub.nnz * 2)
        contributions = {"field.values": delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "laplacian": str(context.config.graph.laplacian),
                "operator_version": state.topology.operator_version,
                "active_operations": operations,
                "sign_convention": "-D*L*Phi; L positive semidefinite",
            },
            proposed_cost=float(operations),
            required_invariants=("stable_timestep", "symmetric_nonnegative_weights"),
            dtype=context.config.dtype,
        )
