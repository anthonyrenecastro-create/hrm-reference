"""Bounded stress-driven continuous geometry adaptation."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import StateBlock


class GeometryMechanism(BaseMechanism):
    name = "geometry"
    reads = frozenset(
        {
            StateBlock.GEOMETRY,
            StateBlock.FIELD,
            StateBlock.COGNITION,
            StateBlock.TOPOLOGY,
            StateBlock.RESOURCES,
        }
    )
    writes = frozenset({StateBlock.GEOMETRY})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state, cfg = snapshot.state, context.config.geometry
        forces = np.zeros_like(state.geometry.coordinates)
        stress_total = 0.0
        for i, j in state.topology.edges:
            vector = state.geometry.coordinates[j] - state.geometry.coordinates[i]
            distance = float(np.linalg.norm(vector)) + 1e-12
            feature_distance = float(np.linalg.norm(state.field.values[i] - state.field.values[j]))
            preferred = min(cfg.max_separation, max(cfg.min_separation, 0.25 + feature_distance))
            mismatch = distance - preferred
            direction = vector / distance
            force = mismatch * direction
            forces[i] += force
            forces[j] -= force
            stress_total += mismatch**2
        forces -= cfg.anchor_penalty * (
            state.geometry.coordinates - state.geometry.anchor_coordinates
        )
        delta = context.dt * cfg.gain * forces
        norms = np.linalg.norm(delta, axis=1)
        over = norms > cfg.max_displacement
        if np.any(over):
            delta[over] *= (cfg.max_displacement / norms[over])[:, None]
        contributions = {"geometry.coordinates": delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "stress": stress_total,
                "maximum_displacement": float(np.max(np.linalg.norm(delta, axis=1))),
                "distortion": float(
                    np.linalg.norm(
                        (state.geometry.coordinates + delta) - state.geometry.anchor_coordinates
                    )
                ),
            },
            proposed_cost=float(
                len(state.topology.edges) * state.geometry.coordinates.shape[1] * 8
            ),
            required_invariants=("coordinate_bounds", "minimum_separation"),
            dtype=context.config.dtype,
        )
