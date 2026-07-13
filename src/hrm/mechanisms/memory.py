"""Associative retrieval, working-memory dynamics, and typed memory events."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.events import StructuralEvent
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.operators.pooling import fit_vector
from hrm.types import EventType, StateBlock


class MemoryMechanism(BaseMechanism):
    name = "memory"
    reads = frozenset(
        {StateBlock.FIELD, StateBlock.COGNITION, StateBlock.MEMORY, StateBlock.RESOURCES}
    )
    writes = frozenset({StateBlock.MEMORY})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state, cfg = snapshot.state, context.config.memory
        target_working = fit_vector(
            observables.retrieved_memory
            + fit_vector(observables.pooled_field, observables.retrieved_memory.size),
            state.memory.working.size,
        )
        working_delta = context.dt * 0.25 * (target_working - state.memory.working)
        reads_delta = np.asarray(1.0 if state.memory.occupancy else 0.0)
        novelty = 1.0
        if state.memory.occupancy:
            novelty = 1.0 - float(np.max(observables.retrieval_weights))
        residual = float(np.mean(np.abs(observables.prediction_residual)))
        uncertainty = float(np.mean(observables.node_uncertainty))
        gate = (
            cfg.novelty_weight * novelty
            + cfg.residual_weight * residual
            + cfg.uncertainty_weight * uncertainty
        )
        events: list[StructuralEvent] = []
        if gate >= cfg.write_threshold:
            free = np.flatnonzero(~state.memory.occupied[: state.resources.memory_allowance])
            if free.size:
                slot = int(free[0])
            else:
                scores = state.memory.strengths + cfg.age_penalty / (1.0 + state.memory.ages)
                slot = int(np.argmin(scores[: state.resources.memory_allowance]))
                events.append(
                    StructuralEvent.create(
                        EventType.MEMORY_EVICT,
                        {"slot": slot, "reason": "capacity_replacement"},
                        (f"slot:{slot}",),
                        gate + 1e-6,
                        gate,
                        1.0,
                        0.05,
                        context.step,
                    )
                )
            key = fit_vector(
                np.concatenate((observables.pooled_field, state.cognition.latent)),
                state.memory.keys.shape[1],
            )
            value = fit_vector(
                np.concatenate((observables.pooled_field, state.cognition.latent)),
                state.memory.values.shape[1],
            )
            basis = {
                "slot": slot,
                "key": key.tolist(),
                "value": value.tolist(),
                "strength": min(cfg.max_strength, max(0.1, gate)),
                "provenance": f"{context.run_id}:{context.step}",
                "write_gate": gate,
            }
            events.append(
                StructuralEvent.create(
                    EventType.MEMORY_WRITE,
                    basis,
                    (f"slot:{slot}",),
                    gate,
                    gate,
                    float(key.size + value.size),
                    0.05,
                    context.step,
                )
            )
        contributions = {"memory.working": working_delta, "memory.read_count": reads_delta}
        return MechanismProposal(
            self.name,
            self.activation(contributions, tuple(events)),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "writes_proposed": int(any(e.event_type == EventType.MEMORY_WRITE for e in events)),
                "retrieval_norm": float(np.linalg.norm(observables.retrieved_memory)),
                "occupancy": state.memory.occupancy,
                "novelty": novelty,
                "write_gate": gate,
                "interference": state.memory.interference,
            },
            proposed_cost=float(
                state.memory.occupancy * (state.memory.keys.shape[1] + state.memory.values.shape[1])
            ),
            candidate_events=tuple(events),
            dtype=context.config.dtype,
        )
