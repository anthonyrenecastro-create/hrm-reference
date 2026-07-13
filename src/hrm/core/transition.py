"""The single authoritative HRM production transition."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from hrm.config import HRMConfig
from hrm.core.contribution import MechanismProposal, sum_contributions, validate_proposal
from hrm.core.determinism import DeterministicContext
from hrm.core.events import StructuralEvent, apply_events, validate_events
from hrm.core.interaction import Interaction, ResidualUncertaintyInteraction
from hrm.core.invariants import apply_deltas, refresh_derived
from hrm.core.ledger import LedgerRecord, build_ledger_record
from hrm.core.observables import observe
from hrm.core.proposal import ExternalInput, Mechanism, TransitionContext
from hrm.core.safety import safe_transition
from hrm.core.snapshot import freeze
from hrm.core.transport import identity_transport
from hrm.mechanisms.base import build_default_mechanisms
from hrm.metrics import compute_metrics
from hrm.operators.graph import effective_weights
from hrm.operators.laplacian import build_laplacian
from hrm.operators.stability import stable_timestep
from hrm.state.digest import numeric_view, state_digest, view_delta
from hrm.state.unified import HRMState
from hrm.state.validation import validate_state
from hrm.types import EventDisposition, StateBlock


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: HRMState
    ledger: LedgerRecord


class TransitionEngine:
    """Authoritative state transition used by every production caller."""

    def __init__(
        self,
        config: HRMConfig,
        mechanisms: Iterable[Mechanism] | None = None,
        interactions: Iterable[Interaction] | None = None,
    ) -> None:
        self.config = config
        self.mechanisms = list(mechanisms) if mechanisms is not None else build_default_mechanisms()
        self.interactions = (
            list(interactions) if interactions is not None else [ResidualUncertaintyInteraction()]
        )
        names = [m.name for m in self.mechanisms]
        if len(names) != len(set(names)):
            raise ValueError("mechanism names must be unique")

    def _effective_timestep(self, state: HRMState) -> tuple[float, str | None, float]:
        weights = effective_weights(state, self.config)
        laplacian = build_laplacian(weights, self.config.graph.laplacian)
        phi = state.field.values[state.field.active_channels]
        empirical_reaction = abs(
            self.config.dynamics.reaction_a
        ) + 3.0 * self.config.dynamics.reaction_b * (float(np.max(phi**2)) if phi.size else 0.0)
        coupling = (
            self.config.dynamics.regional_gain
            + self.config.dynamics.field_coupling_gain
            + self.config.hierarchy.top_down_gain
        )
        stable, eigenvalue = stable_timestep(
            laplacian,
            self.config.dynamics.diffusion,
            empirical_reaction,
            coupling,
            self.config.dynamics.dt,
        )
        if stable < self.config.dynamics.dt:
            if not self.config.safety.adaptive_timestep:
                return (
                    self.config.dynamics.dt,
                    f"bound_exceeded_without_adaptation:{stable:.6g}",
                    eigenvalue,
                )
            effective = max(self.config.safety.min_dt, stable)
            return (
                effective,
                f"reduced_from:{self.config.dynamics.dt:.6g};bound:{stable:.6g}",
                eigenvalue,
            )
        return self.config.dynamics.dt, None, eigenvalue

    @staticmethod
    def _resource_meter(
        snapshot: HRMState, proposals: list[MechanismProposal], context: TransitionContext
    ) -> MechanismProposal:
        cost = float(sum(proposal.proposed_cost for proposal in proposals))
        operations = int(round(cost))
        contributions = {
            "resources.current_use": np.asarray(cost - snapshot.resources.current_use),
            "resources.estimated_cost": np.asarray(cost - snapshot.resources.estimated_cost),
            "resources.measured_cost": np.asarray(cost - snapshot.resources.measured_cost),
            "resources.cumulative_operations": np.asarray(float(operations)),
        }
        return MechanismProposal(
            "resource_meter",
            float(cost > 0),
            contributions,
            frozenset({StateBlock.RESOURCES}),
            frozenset({StateBlock.RESOURCES}),
            diagnostics={
                "measured_active_operations": operations,
                "source_proposals": [p.mechanism for p in proposals],
            },
            proposed_cost=0.0,
            dtype=context.config.dtype,
        )

    def step(
        self, state: HRMState, external_input: ExternalInput, *, run_id: str = "run"
    ) -> TransitionResult:
        started = time.perf_counter()
        validate_state(state, self.config)
        pre_digest = state_digest(state)
        snapshot = freeze(state)
        dt, adaptive_reason, eigenvalue = self._effective_timestep(snapshot.state)
        deterministic = DeterministicContext(self.config.seed, state.step, run_id)
        context = TransitionContext(
            run_id, self.config.seed, state.step, dt, self.config, deterministic, adaptive_reason
        )
        observables = observe(snapshot, external_input)
        metrics_before = compute_metrics(snapshot.state, external_input)

        main: list[MechanismProposal] = []
        for mechanism in self.mechanisms:
            if not self.config.enabled.get(mechanism.name, False):
                continue
            proposal = mechanism.propose(snapshot, external_input, observables, context)
            snapshot.assert_unchanged()
            validate_proposal(proposal, snapshot.state)
            main.append(proposal)

        interactions: list[MechanismProposal] = []
        for interaction in self.interactions:
            if not self.config.enabled.get(interaction.name, False):
                continue
            proposal = interaction.propose(snapshot, external_input, observables, main, context)
            snapshot.assert_unchanged()
            validate_proposal(proposal, snapshot.state)
            interactions.append(proposal)

        meter = self._resource_meter(snapshot.state, [*main, *interactions], context)
        validate_proposal(meter, snapshot.state)
        main.append(meter)

        stochastic: dict[str, np.ndarray] = {}
        if self.config.dynamics.stochastic_scale > 0:
            generator = deterministic.generator("intentional_field_noise")
            noise = generator.normal(
                0.0,
                self.config.dynamics.stochastic_scale * np.sqrt(dt),
                snapshot.state.field.values.shape,
            )
            stochastic["field.values"] = np.where(snapshot.state.field.active_channels, noise, 0.0)

        combined = sum_contributions([*main, *interactions])
        for path, delta in stochastic.items():
            combined[path] = combined.get(path, np.zeros_like(delta)) + delta
        provisional = apply_deltas(snapshot.state, combined)
        if not all(np.all(np.isfinite(value)) for value in combined.values()):
            raise FloatingPointError("aggregate contribution is not finitely representable")
        provisional_finite = all(
            np.all(np.isfinite(value)) for value in numeric_view(provisional).values()
        )

        proposed_events: list[StructuralEvent] = []
        for proposal in main:
            proposed_events.extend(proposal.candidate_events)
        accepted, rejected = validate_events(proposed_events, snapshot.state, self.config)
        if not provisional_finite:
            rejected.extend(
                replace(
                    event,
                    validation_result=False,
                    accepted=False,
                    rejection_reason="numerical_rollback",
                    disposition=EventDisposition.REJECTED,
                )
                for event in accepted
            )
            accepted = []
        transported_state, transported_all, transport = identity_transport(
            provisional, [*main, *interactions], accepted
        )
        transported_main = transported_all[: len(main)]
        transported_interactions = transported_all[len(main) :]
        if provisional_finite:
            event_state, applied_events, event_effects = apply_events(
                transported_state, accepted, self.config
            )
        else:
            event_state, applied_events, event_effects = transported_state, [], {}
        attributed_delta = {
            path: np.asarray(value, dtype=float).copy() for path, value in combined.items()
        }
        for path, value in event_effects.items():
            if path not in attributed_delta:
                attributed_delta[path] = np.zeros_like(np.asarray(value, dtype=float))
            attributed_delta[path] += np.asarray(value, dtype=float)
        if not all(np.all(np.isfinite(value)) for value in attributed_delta.values()):
            raise FloatingPointError("event effect is not finitely representable")
        safe_state, corrections, correction_effects = safe_transition(
            event_state,
            snapshot.state,
            self.config,
            rollback_effect={path: -value for path, value in attributed_delta.items()},
        )

        observed_delta = view_delta(numeric_view(safe_state), numeric_view(snapshot.state))
        changed_paths = set(combined) | set(event_effects) | set(correction_effects)
        refresh_derived(safe_state, self.config, changed_paths)
        safe_state.run_metadata.update(
            {
                "run_id": run_id,
                "seed": self.config.seed,
                "config_digest": self.config.digest,
                "last_lambda_max": eigenvalue,
                "last_dt": dt,
            }
        )
        post_digest = state_digest(safe_state)
        metrics_after = compute_metrics(safe_state, external_input)
        metrics_after.update(
            {
                "event_count": len(applied_events),
                "rejected_event_count": len(rejected),
                "correction_count": len(corrections),
                "correction_cost": float(
                    np.sqrt(sum(np.sum(v**2) for v in correction_effects.values()))
                ),
                "topology_churn": float(
                    sum(event.event_type.value.startswith("edge_") for event in applied_events)
                    / max(1, len(snapshot.state.topology.edges))
                ),
                "active_operations_step": int(
                    safe_state.resources.cumulative_operations
                    - snapshot.state.resources.cumulative_operations
                ),
            }
        )
        versions = {
            "field": safe_state.field.version,
            "geometry": safe_state.geometry.version,
            "topology": safe_state.topology.topology_version,
            "operator": safe_state.topology.operator_version,
            "memory": safe_state.memory.version,
            "cognition": safe_state.cognition.version,
            "hierarchy": safe_state.hierarchy.version,
            "resources": safe_state.resources.version,
        }
        runtime = time.perf_counter() - started
        metrics_after["runtime_seconds"] = runtime
        ledger = build_ledger_record(
            run_id=run_id,
            seed=self.config.seed,
            step=state.step,
            dtype=self.config.dtype,
            solver=self.config.dynamics.solver,
            config_digest=self.config.digest,
            generator_identity=deterministic.identity(),
            input_identity=external_input.input_id,
            perturbation_identity=external_input.perturbation_id,
            pre_digest=pre_digest,
            post_digest=post_digest,
            versions=versions,
            timestep=dt,
            adaptive_reason=adaptive_reason,
            main=transported_main,
            interactions=transported_interactions,
            stochastic=stochastic,
            proposed_events=proposed_events,
            accepted_events=applied_events,
            rejected_events=rejected,
            transport=transport,
            event_effects=event_effects,
            corrections=corrections,
            correction_effects=correction_effects,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            measured_cost=safe_state.resources.measured_cost,
            runtime_seconds=runtime,
            observed_delta=observed_delta,
            tolerance=self.config.ledger.tolerance,
            store_arrays=self.config.ledger.store_arrays,
            schema_version=self.config.ledger.schema_version,
        )
        return TransitionResult(safe_state, ledger)
