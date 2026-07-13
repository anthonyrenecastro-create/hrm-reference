"""Deterministic constrained topology event proposals."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.events import StructuralEvent
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import EventType, StateBlock


class TopologyMechanism(BaseMechanism):
    name = "topology"
    reads = frozenset(
        {
            StateBlock.TOPOLOGY,
            StateBlock.GEOMETRY,
            StateBlock.FIELD,
            StateBlock.COGNITION,
            StateBlock.RESOURCES,
        }
    )
    writes = frozenset({StateBlock.TOPOLOGY})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state, cfg = snapshot.state, context.config.topology
        events: list[StructuralEvent] = []
        existing = set(state.topology.edges)
        n = state.field.values.shape[0]
        candidates: list[tuple[float, int, int]] = []
        for i in range(n):
            if not state.field.active_nodes[i]:
                continue
            for j in range(i + 1, n):
                if not state.field.active_nodes[j] or (i, j) in existing:
                    continue
                utility = 0.5 * (observables.importance[i] + observables.importance[j])
                geometric = float(
                    np.linalg.norm(state.geometry.coordinates[i] - state.geometry.coordinates[j])
                )
                affinity = float(
                    np.exp(-(np.linalg.norm(state.field.values[i] - state.field.values[j]) ** 2))
                )
                score = (
                    utility
                    + cfg.stress_weight
                    * (observables.geometry_stress[i] + observables.geometry_stress[j])
                    / 2
                    + 0.2 * affinity
                    - cfg.cost_weight * geometric
                    - cfg.risk_weight
                )
                candidates.append((score, i, j))
        grow_choice = (
            max(candidates, key=lambda item: (item[0], -item[1], -item[2])) if candidates else None
        )
        if grow_choice is not None:
            score, i, j = grow_choice
            if score >= cfg.grow_threshold:
                distance = float(
                    np.linalg.norm(state.geometry.coordinates[i] - state.geometry.coordinates[j])
                )
                weight = float(
                    np.exp(-(distance**2) / (2 * context.config.graph.distance_scale**2))
                )
                events.append(
                    StructuralEvent.create(
                        EventType.EDGE_GROW,
                        {"edge": [i, j], "weight": weight, "signal": "residual_stress_affinity"},
                        (f"node:{i}", f"node:{j}"),
                        score,
                        score + cfg.cost_weight * distance,
                        distance,
                        cfg.risk_weight,
                        context.step,
                    )
                )
        prune_candidates: list[tuple[float, int, int]] = []
        for i, j in state.topology.edges:
            utility = 0.5 * (observables.importance[i] + observables.importance[j])
            redundancy = float(observables.node_coherence[i] + observables.node_coherence[j]) / 2
            score = redundancy - utility - cfg.cost_weight - cfg.risk_weight
            prune_candidates.append((score, i, j))
        prune_choice = (
            max(prune_candidates, key=lambda item: (item[0], -item[1], -item[2]))
            if prune_candidates
            else None
        )
        if prune_choice is not None:
            score, i, j = prune_choice
            if score >= max(cfg.grow_threshold, -cfg.prune_threshold):
                events.append(
                    StructuralEvent.create(
                        EventType.EDGE_PRUNE,
                        {"edge": [i, j], "signal": "redundancy_low_utility"},
                        (f"node:{i}", f"node:{j}"),
                        score,
                        score,
                        1.0,
                        cfg.risk_weight,
                        context.step,
                    )
                )
        if grow_choice is not None and prune_choice is not None:
            grow_score, add_i, add_j = grow_choice
            prune_score, remove_i, remove_j = prune_choice
            removable_age = state.topology.edge_ages.get(
                f"{min(remove_i, remove_j)}:{max(remove_i, remove_j)}", 0
            )
            if (
                grow_score >= cfg.grow_threshold
                and prune_score >= -cfg.prune_threshold
                and removable_age >= cfg.min_edge_age
            ):
                distance = float(
                    np.linalg.norm(
                        state.geometry.coordinates[add_i] - state.geometry.coordinates[add_j]
                    )
                )
                weight = float(
                    np.exp(-(distance**2) / (2 * context.config.graph.distance_scale**2))
                )
                rewire_score = grow_score + max(0.0, prune_score) + 1e-9
                affected = tuple(
                    f"node:{node}" for node in sorted({add_i, add_j, remove_i, remove_j})
                )
                events.append(
                    StructuralEvent.create(
                        EventType.EDGE_REWIRE,
                        {
                            "remove_edge": [remove_i, remove_j],
                            "add_edge": [add_i, add_j],
                            "weight": weight,
                            "signal": "replace_redundant_edge_with_high_utility_edge",
                        },
                        affected,
                        rewire_score,
                        grow_score + prune_score,
                        distance + 1.0,
                        2.0 * cfg.risk_weight,
                        context.step,
                    )
                )
        diagnostics = {
            "candidate_non_edges": len(candidates),
            "candidate_edges": len(prune_candidates),
            "events_proposed": len(events),
            "expected_future_only_credit": True,
        }
        return MechanismProposal(
            self.name,
            self.activation({}, tuple(events)),
            {},
            self.reads,
            self.writes,
            diagnostics=diagnostics,
            proposed_cost=float(len(candidates) + len(prune_candidates)),
            candidate_events=tuple(events),
            dtype=context.config.dtype,
        )
