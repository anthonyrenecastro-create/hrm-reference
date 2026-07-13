"""Budget-aware allocation and active-width event proposals."""

from __future__ import annotations

import numpy as np

from hrm.core.contribution import MechanismProposal
from hrm.core.events import StructuralEvent
from hrm.core.observables import Observables
from hrm.core.proposal import ExternalInput, TransitionContext
from hrm.core.snapshot import FrozenHRMState
from hrm.mechanisms.base import BaseMechanism
from hrm.types import EventType, StateBlock


class AllocationMechanism(BaseMechanism):
    name = "allocation"
    reads = frozenset(
        {
            StateBlock.RESOURCES,
            StateBlock.FIELD,
            StateBlock.GEOMETRY,
            StateBlock.COGNITION,
            StateBlock.MEMORY,
        }
    )
    writes = frozenset({StateBlock.RESOURCES})

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal:
        if not self.ready(context):
            return self.skipped(context)
        state, cfg = snapshot.state, context.config.resources
        utility = (
            0.30 * observables.node_error
            + 0.20 * observables.node_uncertainty
            + 0.20 * observables.geometry_stress
            + 0.15 * observables.memory_demand
            + 0.15 * observables.importance
        )
        normalized = (utility - np.min(utility)) / (np.ptp(utility) + 1e-12)
        max_width = cfg.max_width or state.field.values.shape[1]
        desired = cfg.min_width + np.rint(normalized * (max_width - cfg.min_width)).astype(int)
        events: list[StructuralEvent] = []
        for node in range(desired.size):
            current = int(state.resources.active_width[node])
            if desired[node] > current:
                channel = current
                events.append(
                    StructuralEvent.create(
                        EventType.CHANNEL_ACTIVATE,
                        {
                            "node": node,
                            "channel": channel,
                            "initial_value": 0.0,
                            "utility": utility[node],
                        },
                        (f"node:{node}", f"channel:{channel}"),
                        float(utility[node]),
                        float(utility[node]),
                        1.0,
                        0.02,
                        context.step,
                    )
                )
            elif desired[node] < current:
                channel = current - 1
                events.append(
                    StructuralEvent.create(
                        EventType.CHANNEL_DEACTIVATE,
                        {"node": node, "channel": channel, "utility": utility[node]},
                        (f"node:{node}", f"channel:{channel}"),
                        float(1.0 - utility[node]),
                        float(1.0 - utility[node]),
                        1.0,
                        0.02,
                        context.step,
                    )
                )
        node_allocation = 0.5 + normalized
        region_allocation = np.asarray(
            [
                np.mean(node_allocation[state.hierarchy.region_assignments == r])
                for r in range(state.hierarchy.coarse_state.shape[0])
            ]
        )
        switching = cfg.switching_penalty * float(
            np.sum(np.abs(desired - state.resources.active_width))
        )
        contributions = {
            "resources.node_allocation": node_allocation - state.resources.node_allocation,
            "resources.region_allocation": region_allocation - state.resources.region_allocation,
            "resources.update_priority": utility - state.resources.update_priority,
            "resources.switching_cost": np.asarray(switching - state.resources.switching_cost),
        }
        return MechanismProposal(
            self.name,
            self.activation(contributions, tuple(events)),
            contributions,
            self.reads,
            self.writes,
            diagnostics={
                "desired_active_width": desired.tolist(),
                "utility_min": float(np.min(utility)),
                "utility_max": float(np.max(utility)),
                "switching_cost": switching,
                "event_candidates": len(events),
            },
            proposed_cost=float(desired.size * 5),
            candidate_events=tuple(events),
            dtype=context.config.dtype,
        )
