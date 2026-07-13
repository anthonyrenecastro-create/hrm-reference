"""Mechanism helpers and the production mechanism registry."""

from __future__ import annotations

from typing import Any

import numpy as np

from hrm.core.contribution import MechanismProposal, zero_proposal
from hrm.core.proposal import Mechanism, TransitionContext


class BaseMechanism:
    name: str
    reads: frozenset[Any]
    writes: frozenset[Any]

    def skipped(self, context: TransitionContext, reason: str = "cadence") -> MechanismProposal:
        return zero_proposal(self.name, self.reads, self.writes, reason=reason)

    def ready(self, context: TransitionContext) -> bool:
        return context.cadence_active(self.name)

    @staticmethod
    def activation(contributions: dict[str, np.ndarray], events: tuple[Any, ...] = ()) -> float:
        return float(
            bool(events) or any(np.any(np.asarray(value) != 0) for value in contributions.values())
        )


def build_default_mechanisms() -> list[Mechanism]:
    from hrm.mechanisms.allocation import AllocationMechanism
    from hrm.mechanisms.cognition import CognitionMechanism
    from hrm.mechanisms.coupling import CouplingMechanism
    from hrm.mechanisms.diffusion import DiffusionMechanism
    from hrm.mechanisms.geometry import GeometryMechanism
    from hrm.mechanisms.guidance import GuidanceMechanism
    from hrm.mechanisms.hierarchy import HierarchyBottomUpMechanism, HierarchyTopDownMechanism
    from hrm.mechanisms.input import InputMechanism
    from hrm.mechanisms.memory import MemoryMechanism
    from hrm.mechanisms.passive import PassiveMechanism
    from hrm.mechanisms.reaction import ReactionMechanism
    from hrm.mechanisms.regional import RegionalMechanism
    from hrm.mechanisms.topology import TopologyMechanism

    return [
        PassiveMechanism(),
        InputMechanism(),
        DiffusionMechanism(),
        ReactionMechanism(),
        RegionalMechanism(),
        MemoryMechanism(),
        CognitionMechanism(),
        GuidanceMechanism(),
        HierarchyBottomUpMechanism(),
        HierarchyTopDownMechanism(),
        GeometryMechanism(),
        TopologyMechanism(),
        AllocationMechanism(),
        CouplingMechanism("field_to_cognition"),
        CouplingMechanism("cognition_to_field"),
        CouplingMechanism("memory_to_cognition"),
        CouplingMechanism("memory_to_field"),
        CouplingMechanism("hierarchy_to_cognition"),
    ]
