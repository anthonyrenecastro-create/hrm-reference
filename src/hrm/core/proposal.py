"""Common mechanism proposal protocol and transition input types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from hrm.config import HRMConfig
from hrm.core.determinism import DeterministicContext
from hrm.core.snapshot import FrozenHRMState
from hrm.types import Diagnostics, StateBlock

if TYPE_CHECKING:
    from hrm.core.contribution import MechanismProposal
    from hrm.core.observables import Observables


@dataclass(frozen=True, slots=True)
class ExternalInput:
    values: NDArray[np.floating]
    target_field: NDArray[np.floating] | None = None
    task_target: NDArray[np.floating] | None = None
    input_id: str = "input"
    perturbation_id: str | None = None
    supervised_guidance: bool = False
    metadata: Diagnostics = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionContext:
    run_id: str
    seed: int
    step: int
    dt: float
    config: HRMConfig
    deterministic: DeterministicContext
    adaptive_timestep_reason: str | None = None

    def cadence_active(self, mechanism_name: str) -> bool:
        cadence = max(1, int(self.config.dynamics.cadence.get(mechanism_name, 1)))
        return self.step % cadence == 0


class Mechanism(Protocol):
    name: str
    reads: frozenset[StateBlock]
    writes: frozenset[StateBlock]

    def propose(
        self,
        snapshot: FrozenHRMState,
        external_input: ExternalInput,
        observables: Observables,
        context: TransitionContext,
    ) -> MechanismProposal: ...
