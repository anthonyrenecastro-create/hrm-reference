"""Block-sparse additive proposals and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hrm.state.digest import numeric_view
from hrm.types import Diagnostics, StateBlock

if TYPE_CHECKING:
    from hrm.core.events import StructuralEvent
    from hrm.state.unified import HRMState


PATH_BLOCK: dict[str, StateBlock] = {
    "field": StateBlock.FIELD,
    "geometry": StateBlock.GEOMETRY,
    "topology": StateBlock.TOPOLOGY,
    "memory": StateBlock.MEMORY,
    "cognition": StateBlock.COGNITION,
    "hierarchy": StateBlock.HIERARCHY,
    "resources": StateBlock.RESOURCES,
}


@dataclass(frozen=True, slots=True)
class MechanismProposal:
    mechanism: str
    activation: float
    contributions: dict[str, np.ndarray]
    reads: frozenset[StateBlock]
    writes: frozenset[StateBlock]
    diagnostics: Diagnostics = field(default_factory=dict)
    proposed_cost: float = 0.0
    required_invariants: tuple[str, ...] = ()
    candidate_events: tuple[StructuralEvent, ...] = ()
    interaction_metadata: Diagnostics = field(default_factory=dict)
    dtype: str = "float64"
    shape_metadata: dict[str, tuple[int, ...]] = field(default_factory=dict)
    skipped_reason: str | None = None

    @property
    def norm(self) -> float:
        return float(
            np.sqrt(
                sum(
                    float(np.sum(np.asarray(value, dtype=float) ** 2))
                    for value in self.contributions.values()
                )
            )
        )


def zero_proposal(
    name: str, reads: frozenset[StateBlock], writes: frozenset[StateBlock], *, reason: str
) -> MechanismProposal:
    return MechanismProposal(
        name, 0.0, {}, reads, writes, diagnostics={"skipped": reason}, skipped_reason=reason
    )


def validate_proposal(proposal: MechanismProposal, state: HRMState) -> None:
    view = numeric_view(state)
    if not np.isfinite(proposal.activation) or proposal.activation < 0:
        raise ValueError(f"{proposal.mechanism}: invalid activation")
    if not np.isfinite(proposal.proposed_cost) or proposal.proposed_cost < 0:
        raise ValueError(f"{proposal.mechanism}: invalid proposed cost")
    if proposal.activation == 0.0 and proposal.norm != 0.0:
        raise ValueError(f"{proposal.mechanism}: zero activation has nonzero contribution")
    for path, delta in proposal.contributions.items():
        if path not in view:
            raise ValueError(f"{proposal.mechanism}: unknown contribution path {path}")
        block = PATH_BLOCK[path.split(".", 1)[0]]
        if block not in proposal.writes:
            raise ValueError(f"{proposal.mechanism}: writes undeclared block {block}")
        value = np.asarray(delta)
        if value.shape != view[path].shape:
            raise ValueError(
                f"{proposal.mechanism}: shape mismatch for {path}: {value.shape} != {view[path].shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{proposal.mechanism}: nonfinite contribution for {path}")


def sum_contributions(proposals: list[MechanismProposal]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for proposal in proposals:
        for path, delta in proposal.contributions.items():
            if path not in result:
                result[path] = np.zeros_like(np.asarray(delta, dtype=float))
            result[path] = result[path] + np.asarray(delta, dtype=float)
    return result


def proposal_arrays(proposal: MechanismProposal) -> dict[str, list[float] | float | list[object]]:
    return {key: np.asarray(value).tolist() for key, value in proposal.contributions.items()}
