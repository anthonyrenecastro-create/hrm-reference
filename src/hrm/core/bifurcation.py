"""Optional bifurcating transition wrapper for lookahead branch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrm.config import HRMConfig
from hrm.core.interaction import Interaction
from hrm.core.ledger import LedgerRecord
from hrm.core.proposal import ExternalInput, Mechanism
from hrm.core.transition import TransitionEngine, TransitionResult
from hrm.state.unified import HRMState


@dataclass(frozen=True, slots=True)
class BifurcationPolicy:
    branches: int = 4
    horizon: int = 2
    trigger_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.branches < 1:
            raise ValueError("branches must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not (0.0 <= self.trigger_threshold <= 1.0):
            raise ValueError("trigger_threshold must be in [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class BranchEvaluation:
    branch: int
    objective: float
    cumulative_objective: float
    horizon: int
    run_id: str


@dataclass(frozen=True, slots=True)
class BifurcatingTransitionResult:
    state: HRMState
    ledger: LedgerRecord
    bifurcated: bool
    selected_branch: int | None
    insufficiency: float
    baseline_objective: float
    branches: tuple[BranchEvaluation, ...]


@dataclass(slots=True)
class _BranchCandidate:
    branch: int
    run_id: str
    objective: float
    cumulative_objective: float
    first_step: TransitionResult


class BifurcatingTransitionEngine:
    """Drop-in transition engine that performs branch lookahead when needed."""

    def __init__(
        self,
        config: HRMConfig,
        policy: BifurcationPolicy,
        mechanisms: list[Mechanism] | None = None,
        interactions: list[Interaction] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self._engine = TransitionEngine(config, mechanisms=mechanisms, interactions=interactions)

    @staticmethod
    def _sanitize(value: Any) -> float:
        scalar = float(value)
        if not np.isfinite(scalar):
            return 0.0
        return scalar

    def _objective(self, ledger: LedgerRecord) -> float:
        metrics = ledger.metrics_after
        reconstruction = max(0.0, self._sanitize(metrics.get("reconstruction_error", 0.0)))
        prediction = max(0.0, self._sanitize(metrics.get("prediction_error", 0.0)))
        uncertainty = max(0.0, self._sanitize(metrics.get("uncertainty", 0.0)))
        collapse_risk = max(0.0, self._sanitize(metrics.get("collapse_risk", 0.0)))
        stress = max(0.0, self._sanitize(metrics.get("geometric_stress", 0.0)))
        cost = max(0.0, self._sanitize(metrics.get("current_cost", 0.0)))
        return reconstruction + prediction + uncertainty + collapse_risk + stress + 0.01 * cost

    @staticmethod
    def _insufficiency(objective: float) -> float:
        clamped = max(0.0, objective)
        return clamped / (1.0 + clamped)

    def _evaluate_branches(
        self, state: HRMState, external_input: ExternalInput, run_id: str
    ) -> list[_BranchCandidate]:
        candidates: list[_BranchCandidate] = []
        for branch in range(self.policy.branches):
            branch_run_id = f"{run_id}:branch:{branch}"
            current_state = state.clone()
            first_step: TransitionResult | None = None
            cumulative = 0.0
            for depth in range(self.policy.horizon):
                step_run_id = f"{branch_run_id}:h{depth}"
                result = self._engine.step(current_state, external_input, run_id=step_run_id)
                if first_step is None:
                    first_step = result
                cumulative += self._objective(result.ledger)
                current_state = result.state
            assert first_step is not None
            candidates.append(
                _BranchCandidate(
                    branch=branch,
                    run_id=branch_run_id,
                    objective=cumulative / float(self.policy.horizon),
                    cumulative_objective=cumulative,
                    first_step=first_step,
                )
            )
        return candidates

    def step(
        self, state: HRMState, external_input: ExternalInput, *, run_id: str = "run"
    ) -> BifurcatingTransitionResult:
        baseline = self._engine.step(state, external_input, run_id=run_id)
        baseline_objective = self._objective(baseline.ledger)
        insufficiency = self._insufficiency(baseline_objective)

        if insufficiency < self.policy.trigger_threshold:
            baseline.state.run_metadata["bifurcation"] = {
                "attempted": False,
                "trigger_threshold": self.policy.trigger_threshold,
                "insufficiency": insufficiency,
                "baseline_objective": baseline_objective,
                "selected_branch": None,
                "branches": (),
            }
            return BifurcatingTransitionResult(
                state=baseline.state,
                ledger=baseline.ledger,
                bifurcated=False,
                selected_branch=None,
                insufficiency=insufficiency,
                baseline_objective=baseline_objective,
                branches=(),
            )

        candidates = self._evaluate_branches(state, external_input, run_id)
        selected = min(candidates, key=lambda candidate: candidate.objective)
        branch_evaluations = tuple(
            BranchEvaluation(
                branch=candidate.branch,
                objective=candidate.objective,
                cumulative_objective=candidate.cumulative_objective,
                horizon=self.policy.horizon,
                run_id=candidate.run_id,
            )
            for candidate in candidates
        )

        selected_state = selected.first_step.state
        selected_state.run_metadata["bifurcation"] = {
            "attempted": True,
            "trigger_threshold": self.policy.trigger_threshold,
            "insufficiency": insufficiency,
            "baseline_objective": baseline_objective,
            "selected_branch": selected.branch,
            "branches": [
                {
                    "branch": branch.branch,
                    "objective": branch.objective,
                    "cumulative_objective": branch.cumulative_objective,
                    "horizon": branch.horizon,
                    "run_id": branch.run_id,
                }
                for branch in branch_evaluations
            ],
        }
        return BifurcatingTransitionResult(
            state=selected_state,
            ledger=selected.first_step.ledger,
            bifurcated=True,
            selected_branch=selected.branch,
            insufficiency=insufficiency,
            baseline_objective=baseline_objective,
            branches=branch_evaluations,
        )
