"""Optional bifurcating transition wrapper for lookahead branch selection."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
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
    improvement_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.branches < 1:
            raise ValueError("branches must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not (0.0 <= self.trigger_threshold <= 1.0):
            raise ValueError("trigger_threshold must be in [0.0, 1.0]")
        if self.improvement_margin < 0.0:
            raise ValueError("improvement_margin must be >= 0.0")


@dataclass(frozen=True, slots=True)
class BranchPolicy:
    branch: int
    control_gain_scale: float
    topology_enabled: bool
    geometry_enabled: bool
    max_width: int
    input_scale: float
    input_channel_shift: int


@dataclass(frozen=True, slots=True)
class BranchEvaluation:
    branch: int
    objective: float
    cumulative_objective: float
    horizon: int
    run_id: str
    post_state_digest: str
    policy: BranchPolicy


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
    post_state_digest: str
    policy: BranchPolicy


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
        self._mechanisms = mechanisms
        self._interactions = interactions
        self._baseline_engine = TransitionEngine(
            config,
            mechanisms=mechanisms,
            interactions=interactions,
        )

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

    def _build_branch_policy(self, branch: int) -> BranchPolicy:
        span = max(1, self.policy.branches - 1)
        phase = (2.0 * branch / float(span)) - 1.0
        max_width = self.config.resources.max_width or self.config.dimensions.channels
        width_delta = (branch % 3) - 1
        adjusted_width = min(
            self.config.dimensions.channels,
            max(self.config.resources.min_width, max_width + width_delta),
        )
        topology_enabled = bool(self.config.enabled.get("topology", False))
        geometry_enabled = bool(self.config.enabled.get("geometry", False))
        if not self.config.fixed_topology and branch % 3 == 1:
            topology_enabled = False
        if not self.config.fixed_geometry and branch % 4 == 2:
            geometry_enabled = False
        return BranchPolicy(
            branch=branch,
            control_gain_scale=max(0.5, 1.0 + 0.2 * phase),
            topology_enabled=topology_enabled,
            geometry_enabled=geometry_enabled,
            max_width=adjusted_width,
            input_scale=max(0.5, 1.0 + 0.25 * phase),
            input_channel_shift=branch % max(1, self.config.dimensions.channels),
        )

    def _branch_config(self, policy: BranchPolicy) -> HRMConfig:
        data = self.config.model_dump(mode="python")
        data["dynamics"]["input_gain"] = max(
            0.0,
            float(data["dynamics"]["input_gain"] * policy.control_gain_scale),
        )
        data["dynamics"]["regional_gain"] = max(
            0.0,
            float(data["dynamics"]["regional_gain"] * policy.control_gain_scale),
        )
        data["dynamics"]["guidance_gain"] = max(
            0.0,
            float(data["dynamics"]["guidance_gain"] * policy.control_gain_scale),
        )
        data["enabled"]["topology"] = policy.topology_enabled
        data["enabled"]["geometry"] = policy.geometry_enabled
        data["resources"]["max_width"] = int(policy.max_width)
        return HRMConfig.model_validate(data)

    @staticmethod
    def _transform_array(value: np.ndarray | None, policy: BranchPolicy) -> np.ndarray | None:
        if value is None:
            return None
        transformed = np.asarray(value, dtype=float).copy()
        if transformed.ndim == 2 and transformed.shape[1] > 0:
            transformed = np.roll(transformed, shift=policy.input_channel_shift, axis=1)
        return policy.input_scale * transformed

    def _branch_input(self, external_input: ExternalInput, policy: BranchPolicy) -> ExternalInput:
        return replace(
            external_input,
            values=self._transform_array(external_input.values, policy),
            target_field=self._transform_array(external_input.target_field, policy),
            task_target=self._transform_array(external_input.task_target, policy),
        )

    def _rollout(
        self,
        *,
        engine: TransitionEngine,
        policy: BranchPolicy,
        state: HRMState,
        external_input: ExternalInput,
        run_id: str,
    ) -> _BranchCandidate:
        current_state = state.clone()
        transformed_input = self._branch_input(external_input, policy)
        first_step: TransitionResult | None = None
        cumulative = 0.0
        for depth in range(self.policy.horizon):
            step_run_id = run_id if depth == 0 else f"{run_id}:h{depth}"
            result = engine.step(current_state, transformed_input, run_id=step_run_id)
            if first_step is None:
                first_step = result
            cumulative += self._objective(result.ledger)
            current_state = result.state
        assert first_step is not None
        return _BranchCandidate(
            branch=policy.branch,
            run_id=run_id,
            objective=cumulative / float(self.policy.horizon),
            cumulative_objective=cumulative,
            first_step=first_step,
            post_state_digest=first_step.ledger.post_state_digest,
            policy=policy,
        )

    def _evaluate_branches(
        self, state: HRMState, external_input: ExternalInput, run_id: str
    ) -> list[_BranchCandidate]:
        candidates: list[_BranchCandidate] = []
        for branch in range(self.policy.branches):
            branch_policy = self._build_branch_policy(branch)
            branch_engine = TransitionEngine(
                self._branch_config(branch_policy),
                mechanisms=self._mechanisms,
                interactions=self._interactions,
            )
            candidates.append(
                self._rollout(
                    engine=branch_engine,
                    policy=branch_policy,
                    state=state,
                    external_input=external_input,
                    run_id=f"{run_id}:branch:{branch}",
                )
            )
        return candidates

    def step(
        self, state: HRMState, external_input: ExternalInput, *, run_id: str = "run"
    ) -> BifurcatingTransitionResult:
        baseline_policy = BranchPolicy(
            branch=-1,
            control_gain_scale=1.0,
            topology_enabled=bool(self.config.enabled.get("topology", False)),
            geometry_enabled=bool(self.config.enabled.get("geometry", False)),
            max_width=self.config.resources.max_width or self.config.dimensions.channels,
            input_scale=1.0,
            input_channel_shift=0,
        )
        baseline = self._rollout(
            engine=self._baseline_engine,
            policy=baseline_policy,
            state=state,
            external_input=external_input,
            run_id=run_id,
        )
        baseline_objective = baseline.objective
        insufficiency = self._insufficiency(baseline_objective)

        if insufficiency < self.policy.trigger_threshold:
            baseline.first_step.state.run_metadata["bifurcation"] = {
                "attempted": False,
                "trigger_threshold": self.policy.trigger_threshold,
                "improvement_margin": self.policy.improvement_margin,
                "insufficiency": insufficiency,
                "baseline_objective": baseline_objective,
                "selected_branch": None,
                "branches": (),
            }
            return BifurcatingTransitionResult(
                state=baseline.first_step.state,
                ledger=baseline.first_step.ledger,
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
                post_state_digest=candidate.post_state_digest,
                policy=candidate.policy,
            )
            for candidate in candidates
        )

        improve_by = baseline_objective - selected.objective
        adopt = improve_by >= self.policy.improvement_margin

        chosen = selected if adopt else baseline
        selected_branch = selected.branch if adopt else None
        decision = "selected_branch" if adopt else "retained_baseline_margin"

        selected_state = chosen.first_step.state
        selected_state.run_metadata["bifurcation"] = {
            "attempted": True,
            "trigger_threshold": self.policy.trigger_threshold,
            "improvement_margin": self.policy.improvement_margin,
            "insufficiency": insufficiency,
            "baseline_objective": baseline_objective,
            "selected_branch": selected_branch,
            "best_branch": selected.branch,
            "best_objective": selected.objective,
            "improvement": improve_by,
            "decision": decision,
            "branches": [
                {
                    "branch": branch.branch,
                    "objective": branch.objective,
                    "cumulative_objective": branch.cumulative_objective,
                    "horizon": branch.horizon,
                    "run_id": branch.run_id,
                    "post_state_digest": branch.post_state_digest,
                    "policy": asdict(branch.policy),
                }
                for branch in branch_evaluations
            ],
        }
        return BifurcatingTransitionResult(
            state=selected_state,
            ledger=chosen.first_step.ledger,
            bifurcated=adopt,
            selected_branch=selected_branch,
            insufficiency=insufficiency,
            baseline_objective=baseline_objective,
            branches=branch_evaluations,
        )
