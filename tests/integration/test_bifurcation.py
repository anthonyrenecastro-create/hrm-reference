from __future__ import annotations

import pytest

from hrm.config import HRMConfig
from hrm.core.bifurcation import BifurcatingTransitionEngine, BifurcationPolicy
from hrm.state.digest import state_digest


def test_bifurcation_engine_falls_back_to_single_path(config, state, external):
    engine = BifurcatingTransitionEngine(
        config,
        BifurcationPolicy(branches=4, horizon=2, trigger_threshold=1.0),
    )

    result = engine.step(state, external, run_id="single-path")

    assert result.state.step == state.step + 1
    assert result.bifurcated is False
    assert result.selected_branch is None
    assert result.branches == ()
    assert "bifurcation" in result.state.run_metadata
    assert result.state.run_metadata["bifurcation"]["attempted"] is False


def test_branches_differ_with_zero_stochastic_noise(config, state, external):
    data = config.model_dump(mode="python")
    data["dynamics"]["stochastic_scale"] = 0.0
    deterministic = HRMConfig.model_validate(data)
    engine = BifurcatingTransitionEngine(
        deterministic,
        BifurcationPolicy(branches=3, horizon=2, trigger_threshold=0.0),
    )

    result = engine.step(state, external, run_id="bifurcate")

    assert len(result.branches) == 3
    digests = {branch.post_state_digest for branch in result.branches}
    assert len(digests) > 1


def test_bifurcation_results_are_deterministic(config, state, external):
    engine = BifurcatingTransitionEngine(
        config,
        BifurcationPolicy(branches=4, horizon=2, trigger_threshold=0.0),
    )

    first = engine.step(state.clone(), external, run_id="deterministic")
    second = engine.step(state.clone(), external, run_id="deterministic")

    assert first.bifurcated == second.bifurcated
    assert first.selected_branch == second.selected_branch
    assert first.baseline_objective == second.baseline_objective
    assert [branch.objective for branch in first.branches] == [
        branch.objective for branch in second.branches
    ]
    assert [branch.post_state_digest for branch in first.branches] == [
        branch.post_state_digest for branch in second.branches
    ]
    assert [branch.policy for branch in first.branches] == [branch.policy for branch in second.branches]
    assert state_digest(first.state) == state_digest(second.state)


def test_branch_policies_are_recorded(config, state, external):
    engine = BifurcatingTransitionEngine(
        config,
        BifurcationPolicy(branches=4, horizon=2, trigger_threshold=0.0),
    )

    result = engine.step(state, external, run_id="metadata")

    assert len(result.branches) == 4
    for branch in result.branches:
        assert branch.policy.branch == branch.branch
        assert branch.policy.max_width >= config.resources.min_width
        assert branch.policy.input_scale > 0.0
    metadata = result.state.run_metadata["bifurcation"]
    assert metadata["attempted"] is True
    assert len(metadata["branches"]) == 4
    assert "policy" in metadata["branches"][0]
    assert metadata["branches"][0]["policy"]["branch"] == metadata["branches"][0]["branch"]


def test_inferior_branch_does_not_replace_baseline(config, state, external):
    engine = BifurcatingTransitionEngine(
        config,
        BifurcationPolicy(
            branches=4,
            horizon=2,
            trigger_threshold=0.0,
            baseline_margin=1e6,
        ),
    )

    result = engine.step(state, external, run_id="margin")

    assert result.bifurcated is False
    assert result.selected_branch is None
    metadata = result.state.run_metadata["bifurcation"]
    assert metadata["attempted"] is True
    assert metadata["decision"] == "retained_baseline_margin"
    assert metadata["baseline_margin"] == 1e6


def test_improvement_margin_alias_still_works(config, state, external):
    engine = BifurcatingTransitionEngine(
        config,
        BifurcationPolicy(
            branches=3,
            horizon=2,
            trigger_threshold=0.0,
            improvement_margin=0.25,
        ),
    )

    result = engine.step(state, external, run_id="alias")

    assert result.state.step == state.step + 1
    assert engine.policy.baseline_margin == 0.25
    assert engine.policy.improvement_margin == 0.25


def test_margin_alias_conflict_raises_configuration_error():
    with pytest.raises(ValueError, match="provide only one of baseline_margin or improvement_margin"):
        BifurcationPolicy(baseline_margin=0.1, improvement_margin=0.2)


def test_branch_post_state_digest_is_deterministic(config, state, external):
    engine = BifurcatingTransitionEngine(
        config,
        BifurcationPolicy(branches=3, horizon=2, trigger_threshold=0.0),
    )

    first = engine.step(state.clone(), external, run_id="digest")
    second = engine.step(state.clone(), external, run_id="digest")

    assert [branch.post_state_digest for branch in first.branches] == [
        branch.post_state_digest for branch in second.branches
    ]
