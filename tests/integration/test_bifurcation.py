from __future__ import annotations

from hrm.config import HRMConfig
from hrm.core.bifurcation import BifurcatingTransitionEngine, BifurcationPolicy


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


def test_bifurcation_engine_selects_branch_when_threshold_reached(config, state, external):
    data = config.model_dump(mode="python")
    data["dynamics"]["stochastic_scale"] = 0.15
    stochastic = HRMConfig.model_validate(data)
    engine = BifurcatingTransitionEngine(
        stochastic,
        BifurcationPolicy(branches=3, horizon=2, trigger_threshold=0.0),
    )

    result = engine.step(state, external, run_id="bifurcate")

    assert result.state.step == state.step + 1
    assert result.bifurcated is True
    assert result.selected_branch is not None
    assert 0 <= result.selected_branch < 3
    assert len(result.branches) == 3
    assert "bifurcation" in result.state.run_metadata
    assert result.state.run_metadata["bifurcation"]["attempted"] is True
    assert result.state.run_metadata["bifurcation"]["selected_branch"] == result.selected_branch
