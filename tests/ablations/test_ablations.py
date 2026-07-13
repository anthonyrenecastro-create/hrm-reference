from __future__ import annotations

from hrm.config import apply_ablation
from hrm.core.transition import TransitionEngine
from hrm.state.digest import numeric_state_digest


def _names(record):
    return {item["mechanism"] for item in record.main_contributions}


def test_no_guidance_removes_production_proposal(config, state, external):
    result = TransitionEngine(apply_ablation(config, "no_guidance")).step(state, external)
    assert "guidance" not in _names(result.ledger)


def test_fixed_geometry_changes_no_coordinates(config, state, external):
    result = TransitionEngine(apply_ablation(config, "fixed_geometry")).step(state, external)
    assert "geometry" not in _names(result.ledger)
    assert (result.state.geometry.coordinates == state.geometry.coordinates).all()


def test_mechanism_ablation_diverges(config, state, external):
    full = TransitionEngine(config).step(state.clone(), external, run_id="matched")
    ablated = TransitionEngine(apply_ablation(config, "no_guidance")).step(
        state.clone(), external, run_id="matched"
    )
    assert numeric_state_digest(full.state) != numeric_state_digest(ablated.state)
