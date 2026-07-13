from __future__ import annotations

import numpy as np

from hrm.core.safety import safe_transition


def test_nonfinite_rolls_back_and_is_attributed(config, state):
    candidate = state.clone()
    candidate.field.values[0, 0] = np.nan
    safe, corrections, effects = safe_transition(candidate, state, config)
    assert np.all(np.isfinite(safe.field.values))
    assert corrections[0].kind == "rollback"
    assert "field.values" in effects
    assert all(np.all(np.isfinite(value)) for value in effects.values())


def test_nonfinite_rollback_accepts_exact_ledger_inverse(config, state):
    candidate = state.clone()
    candidate.field.values[0, 0] = np.inf
    inverse = {"field.values": np.full_like(state.field.values, -0.25)}
    safe, corrections, effects = safe_transition(candidate, state, config, rollback_effect=inverse)
    assert np.array_equal(safe.field.values, state.field.values)
    assert np.array_equal(effects["field.values"], inverse["field.values"])
    assert corrections[0].metadata["exact_ledger_inverse"] is True


def test_field_norm_projection_is_attributed(config, state):
    candidate = state.clone()
    candidate.field.values[:] = 100
    safe, corrections, effects = safe_transition(candidate, state, config)
    assert (
        np.linalg.norm(safe.field.values[safe.field.active_channels])
        <= config.safety.field_norm_bound + 1e-9
    )
    assert any(item.reason in {"field_absolute_bound", "field_norm_bound"} for item in corrections)
    assert "field.values" in effects


def test_budget_correction_is_separate(config, state):
    candidate = state.clone()
    candidate.resources.current_use = 2 * candidate.resources.total_budget
    safe, corrections, effects = safe_transition(candidate, state, config)
    assert safe.resources.current_use == safe.resources.total_budget
    assert any(item.kind == "budget_correction" for item in corrections)
    assert "resources.current_use" in effects
