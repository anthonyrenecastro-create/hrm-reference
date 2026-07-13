from __future__ import annotations

import numpy as np
import pytest

from hrm.state.digest import state_digest
from hrm.state.serialization import load_state, save_state
from hrm.state.validation import validate_state


def test_state_validates_and_clones_independently(config, state):
    assert validate_state(state, config).valid
    clone = state.clone()
    clone.field.values[0, 0] += 1
    assert not np.array_equal(clone.field.values, state.field.values)


def test_state_serialization_preserves_digest(config, state, tmp_path):
    path = tmp_path / "state.npz"
    before = state_digest(state)
    save_state(state, path)
    restored = load_state(path, config.dtype)
    assert state_digest(restored) == before
    restored.validate(config)


def test_invalid_shape_and_nonfinite_fail(config, state):
    state.field.values = state.field.values[:-1]
    with pytest.raises(ValueError):
        state.validate(config)


def test_digest_is_deterministic_and_sensitive(state):
    first = state_digest(state)
    assert state_digest(state.clone()) == first
    state.field.values[0, 0] += 1e-12
    assert state_digest(state) != first
