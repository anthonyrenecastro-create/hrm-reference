from __future__ import annotations

import numpy as np
import pytest

from hrm.core.contribution import MechanismProposal
from hrm.core.snapshot import freeze
from hrm.mechanisms.base import build_default_mechanisms
from hrm.types import StateBlock


def test_snapshot_arrays_are_read_only(state):
    snapshot = freeze(state)
    with pytest.raises(ValueError):
        snapshot.state.field.values[0, 0] = 10


def test_zero_activation_requires_zero_contribution(config, state):
    from hrm.core.contribution import validate_proposal

    proposal = MechanismProposal(
        "bad",
        0.0,
        {"field.values": np.ones_like(state.field.values)},
        frozenset(),
        frozenset({StateBlock.FIELD}),
    )
    with pytest.raises(ValueError):
        validate_proposal(proposal, state)


def test_default_mechanism_names_are_unique():
    names = [item.name for item in build_default_mechanisms()]
    assert len(names) == len(set(names))


def test_undeclared_write_is_rejected(state):
    from hrm.core.contribution import validate_proposal

    proposal = MechanismProposal(
        "bad",
        1.0,
        {"field.values": np.zeros_like(state.field.values)},
        frozenset(),
        frozenset({StateBlock.COGNITION}),
    )
    with pytest.raises(ValueError):
        validate_proposal(proposal, state)
