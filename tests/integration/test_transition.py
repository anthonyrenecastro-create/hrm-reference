from __future__ import annotations

import numpy as np
import pytest

from hrm.core.contribution import MechanismProposal
from hrm.core.transition import TransitionEngine
from hrm.state.digest import numeric_view, state_digest
from hrm.types import StateBlock


def test_transition_advances_once_and_reconstructs(config, state, external):
    result = TransitionEngine(config).step(state, external, run_id="transition")
    assert result.state.step == state.step + 1
    assert result.ledger.relative_residual <= config.ledger.tolerance
    assert result.ledger.pre_state_digest == state_digest(state)
    assert result.ledger.post_state_digest == state_digest(result.state)


def test_same_inputs_reproduce_states_events_and_ledgers(config, state, external):
    first = TransitionEngine(config).step(state.clone(), external, run_id="same")
    second = TransitionEngine(config).step(state.clone(), external, run_id="same")
    assert state_digest(first.state) == state_digest(second.state)
    assert [x["event_id"] for x in first.ledger.accepted_events] == [
        x["event_id"] for x in second.ledger.accepted_events
    ]
    assert first.ledger.observed_delta == second.ledger.observed_delta


def test_common_snapshot_evaluation_order(config, state, external):
    mechanisms = TransitionEngine(config).mechanisms
    forward = TransitionEngine(config, mechanisms=mechanisms).step(
        state.clone(), external, run_id="order"
    )
    reverse = TransitionEngine(config, mechanisms=list(reversed(mechanisms))).step(
        state.clone(), external, run_id="order"
    )
    for path, value in numeric_view(forward.state).items():
        assert np.allclose(value, numeric_view(reverse.state)[path], rtol=0, atol=1e-12)


def test_hidden_snapshot_reassignment_is_detected(config, state, external):
    class Evil:
        name = "input"
        reads = frozenset({StateBlock.FIELD})
        writes = frozenset({StateBlock.FIELD})

        def propose(self, snapshot, external_input, observables, context):
            snapshot.state.field.values = np.zeros_like(snapshot.state.field.values)
            return MechanismProposal(self.name, 0.0, {}, self.reads, self.writes)

    with pytest.raises(RuntimeError, match="snapshot was mutated"):
        TransitionEngine(config, mechanisms=[Evil()], interactions=[]).step(state, external)


def test_duplicate_mechanism_names_are_rejected(config):
    mechanism = TransitionEngine(config).mechanisms[0]
    with pytest.raises(ValueError, match="unique"):
        TransitionEngine(config, mechanisms=[mechanism, mechanism])


def test_disabled_adaptation_reports_exceeded_timestep_bound(config, state):
    data = config.model_dump(mode="python")
    data["dynamics"]["dt"] = 10.0
    data["safety"]["adaptive_timestep"] = False
    unstable = type(config).model_validate(data)
    dt, reason, _ = TransitionEngine(unstable)._effective_timestep(state)
    assert dt == 10.0
    assert reason is not None and reason.startswith("bound_exceeded_without_adaptation")


def test_seeded_stochastic_contribution_is_named_and_reconstructs(config, state, external):
    data = config.model_dump(mode="python")
    data["dynamics"]["stochastic_scale"] = 0.1
    stochastic = type(config).model_validate(data)
    result = TransitionEngine(stochastic).step(state, external, run_id="stochastic")
    assert result.ledger.stochastic_record["kind"] == "intentional_seeded"
    assert result.ledger.stochastic_record["norm"] > 0
    assert result.ledger.relative_residual <= stochastic.ledger.tolerance


def test_nonrepresentable_aggregate_fails_instead_of_becoming_noise(config, state, external):
    class Huge:
        reads = frozenset({StateBlock.FIELD})
        writes = frozenset({StateBlock.FIELD})

        def __init__(self, name):
            self.name = name

        def propose(self, snapshot, external_input, observables, context):
            delta = np.full_like(snapshot.state.field.values, np.finfo(float).max)
            return MechanismProposal(
                self.name, 1.0, {"field.values": delta}, self.reads, self.writes
            )

    with (
        np.errstate(over="ignore"),
        pytest.raises(FloatingPointError, match="aggregate contribution"),
    ):
        TransitionEngine(
            config,
            mechanisms=[Huge("input"), Huge("passive")],
            interactions=[],
        ).step(state, external)
