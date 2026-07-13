from __future__ import annotations

import pytest

from hrm.core.ledger import verify_ledger_file, verify_record, write_ledger
from hrm.core.transition import TransitionEngine


def test_independent_ledger_verification(config, state, external, tmp_path):
    record = TransitionEngine(config).step(state, external).ledger
    path = tmp_path / "ledger.jsonl"
    write_ledger([record], path)
    result = verify_ledger_file(path, config.ledger.tolerance)
    assert result["records"] == 1
    assert result["maximum_relative_residual"] <= config.ledger.tolerance


def test_deliberate_residual_injection_fails(config, state, external):
    record = TransitionEngine(config).step(state, external).ledger.to_dict()
    path = next(iter(record["observed_delta"]))
    value = record["observed_delta"][path]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value[0][0] += 1.0
    elif isinstance(value, list):
        value[0] += 1.0
    else:
        record["observed_delta"][path] = float(value) + 1.0
    with pytest.raises(ValueError):
        verify_record(record, config.ledger.tolerance)


def test_arrays_are_required_for_independent_verification(config, state, external):
    data = config.model_dump(mode="python")
    data["ledger"]["store_arrays"] = False
    from hrm.config import HRMConfig

    record = TransitionEngine(HRMConfig.model_validate(data)).step(state, external).ledger.to_dict()
    with pytest.raises(ValueError, match="does not store arrays"):
        verify_record(record)
