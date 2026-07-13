"""Immutable common-snapshot representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import issparse

from hrm.state.digest import state_digest
from hrm.state.unified import HRMState


def _readonly(value: Any) -> None:
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif issparse(value):
        value.data.setflags(write=False)
        value.indices.setflags(write=False)
        value.indptr.setflags(write=False)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _readonly(getattr(value, name))


@dataclass(frozen=True, slots=True)
class FrozenHRMState:
    state: HRMState
    digest: str

    def assert_unchanged(self) -> None:
        if state_digest(self.state) != self.digest:
            raise RuntimeError("immutable snapshot was mutated")


def freeze(state: HRMState) -> FrozenHRMState:
    cloned = state.clone()
    digest = state_digest(cloned)
    _readonly(cloned)
    return FrozenHRMState(cloned, digest)
