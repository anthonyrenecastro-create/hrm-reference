"""Matched one-step mechanism counterfactual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.core.transition import TransitionEngine, TransitionResult
from hrm.state.digest import numeric_view
from hrm.state.unified import HRMState


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    full: TransitionResult
    ablated: TransitionResult
    state_divergence: float


def one_step_counterfactual(
    state: HRMState, external_input: ExternalInput, config: HRMConfig, mechanism: str
) -> CounterfactualResult:
    full = TransitionEngine(config).step(
        state.clone(), external_input, run_id="counterfactual-full"
    )
    data = config.model_dump(mode="python")
    data["enabled"][mechanism] = False
    ablated_config = HRMConfig.model_validate(data)
    ablated = TransitionEngine(ablated_config).step(
        state.clone(), external_input, run_id="counterfactual-ablated"
    )
    full_view, ablated_view = numeric_view(full.state), numeric_view(ablated.state)
    divergence = float(
        np.sqrt(sum(np.sum((full_view[k] - ablated_view[k]) ** 2) for k in full_view))
    )
    return CounterfactualResult(full, ablated, divergence)
