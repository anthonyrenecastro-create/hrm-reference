"""Transition engine selection from runtime configuration."""

from __future__ import annotations

from collections.abc import Iterable

from hrm.config import HRMConfig
from hrm.core.bifurcation import BifurcatingTransitionEngine, BifurcationPolicy
from hrm.core.interaction import Interaction
from hrm.core.proposal import Mechanism
from hrm.core.transition import TransitionEngine


def build_transition_engine(
    config: HRMConfig,
    mechanisms: Iterable[Mechanism] | None = None,
    interactions: Iterable[Interaction] | None = None,
) -> TransitionEngine | BifurcatingTransitionEngine:
    if config.bifurcation.enabled:
        return BifurcatingTransitionEngine(
            config,
            BifurcationPolicy(
                branches=config.bifurcation.branches,
                horizon=config.bifurcation.horizon,
                trigger_threshold=config.bifurcation.trigger_threshold,
            ),
            mechanisms=list(mechanisms) if mechanisms is not None else None,
            interactions=list(interactions) if interactions is not None else None,
        )
    return TransitionEngine(config, mechanisms=mechanisms, interactions=interactions)
