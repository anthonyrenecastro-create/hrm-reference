"""Authoritative transition, attribution, and structural semantics."""

from hrm.core.bifurcation import BifurcatingTransitionEngine, BifurcationPolicy
from hrm.core.engine_selection import build_transition_engine
from hrm.core.transition import TransitionEngine, TransitionResult

__all__ = [
	"BifurcatingTransitionEngine",
	"BifurcationPolicy",
	"build_transition_engine",
	"TransitionEngine",
	"TransitionResult",
]
