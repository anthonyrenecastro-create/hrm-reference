"""Helical Recursive Manifold scientific reference implementation."""

from hrm.config import HRMConfig, load_config
from hrm.core.transition import TransitionEngine
from hrm.state.unified import HRMState

__all__ = ["HRMConfig", "HRMState", "TransitionEngine", "load_config"]
__version__ = "0.1.1"
