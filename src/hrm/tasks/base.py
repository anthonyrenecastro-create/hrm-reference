"""Canonical task bundle and registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hrm.config import HRMConfig
from hrm.core.proposal import ExternalInput
from hrm.state.unified import HRMState

Perturbation = Callable[[HRMState], HRMState]


@dataclass(slots=True)
class TaskBundle:
    name: str
    initial_state: HRMState
    inputs: list[ExternalInput]
    perturbations: dict[int, tuple[str, Perturbation]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_task(name: str, config: HRMConfig, seed: int) -> TaskBundle:
    normalized = name.replace("_", "-")
    if normalized == "spatial-reconstruction":
        from hrm.tasks.spatial_reconstruction import build
    elif normalized == "sequence-prediction":
        from hrm.tasks.sequence_prediction import build
    elif normalized == "associative-recall":
        from hrm.tasks.associative_recall import build
    elif normalized == "perturbation-recovery":
        from hrm.tasks.perturbation_recovery import build
    elif normalized == "resource-adaptation":
        from hrm.tasks.resource_adaptation import build
    elif normalized == "topology-usefulness":
        from hrm.tasks.topology_usefulness import build
    elif normalized == "topology-irrelevance":
        from hrm.tasks.topology_irrelevance import build
    else:
        raise ValueError(f"unknown task: {name}")
    return build(config, seed)
