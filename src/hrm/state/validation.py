"""State validation facade and explicit invariant report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components

from hrm.config import HRMConfig
from hrm.state.unified import HRMState


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_state(
    state: HRMState, config: HRMConfig, *, raise_on_error: bool = True
) -> ValidationReport:
    violations: list[str] = []
    warnings: list[str] = []
    try:
        state.validate(config)
    except ValueError as exc:
        violations.append(str(exc))
    active = np.flatnonzero(state.field.active_nodes)
    if active.size:
        subgraph = state.topology.weights[active][:, active]
        components, _ = connected_components(subgraph, directed=False)
        if config.graph.require_connected and components > 1:
            violations.append(f"active graph has {components} connected components")
        degree = np.asarray((subgraph > 0).sum(axis=1)).ravel()
        if np.any(degree > config.graph.max_degree):
            violations.append("maximum degree exceeded")
        if active.size > 1 and np.any(degree < config.graph.min_degree):
            violations.append("minimum degree violated")
    field_norm = float(np.linalg.norm(state.field.values[state.field.active_channels]))
    if field_norm > config.safety.field_norm_bound:
        warnings.append("field norm exceeds safe-transition bound")
    if np.linalg.norm(state.cognition.latent) > config.safety.cognitive_norm_bound:
        warnings.append("cognitive norm exceeds safe-transition bound")
    report = ValidationReport(not violations, tuple(violations), tuple(warnings))
    if raise_on_error and violations:
        raise ValueError("; ".join(violations))
    return report
