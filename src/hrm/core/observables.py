"""Shared observables computed once from the immutable snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrm.core.proposal import ExternalInput
from hrm.core.snapshot import FrozenHRMState


@dataclass(frozen=True, slots=True)
class Observables:
    pooled_field: NDArray[np.floating]
    node_error: NDArray[np.floating]
    node_uncertainty: NDArray[np.floating]
    node_coherence: NDArray[np.floating]
    geometry_stress: NDArray[np.floating]
    memory_demand: NDArray[np.floating]
    importance: NDArray[np.floating]
    retrieved_memory: NDArray[np.floating]
    retrieval_weights: NDArray[np.floating]
    prediction_residual: NDArray[np.floating]
    task_target_available: bool
    supervision_authorized: bool


def _fit(vector: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=float)
    count = min(size, vector.size)
    out[:count] = np.ravel(vector)[:count]
    return out


def observe(snapshot: FrozenHRMState, external_input: ExternalInput) -> Observables:
    state = snapshot.state
    values = np.where(state.field.active_channels, state.field.values, 0.0)
    active_count = np.maximum(1, np.sum(state.field.active_channels, axis=0))
    pooled = np.sum(values, axis=0) / active_count
    if external_input.target_field is not None:
        target = np.asarray(external_input.target_field, dtype=float)
        residual_field = np.where(state.field.active_channels, target - values, 0.0)
        node_error = np.sqrt(np.mean(residual_field**2, axis=1))
    else:
        node_error = np.sqrt(np.mean(values**2, axis=1))
    uncertainty_scalar = float(np.mean(state.cognition.uncertainty))
    node_uncertainty = np.full(values.shape[0], uncertainty_scalar)
    weights = state.topology.weights
    np.asarray(weights.sum(axis=1)).ravel()
    disagreement = np.zeros(values.shape[0])
    for i in np.flatnonzero(state.field.active_nodes):
        start, end = weights.indptr[i], weights.indptr[i + 1]
        js, ws = weights.indices[start:end], weights.data[start:end]
        if js.size:
            disagreement[i] = float(
                np.sum(ws[:, None] * (values[js] - values[i]) ** 2)
                / (np.sum(ws) * values.shape[1] + 1e-12)
            )
    coherence = 1.0 / (1.0 + disagreement)
    coords = state.geometry.coordinates
    stress = np.zeros(values.shape[0])
    for i, j in state.topology.edges:
        feature_distance = float(np.linalg.norm(values[i] - values[j]))
        preferred = 0.25 + feature_distance
        geometric = float(np.linalg.norm(coords[i] - coords[j]))
        mismatch = abs(geometric - preferred)
        stress[i] += mismatch
        stress[j] += mismatch
    stress /= np.maximum(1, state.topology.degree)
    query = _fit(np.concatenate((pooled, state.cognition.latent)), state.memory.keys.shape[1])
    occupied = np.flatnonzero(state.memory.occupied)
    retrieval_weights = np.zeros(state.memory.keys.shape[0])
    retrieved = np.zeros(state.memory.values.shape[1])
    if occupied.size:
        keys = state.memory.keys[occupied]
        qnorm = np.linalg.norm(query) + 1e-12
        knorm = np.linalg.norm(keys, axis=1) + 1e-12
        similarity = keys @ query / (knorm * qnorm)
        logits = (
            3.0 * similarity
            + np.log(state.memory.strengths[occupied] + 1e-12)
            - 0.02 * state.memory.ages[occupied]
        )
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        retrieval_weights[occupied] = probs
        retrieved = probs @ state.memory.values[occupied]
    memory_demand = node_error * (1.0 + node_uncertainty)
    importance = 0.4 * node_error + 0.2 * node_uncertainty + 0.2 * stress + 0.2 * (1.0 - coherence)
    target_vector = external_input.task_target if external_input.task_target is not None else pooled
    prediction_residual = (
        _fit(np.asarray(target_vector, dtype=float), state.cognition.prediction.size)
        - state.cognition.prediction
    )
    return Observables(
        pooled,
        node_error,
        node_uncertainty,
        coherence,
        stress,
        memory_demand,
        importance,
        retrieved,
        retrieval_weights,
        prediction_residual,
        external_input.task_target is not None,
        bool(external_input.supervised_guidance),
    )
