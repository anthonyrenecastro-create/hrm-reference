"""Paired statistical reporting with multiplicity corrections."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def paired_statistics(
    full: np.ndarray, ablated: np.ndarray, *, seed: int = 0, bootstrap_samples: int = 4000
) -> dict[str, Any]:
    full = np.asarray(full, dtype=float)
    ablated = np.asarray(ablated, dtype=float)
    if full.shape != ablated.shape or full.ndim != 1:
        raise ValueError("paired samples must be equal one-dimensional arrays")
    difference = full - ablated
    rng = np.random.default_rng(seed)
    if difference.size:
        indices = rng.integers(0, difference.size, size=(bootstrap_samples, difference.size))
        boot = np.mean(difference[indices], axis=1)
        ci = np.quantile(boot, [0.025, 0.975])
    else:
        ci = np.array([np.nan, np.nan])
    mad = np.median(np.abs(difference - np.median(difference))) + 1e-12
    robust_effect = float(np.median(difference) / (1.4826 * mad))
    positives = int(np.sum(difference > 0))
    nonzero = int(np.sum(difference != 0))
    sign_p = float(stats.binomtest(positives, nonzero, 0.5).pvalue) if nonzero else 1.0
    try:
        wilcoxon_p = float(stats.wilcoxon(difference).pvalue) if nonzero else 1.0
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n": int(difference.size),
        "paired_mean_difference": float(np.mean(difference)) if difference.size else float("nan"),
        "paired_median_difference": float(np.median(difference))
        if difference.size
        else float("nan"),
        "bootstrap_ci_95": ci.tolist(),
        "robust_effect_size": robust_effect,
        "sign_test_p": sign_p,
        "wilcoxon_p": wilcoxon_p,
        "per_seed_difference": difference.tolist(),
    }


def holm(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)[::-1]
    adjusted = np.empty(count)
    running = 1.0
    for reverse_rank, index in enumerate(order):
        rank = count - reverse_rank
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()
