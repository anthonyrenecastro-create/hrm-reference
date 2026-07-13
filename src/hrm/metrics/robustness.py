def robustness(before: float, after: float) -> float:
    return float(after / (before + 1e-12))
