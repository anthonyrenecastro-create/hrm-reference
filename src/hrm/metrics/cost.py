def performance_cost(performance: float, cost: float) -> float:
    return float(performance / (cost + 1e-12))
