def topology_churn(added: int, removed: int, edges: int) -> float:
    return float((added + removed) / max(1, edges))
