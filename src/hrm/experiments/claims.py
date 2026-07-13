"""Mechanism evidence-gate classification without inflated claims."""

from __future__ import annotations

from typing import Any

from hrm.core.ledger import LedgerRecord


def mechanism_gate_report(records: list[LedgerRecord]) -> dict[str, dict[str, Any]]:
    names = sorted(
        {
            proposal["mechanism"]
            for record in records
            for group in (record.main_contributions, record.interaction_contributions)
            for proposal in group
        }
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        proposals = [
            proposal
            for record in records
            for group in (record.main_contributions, record.interaction_contributions)
            for proposal in group
            if proposal["mechanism"] == name
        ]
        executed = len(proposals)
        activations = sum(float(item["activation"]) > 0 for item in proposals)
        result[name] = {
            "G1_execution": executed > 0,
            "execution_steps": executed,
            "G2_activation": activations > 0,
            "activation_steps": activations,
            "G3_divergence": None,
            "G4_utility": None,
            "status": "activation_measured; matched ablation required for divergence and utility",
        }
    return result


def apply_matched_gate(
    report: dict[str, dict[str, Any]],
    mechanism: str,
    *,
    diverged: bool,
    utility: float | None,
    cost_equal: bool,
) -> None:
    entry = report.setdefault(mechanism, {})
    entry["G3_divergence"] = diverged
    entry["G4_utility"] = bool(diverged and cost_equal and utility is not None and utility > 0)
    entry["matched_utility_effect"] = utility
    entry["cost_equalized"] = cost_equal
