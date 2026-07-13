from __future__ import annotations

from hrm.config import DEFAULT_ENABLED
from hrm.core.transition import TransitionEngine
from hrm.tasks import build_task


def test_canonical_micro_suite_executes_and_activates_every_enabled_mechanism(config):
    tasks = (
        "spatial-reconstruction",
        "sequence-prediction",
        "associative-recall",
        "perturbation-recovery",
        "resource-adaptation",
        "topology-usefulness",
        "topology-irrelevance",
    )
    executed: set[str] = set()
    activated: set[str] = set()
    for task in tasks:
        bundle = build_task(task, config, config.seed)
        state = bundle.initial_state
        for step, external_input in enumerate(bundle.inputs):
            if step in bundle.perturbations:
                state = bundle.perturbations[step][1](state)
            result = TransitionEngine(config).step(state, external_input, run_id=task)
            state = result.state
            for group in (
                result.ledger.main_contributions,
                result.ledger.interaction_contributions,
            ):
                for proposal in group:
                    name = str(proposal["mechanism"])
                    executed.add(name)
                    if float(proposal["activation"]) > 0:
                        activated.add(name)
    expected = {name for name, enabled in DEFAULT_ENABLED.items() if enabled}
    assert expected <= executed
    assert expected <= activated
