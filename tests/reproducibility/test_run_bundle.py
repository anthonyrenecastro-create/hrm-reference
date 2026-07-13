from __future__ import annotations

from hrm.experiments.replay import replay_run
from hrm.experiments.runner import run_task


def test_run_bundle_replays(config, tmp_path):
    data = config.model_dump(mode="python")
    data["steps"] = 2
    from hrm.config import HRMConfig

    small = HRMConfig.model_validate(data)
    result = run_task(
        small, "spatial-reconstruction", small.seed, tmp_path / "run", run_id="bundle"
    )
    replay = replay_run(result.output_dir)
    assert replay["matched"] and replay["steps"] == 2


def test_configured_perturbation_is_named_and_replayable(config, tmp_path):
    data = config.model_dump(mode="python")
    data["steps"] = 3
    data["perturbation_schedule"] = [
        {
            "step": 1,
            "kind": "field_noise",
            "magnitude": 0.2,
            "label": "test-noise-v1",
        }
    ]
    from hrm.config import HRMConfig

    configured = HRMConfig.model_validate(data)
    result = run_task(
        configured,
        "spatial-reconstruction",
        configured.seed,
        tmp_path / "configured",
        run_id="configured",
    )
    assert result.ledgers[1].perturbation_identity == "test-noise-v1"
    assert replay_run(result.output_dir)["matched"]
