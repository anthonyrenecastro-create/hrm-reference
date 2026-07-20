from __future__ import annotations

from hrm.cli import main
from hrm.config import save_config


def test_cli_run_ledger_replay_and_report(config, tmp_path):
    data = config.model_dump(mode="python")
    data["steps"] = 2
    small = type(config).model_validate(data)
    config_path = tmp_path / "config.yaml"
    run_path = tmp_path / "run"
    save_config(small, config_path)
    assert main(["validate-config", str(config_path)]) == 0
    assert (
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--task",
                "spatial-reconstruction",
                "--output",
                str(run_path),
            ]
        )
        == 0
    )
    assert main(["ledger", "verify", str(run_path)]) == 0
    assert main(["replay", str(run_path)]) == 0
    assert main(["report", str(run_path)]) == 0


def test_cli_reports_invalid_config(config, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")
    assert main(["validate-config", str(path)]) == 2


def test_cli_bifurcation_override_enables_engine(config, tmp_path):
    data = config.model_dump(mode="python")
    data["steps"] = 1
    small = type(config).model_validate(data)
    config_path = tmp_path / "config.yaml"
    run_path = tmp_path / "run"
    save_config(small, config_path)

    assert (
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--task",
                "spatial-reconstruction",
                "--output",
                str(run_path),
                "--bifurcation",
            ]
        )
        == 0
    )

    resolved = (run_path / "config.resolved.yaml").read_text(encoding="utf-8")
    assert "bifurcation:" in resolved
    assert "enabled: true" in resolved
