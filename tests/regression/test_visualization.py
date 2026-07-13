from __future__ import annotations

import pandas as pd

from hrm.core.transition import TransitionEngine
from hrm.visualization import (
    plot_field,
    plot_geometry,
    plot_ledger_norms,
    plot_topology,
    plot_trajectories,
)


def test_all_visualizations_emit_nonempty_pngs(config, state, external, tmp_path):
    result = TransitionEngine(config).step(state, external, run_id="plots")
    records = [result.ledger.to_dict()]
    metrics = pd.DataFrame(
        [
            {"step": 0, "reconstruction_error": 1.0, "prediction_error": 0.5},
            {"step": 1, "reconstruction_error": 0.8, "prediction_error": 0.4},
        ]
    )
    paths = (
        plot_field(result.state, tmp_path / "field.png"),
        plot_geometry(result.state, tmp_path / "geometry.png"),
        plot_topology(result.state, tmp_path / "topology.png"),
        plot_trajectories(metrics, tmp_path / "trajectories.png"),
        plot_ledger_norms(records, tmp_path / "ledger.png"),
    )
    assert all(path.stat().st_size > 0 for path in paths)
