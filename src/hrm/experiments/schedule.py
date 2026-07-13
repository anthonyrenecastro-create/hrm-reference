"""Serializable input and perturbation schedules."""

from __future__ import annotations

from typing import Any

from hrm.config import HRMConfig
from hrm.tasks.base import TaskBundle
from hrm.types import jsonable


def serialize_schedule(bundle: TaskBundle, config: HRMConfig | None = None) -> dict[str, Any]:
    return {
        "task": bundle.name,
        "metadata": jsonable(bundle.metadata),
        "perturbations": {
            str(step): identity for step, (identity, _) in bundle.perturbations.items()
        },
        "configured_perturbations": []
        if config is None
        else [item.model_dump(mode="json") for item in config.perturbation_schedule],
        "inputs": [
            {
                "input_id": item.input_id,
                "perturbation_id": item.perturbation_id,
                "values": item.values.tolist(),
                "target_field": None if item.target_field is None else item.target_field.tolist(),
                "task_target": None if item.task_target is None else item.task_target.tolist(),
                "supervised_guidance": item.supervised_guidance,
                "metadata": jsonable(item.metadata),
            }
            for item in bundle.inputs
        ],
    }
