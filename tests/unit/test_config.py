from __future__ import annotations

import pytest

from hrm.config import HRMConfig, apply_ablation


def test_unknown_configuration_is_rejected():
    with pytest.raises(ValueError):
        HRMConfig.model_validate({"not_a_field": 1})


def test_fixed_graph_enforces_distinct_invariants(config):
    data = config.model_dump(mode="python")
    data["fixed_graph"] = True
    fixed = HRMConfig.model_validate(data)
    assert fixed.fixed_graph and fixed.fixed_topology and fixed.fixed_geometry
    assert not fixed.enabled["topology"] and not fixed.enabled["geometry"]


def test_no_hierarchy_removes_all_paths(config):
    ablated = apply_ablation(config, "no_hierarchy")
    assert not ablated.hierarchy.enabled
    assert not ablated.enabled["hierarchy_bottom_up"]
    assert not ablated.enabled["hierarchy_top_down"]
    assert not ablated.enabled["hierarchy_to_cognition"]


def test_memory_only_has_no_hidden_guidance_or_hierarchy(config):
    ablated = apply_ablation(config, "memory_only")
    assert ablated.enabled["memory"]
    assert not ablated.enabled["guidance"]
    assert not ablated.enabled["hierarchy_bottom_up"]
    assert not ablated.enabled["hierarchy_top_down"]
