"""Typed, strict HRM configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hrm.types import LaplacianKind, ReactionKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class DimensionsConfig(StrictModel):
    n_nodes: int = Field(24, ge=4)
    channels: int = Field(6, ge=1)
    spatial_dim: int = Field(2, ge=1, le=3)
    latent_dim: int = Field(6, ge=1)
    memory_capacity: int = Field(16, ge=1)
    memory_key_dim: int = Field(6, ge=1)
    memory_value_dim: int = Field(6, ge=1)
    regions: int = Field(4, ge=1)

    @model_validator(mode="after")
    def compatible(self) -> DimensionsConfig:
        if self.regions > self.n_nodes:
            raise ValueError("regions cannot exceed n_nodes")
        return self


class InitializationConfig(StrictModel):
    field_scale: float = Field(0.05, ge=0)
    coordinate_scale: float = Field(1.0, gt=0)


class GraphConfig(StrictModel):
    kind: Literal["ring", "grid", "knn"] = "ring"
    neighbors: int = Field(2, ge=1)
    weighted: bool = True
    laplacian: LaplacianKind = LaplacianKind.SYMMETRIC
    distance_scale: float = Field(1.0, gt=0)
    min_degree: int = Field(1, ge=0)
    max_degree: int = Field(8, ge=1)
    require_connected: bool = True


class DynamicsConfig(StrictModel):
    dt: float = Field(0.04, gt=0)
    solver: Literal["explicit_euler"] = "explicit_euler"
    diffusion_kind: Literal["geometry_aware_graph"] = "geometry_aware_graph"
    diffusion: float = Field(0.18, ge=0)
    reaction: ReactionKind = ReactionKind.CUBIC
    reaction_a: float = 0.45
    reaction_b: float = Field(0.35, gt=0)
    reaction_competition: float = Field(0.02, ge=0)
    input_gain: float = 0.35
    passive_decay: float = Field(0.015, ge=0)
    regional_gain: float = Field(0.08, ge=0)
    cognition_gain: float = Field(0.16, ge=0)
    field_coupling_gain: float = Field(0.05, ge=0)
    memory_coupling_gain: float = Field(0.05, ge=0)
    guidance_gain: float = Field(0.08, ge=0)
    guidance_bound: float = Field(0.25, gt=0)
    stochastic_scale: float = Field(0.0, ge=0)
    cadence: dict[str, int] = Field(default_factory=dict)


class MemoryConfig(StrictModel):
    decay: float = Field(0.01, ge=0)
    temperature: float = Field(3.0, gt=0)
    age_penalty: float = Field(0.02, ge=0)
    write_threshold: float = Field(0.18, ge=0)
    novelty_weight: float = Field(0.4, ge=0)
    residual_weight: float = Field(0.4, ge=0)
    uncertainty_weight: float = Field(0.2, ge=0)
    max_strength: float = Field(1.0, gt=0)


class HierarchyConfig(StrictModel):
    enabled: bool = True
    levels: Literal[2] = 2
    bottom_up_gain: float = Field(0.3, ge=0)
    top_down_gain: float = Field(0.06, ge=0)
    max_feedback_gain: float = Field(0.25, ge=0)


class GeometryConfig(StrictModel):
    enabled: bool = True
    gain: float = Field(0.01, ge=0)
    max_displacement: float = Field(0.02, gt=0)
    anchor_penalty: float = Field(0.05, ge=0)
    min_separation: float = Field(1e-3, gt=0)
    max_separation: float = Field(10.0, gt=0)
    adaptive_weights: bool = True


class TopologyConfig(StrictModel):
    enabled: bool = True
    grow_threshold: float = 0.25
    prune_threshold: float = -0.1
    cost_weight: float = Field(0.15, ge=0)
    risk_weight: float = Field(0.1, ge=0)
    stress_weight: float = Field(0.5, ge=0)
    robustness_weight: float = Field(0.2, ge=0)
    min_edge_age: int = Field(3, ge=0)
    cooldown: int = Field(2, ge=0)
    churn_limit: float = Field(0.2, ge=0, le=1)
    repair_enabled: bool = True


class ResourceConfig(StrictModel):
    total_budget: float = Field(10_000.0, gt=0)
    min_width: int = Field(1, ge=1)
    max_width: int | None = None
    switching_penalty: float = Field(0.05, ge=0)
    event_budget: int = Field(2, ge=0)
    memory_allowance: int | None = None
    adaptive: bool = True


class SafetyConfig(StrictModel):
    adaptive_timestep: bool = True
    min_dt: float = Field(1e-5, gt=0)
    field_abs_bound: float = Field(8.0, gt=0)
    field_norm_bound: float = Field(64.0, gt=0)
    cognitive_norm_bound: float = Field(16.0, gt=0)
    coordinate_bound: float = Field(5.0, gt=0)
    max_corrections_per_step: int = Field(8, ge=0)
    rollback_on_nonfinite: bool = True


class LedgerConfig(StrictModel):
    tolerance: float = Field(1e-10, gt=0)
    store_arrays: bool = True
    schema_version: str = "1.0"


class PerturbationConfig(StrictModel):
    step: int = Field(ge=0)
    kind: Literal["field_noise", "field_scale", "geometry_jitter", "memory_attenuation"]
    magnitude: float = Field(ge=0)
    label: str | None = None


DEFAULT_ENABLED: dict[str, bool] = {
    "passive": True,
    "input": True,
    "diffusion": True,
    "reaction": True,
    "regional": True,
    "memory": True,
    "cognition": True,
    "guidance": True,
    "hierarchy_bottom_up": True,
    "hierarchy_top_down": True,
    "geometry": True,
    "topology": True,
    "allocation": True,
    "field_to_cognition": True,
    "cognition_to_field": True,
    "memory_to_cognition": True,
    "memory_to_field": True,
    "hierarchy_to_cognition": True,
    "residual_uncertainty_interaction": True,
}


class HRMConfig(StrictModel):
    name: str = "hrm-reference"
    seed: int = 7
    dtype: Literal["float64", "float32"] = "float64"
    steps: int = Field(20, ge=1)
    dimensions: DimensionsConfig = Field(default_factory=DimensionsConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    perturbation_schedule: tuple[PerturbationConfig, ...] = ()
    enabled: dict[str, bool] = Field(default_factory=lambda: dict(DEFAULT_ENABLED))
    fixed_topology: bool = False
    fixed_geometry: bool = False
    fixed_graph: bool = False
    fixed_allocation: bool = False
    allow_supervised_guidance: bool = False
    output_root: str = "artifacts"

    @model_validator(mode="after")
    def validate_contract(self) -> HRMConfig:
        unknown = set(self.enabled) - set(DEFAULT_ENABLED)
        if unknown:
            raise ValueError(f"unknown mechanisms: {sorted(unknown)}")
        merged = dict(DEFAULT_ENABLED)
        merged.update(self.enabled)
        self.enabled = merged
        max_width = self.resources.max_width or self.dimensions.channels
        if max_width > self.dimensions.channels:
            raise ValueError("resources.max_width cannot exceed dimensions.channels")
        if self.resources.min_width > max_width:
            raise ValueError("resources.min_width cannot exceed max_width")
        if (
            self.resources.memory_allowance is not None
            and self.resources.memory_allowance > self.dimensions.memory_capacity
        ):
            raise ValueError("memory allowance exceeds memory capacity")
        if self.graph.max_degree >= self.dimensions.n_nodes:
            self.graph.max_degree = self.dimensions.n_nodes - 1
        if any(item.step >= self.steps for item in self.perturbation_schedule):
            raise ValueError("perturbation step must be smaller than configured steps")
        if self.fixed_graph:
            self.fixed_topology = True
            self.fixed_geometry = True
        if self.fixed_topology:
            self.enabled["topology"] = False
        if self.fixed_geometry:
            self.enabled["geometry"] = False
        if self.fixed_allocation:
            self.enabled["allocation"] = False
        if not self.hierarchy.enabled:
            self.enabled["hierarchy_bottom_up"] = False
            self.enabled["hierarchy_top_down"] = False
            self.enabled["hierarchy_to_cognition"] = False
        return self

    @property
    def numpy_dtype(self) -> str:
        return self.dtype

    @property
    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: str | Path) -> HRMConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return HRMConfig.model_validate(raw)


def save_config(config: HRMConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)


def apply_ablation(config: HRMConfig, name: str) -> HRMConfig:
    data = config.model_dump(mode="python")
    enabled = dict(data["enabled"])
    if name == "no_guidance":
        enabled["guidance"] = False
    elif name == "no_hrm_adaptation":
        enabled["geometry"] = False
        enabled["topology"] = False
        enabled["cognition_to_field"] = False
        enabled["field_to_cognition"] = False
    elif name == "fixed_topology":
        data["fixed_topology"] = True
    elif name == "fixed_geometry":
        data["fixed_geometry"] = True
    elif name == "fixed_graph":
        data["fixed_graph"] = True
    elif name == "no_hierarchy":
        data["hierarchy"]["enabled"] = False
    elif name == "memory_only":
        enabled = {
            key: key in {"passive", "input", "memory", "memory_to_cognition", "memory_to_field"}
            for key in enabled
        }
    elif name == "fixed_allocation":
        data["fixed_allocation"] = True
    elif name == "no_memory":
        for key in ("memory", "memory_to_cognition", "memory_to_field"):
            enabled[key] = False
    elif name == "no_cognition_to_field":
        enabled["cognition_to_field"] = False
    elif name == "no_field_to_cognition":
        enabled["field_to_cognition"] = False
    elif name == "no_regional_coupling":
        enabled["regional"] = False
    elif name == "no_topology_repair":
        data["topology"]["repair_enabled"] = False
    elif name == "no_adaptive_timestep":
        data["safety"]["adaptive_timestep"] = False
    else:
        raise ValueError(f"unknown ablation: {name}")
    data["enabled"] = enabled
    return HRMConfig.model_validate(data)
