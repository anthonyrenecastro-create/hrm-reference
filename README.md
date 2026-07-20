# Helical Recursive Manifold reference implementation

[![tests](https://github.com/anthonyrenecastro-create/hrm-reference/actions/workflows/tests.yml/badge.svg)](https://github.com/anthonyrenecastro-create/hrm-reference/actions/workflows/tests.yml)
[![lint](https://github.com/anthonyrenecastro-create/hrm-reference/actions/workflows/lint.yml/badge.svg)](https://github.com/anthonyrenecastro-create/hrm-reference/actions/workflows/lint.yml)
[![reproducibility](https://github.com/anthonyrenecastro-create/hrm-reference/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/anthonyrenecastro-create/hrm-reference/actions/workflows/reproducibility.yml)

`hrm-reference` is a deterministic scientific reference implementation of the Helical Recursive Manifold (HRM): an attributed hybrid state-space system combining multidimensional graph fields, geometry, topology, memory, bounded predictive state, hierarchy, and measured resources.

The repository is designed to answer a narrower and more useful question than “does a complex simulation run?”: **which named mechanisms execute, activate, change matched trajectories, and improve an appropriate outcome at comparable cost?**

## Current evidence status

| Level | Status | Meaning |
|---|---:|---|
| Specified | Yes | The theory and implementation contracts are documented. |
| Implemented | Yes | The reference transition, state blocks, mechanisms, tasks, and controls execute. |
| Activated | Measured | Activation is reported per mechanism and task; it is not assumed. |
| Attributable | Yes | Every accepted production transition must pass ledger reconstruction. |
| Synthetically supported | Not presumed | Generated experiment reports determine this per mechanism/task. |
| Externally supported | No | No independent external workload evidence is claimed. |
| Validated | No | This repository is an instrument for validation, not proof of validation. |

HRM is not presented here as consciousness, biological equivalence, a physical law, general intelligence, or established superiority over conventional computing.

The retained 0.1.0 calibration evidence comprises 21 runs across all seven synthetic tasks (three seeds each), 42 paired ablation comparisons across 14 interventions, and one disclosed single-seed baseline comparison. All 21 canonical ledgers and all 42 ablation-pair ledgers passed. The worst canonical relative reconstruction residual was `3.37e-19`. These are implementation/calibration results, not broad scientific validation.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quick start

```bash
hrm validate-config configs/minimal.yaml
hrm run --config configs/minimal.yaml --output artifacts/smoke
hrm ledger verify artifacts/smoke
hrm replay artifacts/smoke
```

Canonical task and ablation examples:

```bash
hrm experiment spatial-reconstruction --seeds 5 --config configs/full_hrm.yaml
hrm experiment sequence-prediction --seeds 5 --config configs/full_hrm.yaml
hrm ablate --config configs/full_hrm.yaml --suite canonical --seeds 5
```

The default smoke configuration is intentionally small and CPU-friendly. Larger runs should be treated as experiments and preserve their resolved configuration, schedules, per-seed records, environment, and source revision.

## Defining contracts

- One authoritative `TransitionEngine` is used by commands, tasks, benchmarks, and tests.
- All main proposals read the same immutable pre-transition snapshot.
- Structural changes are immutable events with acceptance or rejection records.
- Fixed maximum carriers and active masks provide exact initial transport semantics.
- Safety actions are corrections, never hidden inside ordinary mechanisms.
- Intentional randomness is counter-derived from run seed, step, and stream name.
- Ledger failure invalidates downstream task interpretation.
- Fixed topology, fixed geometry, fixed graph, and fixed allocation are distinct interventions.

See [architecture](docs/architecture.md), [mathematical contract](docs/mathematical_contract.md), [transition protocol](docs/transition_protocol.md), and [claims boundary](docs/claims_boundary.md).

## Release evidence

The 0.1.0 release was installed in a clean Python 3.12 environment and verified with:

- 55 passing tests;
- Ruff check and format verification;
- mypy with no issues across 94 source files;
- 83% total branch-aware coverage, including 93% for the authoritative transition and 98% for the ledger;
- installed-CLI ledger verification and exact deterministic replay;
- execution and activation evidence for every enabled named mechanism across the canonical micro-suite.

The short baseline calibration produced reconstruction MSE `0.32712` for HRM at measured cost `7457`, versus `0.32834` at cost `2880` for the reaction–diffusion control. That small error difference at substantially higher reported cost is not evidence of superiority. In the three-seed ablations, no Holm-adjusted result was significant; no-topology-repair and no-adaptive-timestep did not diverge on this workload. See [release evidence](docs/release_evidence.md) and the machine-readable files under `artifacts/reference_results`.

## Repository map

- `src/hrm/state`: seven typed state blocks, validation, digest, and serialization.
- `src/hrm/core`: snapshots, proposals, transition, events, transport, safety, and ledger.
- `src/hrm/mechanisms`: attributable HRM–RCM mechanisms.
- `src/hrm/tasks`, `baselines`, `experiments`: matched scientific evaluation.
- `configs`: validated reference configurations and interventions.
- `tests`: unit, structural, ledger, ablation, and replay checks.
- `artifacts/reference_results`: compact, retained calibration reports; bulky per-run bundles are regenerated on demand.

## Extending HRM

New mechanisms must declare read/write ownership, produce side-effect-free proposals, expose configuration, have activation and ablation tests, and preserve the ledger identity. Accelerated backends may replace kernels but not transition semantics.

## Bifurcating transition engine

Use the bifurcating engine in place of the ordinary transition engine:

```python
from hrm.core.bifurcation import (
	BifurcatingTransitionEngine,
	BifurcationPolicy,
)

engine = BifurcatingTransitionEngine(
	config,
	BifurcationPolicy(
		branches=4,
		horizon=2,
		trigger_threshold=0.55,
		baseline_margin=0.0,
	),
)

result = engine.step(state, external_input, run_id="experiment")
state = result.state
```

The returned result still exposes:

- `result.state`
- `result.ledger`

And adds:

- `result.bifurcated`
- `result.selected_branch`
- `result.insufficiency`
- `result.baseline_objective`
- `result.branches`

Branch decisions and objective scores are also written into:

- `state.run_metadata["bifurcation"]`

Compatibility notes: bifurcation is only attempted when the step insufficiency reaches the configured `trigger_threshold`; otherwise the engine follows the ordinary single-path transition for that step (`result.bifurcated == False`). In the non-bifurcated case, `result.selected_branch` is `None` to indicate that no branch was chosen.

Margin notes: `baseline_margin` controls how much better a branch must score before replacing the ordinary baseline step. `improvement_margin` remains accepted as a deprecated alias in `BifurcationPolicy` for backwards compatibility.

## License

The public source is available under the
[PolyForm Noncommercial License 1.0.0](LICENSE) for the permitted noncommercial purposes defined
in that license. This is a source-available project, not OSI open source. The required provenance
and attribution notices are in [`NOTICE`](NOTICE).

Commercial use requires a separate written license. See
[`LICENSING.md`](LICENSING.md) for the licensing model and inquiry process. Repository provenance
and the recorded pre-publication ownership attestation are documented in
[`PROVENANCE.md`](PROVENANCE.md).
