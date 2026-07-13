# Release evidence — 0.1.0

Date: 2026-07-13. Numeric mode: CPU float64. This page records release calibration, not a claim of
external support or broad validation.

## Engineering gates

| Gate | Result |
|---|---|
| Clean editable install | passed in a newly created Python 3.12 environment |
| Tests | 55 passed |
| Lint/format | Ruff passed |
| Static typing | mypy: no issues in 94 source files |
| Total branch-aware coverage | 83% |
| Authoritative transition coverage | 93% |
| Ledger coverage | 98% |
| Installed CLI smoke | 6 steps, 12 accepted events, no corrections |
| Independent ledger verification | 6/6 records; max relative residual `1.8875671021266e-19` |
| Replay | exact final state digest match |

## Canonical micro-suite

All seven tasks ran for seeds 0, 1, and 2: 21 complete synthetic runs. Every run passed its ledger;
the worst relative residual was `3.3649457252762495e-19`. The suite demonstrated execution and at
least one nonzero activation for every enabled named mechanism. Full per-run summaries are retained
in `artifacts/reference_results/canonical-micro/index.json`.

## Matched ablations

The spatial-reconstruction calibration ran 14 canonical interventions across three seeds, producing
42 matched comparisons. Every full and ablated ledger passed. Numerical-carrier digests—not complete
digests containing configuration metadata—define trajectory divergence.

- `no_topology_repair` and `no_adaptive_timestep` produced no numerical divergence; utility is
  therefore uninterpretable on this workload.
- Several structural/capacity interventions failed the 5% cost-equality gate.
- All Holm-adjusted p-values were 1.0 with only three seeds.
- The full system was slightly worse than `no_guidance` in mean reconstruction error in this short
  run (`-5.41e-05` utility effect under the report sign convention).
- No synthetic mechanism-support claim is made from this calibration.

Machine-readable configuration differences, paired rows, statistics, and the Markdown report are
under `artifacts/reference_results/canonical-ablations`.

## Baseline calibration

| Condition | Reconstruction MSE | Reported operation cost |
|---|---:|---:|
| Persistence | 0.341432 | 48 |
| Graph diffusion | 0.342747 | 1,728 |
| Reaction–diffusion | 0.328336 | 2,880 |
| HRM | 0.327120 | 7,457 |

This is one seed and the operation models are not identical. HRM's small raw error advantage over
reaction–diffusion came at substantially higher reported cost, so the result does not establish
superiority.

## Reproduce

```bash
python -m pip install -e '.[dev]'
make check
make coverage
python benchmarks/run_canonical.py --config configs/minimal.yaml --seeds 3
python benchmarks/compare_baselines.py --config configs/minimal.yaml
hrm ablate --config configs/minimal.yaml --suite canonical --seeds 3
```

Priorities for a subsequent study are preregistered task-specific outcome gates, at least 30 matched
seeds, cost-equalized structural controls, independent workloads, backend/platform replication, and
external review of the mathematical-to-computational mapping.
