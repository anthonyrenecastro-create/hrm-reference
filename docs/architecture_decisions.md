# Reference implementation decisions

These are conservative implementation choices, not consequences proven by HRM theory.

1. **Fixed maximum carrier.** Node and channel capacities are fixed arrays with active masks. Activation and deactivation are structural events, but transport remains an exact identity on the carrier. This avoids falsely additive nonlinear transport in the first reference system.
2. **Canonical graph ownership.** Topology owns sparse edge support and edge confidence. Geometry owns coordinates and metric variables. Effective weights and Laplacians are versioned derived operators computed from both; they are not hidden mutable state.
3. **Explicit Euler reference solver.** Continuous proposals are increments computed with a declared effective timestep. A spectral bound can reduce the timestep before proposal evaluation. Alternate solvers must preserve proposal and ledger contracts.
4. **Diagnostic caches are derived.** Field energy/coherence, graph degrees/components, and similar cached diagnostics are recomputed deterministically and cannot influence results without fresh observation.
5. **Structural safety is explicit.** Invalid candidate events are rejected. Optional graph repair is recorded as a correction; it is disabled by fixed-topology/no-repair interventions.
6. **Perturbation stage.** Experiments apply named perturbations between committed steps inside a declared recoverable envelope and record them separately in the schedule. Production transitions still start from a state that passes structural and finite validation.
7. **Supervision is typed.** External observations and task targets are separate. A task must explicitly authorize target-dependent guidance; the ledger records that access.
8. **Multirate updates use schedules.** Every mechanism has a deterministic cadence. A skipped mechanism executes with zero activation and a recorded reason.
9. **Runtime is observational.** Runtime is recorded but excluded from deterministic state/ledger identity comparisons.
