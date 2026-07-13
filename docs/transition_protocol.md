# Transition protocol

`hrm.core.transition.TransitionEngine.step` is the only production transition. Tasks, CLI runs,
experiments, examples, benchmarks, replay, and tests call it.

| Phase | Contract | Failure behavior |
|---|---|---|
| Validate | state and config agree | reject before work |
| Freeze | recursive arrays and sparse storage become read-only | hidden mutation raises |
| Observe | one shared observable bundle | no mechanism-local recomputation with privileged targets |
| Propose | mechanisms declare read/write blocks, costs, invariants, and event candidates | invalid shape, nonfinite value, undeclared write, or zero-activation contribution raises |
| Interact | explicit interaction terms consume named proposals | separately attributed |
| Aggregate | deterministic addition plus named seeded streams | unrepresentable aggregate raises |
| Events | validate, sort, enforce degree/connectivity/age/cooldown/churn/budget | rejection is recorded |
| Transport | identity on the fixed maximum carrier | metadata records carrier and event IDs |
| Safety | rollback, projection, repair, gain and budget correction | numerical effect is separate |
| Ledger | reconstruct observed numerical delta | transition fails above tolerance |
| Commit | refresh diagnostics, versions, metadata, and step once | returns new state; input remains unchanged |

Multi-rate execution uses a deterministic per-mechanism cadence. A skipped cadence emits a
zero-contribution proposal with a reason. Evaluation order cannot change the numerical result because
all main proposals use the common snapshot and event ordering is canonical.

Experimental perturbations occur between committed steps. They are named in the schedule and ledger
input identity and are not retroactively credited to a mechanism.
