# Architecture

HRM is implemented as a deterministic hybrid state-space system. Seven typed state blocks are
committed together by one `TransitionEngine`; ordinary mechanisms can only propose additive
changes from the same frozen snapshot. Discrete structure changes pass through an immutable event
pipeline, and safety changes are recorded as corrections.

## Ownership

| Block | Owns | Does not own |
|---|---|---|
| Field | values, active node/channel masks, field diagnostics | graph support |
| Geometry | coordinates, metric, anchors, stress/curvature diagnostics | accepted edges |
| Topology | sparse edge support, confidence weights, ages, versions | coordinates |
| Memory | working state, key/value slots, strength, age, provenance | task target |
| Cognition | latent state, predictions, residual, uncertainty, guidance precursor | supervision policy |
| Hierarchy | region assignment, restriction/prolongation, coarse state, routing | graph mutation |
| Resources | budgets, allocations, widths, operation and switching cost | mechanism outcomes |

`effective_weights(T, G)` is derived from topology confidence and geometric distance. This prevents
geometry and topology from silently overwriting one another. Operator versions are invalidated on
accepted topology changes and, when weights are geometry-adaptive, geometry changes.

## Data flow

1. Validate and freeze the authoritative state.
2. Compute shared observables once.
3. Collect independent main proposals.
4. Compute declared interaction proposals.
5. Aggregate continuous and seeded stochastic contributions.
6. Validate, sort, and budget structural events.
7. Transport contributions on the fixed carrier and apply accepted events.
8. Apply separately attributed safety corrections.
9. Verify the numerical reconstruction identity.
10. Refresh derived diagnostics and commit once.

The fixed maximum carrier plus active masks is a conservative reference choice. Activation and
deactivation therefore use exact identity transport; a variable-carrier backend must implement a
different transport record without changing the ledger contract.

## Determinism boundary

State, resolved configuration, serialized input schedule, perturbation identity, seed, step, and
named random stream determine a transition. Wall-clock runtime and platform metadata are recorded
but excluded from trajectory identity. Floating-point bit identity is guaranteed only within a
platform-compatible numeric mode.

See [architecture decisions](architecture_decisions.md) for choices left unspecified by the theory.
