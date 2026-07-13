# Extension guide

To add a mechanism:

1. Add strict typed configuration with conservative defaults.
2. Implement the common proposal interface and declare read/write blocks.
3. Read only the frozen snapshot and shared observables.
4. Return additive arrays on existing numerical carrier paths; use typed events for structure.
5. Report activation, cost, invariants, and diagnostic basis without hidden target access.
6. Register the mechanism once in `build_default_mechanisms` and expose an enabled flag.
7. Add unit tests for zero activation, shape, finiteness, ownership, and immutability.
8. Add a matched ablation that demonstrates execution, activation, and trajectory divergence.
9. Run ledger verification and replay before interpreting outcome utility.

New event types need deterministic IDs/order, basis validation, rejection reasons, transport
semantics, application effects, cooldown/churn interaction, serialization, and adversarial tests.

Alternative numerical backends may replace operator kernels behind the existing interfaces. They
must preserve active-mask work avoidance, state schema, event protocol, ledger reconstruction, and
deterministic-stream identity. Backend-specific tolerances and platform compatibility must be
declared rather than silently weakening validation.
