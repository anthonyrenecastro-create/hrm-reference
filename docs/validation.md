# Validation

The repository uses four complementary checks:

1. Unit and property tests validate state, operators, proposal ownership, finite values, masks, and
   serialization.
2. Integration tests attack the transition, event constraints, transport, safety rollback, ledger
   reconstruction, hidden mutation, and ablation behavior.
3. Replay regenerates a task from its resolved configuration and serialized schedule identities,
   then compares every pre/post digest, event ID, and ledger-observed delta.
4. Experiment gates measure execution, activation, divergence, outcome, and cost separately.

Local release checks:

```bash
make check
make coverage
make smoke
```

CI separates tests, lint/type checking, and reproducibility. The reference numerical mode is CPU
float64. Exact cross-platform bit identity is not promised across different BLAS or sparse-library
implementations; the environment record captures those boundaries.

Known validation gaps are maintained in [claims boundary](claims_boundary.md), not hidden in a
single composite score.
