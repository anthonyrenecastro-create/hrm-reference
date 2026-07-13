# Attribution ledger

Every timestep writes one JSON Lines record under schema `1.0`. A record contains run and generator
identity, configuration and state digests, block versions, solver/timestep information, main and
interaction proposals, intentional stochastic contributions, event lifecycle records, transport,
corrections, raw metrics, cost, runtime, reconstructed and observed arrays, and residual norms.

Contribution arrays are stored by default. Setting `ledger.store_arrays: false` reduces output size
but intentionally makes independent numerical verification impossible for nonzero proposals. Such a
run remains useful for diagnostics but does not satisfy the strongest attribution gate.

Verify a run with:

```bash
hrm ledger verify artifacts/run-id
```

The verifier reads only the ledger file, sums stored main, interaction, stochastic, event, and
correction arrays, and compares the sum with the observed delta. It rejects missing arrays when a
nonzero norm is claimed and rejects deliberate residual injection.

Rollbacks invert every already-attributed finite contribution. Projection, structural repair,
resource correction, and gain reduction are never credited to the mechanism that happened to
precede them. Derived cache refreshes are excluded from the numerical carrier and identified as
deterministic commit diagnostics.
