# Contributing

Changes must preserve the authoritative transition and claims boundary.

1. Add or update a typed configuration field.
2. Declare mechanism read and write sets.
3. Keep proposals side-effect free and common-snapshot compatible.
4. Add activation, ablation, reconstruction, and determinism tests.
5. Report negative or null outcomes without weakening baselines.
6. Run `make check` before proposing a change.

Architectural changes should include a short decision record in `docs/`.

## Temporary code-contribution policy

Outside code contributions will **not be merged** until a contributor license agreement is
operational. The agreement must grant the project owner sufficient rights to distribute accepted
contributions under both the public PolyForm Noncommercial license and separate commercial terms.

Until that process is published, community members are welcome to open issues, report reproduction
results, discuss designs, and propose experiments, but maintainers must not merge third-party code.
This temporary restriction prevents contributor-owned code from creating uncertainty in the
commercially licensed edition.
