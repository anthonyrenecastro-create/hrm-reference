# Contributing

Changes must preserve the authoritative transition and claims boundary.

1. Add or update a typed configuration field.
2. Declare mechanism read and write sets.
3. Keep proposals side-effect free and common-snapshot compatible.
4. Add activation, ablation, reconstruction, and determinism tests.
5. Report negative or null outcomes without weakening baselines.
6. Run `make check` before proposing a change.

Architectural changes should include a short decision record in `docs/`.

## Contribution licensing

Unless a separate written agreement applies, accepted contributions are distributed under the
repository's PolyForm Noncommercial License 1.0.0. Contributors retain their own copyrights.
Commercial licenses offered by the project owner cover only rights the project owner controls, so
a separate contributor agreement may be required before incorporating a contribution into a
commercially licensed distribution.
