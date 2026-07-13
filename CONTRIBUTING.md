# Contributing

Changes must preserve the authoritative transition and claims boundary.

1. Add or update a typed configuration field.
2. Declare mechanism read and write sets.
3. Keep proposals side-effect free and common-snapshot compatible.
4. Add activation, ablation, reconstruction, and determinism tests.
5. Report negative or null outcomes without weakening baselines.
6. Run `make check` before proposing a change.

Architectural changes should include a short decision record in `docs/`.

## Continuous integration

Every pull request runs three required workflow groups:

- static analysis with Ruff formatting, Ruff lint, and mypy;
- the complete test suite with at least 80% coverage on Python 3.11 and 3.12;
- an installed-CLI smoke run with attribution-ledger verification and exact replay.

Run the same checks locally before opening a pull request:

```bash
make check
python -m pytest --cov=hrm --cov-report=term-missing --cov-fail-under=80
make smoke
```

Maintainers can also run each workflow manually from the GitHub Actions page. Ledger or replay
failure is a release blocker even when every other test passes. Reproducibility evidence is retained
as a workflow artifact for 14 days to support diagnosis.
