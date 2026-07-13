.PHONY: install test coverage lint typecheck check smoke canonical

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

coverage:
	python -m pytest --cov=hrm --cov-report=term-missing

lint:
	python -m ruff check src tests examples benchmarks
	python -m ruff format --check src tests examples benchmarks

typecheck:
	python -m mypy src

check: lint typecheck test

smoke:
	hrm run --config configs/minimal.yaml --output artifacts/smoke
	hrm ledger verify artifacts/smoke
	hrm replay artifacts/smoke

canonical:
	hrm ablate --config configs/full_hrm.yaml --suite canonical --seeds 10
