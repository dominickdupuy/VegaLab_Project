.PHONY: check lint typecheck test format bench gate

PY ?= .venv/Scripts/python

check: lint typecheck test

lint:
	$(PY) -m ruff check optspread tests scripts
	$(PY) -m ruff format --check optspread tests scripts

format:
	$(PY) -m ruff format optspread tests scripts
	$(PY) -m ruff check --fix optspread tests scripts

typecheck:
	$(PY) -m mypy --strict optspread scripts

test:
	$(PY) -m pytest tests/ --cov=optspread --cov-report=term-missing --cov-fail-under=85

bench:
	$(PY) scripts/bench.py

gate:
	$(PY) scripts/wave0_gate.py
