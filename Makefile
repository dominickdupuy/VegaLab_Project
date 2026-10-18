.PHONY: check lint typecheck test format bench gate drag leaks m1 m2 pilot sweep replay train-smoke

PY ?= .venv/Scripts/python
ARGS ?=
RUN ?=
SMOKE_ROOT ?= runs/smoke

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

drag:
	$(PY) scripts/cost_drag.py

leaks:
	$(PY) -m scripts.leak_report

m1:
	$(PY) -m scripts.m1_gate_report

m2:
	$(PY) -m scripts.m2_gate_report

# The M2 runs. Flags go in ARGS, e.g.
#   make pilot ARGS="--root $$VEGALAB_DATA/runs/pilot --seeds 8 --episodes 200 --timesteps 1000000 --workers 16"
pilot:
	$(PY) -m scripts.pilot $(ARGS)

sweep:
	$(PY) -m scripts.sweep $(ARGS)

# make replay RUN=runs/pilot/<run id>   (or a folder of runs; add ARGS="--max 4")
replay:
	$(PY) -m scripts.replay $(RUN) $(ARGS)

# Two seeds x three episodes x one PPO rollout on two workers, then every run replayed.
train-smoke:
	$(PY) -m scripts.pilot --root $(SMOKE_ROOT) --seeds 2 --episodes 3 --timesteps 2048 --workers 2
	$(PY) -m scripts.replay $(SMOKE_ROOT)
