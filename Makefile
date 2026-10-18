.PHONY: check lint typecheck test format bench gate drag leaks m1 m2 pilot sweep replay train-smoke web serve web-dev

PY ?= .venv/Scripts/python
ARGS ?=
RUN ?=
SMOKE_ROOT ?= runs/smoke
NPM ?= npm
PORT ?= 8600

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

# The dashboard: `make web` builds the static site into web/out, `make serve` serves
# it beside the JSON API (needs the dashboard extra), `make web-dev` runs the site with
# hot reload against a `make serve` already running on $(PORT).
web:
	cd web && $(NPM) ci && $(NPM) run build && $(NPM) run typecheck

serve:
	$(PY) -m optspread.server --port $(PORT)

web-dev:
	cd web && OPTSPREAD_API=http://127.0.0.1:$(PORT) $(NPM) run dev
