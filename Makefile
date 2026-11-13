.PHONY: check lint typecheck test format bench gate drag leaks m1 m2 m3gate reproduce vrp m4 m4gate edge showcase run pilot sweep replay train-smoke web serve web-dev

# The venv's interpreter: Scripts/ on Windows, bin/ everywhere else.
PY ?= $(if $(wildcard .venv/Scripts/python .venv/Scripts/python.exe),.venv/Scripts/python,.venv/bin/python)
# What runs scripts/reproduce.py before any venv exists: python on Windows; elsewhere the
# first of python3, python3.14, python3.13, python3.12 on the path that is at least the
# project's requires-python (MIN_PYTHON), since python3 can be an older system Python.
MIN_PYTHON := 3.12
BOOTSTRAP_PY ?= $(if $(filter Windows_NT,$(OS)),python,$(or $(shell for p in python3 python3.14 python3.13 python3.12; do $$p -c 'import sys; sys.exit(sys.version_info < tuple(map(int, "$(MIN_PYTHON)".split("."))))' >/dev/null 2>&1 && { echo $$p; break; }; done),python3))
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

# Parallel runs, beside `test`. optspread/reproduction.py's STEP_COMMANDS must match what
# `make -n test` prints, so `test` stays one serial command and these are targets of their
# own. Each of the TEST_WORKERS xdist workers imports the whole suite.
TEST_WORKERS ?= 4
XDIST = -n $(TEST_WORKERS) --dist loadfile --no-loadscope-reorder
.PHONY: test-par test-fast

# Everything, with the coverage floor on the combined result, in two phases. First the
# suite on $(TEST_WORKERS) workers, one file to a worker at a time (--dist loadfile, so a
# file's module fixtures are built once), without the tests that start processes of their
# own; then those (spawns_processes) in one process, appending to the same coverage data.
# tests/conftest.py keeps them out of any run under -n, and sends the slow files first.
test-par:
	$(PY) -m pytest tests/ $(XDIST) -m "not spawns_processes" --cov=optspread --cov-report=
	$(PY) -m pytest tests/ -m spawns_processes --cov=optspread --cov-append --cov-report=term-missing --cov-fail-under=85

# The quick loop: parallel, without the slow and the process-spawning tests, no coverage.
test-fast:
	$(PY) -m pytest tests/ $(XDIST) -m "not slow and not spawns_processes"

# One BLAS thread per test process: the parallelism is the workers', and every idle
# OpenBLAS thread pool commits memory a test run never uses.
test-par test-fast: export OPENBLAS_NUM_THREADS = 1
test-par test-fast: export OMP_NUM_THREADS = 1
test-par test-fast: export MKL_NUM_THREADS = 1

# As modules, so the scripts import this checkout's optspread, not whichever one an
# editable install in a shared venv points at.
bench:
	$(PY) -m scripts.bench

gate:
	$(PY) -m scripts.wave0_gate

drag:
	$(PY) -m scripts.cost_drag

leaks:
	$(PY) -m scripts.leak_report

m1:
	$(PY) -m scripts.m1_gate_report

m2:
	$(PY) -m scripts.m2_gate_report $(ARGS)

# docs/M3_GATE_REPORT.md and .json: the sweep, the leak floor, the demo and the partner's
# record, e.g. make m3gate ARGS="--runs-root $$VEGALAB_DATA/runs --replay 2"
m3gate:
	$(PY) -m scripts.m3_gate_report $(ARGS)

# A partner team's one command (#10): venv, install, every gate, sweep cells regenerated,
# docs/M3_PARTNER_REPRODUCTION.json written, e.g. make reproduce ARGS="--with-ppo"
reproduce:
	$(BOOTSTRAP_PY) scripts/reproduce.py $(ARGS)

# docs/VRP_MAPPING.md and .json from $$VEGALAB_DATA/panel/spx_daily.parquet, e.g.
#   make vrp ARGS="--daily /scratch/spx_daily.parquet"   (a local copy of the same file)
vrp:
	$(PY) -m scripts.vrp_mapping $(ARGS)
# The M4 analysis of the confirmatory sweep, e.g. make m4 ARGS="--root $$VEGALAB_DATA/runs"
m4:
	$(PY) -m scripts.m4_analysis $(ARGS)
# docs/M4_GATE_REPORT.md and .json from docs/M4_ANALYSIS.json, the VRP mapping, the sweep's
# grid record and attribution, and the built site, e.g. make m4gate ARGS="--runs-root $$VEGALAB_DATA/runs"
m4gate:
	$(PY) -m scripts.m4_gate_report $(ARGS)
# Which structures a policy trades, re-simulated from runs or a sweep's grid, written under
# actionmix/, e.g. make actionmix ARGS="--grid $$VEGALAB_DATA/runs --arms delta_seller --workers 8"
# Phony on its own line: the target shares its name with the folder it writes.
.PHONY: actionmix
actionmix:
	$(PY) -m scripts.action_mix $(ARGS)

# Which components and structures carry the detected edge (A1/A2 per component), written under
# attribution/edge/, e.g. make edge ARGS="--grid $$VEGALAB_DATA/runs --first 50 --workers 2"
edge:
	$(PY) -m scripts.edge_attribution $(ARGS)

# The showcase runs every view of the dashboard reads, and their catalog, e.g.
#   make showcase ARGS="--root runs/showcase"   (--dry-run to list them)
showcase:
	$(PY) -m scripts.showcase $(ARGS)

# One run, written as a run directory. Flags go in ARGS, e.g.
#   make run ARGS="--policy delta_seller --episodes 200 --source optionmetrics"
run:
	$(PY) -m scripts.run $(ARGS)

# The M2 runs. Flags go in ARGS (add --source optionmetrics for the real panel), e.g.
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
#
# Before the build, `make web` writes the server's OpenAPI document (needs the dashboard
# extra) and regenerates the site's API types from it into web/lib/schema.d.ts, so the
# build and the typecheck fail when a page no longer fits the API. The file is committed:
# commit it again whenever this changes it.
web:
	cd web && $(NPM) ci
	$(PY) -m scripts.write_openapi web/node_modules/.cache/optspread/openapi.json
	cd web && $(NPM) run api-types && $(NPM) run build && $(NPM) run typecheck

serve:
	$(PY) -m optspread.server --port $(PORT)

web-dev:
	cd web && OPTSPREAD_API=http://127.0.0.1:$(PORT) $(NPM) run dev
