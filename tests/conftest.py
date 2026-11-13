"""Hooks and fixtures every test folder shares, and nothing heavier than they need.

The tests are grouped by area of optspread under ``tests/<area>/``. The folders are
not packages (no ``__init__.py``): pytest imports a test module by its bare name, so
file basenames must stay unique across folders. The helpers stay here at ``tests/``
(``fake_learner``, ``dashboard_fixtures``, ``dying_jobs``, ``m3_fixtures``) and are
imported bare from every folder: pytest puts this file's folder on ``sys.path`` when
it loads it, in an xdist worker too, and a spawned worker inherits that path.

Nothing heavy is imported at module level, so collecting ``tests/core`` loads neither
pandas, pyarrow nor the sweep. The fixtures the dashboard and API tests share (tiny
run roots, gate docs and an SPX panel) import what they need when a test first asks
for one. The run root is written by the real sweep, so the browser is tested on the
same files a sweep leaves behind. The panel builders live in ``dashboard_fixtures``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from optspread.experiment.sweep import SweepConfig


# -- under xdist: process-spawning tests left out, the heaviest files sent first ---- #

LEFT_OUT = "spawns_processes_left_out"
_left_out = pytest.StashKey[int]()


@pytest.hookimpl(trylast=True)  # after -m and -k have deselected what they will
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """In an xdist worker: leave out the spawns_processes tests, put slow files first.

    A test that starts processes of its own inside an xdist worker multiplies the
    processes on the machine and competes with the other workers for the cores. So
    under ``-n`` those tests are left out, and ``make test-par`` runs them afterwards
    in one process (``-m spawns_processes``, no ``-n``). A plain ``pytest`` runs them all.

    xdist hands out whole files in the order the workers collected them (with
    ``--no-loadscope-reorder``; by default it sends the files with the most tests
    first). A long file sent last keeps one worker busy after the others are done,
    so the files holding slow tests go first, fewest tests first: there each test is
    likely the longest. Every other file, and the order inside each file, is kept.
    Runs without xdist keep the collection order untouched.
    """
    if not hasattr(config, "workerinput"):  # the main process, or no xdist at all
        return
    spawning = [item for item in items if item.get_closest_marker("spawns_processes")]
    config.workeroutput[LEFT_OUT] = len(spawning)  # type: ignore[attr-defined]
    if spawning:
        config.hook.pytest_deselected(items=spawning)
        keep = {id(item) for item in items} - {id(item) for item in spawning}
        items[:] = [item for item in items if id(item) in keep]

    def file_of(item: pytest.Item) -> str:
        return item.nodeid.split("::", 1)[0]

    sizes = Counter(file_of(item) for item in items)
    slow = {file_of(item) for item in items if item.get_closest_marker("slow")}
    items.sort(key=lambda item: (0, sizes[file_of(item)]) if file_of(item) in slow else (1, 0))


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    """xdist, in the main process: each worker reports how many it left out."""
    left_out = int(getattr(node, "workeroutput", {}).get(LEFT_OUT, 0))
    node.config.stash[_left_out] = max(node.config.stash.get(_left_out, 0), left_out)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    left_out = config.stash.get(_left_out, 0)
    if left_out:
        terminalreporter.write_line(
            f"{left_out} spawns_processes test(s) left out under -n: run them with "
            "`pytest -m spawns_processes` and no -n, as `make test-par` does"
        )


# -- the dashboard and API tests: tiny run roots, gate docs and an SPX panel -------- #


@pytest.fixture(scope="session")
def tiny_sweep() -> SweepConfig:
    """Two premiums by two arms on a three-week world: the sweep in every run root."""
    from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
    from optspread.experiment.sweep import SweepConfig

    tiny = RunConfig(
        seed=7,
        n_episodes=3,
        market=MarketConfig(n_days=21, tenor_days=(7, 14)),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
    )
    return SweepConfig(base=tiny, premiums=(0.0, 2.0), arms=("flat", "delta_seller"))


@pytest.fixture(scope="session")
def run_root(tmp_path_factory: pytest.TempPathFactory, tiny_sweep: SweepConfig) -> Path:
    """A runs root with one real sweep (four cells) plus a broken and a tampered run."""
    from optspread.experiment.sweep import run_sweep

    root = tmp_path_factory.mktemp("runs")
    run_sweep(tiny_sweep, root / "sweeps" / "tiny")
    # A manifest with no git SHA: refused by the gate, so listed as INVALID.
    broken = root / "adhoc" / "broken-run"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(json.dumps({"run_id": "broken-run", "config": {}}))
    # A real run whose results file was edited after its manifest was written.
    cells = sorted((root / "sweeps" / "tiny").glob("*-s7"))
    tampered = root / "adhoc" / cells[0].name
    tampered.mkdir(parents=True)
    for name in ("manifest.json", "run.json"):
        (tampered / name).write_bytes((cells[0] / name).read_bytes())
    record = json.loads((tampered / "run.json").read_text())
    record["mean_pnl"] = 1e9
    (tampered / "run.json").write_text(json.dumps(record))
    return root


def gate_json(milestone: str, items: list[dict[str, str]], verdict: str) -> str:
    return json.dumps(
        {"milestone": milestone, "report_date": "2026-10-26", "verdict": verdict, "items": items}
    )


@pytest.fixture()
def gate_docs(tmp_path: Path) -> Path:
    """M1 as markdown only (the repository's own file) and M2 as a JSON sidecar."""
    docs = tmp_path / "docs"
    docs.mkdir()
    repo_docs = Path(__file__).resolve().parent.parent / "docs"
    (docs / "M1_GATE_REPORT.md").write_text((repo_docs / "M1_GATE_REPORT.md").read_text())
    items = [
        {
            "name": "Analysis pipeline validated",
            "owner": "Dominick Dupuy",
            "verdict": "PASS",
            "criterion": "recovers the closed-form detection probability within 2 se",
            "evidence": "200 synthetic replicates; run 0123456789abcdef-s3",
            "failed_first": "",
        },
        {
            "name": "Replay",
            "owner": "Ilan Danial",
            "verdict": "NOT CLEARED",
            "criterion": "a replay reproduces every artifact digest",
            "evidence": "",
            "failed_first": "no replay has been run",
        },
    ]
    (docs / "M2_GATE_REPORT.json").write_text(gate_json("M2", items, "NOT CLEARED"))
    (docs / "M2_GATE_REPORT.md").write_text("# M2 gate report\n")  # the JSON wins
    return docs


@pytest.fixture(scope="session")
def panel_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture SPX panel, written as the build writes it (it skips without pyarrow)."""
    from dashboard_fixtures import write_panel

    return write_panel(tmp_path_factory.mktemp("vegalab"))


@pytest.fixture(autouse=True)
def _no_shared_data_folder(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No test reads the team's shared data folder: VEGALAB_DATA points at an empty dir.

    A test that wants data builds it under its own tmp_path and sets VEGALAB_DATA
    itself; the default (the synced OneDrive folder) is never reached from the suite.
    """
    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path_factory.mktemp("no_shared_data")))


# The M4 tests' grids are trained by ``tests/fake_learner.py``. ``optspread.m4analysis`` reads
# a grid with the rule every reader shares (``sweep.valid_run``), which holds a learned cell
# to the real learner unless told otherwise; in those modules it is told the stand-in.
STAND_IN_MODULES = ("test_m4analysis", "test_m4gate", "test_webviews")


@pytest.fixture(scope="module", autouse=True)
def _the_m4_grids_stand_in_learner(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.module.__name__ not in STAND_IN_MODULES:
        yield
        return
    from fake_learner import fake_trainer

    from optspread import m4analysis
    from optspread.manifest import callable_name
    from optspread.runs.rundir import valid_run

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m4analysis, "valid_run", partial(valid_run, trainer=callable_name(fake_trainer)))
        yield
