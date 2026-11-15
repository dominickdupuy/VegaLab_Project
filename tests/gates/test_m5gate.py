"""Hemdutt Rao: the M5 gate items (#18).

Every item is tested both ways: with its deliverable absent or wrong it must read NOT
CLEARED and say what failed first, and with evidence that meets its criterion it must
pass. The dashboard server is stood in for by a small FastAPI app registered under the
name the gate imports, as the M3 and M4 tests do, and the real one is built once to
check the views match its routes. The grid record is written by hand for a tiny grid;
the usability file is the committed template with its tables filled in; the cited
commits are in a throwaway repository. Nothing here reads the shared folder.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pytest

from optspread import data as data_layer
from optspread import m5gate
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.config import config_hash as hash_of
from optspread.dashboard_data import REPO_ROOT
from optspread.experiment.grid import GridCell, GridConfig, GridRecord, grid_cells, grid_filename
from optspread.gatereport import GateItem, GateReport

TINY = GridConfig(
    base=RunConfig(
        seed=7000,
        n_episodes=3,
        market=MarketConfig(n_days=21, tenor_days=(10, 21)),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
    ),
    premiums=(0.0, 2.0),
    arms=("delta_seller",),
    n_seeds=2,
)
RECORD = (REPO_ROOT / "docs" / m5gate.USABILITY_FILE).read_text(encoding="utf-8")
# The kit as it was before any session: the committed record with its rows emptied.
TEMPLATE = re.sub(r"(?m)^\| (\d) \| 2026-[^\n]*$", r"| \1 | | | | | |", RECORD)
TEMPLATE = re.sub(r"(?m)^\| \d+ \| [\d, ]+ \| [^\n]*\n", "", TEMPLATE)


def blocked(item: GateItem, contains: str) -> None:
    assert not item.passed and item.verdict == "FAIL"
    assert contains in item.failed_first, item.failed_first
    assert item.has_criterion


def passes(item: GateItem) -> None:
    assert item.passed and item.verdict == "PASS", item.failed_first
    assert item.evidence


# -- 1. the dashboard's views ------------------------------------------------------- #


def write_grid(root: Path, grid: GridConfig = TINY) -> GridRecord:
    """A grid record of ``grid`` under ``root``, its cells' headlines made up."""
    cells = tuple(
        GridCell(
            premium=c.premium,
            arm=c.arm,
            block=c.block,
            run_id=c.run_id,
            config_hash=c.run_id.split("-")[0],
            seed=c.config.seed,
            n_episodes=c.config.n_episodes,
            train_seed=None,
            mean_pnl=1.0,
            se_pnl=1.0,
            mean_cost=1.0,
            mean_trades=4.0,
        )
        for c in grid_cells(grid)
    )
    record = GridRecord(grid_hash=hash_of(grid), grid=grid, cells=cells)
    root.mkdir(parents=True, exist_ok=True)
    (root / grid_filename(grid)).write_text(record.model_dump_json())
    return record


def answers(record: GridRecord) -> dict[str, Any]:
    """What a working server answers on the sweep, the M4 analysis and the edge."""
    cells = [{"run_id": c.run_id, "present": True} for c in record.cells]
    return {
        m5gate.SWEEP_ROUTE: {"grid": {"cells": cells}},
        m5gate.M4_ROUTE: {
            "available": True,
            "file": "M4_ANALYSIS.json",
            "analysis": {"report_date": "2026-11-14", "arm_order": ["delta_seller"]},
        },
        m5gate.EDGE_ROUTE: {
            "edge": {
                "grid_hash": record.grid_hash,
                "arms": ["delta_seller"],
                "cells": cells,
                "reproduction": "reproduces",
            }
        },
    }


ROUTES = (
    "/api/health",
    "/api/overview",
    "/api/runs",
    "/api/launch/options",
    "/api/power",
    "/api/data/status",
    "/api/leaks/demo",
    "/api/sweeps",
    m5gate.SWEEP_ROUTE,
    m5gate.M4_ROUTE,
    "/api/edge",
    m5gate.EDGE_ROUTE,
)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Registers a stand-in ``optspread.server`` whose app answers ``bodies`` on ``routes``."""
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    def install(
        bodies: dict[str, Any] | None = None,
        *,
        routes: tuple[str, ...] = ROUTES,
        status: dict[str, int] | None = None,
        raises: Exception | None = None,
        factory: bool = True,
    ) -> None:
        module = types.ModuleType(m5gate.SERVER_MODULE)

        @dataclass
        class Settings:
            runs: Path
            docs: Path
            data: Path
            web: Path
            readme: Path
            attribution: Path | None = None
            actionmix: Path | None = None

        def create_app(settings: Settings) -> Any:
            if raises is not None:
                raise raises
            app = fastapi.FastAPI()
            for path in routes:

                def answer(path: str = path) -> Any:
                    code = (status or {}).get(path, 200)
                    if code != 200:
                        raise fastapi.HTTPException(status_code=code, detail="no such thing")
                    return (bodies or {}).get(path, {"ok": True})

                app.get(path)(answer)
            return app

        setattr(module, "Settings", Settings)  # noqa: B010
        if factory:
            setattr(module, "create_app", create_app)  # noqa: B010
        monkeypatch.setitem(sys.modules, m5gate.SERVER_MODULE, module)

    yield install


def built_site(web: Path, skip: tuple[str, ...] = ()) -> Path:
    """A static export with every view's page but those in ``skip``."""
    for _, page, _ in m5gate.VIEWS:
        if page not in skip:
            (web / page).parent.mkdir(parents=True, exist_ok=True)
            (web / page).write_text("<html></html>")
    return web


def item_on(root: Path, web: Path) -> GateItem:
    return m5gate.dashboard_item(root, root, web, root / "attribution", root / "mix", TINY)


@pytest.mark.behaviour
def test_the_dashboard_item_queries_the_sweep_m4_and_edge_and_finds_every_page(
    server: Callable[..., None], tmp_path: Path
) -> None:
    record = write_grid(tmp_path)
    server(answers(record))
    item = item_on(tmp_path, built_site(tmp_path / "web"))
    passes(item)
    assert item.evidence.startswith("12 API routes; home: /api/overview; runs: /api/runs;")
    assert "attribution: /api/edge, /api/edge/detail;" in item.evidence
    assert f"?key={grid_filename(TINY)} 200: 4 cells, 4 present" in item.evidence
    assert "GET /api/m4 200: M4_ANALYSIS.json of 2026-11-14 (delta_seller)" in item.evidence
    assert f"?grid={record.grid_hash} 200: the edge of 4 cells (delta_seller)" in item.evidence
    assert "<built site> has all 8 views' index.html" in item.evidence
    assert str(tmp_path) not in item.evidence


@pytest.mark.behaviour
def test_the_dashboard_item_names_routes_beyond_three_by_count() -> None:
    assert m5gate._shown(["/a", "/b", "/c", "/d", "/e"]) == "/a, /b, /c and 2 more"
    assert m5gate._shown([]) == "no route"


@pytest.mark.behaviour
def test_the_dashboard_item_names_the_view_with_no_route(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(routes=tuple(r for r in ROUTES if not r.startswith(("/api/leaks", "/api/power"))))
    blocked(item_on(tmp_path, tmp_path), "routes no API for the power, leaks view")
    server(routes=tuple(r for r in ROUTES if r != m5gate.M4_ROUTE))
    blocked(item_on(tmp_path, tmp_path), "the app routes no /api/m4")


@pytest.mark.behaviour
def test_the_dashboard_item_says_when_there_is_no_sweep(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server()
    item = item_on(tmp_path, built_site(tmp_path / "web"))
    blocked(item, f"no grid record {grid_filename(TINY)} under <runs root>, so no sweep")
    (tmp_path / grid_filename(TINY)).write_text("{}")
    blocked(item_on(tmp_path, tmp_path), f"{grid_filename(TINY)} is refused")


@pytest.mark.behaviour
@pytest.mark.parametrize(
    ("edit", "why"),
    [
        ({"status": {m5gate.SWEEP_ROUTE: 404}}, "/api/sweeps/detail?key="),
        ({"status": {m5gate.EDGE_ROUTE: 404}}, "answered 404: no such thing"),
        ({m5gate.SWEEP_ROUTE: {"grid": {"cells": []}}}, "lists 0 cells, the grid record 4"),
        ({m5gate.M4_ROUTE: {"available": False, "waiting_for": "make m4"}}, "no analysis: make m4"),
        ({m5gate.M4_ROUTE: {"available": True, "analysis": {}}}, "no reason given"),
        ({m5gate.EDGE_ROUTE: {"edge": {"grid_hash": "0" * 16}}}, "answers no edge of grid"),
        ({m5gate.EDGE_ROUTE: ["not", "an", "object"]}, "answers no edge of grid"),
    ],
)
def test_the_dashboard_item_needs_each_answer_with_content(
    server: Callable[..., None], tmp_path: Path, edit: dict[str, Any], why: str
) -> None:
    record = write_grid(tmp_path)
    edit = dict(edit)
    status = edit.pop("status", None)
    server({**answers(record), **edit}, status=status)
    blocked(item_on(tmp_path, built_site(tmp_path / "web")), why)


@pytest.mark.behaviour
def test_the_dashboard_item_needs_every_page_built(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(answers(write_grid(tmp_path)))
    web = built_site(tmp_path / "web", skip=("sweep/index.html", "data/index.html"))
    item = item_on(tmp_path, web)
    blocked(item, "the built site has no data/index.html, sweep/index.html (run make web)")
    assert "GET /api/m4 200" in item.evidence  # the answers are in the report either way


@pytest.mark.behaviour
def test_the_dashboard_item_without_a_server_or_with_a_broken_one(
    server: Callable[..., None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server(factory=False)
    blocked(item_on(tmp_path, tmp_path), "has no create_app")
    server(raises=RuntimeError("half wired"))
    blocked(item_on(tmp_path, tmp_path), "half wired")
    server(answers(write_grid(tmp_path)))
    monkeypatch.setattr(m5gate, "_m4_answer", lambda client: 1 / 0)
    blocked(item_on(tmp_path, tmp_path), "the app raised ZeroDivisionError")
    monkeypatch.setitem(sys.modules, "fastapi.testclient", None)
    blocked(item_on(tmp_path, tmp_path), "(the dashboard extra)")


@pytest.mark.behaviour
def test_every_view_matches_a_route_of_the_real_app(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from optspread.server import Settings, create_app

    app = create_app(Settings(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path))
    paths = m5gate._api_paths(app)
    for view, _, route in m5gate.VIEWS:
        assert any(route.search(p) for p in paths), view
    assert {m5gate.SWEEP_ROUTE, m5gate.M4_ROUTE, m5gate.EDGE_ROUTE} <= set(paths)
    item = m5gate.dashboard_item(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    blocked(item, "no grid record grid-7d9d53d340006f02.json under <runs root>")


@pytest.mark.behaviour
def test_every_view_has_a_page_in_the_site_source() -> None:
    app_dir = REPO_ROOT / "web" / "app"
    for _, page, _ in m5gate.VIEWS:
        assert (app_dir / page.replace("index.html", "page.tsx")).is_file(), page


# -- 2. usability testing ------------------------------------------------------------ #

SESSION_HEAD = (
    "| Session | Date | Tester (role, no name) | Outside the team | Tasks finished (of 8) |"
)
FINDING_HEAD = "| # | Session | Task | What happened | Severity | Response | Commit or reason |"


def usability(
    docs: Path, sessions: Sequence[tuple[str, ...]], findings: Sequence[tuple[str, ...]] = ()
) -> Path:
    """The committed template with its two tables' rows replaced."""
    lines = TEMPLATE.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith(SESSION_HEAD) or line.startswith(FINDING_HEAD):
            out += [line, lines[lines.index(line) + 1]]
            rows = sessions if line.startswith(SESSION_HEAD) else findings
            out += ["| " + " | ".join(r) + " |" for r in rows]
            skipping = True
            continue
        if skipping and line.startswith("|"):
            continue
        skipping = False
        out.append(line)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / m5gate.USABILITY_FILE).write_text("\n".join(out) + "\n")
    return docs


def session(n: str, outside: str = "yes", when: str = "2026-11-20") -> tuple[str, ...]:
    return (n, when, "CS undergraduate, no finance background", outside, "7", "Hemdutt Rao")


def finding(n: str, at: str = "1", response: str = "relabelled", why: str = "x") -> tuple[str, ...]:
    return (n, at, "3", "looked for the floor on Power", "medium", response, why)


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A throwaway repository with one commit, and that commit's sha."""
    root = tmp_path_factory.mktemp("repo")

    def git(*args: str) -> str:
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args]
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True).stdout

    git("init", "-q")
    (root / "a.txt").write_text("a\n")
    git("add", "a.txt")
    git("commit", "-q", "-m", "a")
    return root, git("rev-parse", "HEAD").strip()


@pytest.mark.invariant
def test_the_committed_record_holds_three_simulated_sessions_and_answers_every_finding() -> None:
    item = m5gate.usability_item(REPO_ROOT / "docs", REPO_ROOT)
    assert "9 sessions (all simulated)" in item.evidence
    assert "59 findings" in item.evidence
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # a shallow clone (CI's) lacks the commits the findings cite; a full one resolves them
    assert item.passed or (shallow == "true" and "cites commit" in (item.failed_first or ""))


@pytest.mark.behaviour
def test_the_kit_before_any_session_reads_no_session_recorded(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    (tmp_path / m5gate.USABILITY_FILE).write_text(TEMPLATE, encoding="utf-8")
    item = m5gate.usability_item(tmp_path, repo[0])
    blocked(item, "no session recorded")
    assert item.evidence == (
        "docs/USABILITY.md: 9 session rows, 0 recorded, none counting; 0 findings "
        "(0 high, 0 medium, 0 low)"
    )


@pytest.mark.behaviour
def test_two_sessions_outside_the_team_with_answered_findings_pass(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    root, sha = repo
    findings = [
        finding("1", response=f"renamed the tab ({sha[:8]})", why=sha[:7]),
        finding("2", at="2", response="left as is", why="the term is the study's"),
    ]
    docs = usability(tmp_path, [session("1"), session("2", "Yes"), session("3", "", "")], findings)
    item = m5gate.usability_item(docs, root)
    passes(item)
    assert "3 session rows, 2 recorded, 2 sessions (all outside the team)" in item.evidence
    assert "2 findings (0 high, 2 medium, 0 low)" in item.evidence
    assert "2 responses, citing 2 commits, every one in the repository" in item.evidence


@pytest.mark.behaviour
@pytest.mark.parametrize(
    ("outside", "says"),
    [
        (("simulated", "Simulated"), "2 sessions (all simulated)"),
        (("yes", "simulated"), "2 sessions (1 outside the team, 1 simulated)"),
    ],
)
def test_a_simulated_session_counts_and_the_evidence_says_so(
    tmp_path: Path, repo: tuple[Path, str], outside: tuple[str, str], says: str
) -> None:
    docs = usability(tmp_path, [session("1", outside[0]), session("2", outside[1])])
    item = m5gate.usability_item(docs, repo[0])
    passes(item)
    assert says in item.evidence and "no finding recorded, so none to answer" in item.evidence


@pytest.mark.behaviour
def test_the_session_texts_read_one_and_many() -> None:
    def s(kind: str) -> m5gate.Session:
        return m5gate.Session("1", "2026-11-20", "tester", kind)

    assert m5gate._sessions_text([s("yes")]) == "1 session (outside the team)"
    assert m5gate._sessions_text([s("simulated")]) == "1 session (simulated)"
    assert m5gate._sessions_text([s("simulated")] * 3) == "3 sessions (all simulated)"
    assert s("simulator").kind == "" and s("yesterday").kind == ""


@pytest.mark.behaviour
def test_a_session_counts_only_with_yes_or_simulated(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    docs = usability(tmp_path, [session("1"), session("2", "no (a teammate's flatmate)")])
    item = m5gate.usability_item(docs, repo[0])
    blocked(item, "1 session(s) count, at least 2 needed")
    assert "session 2 has neither" in item.failed_first
    docs = usability(tmp_path, [session("1"), session("2"), session("3", "no")])
    item = m5gate.usability_item(docs, repo[0])
    passes(item)
    assert item.evidence.endswith("; session 3 recorded but not counted")
    docs = usability(tmp_path, [session("1"), session("2", when="")])
    blocked(m5gate.usability_item(docs, repo[0]), "1 session(s) count")


@pytest.mark.behaviour
@pytest.mark.parametrize(
    ("row", "why"),
    [
        (finding("1", at="4"), "finding 1 names session '4', not one recorded"),
        (finding("1", at="1, 4"), "finding 1 names session '1, 4', not one recorded"),
        (finding("2", response=""), "finding 2 has no Response"),
        (finding("3", why=""), "finding 3 has no Commit or reason"),
        (finding("4", why="fixed in 0123abc"), "finding 4 cites commit 0123abc, which is not in"),
        (finding("5", response="see deadbeef1"), "cites commit deadbeef1"),
    ],
)
def test_every_finding_needs_a_recorded_session_a_response_and_real_commits(
    tmp_path: Path, repo: tuple[Path, str], row: tuple[str, ...], why: str
) -> None:
    docs = usability(tmp_path, [session("1"), session("2")], [row])
    blocked(m5gate.usability_item(docs, repo[0]), why)


@pytest.mark.behaviour
def test_a_hex_word_without_a_digit_is_not_read_as_a_commit(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    row = finding("1", response="the label was defaced by the theme", why="cafebabe looks fine")
    docs = usability(tmp_path, [session("1"), session("2")], [row])
    item = m5gate.usability_item(docs, repo[0])
    passes(item)
    assert "1 response, citing 0 commits" in item.evidence


@pytest.mark.behaviour
def test_the_usability_file_must_exist_and_hold_both_tables(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    blocked(m5gate.usability_item(tmp_path, repo[0]), "no docs/USABILITY.md")
    (tmp_path / m5gate.USABILITY_FILE).write_text("# Usability\n")
    blocked(m5gate.usability_item(tmp_path, repo[0]), "the Sessions table: no table headed")
    (tmp_path / m5gate.USABILITY_FILE).write_text(f"{SESSION_HEAD}\n|---|\n| 1 | x |\n")
    blocked(m5gate.usability_item(tmp_path, repo[0]), "a 'Session' table row has 2 cells, not 5")
    (tmp_path / m5gate.USABILITY_FILE).write_text(f"{SESSION_HEAD}\n|---|\n")
    blocked(m5gate.usability_item(tmp_path, repo[0]), "the Findings table: no table headed '#'")


@pytest.mark.behaviour
def test_a_severity_outside_the_three_is_counted_apart(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    rows = [finding("1"), (*finding("2")[:4], "blocker", *finding("2")[5:])]
    docs = usability(tmp_path, [session("1"), session("2")], rows)
    item = m5gate.usability_item(docs, repo[0])
    passes(item)
    assert "2 findings (0 high, 1 medium, 0 low, 1 of another severity)" in item.evidence


# -- 3. documentation ------------------------------------------------------------------ #


@pytest.mark.invariant
def test_the_committed_documentation_is_drafted() -> None:
    item = m5gate.documentation_item(REPO_ROOT, REPO_ROOT / "docs")
    passes(item)
    assert item.evidence.startswith("10 of 10 files written (lines: README.md ")


@pytest.mark.behaviour
def test_the_documentation_item_lists_what_is_missing_or_empty(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / m5gate.README).write_text("# optspread\n\nWhat it is.\n")
    for name in m5gate.DOC_FILES[1:]:
        (docs / name).write_text(f"# {name}\n")
    (docs / "DATA.md").write_text(" \n\n")
    item = m5gate.documentation_item(tmp_path, docs)
    blocked(item, "missing or empty: docs/REPRODUCE.md (absent), docs/DATA.md")
    assert item.evidence.startswith("8 of 10 files written (lines: README.md 3, docs/EXTENDING")
    (docs / "REPRODUCE.md").write_text("# Reproduce\n")
    (docs / "DATA.md").write_text("# Data\n")
    passes(m5gate.documentation_item(tmp_path, docs))


# -- 4. the adapter conformance suite and the OptionMetrics stub ------------------------- #


@pytest.fixture
def no_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real source unavailable, so nothing reads the shared folder."""
    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path / "nothing"))


@pytest.mark.invariant
@pytest.mark.usefixtures("no_panel")
def test_the_stub_and_the_suite_pass_and_a_missing_panel_is_reported() -> None:
    item = m5gate.adapter_item()
    passes(item)
    assert item.evidence.startswith(
        "SourceName names optionmetrics_stub; make_source('optionmetrics_stub') builds "
        "OptionMetricsStubSource, its provider a SyntheticChainProvider of "
    )
    assert item.evidence.endswith(
        "conformance_suite: synthetic PASS (64 chains); optionmetrics_stub PASS (64 chains); "
        "optionmetrics SKIPPED"
    )


def report(source: str, checked: bool = True, *violations: str) -> data_layer.ConformanceReport:
    return data_layer.ConformanceReport(source, checked, 64 if checked else 0, violations)


@pytest.mark.behaviour
@pytest.mark.parametrize(
    ("suite", "why"),
    [
        (
            (report("synthetic"), report("optionmetrics_stub", False), report("optionmetrics")),
            "the suite does not pass optionmetrics_stub",
        ),
        ((report("synthetic"),), "the suite does not pass optionmetrics_stub"),
        (
            (report("optionmetrics_stub"), report("optionmetrics", True, "reset at t=3")),
            "optionmetrics: reset at t=3",
        ),
    ],
)
@pytest.mark.usefixtures("no_panel")
def test_the_adapter_item_fails_on_a_failed_or_skipped_stub_or_any_failure(
    monkeypatch: pytest.MonkeyPatch,
    suite: tuple[data_layer.ConformanceReport, ...],
    why: str,
) -> None:
    monkeypatch.setattr(data_layer, "conformance_suite", lambda: suite)
    blocked(m5gate.adapter_item(), why)


@pytest.mark.behaviour
@pytest.mark.usefixtures("no_panel")
def test_the_adapter_item_without_the_stub_or_with_a_broken_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("no provider today")

    monkeypatch.setattr(data_layer, "conformance_suite", broken)
    blocked(m5gate.adapter_item(), "conformance_suite raised RuntimeError('no provider today')")
    monkeypatch.setattr(data_layer, "make_source", broken)
    blocked(m5gate.adapter_item(), "it raised RuntimeError('no provider today')")
    stub = types.SimpleNamespace(name="optionmetrics", available=True, provider=lambda: None)
    monkeypatch.setattr(data_layer, "make_source", lambda name: stub)
    blocked(m5gate.adapter_item(), "it is named 'optionmetrics'")
    stub.name, stub.available = "optionmetrics_stub", False
    blocked(m5gate.adapter_item(), "it is not available")
    monkeypatch.setattr(data_layer, "SourceName", Literal["synthetic"])
    blocked(m5gate.adapter_item(), "no optionmetrics_stub source")


# -- the report command ---------------------------------------------------------------- #


@pytest.mark.behaviour
@pytest.mark.usefixtures("no_panel")
def test_the_report_names_every_root_portably_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import m5_gate_report

    data = tmp_path / "shared"
    monkeypatch.setenv("VEGALAB_DATA", str(data))
    docs = data / "docs"
    docs.mkdir(parents=True)
    (docs / m5gate.USABILITY_FILE).write_text(TEMPLATE)
    out = data / "out" / "M5_GATE_REPORT.md"
    argv = ["--out", str(out), "--docs", str(docs), "--attribution", str(data / "attribution")]
    argv += ["--actionmix", str(data / "actionmix"), "--web", str(data / "web")]
    assert m5_gate_report.main(argv, on=date(2026, 11, 24)) == 1
    text = out.read_text()
    assert "Runs root: `$VEGALAB_DATA/runs` (grid `grid-7d9d53d340006f02.json`)" in text
    assert (
        "docs: `$VEGALAB_DATA/docs`; attribution root: `$VEGALAB_DATA/attribution`; action mix "
        "root: `$VEGALAB_DATA/actionmix`; built site: `$VEGALAB_DATA/web`"
    ) in text
    assert "Built by `python -m scripts.m5_gate_report` at " in text and str(data) not in text
    report = GateReport.from_markdown(text)  # the committed file re-renders byte for byte
    assert report.milestone == "M5" and len(report.items) == 4 and not report.passed
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["report_date"] == "2026-11-24" and sidecar["verdict"] == "NOT CLEARED"
    assert [i["name"] for i in sidecar["items"]] == [
        "Dashboard views on sweep data",
        "Usability testing conducted, findings applied",
        "Documentation drafted",
        "Adapter conformance suite and OptionMetrics stub",
    ]
    assert sidecar["items"][1]["failed_first"] == "no session recorded"


@pytest.mark.behaviour
@pytest.mark.usefixtures("no_panel")
def test_the_report_clears_when_every_item_does(
    tmp_path: Path,
    server: Callable[..., None],
    repo: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import m5_gate_report

    runs = tmp_path / "runs"
    server(answers(write_grid(runs)))
    docs = usability(tmp_path / "docs", [session("1", "simulated"), session("2", "simulated")])
    root = repo[0]
    (root / m5gate.README).write_text("# optspread\n")
    for name in m5gate.DOC_FILES:
        (docs / name).write_text(f"# {name}\n")
    monkeypatch.setattr(
        m5_gate_report,
        "dashboard_item",
        lambda r, d, w, a, m: m5gate.dashboard_item(r, d, w, a, m, TINY),
    )
    argv = ["--docs", str(docs), "--runs-root", str(runs), "--repo", str(root)]
    argv += ["--web", str(built_site(tmp_path / "web")), "--out", "-"]
    assert m5_gate_report.main(argv) == 0
    text = capsys.readouterr().out
    assert "**Verdict: PASS** (4 items, 0 failed, 0 without a stated criterion)" in text
    assert "2 sessions (all simulated)" in text


@pytest.mark.behaviour
def test_a_finding_several_testers_hit_names_each_session(
    tmp_path: Path, repo: tuple[Path, str]
) -> None:
    docs = usability(tmp_path, [session("1"), session("2")], [finding("1", at="1, 2")])
    passes(m5gate.usability_item(docs, repo[0]))
