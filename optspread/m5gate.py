"""The M5 gate items: one small evidence function per deliverable.

Lane: Hemdutt Rao (market layer).

The M5 row of the schedule (project document section 12) is: all required dashboard
views running end to end on sweep data; usability testing conducted and findings
applied; documentation drafted; the adapter conformance suite and the OptionMetrics
stub. Section 10.10 says what the usability testing is: two to three people outside
the team, a short task list and no instruction, the findings and the changes made in
response recorded. The course allows the team to simulate those testers, so a session
may be marked "simulated" instead, and the evidence says how many of each there were.

Each function returns a ``GateItem`` and never raises for a missing deliverable.

- **The dashboard** is another lane's (Ilan Danial's), so its item imports the server by
  the names m4gate uses and matches each view's API by name (``VIEWS``). Given the A7 grid
  record under the runs root, the sweep detail, the M4 answer and the sweep's attribution
  (the edge by component, what the attribution page reads) are fetched in-process through
  FastAPI's TestClient, as the M3 demo item does, and must answer 200 with content. The
  built site must hold each view's page.
- **Usability** is read from ``docs/USABILITY.md``: its Sessions and Findings tables, in
  the form that file states. A commit a finding's response cites must resolve in the
  repository. A sha is a run of 7 to 40 lowercase hex characters holding a digit: a
  word such as "defaced" is hex too, and a sha with no digit (one in a thousand) is left
  unchecked rather than refused.
- **Documentation** is the files ``DOC_FILES`` names, present and more than whitespace.
- **The adapter** is the ``optionmetrics_stub`` source of ``optspread.data`` and the
  conformance suite m2gate's adapter item runs (``data.conformance_suite``), run here in
  process. The real source is reported beside the stub; it was M2's item, so its panel
  being absent here (SKIPPED) is stated, not failed. A source that is checked and fails
  does fail the item.

The evidence names roots as ``<runs root>``, ``<attribution root>`` and ``<built site>``,
as the server's answers do, and files by their place in the repository; the report's
header names the roots themselves. ``scripts/m5_gate_report.py`` assembles these into
``docs/M5_GATE_REPORT.md``.
"""

from __future__ import annotations

import re
import typing
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optspread import data as data_layer
from optspread.experiment.grid import GridConfig, grid_filename, read_grid
from optspread.gatereport import GateItem
from optspread.m2gate import HEMDUTT, ILAN, _git, _load, _not_cleared
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.m4gate import APP_FACTORY, SERVER_MODULE, SETTINGS_CLASS, _api_paths, _one_line

TEAM = "All members"

# -- 1. the dashboard's views ------------------------------------------------------ #

# (view, its page in the static export, the API it reads, matched by name). The export
# writes every page as a folder with an index.html (trailingSlash, web/next.config.mjs).
VIEWS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("home", "index.html", re.compile(r"^/api/overview$")),
    ("runs", "runs/index.html", re.compile(r"^/api/runs$")),
    ("launch", "launch/index.html", re.compile(r"^/api/launch/")),
    ("power", "power/index.html", re.compile(r"^/api/power$")),
    ("real data", "data/index.html", re.compile(r"^/api/data/")),
    ("leaks", "leaks/index.html", re.compile(r"^/api/leaks/")),
    ("sweep", "sweep/index.html", re.compile(r"^/api/sweeps(/|$)")),
    ("attribution", "attribution/index.html", re.compile(r"^/api/(edge|runs/attribution)(/|$)")),
)
SWEEP_ROUTE = "/api/sweeps/detail"  # ?key=<grid file under the runs root>
M4_ROUTE = "/api/m4"
EDGE_ROUTE = "/api/edge/detail"  # ?grid=<grid hash>: what the attribution page draws

DASHBOARD_CRITERION = (
    f"{SERVER_MODULE}.{APP_FACTORY} builds the dashboard's app, which routes an API for "
    "every required view (home /api/overview, runs /api/runs, launch /api/launch/..., power "
    "/api/power, real data /api/data/..., leaks /api/leaks/..., sweep /api/sweeps..., "
    "attribution /api/edge... or /api/runs/attribution), matched by name and named in the "
    "evidence; with the A7 grid record under the runs root, the app queried in-process "
    f"(FastAPI's TestClient) answers 200 with content for the sweep's detail ({SWEEP_ROUTE}: "
    f"every cell of the grid record listed), the M4 analysis ({M4_ROUTE}: available) and the "
    f"sweep's attribution ({EDGE_ROUTE}: the grid's edge with its cells); and the built site "
    "(web/out) has each view's index.html"
)


def _shown(paths: Sequence[str], most: int = 3) -> str:
    """The routes matched, the first ``most`` of them named."""
    if len(paths) <= most:
        return ", ".join(paths) or "no route"
    return f"{', '.join(paths[:most])} and {len(paths) - most} more"


def _json(response: Any) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _refused(asked: str, response: Any) -> str:
    detail = (_json(response) or {}).get("detail", response.text[:200])
    return f"{asked} answered {response.status_code}: {_one_line(Exception(str(detail)))}"


def _sweep_answer(client: Any, key: str, n_cells: int) -> tuple[str, str]:
    """``(evidence, "")`` from the sweep detail, or ``("", what failed first)``."""
    asked = f"GET {SWEEP_ROUTE}?key={key}"
    response = client.get(SWEEP_ROUTE, params={"key": key})
    if response.status_code != 200:
        return "", _refused(asked, response)
    body = _json(response) or {}
    cells = [c for c in (body.get("grid") or {}).get("cells") or () if isinstance(c, dict)]
    if len(cells) != n_cells:
        return "", f"{asked} lists {len(cells)} cells, the grid record {n_cells}"
    present = sum(c.get("present") is True for c in cells)
    return f"{asked} 200: {len(cells)} cells, {present} present", ""


def _m4_answer(client: Any) -> tuple[str, str]:
    asked = f"GET {M4_ROUTE}"
    response = client.get(M4_ROUTE)
    if response.status_code != 200:
        return "", _refused(asked, response)
    body = _json(response) or {}
    analysis = body.get("analysis")
    if body.get("available") is not True or not isinstance(analysis, dict) or not analysis:
        why = body.get("error") or body.get("waiting_for") or "no reason given"
        return "", f"{asked} answers no analysis: {_one_line(Exception(str(why)))}"
    arms = ", ".join(str(a) for a in analysis.get("arm_order") or ()) or "no arms"
    return f"{asked} 200: {body.get('file')} of {analysis.get('report_date')} ({arms})", ""


def _edge_answer(client: Any, grid_hash: str) -> tuple[str, str]:
    asked = f"GET {EDGE_ROUTE}?grid={grid_hash}"
    response = client.get(EDGE_ROUTE, params={"grid": grid_hash})
    if response.status_code != 200:
        return "", _refused(asked, response)
    edge = (_json(response) or {}).get("edge") or {}
    cells = [c for c in edge.get("cells") or () if isinstance(c, dict)]
    if edge.get("grid_hash") != grid_hash or not cells:
        return "", f"{asked} answers no edge of grid {grid_hash} with cells"
    arms = ", ".join(str(a) for a in edge.get("arms") or ())
    return (
        f"{asked} 200: the edge of {len(cells)} cells ({arms}), re-simulated against "
        f"run.json: {edge.get('reproduction')}",
        "",
    )


def dashboard_item(
    runs_root: Path,
    docs_dir: Path,
    web_dir: Path,
    attribution_root: Path,
    actionmix_root: Path,
    grid: GridConfig = CONFIRMATORY_GRID,
) -> GateItem:
    name = "Dashboard views on sweep data"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, ILAN, DASHBOARD_CRITERION, evidence, why)

    factory, why = _load(SERVER_MODULE, APP_FACTORY)
    settings, why_settings = _load(SERVER_MODULE, SETTINGS_CLASS)
    if factory is None or settings is None:
        return fail("no dashboard server", why or why_settings)
    with warnings.catch_warnings():  # starlette's notice about its httpx client, not ours
        warnings.simplefilter("ignore")
        client_class, why = _load("fastapi.testclient", "TestClient")
    if client_class is None:
        return fail("no TestClient to call the app with", f"{why} (the dashboard extra)")
    try:
        # The panel is read lazily, so a folder with no panel is enough to build the app.
        app = factory(
            settings(
                runs_root,
                docs_dir,
                docs_dir,
                web_dir,
                docs_dir / "README.md",
                attribution=attribution_root,
                actionmix=actionmix_root,
            )
        )
        paths = _api_paths(app)
    except Exception as exc:  # a broken server is a finding, not a crash of the report
        return fail(f"{SERVER_MODULE}.{APP_FACTORY} was called", f"it raised {exc!r}")
    found = {view: [p for p in paths if route.search(p)] for view, _, route in VIEWS}
    evidence = f"{len(paths)} API routes; " + "; ".join(
        f"{view}: {_shown(p)}" for view, p in found.items()
    )
    absent = [view for view, p in found.items() if not p]
    if absent:
        return fail(evidence, f"the app routes no API for the {', '.join(absent)} view")
    unrouted = [r for r in (SWEEP_ROUTE, M4_ROUTE, EDGE_ROUTE) if r not in paths]
    if unrouted:
        return fail(evidence, f"the app routes no {' and no '.join(unrouted)}")
    key = grid_filename(grid)
    grid_path = runs_root / key
    if not grid_path.is_file():
        return fail(evidence, f"no grid record {key} under <runs root>, so no sweep to show")
    try:
        record = read_grid(grid_path)
    except (OSError, ValueError) as exc:  # SchemaVersionError and pydantic's are ValueErrors
        return fail(evidence, f"{key} is refused: {_one_line(exc)}")
    answers: list[str] = []
    try:
        client = client_class(app, base_url="http://127.0.0.1")
        asks: tuple[Callable[[], tuple[str, str]], ...] = (
            lambda: _sweep_answer(client, key, len(record.cells)),
            lambda: _m4_answer(client),
            lambda: _edge_answer(client, record.grid_hash),
        )
        for ask in asks:
            text, why = ask()
            if why:
                return fail("; ".join([evidence, *answers]), why)
            answers.append(text)
    except Exception as exc:  # a route that raises is a finding, not a crash of the report
        return fail("; ".join([evidence, *answers]), f"the app raised {exc!r}")
    evidence = "; ".join([evidence, *answers])
    missing = [page for _, page, _ in VIEWS if not (web_dir / page).is_file()]
    if missing:
        return fail(evidence, f"the built site has no {', '.join(missing)} (run make web)")
    evidence += f"; <built site> has all {len(VIEWS)} views' index.html"
    return GateItem(name, ILAN, DASHBOARD_CRITERION, evidence, passed=True)


# -- 2. usability testing ---------------------------------------------------------- #

USABILITY_FILE = "USABILITY.md"
MIN_SESSIONS = 2
SEVERITIES = ("high", "medium", "low")
SHA = re.compile(r"\b(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b")
# What the Outside the team cell must begin with for a session to count.
REAL, SIMULATED = "yes", "simulated"
_OUTSIDE = re.compile(rf"^({REAL}|{SIMULATED})\b", re.IGNORECASE)

USABILITY_CRITERION = (
    f"docs/{USABILITY_FILE} records at least {MIN_SESSIONS} sessions in its Sessions table "
    "with a date, a tester's role and 'yes' (or 'simulated') in Outside the team, the "
    "evidence saying how many of each; "
    "and every row of its Findings table names a recorded session and has a Response and a "
    "Commit or reason, every commit cited (a 7 to 40 hex sha) resolving in the repository "
    "(section 10.10)"
)


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def read_table(text: str, first_column: str) -> tuple[list[dict[str, str]], str]:
    """The rows of the Markdown table whose header begins ``| <first_column> |``, keyed by
    the header's cells, or ``([], why not)``. Rows of the wrong width are refused."""
    lines = text.splitlines()
    head = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.strip().startswith("|") and _cells(ln)[0] == first_column
        ),
        None,
    )
    if head is None:
        return [], f"no table headed '{first_column}'"
    columns = _cells(lines[head])
    rows: list[dict[str, str]] = []
    for line in lines[head + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = _cells(line)
        if len(cells) != len(columns):
            return [], f"a '{first_column}' table row has {len(cells)} cells, not {len(columns)}"
        rows.append(dict(zip(columns, cells, strict=True)))
    return rows, ""


@dataclass(frozen=True, slots=True)
class Session:
    number: str
    date: str
    tester: str
    outside: str

    @property
    def recorded(self) -> bool:
        return bool(self.date and self.tester)

    @property
    def kind(self) -> str:
        """``yes``, ``simulated``, or "" for a session that does not count."""
        match = _OUTSIDE.match(self.outside)
        return match[1].lower() if self.recorded and match else ""


def read_sessions(rows: Sequence[dict[str, str]]) -> list[Session]:
    def cell(row: dict[str, str], prefix: str) -> str:
        return next((v for k, v in row.items() if k.lower().startswith(prefix)), "")

    return [
        Session(cell(r, "session"), cell(r, "date"), cell(r, "tester"), cell(r, "outside"))
        for r in rows
    ]


def _sessions_text(sessions: Sequence[Session]) -> str:
    real = sum(s.kind == REAL for s in sessions)
    simulated = sum(s.kind == SIMULATED for s in sessions)
    n = real + simulated
    if not simulated:
        how = "all outside the team" if n > 1 else "outside the team"
    elif not real:
        how = "all simulated" if n > 1 else "simulated"
    else:
        how = f"{real} outside the team, {simulated} simulated"
    return f"{n} session{'s' if n != 1 else ''} ({how})"


def usability_item(docs_dir: Path, repo: Path) -> GateItem:
    name = "Usability testing conducted, findings applied"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, TEAM, USABILITY_CRITERION, evidence, why)

    path = docs_dir / USABILITY_FILE
    shown = f"docs/{USABILITY_FILE}"
    if not path.is_file():
        return fail("not written", f"no {shown}")
    text = path.read_text(encoding="utf-8", errors="replace")
    rows, why = read_table(text, "Session")
    if why:
        return fail(f"{shown} read", f"the Sessions table: {why}")
    findings, why = read_table(text, "#")
    if why:
        return fail(f"{shown} read", f"the Findings table: {why}")
    sessions = read_sessions(rows)
    recorded = [s for s in sessions if s.recorded]
    counted = [s for s in recorded if s.kind]
    severity = Counter(f.get("Severity", "").strip().lower() for f in findings)
    graded = ", ".join(f"{severity[s]} {s}" for s in SEVERITIES)
    other = sum(n for s, n in severity.items() if s not in SEVERITIES)
    evidence = (
        f"{shown}: {len(sessions)} session rows, {len(recorded)} recorded, "
        f"{_sessions_text(counted) if counted else 'none counting'}; {len(findings)} "
        f"finding{'s' if len(findings) != 1 else ''} ({graded}"
        f"{f', {other} of another severity' if other else ''})"
    )
    if not recorded:
        return fail(evidence, "no session recorded")
    not_counted = [s.number for s in recorded if not s.kind]
    if len(counted) < MIN_SESSIONS:
        why = (
            f"{len(counted)} session(s) count, at least {MIN_SESSIONS} needed (a session counts "
            f"with '{REAL}' or '{SIMULATED}' in Outside the team"
            + (f"; session {', '.join(not_counted)} has neither" if not_counted else "")
            + ")"
        )
        return fail(evidence, why)
    numbers = {s.number for s in recorded}
    cited: set[str] = set()
    responded = 0
    for f in findings:
        at = f"finding {f.get('#') or '?'}"
        # A finding more than one tester hit names each session: "1, 3".
        named = [n.strip() for n in f.get("Session", "").split(",")]
        if not all(n in numbers for n in named):
            return fail(evidence, f"{at} names session {f.get('Session')!r}, not one recorded")
        response, commit = f.get("Response", ""), f.get("Commit or reason", "")
        if not response or not commit:
            return fail(
                evidence, f"{at} has no {'Response' if not response else 'Commit or reason'}"
            )
        responded += 1
        for sha in SHA.findall(f"{response} {commit}"):
            if _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
                return fail(evidence, f"{at} cites commit {sha}, which is not in the repository")
            cited.add(sha)
    evidence += (
        f"; {responded} response{'s' if responded != 1 else ''}, citing {len(cited)} "
        f"commit{'s' if len(cited) != 1 else ''}, every one in the repository"
        if findings
        else "; no finding recorded, so none to answer"
    )
    if not_counted:
        evidence += f"; session {', '.join(not_counted)} recorded but not counted"
    return GateItem(name, TEAM, USABILITY_CRITERION, evidence, passed=True)


# -- 3. documentation ---------------------------------------------------------------- #

README = "README.md"  # at the repository root; the rest are under docs/
DOC_FILES = (
    "REPRODUCE.md",
    "EXTENDING.md",
    "DATA.md",
    "DEFECTS.md",
    "PRE_REGISTRATION.md",
    "M1_GATE_REPORT.md",
    "M2_GATE_REPORT.md",
    "M3_GATE_REPORT.md",
    "M4_GATE_REPORT.md",
)

DOCUMENTATION_CRITERION = (
    f"{README} and docs/" + ", docs/".join(DOC_FILES) + " exist and each holds more than "
    "whitespace; the evidence lists what is missing"
)


def documentation_item(repo: Path, docs_dir: Path) -> GateItem:
    name = "Documentation drafted"
    files = [(README, repo / README), *((f"docs/{f}", docs_dir / f) for f in DOC_FILES)]
    lines: list[str] = []
    missing: list[str] = []
    for shown, path in files:
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if not text.strip():
            missing.append(shown + ("" if path.is_file() else " (absent)"))
            continue
        lines.append(f"{shown} {len(text.splitlines()):,}")
    evidence = f"{len(lines)} of {len(files)} files written (lines: {', '.join(lines) or 'none'})"
    if missing:
        why = f"missing or empty: {', '.join(missing)}"
        return _not_cleared(name, TEAM, DOCUMENTATION_CRITERION, evidence, why)
    return GateItem(name, TEAM, DOCUMENTATION_CRITERION, evidence, passed=True)


# -- 4. the adapter conformance suite and the OptionMetrics stub ------------------- #

STUB: data_layer.SourceName = "optionmetrics_stub"
REAL_SOURCE = "optionmetrics"

ADAPTER_CRITERION = (
    f"optspread.data's SourceName names {STUB}, make_source builds it as that source, "
    "available, and its provider builds; and the conformance suite (data.conformance_suite, "
    "the check m2gate's adapter item runs: the provider contract and the leak suite's "
    f"structural invariants) passes {STUB} and every other source it checks. The real "
    f"{REAL_SOURCE} source is reported beside them: SKIPPED where its panel is absent, which "
    "is not a failure of this item (its conformance was M2's)"
)


def _report_text(r: data_layer.ConformanceReport) -> str:
    verdict = "PASS" if r.passed else "SKIPPED" if not r.checked else "FAIL"
    return f"{r.source} {verdict}" + (f" ({r.n_chains} chains)" if r.checked else "")


def adapter_item() -> GateItem:
    name = "Adapter conformance suite and OptionMetrics stub"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, HEMDUTT, ADAPTER_CRITERION, evidence, why)

    if STUB not in typing.get_args(data_layer.SourceName):
        return fail(f"SourceName is {typing.get_args(data_layer.SourceName)}", f"no {STUB} source")
    try:
        source = data_layer.make_source(STUB)
        provider = source.provider() if source.available else None
    except Exception as exc:  # a broken source is a finding, not a crash of the report
        return fail(f"make_source({STUB!r}) was called", f"it raised {exc!r}")
    built = f"make_source({STUB!r}) builds {type(source).__name__}"
    if source.name != STUB or provider is None:
        why = f"it is named {source.name!r}" if source.name != STUB else "it is not available"
        return fail(built, why)
    evidence = (
        f"SourceName names {STUB}; {built}, its provider a {type(provider).__name__} of "
        f"{provider.n_days} days"
    )
    try:
        reports = data_layer.conformance_suite()
    except Exception as exc:  # a suite that raises is a finding, not a crash of the report
        return fail(evidence, f"conformance_suite raised {exc!r}")
    evidence += "; conformance_suite: " + "; ".join(_report_text(r) for r in reports)
    by_source = {r.source: r for r in reports}
    if STUB not in by_source or not by_source[STUB].passed:
        return fail(evidence, f"the suite does not pass {STUB}")
    failed = [r for r in reports if r.checked and not r.passed]
    if failed:
        return fail(evidence, f"{failed[0].source}: {failed[0].violations[0]}")
    return GateItem(name, HEMDUTT, ADAPTER_CRITERION, evidence, passed=True)
