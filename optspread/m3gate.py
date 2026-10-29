"""The M3 gate items: one small evidence function per deliverable.

Lane: Dominick Dupuy (statistical spine).

The M3 row of the schedule (project document section 12) is: sweep complete; leak
detection floor measured; poisoned-environment demo live; partner team clones the
repository and independently runs the gate commands (#11, #10).

Each function returns a ``GateItem`` and never raises for a missing deliverable.

- **The sweep** is the A7 grid (``m4analysis.CONFIRMATORY_GRID``, ``grid-7d9d53d340006f02.json``)
  under the runs root, every cell valid by the rule every reuse check shares
  (``sweep.valid_run``: manifest and results keyed by the cell's hash, on the generator,
  every recorded digest matching its file, a PPO cell by the real learner), its headline in
  the record equal to its ``run.json``, and written from a clean tree at or after the cutoff
  tag, so the numbers belong to the frozen environment (12.1). The replay spot-check is read
  from ``replay.log`` beside the record (what ``make replay ... > replay.log`` leaves) and
  from replays run here (``--replay N``); an IDENTICAL line must name the sha256 of the
  ``run.json`` now on disk, or it is a line about another file. A cell replayed here that
  was written on another platform (or one its manifest does not record) is compared within
  the seller tolerance (``replay_here``), since libm's last bit differs between platforms;
  a PPO cell written elsewhere is not replayed at all, because its training is not
  reproducible across platforms at any tolerance, and the evidence says so.
- **The floor** is measured again, not quoted: ``leaks.leak_floor`` on the M1 grid and seeds,
  with the clean world quiet at the same seeds and the caught set a threshold (every leak at
  or above the floor caught). The detector and the world are deterministic in the seed, so
  the floor must equal the one ``docs/M1_GATE_REPORT.json`` records; a different one means
  something moved since M1.
- **The demo** is called, not looked for: the server's app is built, its leak demo route
  posted to in-process through FastAPI's TestClient (no port, no job pool), and the answer
  must fail the poisoned variant on purpose (flagged by the detector, its past moved by
  resampling the future) and pass the clean one; the built site has the page and calls the
  route.
- **The partner reproduction** is ``docs/M3_PARTNER_REPRODUCTION.json``, the record
  ``scripts/reproduce.py`` writes on the partner's clone (its schema in
  ``optspread.reproduction``): a clean checkout of a commit of this repository at or after the
  cutoff tag, every step present and passed, every regenerated cell a cell of the A7 grid
  within the tolerance stated in ``reproduction``, compared against the
  ``docs/SWEEP_EXPECTED.json`` of that commit, whose digests must be the sweep's on disk.

``scripts/m3_gate_report.py`` assembles these into ``docs/M3_GATE_REPORT.md``.
"""

from __future__ import annotations

import re
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import Random
from typing import Any

from optspread import reproduction as rp
from optspread.artifacts import RUN_FILENAME, read_run
from optspread.config import config_hash
from optspread.env import SpreadEnv
from optspread.gatereport import GateItem
from optspread.leaks import default_env, detect_leak, leak_floor
from optspread.m2gate import DOMINICK, ILAN, _git, _load, _not_cleared
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.m4gate import (
    APP_FACTORY,
    SERVER_MODULE,
    SETTINGS_CLASS,
    _api_paths,
    _callers,
    _one_line,
    read_json,
)
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256
from optspread.orchestrate import Cell, GridConfig, grid_cells, grid_filename, read_grid
from optspread.replay import ReplayResult, platform_differs, replay_run
from optspread.sweep import (
    REAL_TRAINER,
    changed_artifacts,
    recorded_trainer,
    run_id,
    source_mismatch,
    valid_run,
)

TEAM = "All members"
CUTOFF_TAG = rp.CUTOFF_TAG
REPLAY_LOG = "replay.log"  # the spot-check's output, beside the grid record
REPLAY_SAMPLE_SEED = 3  # which cells --replay N picks: a fixed draw, one arm after the other
_LOG_LINE = re.compile(r"^(IDENTICAL|DIFFERS)\s+(\S+)\s+\(([\d.]+)s\)\s*(.*)$")
_LOG_SHA = re.compile(r"run\.json sha256 ([0-9a-f]{8,64}) reproduced")

# The M1 floor's design (scripts/m1_gate_report.py), measured again here.
FLOOR_GRID = (1.0, 0.5, 0.25, 0.1, 0.05)
FLOOR_SEEDS = 10
M1_SIDECAR = "M1_GATE_REPORT.json"
M1_FLOOR_ITEM = "Leak detection floor"
_M1_FLOOR = re.compile(r"floor eps = ([0-9.]+)")

DEMO_ROUTE = re.compile(r"leak", re.IGNORECASE)
DEMO_EPS = 0.5  # the endpoint's default, the README's demo
DEMO_SEEDS = 10
LEAK_PAGES = ("leaks/index.html", "leaks.html")  # the export writes the first (trailingSlash)


def _premiums(values: Sequence[float]) -> str:
    return ", ".join(f"{p:g}" for p in values)


def _short(sha: str) -> str:
    return sha[:8] if sha else "?"


def _tag_commit(repo: Path, tag: str) -> str:
    done = _git(repo, "rev-parse", f"{tag}^{{commit}}")
    return done.stdout.strip() if done.returncode == 0 else ""


def _at_or_after(repo: Path, tag_commit: str, sha: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", tag_commit, sha).returncode == 0


# -- 1. the sweep ------------------------------------------------------------------- #

SWEEP_CRITERION = (
    f"the A7 grid record ({grid_filename(CONFIRMATORY_GRID)}) under the runs root is readable, "
    f"is of the A7 design (grid hash {config_hash(CONFIRMATORY_GRID)}) and lists exactly its "
    f"{len(grid_cells(CONFIRMATORY_GRID))} cells; every cell is valid under the runs root by "
    "sweep.valid_run (manifest and run.json keyed by the cell's config hash and run id, run "
    f"on the generator, every recorded digest matching its file, a PPO cell trained by "
    f"{REAL_TRAINER}) with its headline in the record equal to its run.json; every cell was "
    f"written from a clean tree at a commit at or after the {CUTOFF_TAG} tag; and every grid "
    f"cell a replay spot-check replayed ({REPLAY_LOG} beside the record, an IDENTICAL line "
    "naming the sha256 of the run.json on disk, or --replay N run by the report) reproduced. "
    "The evidence says what the spot-check covered, or that none was run"
)


def why_invalid(run_dir: Path, cell: Cell) -> str:
    """Why ``sweep.valid_run`` refuses the cell's run directory, in its own order."""
    config = cell.config
    if not (run_dir / RUN_FILENAME).exists() or not (run_dir / MANIFEST_FILENAME).exists():
        return f"no {RUN_FILENAME} and {MANIFEST_FILENAME} at {run_dir}"
    try:
        record, manifest = read_run(run_dir), RunManifest.read(run_dir)
    except (OSError, ValueError) as exc:
        return f"unreadable: {_one_line(exc)}"
    want = config_hash(config)
    if record.config_hash != want or manifest.config_hash != want:
        return f"keyed by {record.config_hash} (manifest {manifest.config_hash}), not {want}"
    if record.run_id != run_id(config) or manifest.run_id != run_id(config):
        return f"run id {record.run_id} (manifest {manifest.run_id}), not {run_id(config)}"
    if why := source_mismatch(manifest, "synthetic"):
        return why
    if changed := changed_artifacts(run_dir, manifest):
        return f"{', '.join(changed)} no longer match the manifest's digests"
    if config.train is not None:
        by = recorded_trainer(run_dir, manifest)
        if by != REAL_TRAINER:
            return f"trained by {by or 'no recorded trainer'}, not {REAL_TRAINER}"
    return "refused by sweep.valid_run"


@dataclass(frozen=True, slots=True)
class Replayed:
    run_id: str
    identical: bool
    reason: str
    source: str  # "log" or "here"


def read_replay_log(path: Path) -> list[Replayed]:
    """The verdict lines of ``scripts.replay``'s output; every other line is ignored."""
    if not path.is_file():
        return []
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LOG_LINE.match(line.strip())
        if m:
            found.append(Replayed(m[2], m[1] == "IDENTICAL", m[4], "log"))
    return found


def replay_sample(cells: Sequence[Cell], n: int) -> list[Cell]:
    """``n`` cells to replay: a fixed draw per arm, the arms taken in turn (the grid's
    first arm first, the fixed-delta seller in A7, so one cell costs seconds, not a
    training run)."""
    arms = list(dict.fromkeys(c.arm for c in cells))
    draws = []
    for a in arms:
        mine = [c for c in cells if c.arm == a]
        draws.append(Random(REPLAY_SAMPLE_SEED).sample(mine, k=len(mine)))
    turns = [
        draw[i] for i in range(max(map(len, draws), default=0)) for draw in draws if i < len(draw)
    ]
    return turns[: max(n, 0)]


def replay_here(run_dir: Path) -> ReplayResult:
    """The spot-check's replay of one cell: byte for byte on the platform it was written on,
    within ``reproduction.SELLER_REL_TOL`` when it was written on another (or an unrecorded)
    one."""
    if platform_differs(run_dir):
        return replay_run(run_dir, rtol=rp.SELLER_REL_TOL)
    return replay_run(run_dir)


def sweep_item(
    runs_root: Path,
    repo: Path,
    *,
    grid: GridConfig = CONFIRMATORY_GRID,
    tag: str = CUTOFF_TAG,
    replays: int = 0,
    replay: Callable[[Path], ReplayResult] = replay_here,
) -> GateItem:
    name = "Sweep complete"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, SWEEP_CRITERION, evidence, why)

    grid_path = runs_root / grid_filename(grid)
    if not grid_path.is_file():
        return fail(
            f"runs root {runs_root}", f"no grid record at {grid_path}: the sweep has not run"
        )
    try:
        record = read_grid(grid_path)
    except (OSError, ValueError) as exc:
        return fail(f"{grid_path} found", f"{grid_path} is refused: {_one_line(exc)}")
    want = config_hash(grid)
    if record.grid_hash != want or config_hash(record.grid) != want:
        why = (
            f"the record is of grid {record.grid_hash} (its config hashes to "
            f"{config_hash(record.grid)}), not the A7 grid {want}"
        )
        return fail(f"{grid_path.name} read", why)
    cells = grid_cells(grid)
    listed = {(c.premium, c.arm, c.block): c for c in record.cells}
    absent = [c for c in cells if (c.premium, c.arm, c.block) not in listed]
    if absent or len(record.cells) != len(cells):
        why = f"the record lists {len(record.cells)} cells, the grid has {len(cells)}"
        if absent:
            why += f"; first missing: {absent[0].label.strip()} ({absent[0].run_id})"
        return fail(f"{grid_path.name}: grid {want}", why)
    head = (
        f"{grid_path.name}: the A7 design, {len(cells)} of {len(cells)} cells "
        f"({len(grid.premiums)} premiums x {len(grid.arms)} arms x {grid.n_seeds} blocks)"
    )
    invalid: list[str] = []
    manifests: dict[str, RunManifest] = {}
    for cell in cells:
        run_dir = runs_root / cell.run_id
        trainer = REAL_TRAINER if cell.config.train is not None else None
        run = valid_run(run_dir, cell.config, trainer=trainer)
        row = listed[(cell.premium, cell.arm, cell.block)]
        if run is None:
            invalid.append(f"{cell.label.strip()} ({cell.run_id}): {why_invalid(run_dir, cell)}")
            continue
        headline = (row.run_id, row.n_episodes, row.mean_pnl, row.se_pnl, row.mean_cost)
        if headline != (run.run_id, run.n_episodes, run.mean_pnl, run.se_pnl, run.mean_cost):
            invalid.append(
                f"{cell.label.strip()} ({cell.run_id}): the record's headline is not its run.json's"
            )
            continue
        manifests[cell.run_id] = RunManifest.read(run_dir)
    if invalid:
        why = f"{len(invalid)} of {len(cells)} cells are not valid; first: {invalid[0]}"
        return fail(f"{head} under {runs_root}", why)
    ppo = [m for m in manifests.values() if m.config.train is not None]
    by_archive = sum(m.trainer is None for m in ppo)
    head += (
        f", all valid under {runs_root} (digests match, headlines agree; {len(ppo)} PPO cells "
        f"by {REAL_TRAINER}"
        + (
            f", {by_archive} of them told by their Stable-Baselines3 policy.zip"
            if by_archive
            else ""
        )
        + ")"
    )
    dirty = sorted(m.run_id for m in manifests.values() if m.git_dirty)
    if dirty:
        return fail(head, f"{len(dirty)} cell(s) written from a dirty tree, first {dirty[0]}")
    shas = sorted({m.git_sha for m in manifests.values()})
    tag_commit = _tag_commit(repo, tag)
    if not tag_commit:
        return fail(head, f"no {tag} tag in {repo} to date the cells against")
    early = [s for s in shas if not _at_or_after(repo, tag_commit, s)]
    head += f"; written at {', '.join(_short(s) for s in shas)} (clean"
    if early:
        why = (
            f"commit {_short(early[0])} is not at or after {tag} ({_short(tag_commit)}), "
            "or is unknown here"
        )
        return fail(head + ")", why)
    head += f", at or after {tag} {_short(tag_commit)})"
    # The spot-check: the log beside the record, then any replays asked for here.
    by_id = {c.run_id: c for c in cells}
    log = read_replay_log(runs_root / REPLAY_LOG)
    # A PPO cell trained on another platform does not retrain to the same policy here.
    elsewhere = [
        c for c in cells if c.config.train is not None and platform_differs(runs_root / c.run_id)
    ]
    comparable = [c for c in cells if c not in elsewhere]
    fresh = (
        [replay(runs_root / c.run_id) for c in replay_sample(comparable, replays)]
        if replays
        else []
    )
    within = sum(r.within and not r.matches for r in fresh)
    checked = [r for r in log if r.run_id in by_id] + [
        Replayed(r.run_id, r.reproduced, r.reason, "here") for r in fresh
    ]
    for r in checked:
        if not r.identical:
            return fail(head, f"the replay ({r.source}) of {r.run_id} differs: {r.reason}")
        if r.source == "log":
            m = _LOG_SHA.search(r.reason)
            now = file_sha256(runs_root / r.run_id / RUN_FILENAME)
            if m is None or not now.startswith(m[1]):
                why = (
                    f"{REPLAY_LOG} says {r.run_id} reproduced a run.json with sha256 "
                    f"{m[1] if m else '(none named)'}, the file on disk is {now[:16]}: stale"
                )
                return fail(head, why)
    if not checked:
        evidence = (
            f"{head}; no replay spot-check ({REPLAY_LOG} absent or naming no grid cell, "
            "--replay not given)"
        )
        return GateItem(name, DOMINICK, SWEEP_CRITERION, evidence, passed=True)
    arms = sorted({by_id[r.run_id].arm for r in checked})
    counts = ", ".join(f"{sum(by_id[r.run_id].arm == a for r in checked)} {a}" for a in arms)
    sources = [
        f"{sum(r.source == s for r in checked)} {label}"
        for s, label in (("log", f"in {REPLAY_LOG}"), ("here", "replayed here"))
        if any(r.source == s for r in checked)
    ]
    if within:
        sources.append(
            f"{within} of those within {rp.SELLER_REL_TOL:g}, written on another platform"
        )
    verdict = "reproduced" if within else "bit-identical"
    evidence = (
        f"{head}; replay spot-check: {len(checked)} cell(s) {verdict} ({counts}; "
        f"{', '.join(sources)})"
    )
    uncovered = [a for a in grid.arms if a not in arms]
    if uncovered:
        evidence += f"; no {', '.join(uncovered)} cell replayed"
    if replays and elsewhere:
        evidence += (
            f" ({len(elsewhere)} PPO cell(s) trained on another platform are not comparable "
            "here: PPO training is not reproducible across platforms)"
        )
    return GateItem(name, DOMINICK, SWEEP_CRITERION, evidence, passed=True)


# -- 2. the leak detection floor ------------------------------------------------------ #

FLOOR_CRITERION = (
    f"leaks.leak_floor on the M1 design (spot leaks eps = {_premiums(FLOOR_GRID)} of tomorrow's "
    f"return planted by PoisonedChainProvider, {FLOOR_SEEDS} seeds, Bonferroni alpha 0.01) is "
    "measured by this report: some leak is caught, the caught set is every eps at or above the "
    "smallest caught (the floor), the clean world (eps = 0) is quiet at the same seeds, and the "
    f"floor equals the one docs/{M1_SIDECAR} records (the detector is deterministic in the seed, "
    "so a different floor means the detector or the world moved since M1)"
)


def m1_floor(docs_dir: Path) -> tuple[float | None, str]:
    """The floor the M1 sidecar records, or ``(None, why not)``."""
    data, why = read_json(docs_dir / M1_SIDECAR)
    if data is None:
        return None, why
    item = next((i for i in data.get("items") or () if i.get("name") == M1_FLOOR_ITEM), None)
    m = _M1_FLOOR.search(str((item or {}).get("evidence", "")))
    if m is None:
        return None, f"docs/{M1_SIDECAR} records no '{M1_FLOOR_ITEM}' item stating a floor"
    return float(m[1]), str(data.get("report_date", "?"))


def floor_item(
    docs_dir: Path,
    *,
    seeds: int = FLOOR_SEEDS,
    make_env: Callable[[float], SpreadEnv] = default_env,
    grid: Sequence[float] = FLOOR_GRID,
) -> GateItem:
    name = "Leak detection floor measured"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, DOMINICK, FLOOR_CRITERION, evidence, why)

    start = time.perf_counter()
    report = leak_floor(make_env, grid, range(seeds))
    clean = detect_leak(partial(make_env, 0.0), range(seeds))
    seconds = time.perf_counter() - start
    caught = ", ".join(
        f"{e:g} {'caught' if f else 'missed'}"
        for e, f in zip(report.eps_grid, report.flagged, strict=True)
    )
    evidence = (
        f"leak_floor over {seeds} seeds ({clean.n_pairs:,} pairs a level, {seconds:.1f}s): {caught}"
    )
    if clean.flagged:
        why = (
            f"the clean world is flagged at these seeds: {clean.worst_feature} p = "
            f"{clean.p_value:.2g} under {clean.threshold:.2g}"
        )
        return fail(evidence, why)
    evidence += (
        f"; clean world quiet (worst {clean.worst_feature} p = {clean.p_value:.2g}, "
        f"threshold {clean.threshold:.2g})"
    )
    if report.floor is None:
        return fail(evidence, "no planted leak was caught: the detector has no floor on this grid")
    above = [f for e, f in zip(report.eps_grid, report.flagged, strict=True) if e >= report.floor]
    if not all(above):
        missed = [
            e
            for e, f in zip(report.eps_grid, report.flagged, strict=True)
            if e >= report.floor and not f
        ]
        return fail(
            evidence,
            f"eps = {_premiums(missed)} is missed above the smallest caught "
            f"{report.floor:g}: not a threshold",
        )
    below = [e for e in report.eps_grid if e < report.floor]
    evidence += f"; floor eps = {report.floor:g}" + (
        f" (eps = {max(below):g} missed)" if below else " (the smallest on the grid)"
    )
    recorded, when = m1_floor(docs_dir)
    if recorded is None:
        return fail(evidence, when)
    if recorded != report.floor:
        return fail(
            evidence,
            f"docs/{M1_SIDECAR} ({when}) records floor eps = {recorded:g}, measured "
            f"{report.floor:g}",
        )
    evidence += f", the floor docs/{M1_SIDECAR} records ({when})"
    return GateItem(name, DOMINICK, FLOOR_CRITERION, evidence, passed=True)


# -- 3. the poisoned-environment demo ------------------------------------------------- #

DEMO_CRITERION = (
    f"{SERVER_MODULE}.{APP_FACTORY} builds the dashboard's app, which routes a leak demo "
    f"(an API path holding 'leak'); posted to in-process (FastAPI's TestClient) at eps = "
    f"{DEMO_EPS:g} over {DEMO_SEEDS} seeds it answers 200 and fails the poisoned variant on "
    "purpose (the detector flags it and resampling the future moves its past) while passing the "
    "clean one (not flagged, its past unchanged), its caught flag true; and the built site "
    f"(web/out) has its index.html, the leak page ({' or '.join(LEAK_PAGES)}) and calls the route"
)


def _g(x: Any) -> str:
    return f"{x:.2g}" if isinstance(x, int | float) and not isinstance(x, bool) else str(x)


def _side(payload: dict[str, Any], name: str) -> dict[str, Any]:
    side = payload.get(name)
    return side if isinstance(side, dict) else {}


def demo_item(runs_root: Path, docs_dir: Path, web_dir: Path) -> GateItem:
    name = "Poisoned-environment demo live"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, ILAN, DEMO_CRITERION, evidence, why)

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
        app = factory(settings(runs_root, docs_dir, docs_dir, web_dir, docs_dir / "README.md"))
        routes = [p for p in _api_paths(app) if DEMO_ROUTE.search(p)]
        if not routes:
            return fail(
                f"{len(_api_paths(app))} API routes", "the app routes no leak demo endpoint"
            )
        path = next((p for p in routes if "demo" in p), routes[0])
        start = time.perf_counter()
        response = client_class(app, base_url="http://127.0.0.1").post(
            path, json={"eps": DEMO_EPS, "n_seeds": DEMO_SEEDS}
        )
        seconds = time.perf_counter() - start
    except Exception as exc:  # a broken server is a finding, not a crash of the report
        return fail(f"{SERVER_MODULE}.{APP_FACTORY} was called", f"it raised {exc!r}")
    asked = f"POST {path} (eps {DEMO_EPS:g}, {DEMO_SEEDS} seeds)"
    if response.status_code != 200:
        return fail(
            asked, f"answered {response.status_code}: {_one_line(Exception(response.text[:200]))}"
        )
    try:
        payload = response.json()
    except ValueError:
        return fail(asked, "answered 200 with a body that is not JSON")
    poisoned, clean = _side(payload, "poisoned"), _side(payload, "clean")
    evidence = (
        f"{asked} answered 200 in {seconds:.1f}s: poisoned flagged {poisoned.get('flagged')} "
        f"(worst {poisoned.get('worst_feature')}, p = {_g(poisoned.get('p_value'))}), past moved "
        f"{poisoned.get('past_unchanged') is False}; clean flagged {clean.get('flagged')} (worst "
        f"p = {_g(clean.get('p_value'))} against {_g(clean.get('threshold'))}), past moved "
        f"{clean.get('past_unchanged') is False}; caught {payload.get('caught')}"
    )
    if poisoned.get("flagged") is not True:
        return fail(evidence, "the detector does not flag the poisoned variant")
    if poisoned.get("past_unchanged") is not False:
        return fail(evidence, "the perturbation check does not catch the poisoned variant")
    if clean.get("flagged") is not False or clean.get("past_unchanged") is not True:
        return fail(evidence, "the clean variant does not pass: flagged, or its past moved")
    if payload.get("caught") is not True:
        return fail(evidence, "the demo does not report the leak caught")
    if not (web_dir / "index.html").is_file():
        return fail(evidence, f"no built site at {web_dir} (run make web)")
    page = next((p for p in LEAK_PAGES if (web_dir / p).is_file()), None)
    if page is None:
        return fail(
            evidence, f"the built site at {web_dir} has no leak page ({' or '.join(LEAK_PAGES)})"
        )
    callers = _callers(web_dir, path)
    if not callers:
        return fail(evidence, f"the built site at {web_dir} never calls {path}")
    shown = ", ".join(callers[:3]) + (", ..." if len(callers) > 3 else "")
    evidence += f"; the built site at {web_dir} has {page} and calls {path} from {shown}"
    return GateItem(name, ILAN, DEMO_CRITERION, evidence, passed=True)


# -- 4. the partner reproduction ------------------------------------------------------ #

PARTNER_CRITERION = (
    f"docs/{rp.RECORD_FILE}, written by scripts/reproduce.py on the partner team's clone, "
    f"reads as schema {rp.SCHEMA_VERSION} and names a commit of this repository at or after "
    f"the {CUTOFF_TAG} tag, checked out clean; every step ({', '.join(rp.STEPS)}) is "
    "present and passed, and any other recorded (the PPO cell) passed too; every regenerated "
    "cell is a cell of the A7 grid at its place and within the stated tolerance (a fixed-delta "
    f"seller relative {rp.SELLER_REL_TOL:g}, {rp.ABS_TOL:g} dollars near zero, counts and "
    "identifiers exact; a PPO cell statistically, since PPO training is not bit-reproducible "
    f"across platforms: its mean within {rp.PPO_K_SE:g} standard errors of the reference's, "
    "trading, no refused open and the residual share under the M4 bound), at least one seller "
    f"cell; and it was compared against docs/{rp.EXPECTED_FILE} as committed at that commit "
    "(its sha256 with git's LF line endings), whose every run.json digest is the sweep's under "
    "the runs root. Bytes are reported, not required: the sweep ran on Windows and a last-bit "
    "libm difference reaches the printed digits"
)


def _expected_at(repo: Path, commit: str) -> tuple[bytes | None, str]:
    """The expected file's bytes as committed at ``commit``, or ``(None, why not)``."""
    done = subprocess.run(
        ["git", "show", f"{commit}:docs/{rp.EXPECTED_FILE}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        return None, f"docs/{rp.EXPECTED_FILE} is not in commit {_short(commit)}"
    return done.stdout, ""


def expected_problem(expected: rp.SweepExpected, runs_root: Path, grid: GridConfig) -> str:
    """Why the expected file is not the sweep under ``runs_root``; "" when it is."""
    want = config_hash(grid)
    if expected.grid_hash != want:
        return f"docs/{rp.EXPECTED_FILE} is of grid {expected.grid_hash}, not {want}"
    if not (runs_root / grid_filename(grid)).is_file():
        return f"no sweep at {runs_root} ({grid_filename(grid)}) to tie docs/{rp.EXPECTED_FILE} to"
    ids = {c.run_id for c in grid_cells(grid)}
    if {c.run_id for c in expected.cells} != ids:
        return f"docs/{rp.EXPECTED_FILE} does not list exactly the grid's {len(ids)} cells"
    for c in expected.cells:
        path = runs_root / c.run_id / RUN_FILENAME
        if not path.is_file() or file_sha256(path) != c.run_json_sha256:
            return (
                f"{c.run_id}'s run.json under {runs_root} is not the one "
                f"docs/{rp.EXPECTED_FILE} records"
            )
    return ""


def partner_item(
    docs_dir: Path,
    runs_root: Path,
    repo: Path,
    *,
    grid: GridConfig = CONFIRMATORY_GRID,
    tag: str = CUTOFF_TAG,
) -> GateItem:
    name = "Partner team reproduces the gates"

    def fail(evidence: str, why: str) -> GateItem:
        return _not_cleared(name, TEAM, PARTNER_CRITERION, evidence, why)

    path = docs_dir / rp.RECORD_FILE
    if not path.is_file():
        why = (
            f"no {path}: the partner team has not sent back what python scripts/reproduce.py wrote"
        )
        return fail("not received", why)
    try:
        record = rp.PartnerRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fail(f"{path} found", f"{path} is refused: {_one_line(exc)}")
    if record.schema_version != rp.SCHEMA_VERSION:
        return fail(f"{path} found", f"schema {record.schema_version}, not {rp.SCHEMA_VERSION}")
    who = record.team or "an unnamed team"
    head = (
        f"{path.name} from {who} on {record.platform.get('platform', '?')}, Python "
        f"{record.platform.get('python', '?')}, generated {record.generated}, at "
        f"{_short(record.commit)}"
    )
    if _git(repo, "cat-file", "-e", f"{record.commit}^{{commit}}").returncode != 0:
        return fail(head, f"commit {record.commit} is not in this repository")
    tag_commit = _tag_commit(repo, tag)
    if not tag_commit:
        return fail(head, f"no {tag} tag in {repo}")
    if not _at_or_after(repo, tag_commit, record.commit):
        return fail(
            head, f"{_short(record.commit)} is not at or after {tag} ({_short(tag_commit)})"
        )
    if record.dirty:
        return fail(head, "the checkout had uncommitted changes: not the repository as cloned")
    head += f" (clean, at or after {tag} {_short(tag_commit)})"
    if record.missing_steps:
        return fail(head, f"step(s) not run: {', '.join(record.missing_steps)}")
    timings = ", ".join(f"{s.name} {s.seconds:.0f}s" for s in record.steps)
    head += (
        f"; {sum(s.passed for s in record.steps)} of {len(record.steps)} steps passed ({timings})"
    )
    if record.failed_steps:
        s = record.failed_steps[0]
        return fail(head, f"{s.name} failed: {s.detail or f'exit {s.returncode}'}")
    if record.tolerance != rp.tolerances():
        return fail(head, f"compared at tolerance {record.tolerance}, not {rp.tolerances()}")
    places = {c.run_id: (c.premium, c.arm, c.block) for c in grid_cells(grid)}
    for c in record.cells:
        if places.get(c.run_id) != (c.premium, c.arm, c.block):
            return fail(
                head, f"{c.run_id} is not the A7 cell {c.arm} at {c.premium:g} block {c.block}"
            )
        if c.arm == rp.PPO and (c.check != "statistical" or c.k_se != rp.PPO_K_SE):
            return fail(head, f"{c.run_id} was not compared statistically at {rp.PPO_K_SE:g} se")
        if c.arm != rp.PPO and (c.check != "numeric" or c.rel_tol != rp.SELLER_REL_TOL):
            return fail(
                head, f"{c.run_id} was compared at {c.rel_tol:g}, not {rp.SELLER_REL_TOL:g}"
            )
        if not c.within_tolerance:
            return fail(head, f"{c.run_id} is outside the tolerance: {c.first_difference}")
    sellers = [c for c in record.cells if c.arm == rp.SELLER]
    if not sellers:
        return fail(head, "no fixed-delta seller cell was regenerated")
    raw, why = _expected_at(repo, record.commit)
    if raw is None:
        return fail(head, why)
    if rp.text_sha256(raw) != record.expected_sha256:
        why = (
            f"compared against an expected file with sha256 {record.expected_sha256[:16]}, "
            f"commit {_short(record.commit)} has {rp.text_sha256(raw)[:16]}"
        )
        return fail(head, why)
    try:
        expected = rp.SweepExpected.model_validate_json(raw)
    except ValueError as exc:
        return fail(
            head, f"docs/{rp.EXPECTED_FILE} at {_short(record.commit)} is refused: {_one_line(exc)}"
        )
    if why := expected_problem(expected, runs_root, grid):
        return fail(head, why)
    worst = max(c.max_rel_diff for c in record.cells)
    identical = sum(c.byte_identical for c in record.cells)
    arms = ", ".join(
        f"{sum(c.arm == a for c in record.cells)} {a}"
        for a in grid.arms
        if any(c.arm == a for c in record.cells)
    )
    evidence = (
        f"{head}; {len(record.cells)} sweep cell(s) regenerated ({arms}; premiums "
        f"{_premiums(sorted({c.premium for c in record.cells}))}), every one within tolerance, "
        f"largest relative difference {worst:.2g}, {identical} byte-identical; against "
        f"docs/{rp.EXPECTED_FILE} sha256 {record.expected_sha256[:16]}, whose "
        f"{len(expected.cells)} run.json digests are the sweep's under {runs_root}"
    )
    return GateItem(name, TEAM, PARTNER_CRITERION, evidence, passed=True)
