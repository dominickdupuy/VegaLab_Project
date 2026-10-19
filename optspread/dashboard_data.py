"""What the dashboard reads: the run index, the gate board and the live leak demo.

Lane: Ilan Danial (machinery).

The dashboard only shows what the harness recorded, so every number on it traces
back to a run. That is a property of this module rather than of the pages: every
loader here reads a file the harness or a gate report wrote, and nothing here
computes a result that is not already on disk, with one deliberate exception, the
leak demo, which runs the detector live because watching it fire is the point.

Nothing here needs the web stack, so it is tested and covered on the plain ``dev``
install the CI runs. The API (``webapi``, ``server``) and the site in ``web/`` are
thin over it.

A run directory that does not read cleanly is listed with what is wrong rather
than dropped. A browser that silently skipped broken runs would make the results
folder look healthier than it is, which is the failure the manifest gate exists to
prevent.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from optspread.artifacts import RUN_FILENAME, RunRecord, SchemaVersionError, read_run
from optspread.config import MarketConfig
from optspread.leaks import (
    PASS_CRITERION,
    LeakReport,
    PoisonedChainProvider,
    default_env,
    detect_leak,
    perturbation_check,
)
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256
from optspread.market import SyntheticChainProvider
from optspread.sweep import SWEEP_FILENAME
from optspread.types import ChainProvider

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ENV = "OPTSPREAD_RUNS"
DOCS_ENV = "OPTSPREAD_DOCS"


def runs_root(env: Mapping[str, str] | None = None) -> Path:
    """``$OPTSPREAD_RUNS`` if set, else ``runs/`` at the repository root."""
    value = (os.environ if env is None else env).get(RUNS_ENV, "")
    return Path(value).expanduser() if value else REPO_ROOT / "runs"


def docs_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$OPTSPREAD_DOCS`` if set, else ``docs/`` at the repository root."""
    value = (os.environ if env is None else env).get(DOCS_ENV, "")
    return Path(value).expanduser() if value else REPO_ROOT / "docs"


# -- the run index ------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class RunRow:
    """One run directory as the browser lists it: provenance, headline, and health.

    ``problems`` is empty for a run that reads cleanly. A row with problems is still
    listed, so a broken run is visible rather than missing. A run is OK only when the
    manifest records ``run.json`` among its artifacts and every recorded digest
    matches; one that records nothing, or not the results file, is UNVERIFIED, since
    nothing ties the numbers shown to the numbers the manifest was written over.
    """

    run_id: str
    path: Path
    config_hash: str
    seed: int | None
    policy: str
    premium: float | None
    git_sha: str
    git_dirty: bool | None
    created_at: str
    n_episodes: int | None
    mean_pnl: float | None
    se_pnl: float | None
    t_stat: float | None
    mean_cost: float | None
    mean_trades: float | None
    sweep: str = ""  # the parent sweep's directory name, when the run is a sweep cell
    digests_ok: bool | None = None  # None: the manifest records no artifacts
    recorded: tuple[str, ...] = ()  # the artifact paths the manifest records
    problems: tuple[str, ...] = ()

    @property
    def commit(self) -> str:
        return self.git_sha[:8]

    @property
    def status(self) -> str:
        if self.problems:
            return "INVALID"
        if self.digests_ok is False:
            return "DIGEST MISMATCH"
        if not self.digests_ok or RUN_FILENAME not in self.recorded:
            return "UNVERIFIED"
        return "OK"


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _first_line(exc: Exception) -> str:
    text = str(exc).strip()
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        where = ".".join(str(p) for p in errors[0]["loc"]) if errors else ""
        return f"{where}: {errors[0]['msg']}" if errors else text
    return text.splitlines()[0] if text else type(exc).__name__


@dataclass(frozen=True, slots=True)
class ArtifactCheck:
    path: str
    recorded: str
    actual: str  # empty when the file is missing

    @property
    def ok(self) -> bool:
        return bool(self.actual) and self.actual == self.recorded


def check_artifacts(run_dir: Path, artifacts: Mapping[str, str]) -> tuple[ArtifactCheck, ...]:
    """Recompute every digest the manifest recorded and compare."""
    checks: list[ArtifactCheck] = []
    for rel, recorded in sorted(artifacts.items()):
        target = run_dir / rel
        actual = file_sha256(target) if target.is_file() else ""
        checks.append(ArtifactCheck(rel, recorded, actual))
    return tuple(checks)


def load_run_row(run_dir: Path) -> RunRow:
    """Read one run directory without trusting it: every failure becomes a problem."""
    problems: list[str] = []
    try:
        raw: Any = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    except (OSError, ValueError) as exc:
        raw = {}
        problems.append(f"manifest.json unreadable: {_first_line(exc)}")
    if raw:
        try:
            RunManifest.model_validate(raw).validate_complete()
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            problems.append(f"manifest: {_first_line(exc)}")

    record: RunRecord | None = None
    if (run_dir / RUN_FILENAME).exists():
        try:
            record = read_run(run_dir)
        except (OSError, ValueError) as exc:  # includes SchemaVersionError
            problems.append(f"run.json: {_first_line(exc)}")
    else:
        problems.append("run.json is missing")

    results = _get(raw, "results") or {}
    artifacts = _get(raw, "artifacts")
    checks = (
        check_artifacts(run_dir, {str(k): str(v) for k, v in artifacts.items()})
        if isinstance(artifacts, dict)
        else ()
    )
    seed = _as_int(_get(raw, "config", "seed"))
    return RunRow(
        run_id=str(_get(raw, "run_id") or (record.run_id if record else run_dir.name)),
        path=run_dir,
        config_hash=str(_get(raw, "config_hash") or (record.config_hash if record else "")),
        seed=seed if seed is not None else (record.seed if record else None),
        policy=str(_get(raw, "config", "policy") or (record.policy if record else "")),
        premium=_as_float(_get(raw, "config", "market", "iv_premium_vol_pts")),
        git_sha=str(_get(raw, "git_sha") or ""),
        git_dirty=_get(raw, "git_dirty") if isinstance(_get(raw, "git_dirty"), bool) else None,
        created_at=str(_get(raw, "created_at") or ""),
        n_episodes=record.n_episodes if record else _as_int(_get(raw, "config", "n_episodes")),
        mean_pnl=record.mean_pnl if record else _as_float(_get(results, "mean_pnl")),
        se_pnl=record.se_pnl if record else _as_float(_get(results, "se_pnl")),
        t_stat=record.t_stat if record else _as_float(_get(results, "t_stat")),
        mean_cost=record.mean_cost if record else _as_float(_get(results, "mean_cost")),
        mean_trades=record.mean_trades if record else _as_float(_get(results, "mean_trades")),
        sweep=run_dir.parent.name if (run_dir.parent / SWEEP_FILENAME).exists() else "",
        digests_ok=all(c.ok for c in checks) if checks else None,
        recorded=tuple(c.path for c in checks),
        problems=tuple(problems),
    )


def is_matrix_manifest(run_dir: Path) -> bool:
    """The pilot's own manifest: at the pilot root, hashing ``.npy`` matrices, no ``run.json``.

    ``orchestrate.write_pilot_matrices`` writes it so the matrices can be checked; it
    describes no run, so the run index leaves it out rather than listing it INVALID.
    """
    if (run_dir / RUN_FILENAME).exists():
        return False
    try:
        artifacts: Any = json.loads((run_dir / MANIFEST_FILENAME).read_text()).get("artifacts")
    except (OSError, ValueError, AttributeError):
        return False
    return (
        isinstance(artifacts, dict)
        and bool(artifacts)
        and all(str(rel).endswith(".npy") for rel in artifacts)
    )


def scan_runs(root: Path) -> tuple[RunRow, ...]:
    """Every run directory under ``root`` holding a ``manifest.json``, newest first.

    The pilot's matrix manifest is not a run and is skipped (``is_matrix_manifest``).
    """
    if not root.is_dir():
        return ()
    found = (m.parent for m in root.rglob(MANIFEST_FILENAME) if m.is_file())
    rows = [load_run_row(d) for d in found if not is_matrix_manifest(d)]
    return tuple(sorted(rows, key=lambda r: (r.created_at, r.run_id), reverse=True))


@dataclass(frozen=True, slots=True)
class RunFacets:
    """The distinct values each filter can take, in display order."""

    config_hashes: tuple[str, ...]
    seeds: tuple[int, ...]
    policies: tuple[str, ...]
    premiums: tuple[float, ...]
    commits: tuple[str, ...]
    sweeps: tuple[str, ...]


def run_facets(rows: Iterable[RunRow]) -> RunFacets:
    rows = tuple(rows)
    return RunFacets(
        config_hashes=tuple(sorted({r.config_hash for r in rows if r.config_hash})),
        seeds=tuple(sorted({r.seed for r in rows if r.seed is not None})),
        policies=tuple(sorted({r.policy for r in rows if r.policy})),
        premiums=tuple(sorted({r.premium for r in rows if r.premium is not None})),
        commits=tuple(sorted({r.commit for r in rows if r.commit})),
        sweeps=tuple(sorted({r.sweep for r in rows if r.sweep})),
    )


def filter_runs(
    rows: Iterable[RunRow],
    *,
    config_hashes: Sequence[str] = (),
    seeds: Sequence[int] = (),
    policies: Sequence[str] = (),
    premiums: Sequence[float] = (),
    commits: Sequence[str] = (),
    sweeps: Sequence[str] = (),
    text: str = "",
) -> tuple[RunRow, ...]:
    """Rows matching every non-empty filter. An empty filter does not filter.

    ``text`` is a case-insensitive substring match on the run id, config hash and
    commit, for pasting a hash from a gate report or a figure caption.
    """
    needle = text.strip().lower()
    out: list[RunRow] = []
    for r in rows:
        if config_hashes and r.config_hash not in config_hashes:
            continue
        if seeds and r.seed not in seeds:
            continue
        if policies and r.policy not in policies:
            continue
        if premiums and r.premium not in premiums:
            continue
        if commits and r.commit not in commits:
            continue
        if sweeps and r.sweep not in sweeps:
            continue
        if needle and not any(needle in s.lower() for s in (r.run_id, r.config_hash, r.git_sha)):
            continue
        out.append(r)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class RunDetail:
    """One run opened: the manifest as written, the record, and per-episode arrays."""

    row: RunRow
    manifest: dict[str, Any]
    record: RunRecord | None
    pnl: NDArray[np.float64]
    cost: NDArray[np.float64]
    trades: NDArray[np.float64]
    artifacts: tuple[ArtifactCheck, ...]


def load_run_detail(run_dir: Path) -> RunDetail:
    row = load_run_row(run_dir)
    try:
        manifest: Any = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    except (OSError, ValueError):
        manifest = {}
    manifest = manifest if isinstance(manifest, dict) else {}
    try:
        record: RunRecord | None = read_run(run_dir)
    except (OSError, ValueError, SchemaVersionError):
        record = None
    episodes = record.episodes if record else ()
    artifacts = _get(manifest, "artifacts")
    return RunDetail(
        row=row,
        manifest=manifest,
        record=record,
        pnl=np.asarray([e.pnl for e in episodes], dtype=np.float64),
        cost=np.asarray([e.cost for e in episodes], dtype=np.float64),
        trades=np.asarray([e.n_trades for e in episodes], dtype=np.float64),
        artifacts=check_artifacts(run_dir, {str(k): str(v) for k, v in artifacts.items()})
        if isinstance(artifacts, dict)
        else (),
    )


@dataclass(frozen=True, slots=True)
class RiskStats:
    """Per-episode risk figures for one run, computed from the episodes in its ``run.json``.

    Every figure is over episodes, the grain the run records. ``max_drawdown`` is the
    worst peak-to-trough of cumulative P&L with the episodes taken in seed order: it
    describes the sample, not a trading account, because the episodes are independent
    paths rather than consecutive ones. A figure the sample cannot support is None.
    """

    n: int
    mean: float | None
    se: float | None
    sortino: float | None  # mean over downside deviation against zero
    cvar5: float | None  # mean of the worst 5% of episodes (at least one)
    max_drawdown: float | None
    hit_rate: float | None  # share of episodes with positive P&L
    turnover: float | None  # mean trades per episode


def risk_stats(pnl: NDArray[np.float64], trades: NDArray[np.float64] | None = None) -> RiskStats:
    n = len(pnl)
    if n == 0:
        return RiskStats(0, None, None, None, None, None, None, None)
    downside = float(np.sqrt(np.mean(np.minimum(pnl, 0.0) ** 2)))
    worst = np.sort(pnl)[: max(1, int(np.ceil(0.05 * n)))]
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    return RiskStats(
        n=n,
        mean=float(pnl.mean()),
        se=float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        sortino=float(pnl.mean() / downside) if downside > 0.0 else None,
        cvar5=float(worst.mean()),
        max_drawdown=float((np.maximum.accumulate(equity) - equity).max()),
        hit_rate=float((pnl > 0.0).mean()),
        turnover=float(trades.mean()) if trades is not None and len(trades) else None,
    )


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    else:
        out[prefix] = value


def flatten_config(config: Any) -> dict[str, Any]:
    """``{"market.sigma": 0.2, ...}``: a config as dotted keys, for a diff a person can read."""
    out: dict[str, Any] = {}
    _flatten("", config if isinstance(config, dict) else {}, out)
    return out


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Two runs side by side, and their paired difference on the seeds they share.

    Runs on the same seeds saw the same paths, so ``b - a`` per seed is a paired
    difference and its standard error is the honest one. Unshared seeds are counted,
    not used.
    """

    a: RunDetail
    b: RunDetail
    config_diff: tuple[tuple[str, Any, Any], ...]  # (key, a's value, b's value)
    shared_seeds: tuple[int, ...]
    diff_mean: float | None
    diff_se: float | None

    @property
    def diff_t(self) -> float | None:
        if self.diff_mean is None or not self.diff_se:
            return None
        return self.diff_mean / self.diff_se


def compare_runs(a: RunDetail, b: RunDetail) -> RunComparison:
    fa, fb = flatten_config(a.manifest.get("config")), flatten_config(b.manifest.get("config"))
    diff = tuple(
        (k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)
    )
    pa = {e.seed: e.pnl for e in a.record.episodes} if a.record else {}
    pb = {e.seed: e.pnl for e in b.record.episodes} if b.record else {}
    shared = tuple(sorted(set(pa) & set(pb)))
    d = np.asarray([pb[s] - pa[s] for s in shared], dtype=np.float64)
    return RunComparison(
        a=a,
        b=b,
        config_diff=diff,
        shared_seeds=shared,
        diff_mean=float(d.mean()) if len(d) else None,
        diff_se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
    )


@dataclass(frozen=True, slots=True)
class CurvePoint:
    policy: str
    premium: float
    mean_pnl: float
    se_pnl: float
    run_id: str


def premium_curve(rows: Iterable[RunRow]) -> dict[str, tuple[CurvePoint, ...]]:
    """Mean P&L along the planted-premium axis, per arm, from runs that read cleanly.

    Only meaningful within one sweep (one base config, paired seeds); the caller
    passes the rows of a single sweep. Within an arm the points are in premium order.
    """
    by_arm: dict[str, list[CurvePoint]] = {}
    for r in rows:
        if r.status != "OK" or r.premium is None or r.mean_pnl is None or r.se_pnl is None:
            continue
        by_arm.setdefault(r.policy, []).append(
            CurvePoint(r.policy, r.premium, r.mean_pnl, r.se_pnl, r.run_id)
        )
    return {arm: tuple(sorted(pts, key=lambda p: p.premium)) for arm, pts in sorted(by_arm.items())}


# -- the gate board ----------------------------------------------------------- #

GATE_STEM = "_GATE_REPORT"
VERDICTS = ("PASS", "FAIL", "NO CRITERION", "NOT CLEARED")
RUN_ID_PATTERN = re.compile(r"\b[0-9a-f]{16}-s\d+\b")


class GateReportFormatError(ValueError):
    """A gate report this reader cannot parse. Raised, never guessed around."""


@dataclass(frozen=True, slots=True)
class GateRow:
    name: str
    owner: str
    verdict: str
    criterion: str
    evidence: str
    failed_first: str = ""

    @property
    def cited_runs(self) -> tuple[str, ...]:
        """Run ids quoted in the evidence, so the board can link to the runs behind it."""
        return tuple(RUN_ID_PATTERN.findall(self.evidence))


@dataclass(frozen=True, slots=True)
class GateBoard:
    """One gate report as the board shows it, with its verdict re-derived from its items.

    ``stated_verdict`` is whatever the file says. ``verdict`` is what the items support:
    PASS only when there are items and every one is a PASS with a criterion and
    evidence written down, NOT CLEARED otherwise, which is ``GateReport.passed`` read
    back. The board shows the derived one, so a hand-edited verdict line or an empty
    item list cannot put a PASS on the page; ``verdict_mismatch`` says the two differ.
    """

    milestone: str
    report_date: str
    stated_verdict: str
    items: tuple[GateRow, ...]
    source: Path
    from_json: bool

    @property
    def verdict(self) -> str:
        cleared = bool(self.items) and all(
            i.verdict == "PASS" and i.criterion.strip() and i.evidence.strip() for i in self.items
        )
        return "PASS" if cleared else "NOT CLEARED"

    @property
    def verdict_mismatch(self) -> bool:
        return self.stated_verdict != self.verdict

    @property
    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for item in self.items:
            out[item.verdict] = out.get(item.verdict, 0) + 1
        return out

    @property
    def not_passing(self) -> tuple[GateRow, ...]:
        return tuple(i for i in self.items if i.verdict != "PASS")


def gate_milestones(docs: Path) -> tuple[str, ...]:
    """Milestones with a report on disk, JSON or markdown, in milestone order."""
    if not docs.is_dir():
        return ()
    names = {
        p.name.split(GATE_STEM)[0]
        for p in docs.glob(f"M*{GATE_STEM}.*")
        if p.suffix in (".json", ".md")
    }

    def order(name: str) -> tuple[int, str]:
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 0, name)

    return tuple(sorted(names, key=order))


def parse_gate_json(text: str, source: Path) -> GateBoard:
    """The JSON sidecar ``optspread.gatereport`` writes beside the markdown."""
    try:
        raw: Any = json.loads(text)
        items = tuple(
            GateRow(
                name=str(i["name"]),
                owner=str(i.get("owner", "")),
                verdict=str(i["verdict"]),
                criterion=str(i.get("criterion", "")),
                evidence=str(i.get("evidence", "")),
                failed_first=str(i.get("failed_first") or ""),
            )
            for i in raw["items"]
        )
        return GateBoard(
            milestone=str(raw["milestone"]),
            report_date=str(raw.get("report_date", "")),
            stated_verdict=str(raw["verdict"]),
            items=items,
            source=source,
            from_json=True,
        )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise GateReportFormatError(f"{source.name}: {exc}") from exc


_DATE_LINE = re.compile(r"^Report date:\s*(\S+)")
_VERDICT_LINE = re.compile(r"^\*\*Verdict:\s*([A-Z ]+?)\*\*")
_TITLE_LINE = re.compile(r"^#\s+(\S+)\s+gate report", re.IGNORECASE)


def _split_row(line: str) -> list[str]:
    # Cells are separated by " | " as ``GateReport.render`` writes them. A bare pipe
    # inside a cell, such as "|t| <= 3", is not a separator.
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [c.strip() for c in inner.split(" | ")]


def parse_gate_markdown(text: str, source: Path) -> GateBoard:
    """The rendered markdown table, for reports written before the JSON sidecar."""
    milestone, report_date, verdict = source.name.split(GATE_STEM)[0], "", ""
    items: list[GateRow] = []
    header_seen = False
    for line in text.splitlines():
        if m := _TITLE_LINE.match(line):
            milestone = m.group(1)
        elif m := _DATE_LINE.match(line):
            report_date = m.group(1)
        elif m := _VERDICT_LINE.match(line):
            verdict = m.group(1).strip()
        elif line.startswith("| Item |"):
            header_seen = True
        elif header_seen and line.startswith("|") and not line.startswith("| ---"):
            cells = _split_row(line)
            if len(cells) < 6:
                raise GateReportFormatError(f"{source.name}: a table row has {len(cells)} cells")
            # name, owner and verdict lead; evidence and failed-first close the row; a
            # stray separator can only have come from the criterion, so it is rejoined.
            name, owner, item_verdict = cells[0], cells[1], cells[2]
            criterion = " | ".join(cells[3:-2])
            failed = "" if cells[-1] in ("--", "") else cells[-1]
            items.append(GateRow(name, owner, item_verdict, criterion, cells[-2], failed))
        elif header_seen and not line.startswith("|"):
            header_seen = False
    if not items or not verdict:
        raise GateReportFormatError(f"{source.name}: no verdict line or no item table")
    return GateBoard(milestone, report_date, verdict, tuple(items), source, from_json=False)


def load_gate_board(docs: Path, milestone: str) -> GateBoard | None:
    """The JSON sidecar when present, else the markdown table; None when neither exists."""
    sidecar = docs / f"{milestone}{GATE_STEM}.json"
    if sidecar.is_file():
        return parse_gate_json(sidecar.read_text(), sidecar)
    markdown = docs / f"{milestone}{GATE_STEM}.md"
    if markdown.is_file():
        return parse_gate_markdown(markdown.read_text(), markdown)
    return None


def load_gate_boards(docs: Path) -> tuple[GateBoard, ...]:
    """Every board that reads; an unreadable one is named by ``unreadable_gate_reports``.

    One malformed sidecar must not take down every page that shows a board, so it
    is left out here and reported there, rather than raised through the API.
    """
    boards: list[GateBoard] = []
    for m in gate_milestones(docs):
        try:
            board = load_gate_board(docs, m)
        except (GateReportFormatError, ValueError):
            continue
        if board is not None:
            boards.append(board)
    return tuple(boards)


def unreadable_gate_reports(docs: Path) -> tuple[str, ...]:
    """The milestones whose report exists but does not parse."""
    bad: list[str] = []
    for m in gate_milestones(docs):
        try:
            load_gate_board(docs, m)
        except (GateReportFormatError, ValueError):
            bad.append(m)
    return tuple(bad)


# -- the poisoned-environment demo ------------------------------------------ #


@dataclass(frozen=True, slots=True)
class LeakDemo:
    """The detector run live on the clean env and on its spot-poisoned twin.

    Two layers, because they fail differently. The perturbation property is exact:
    resample the future and the poisoned chain at ``t`` moves. The statistical
    detector is what would catch a leak in a world we did not build ourselves.
    """

    eps: float
    seeds: tuple[int, ...]
    clean: LeakReport
    poisoned: LeakReport
    clean_past_unchanged: bool
    poisoned_past_unchanged: bool
    seconds: float
    criterion: str = field(default=PASS_CRITERION, repr=False)

    @property
    def caught(self) -> bool:
        """The demo's claim: the clean world is quiet and the poisoned one is caught twice."""
        return (
            not self.clean.flagged
            and self.clean_past_unchanged
            and self.poisoned.flagged
            and not self.poisoned_past_unchanged
        )


def _clean_provider(market: MarketConfig) -> ChainProvider:
    return SyntheticChainProvider(market)


def _poisoned_provider(market: MarketConfig, eps: float) -> ChainProvider:
    return PoisonedChainProvider(market, eps)


def run_leak_demo(
    eps: float = 0.5, n_seeds: int = 10, split_t: int = 10, market: MarketConfig | None = None
) -> LeakDemo:
    """Run both detectors on the clean and the poisoned env, on the same seeds."""
    if eps <= 0.0:
        raise ValueError(f"the poisoned variant needs eps > 0, got {eps}")
    cfg = market if market is not None else MarketConfig()
    seeds = tuple(range(n_seeds))
    start = time.perf_counter()
    clean = detect_leak(partial(default_env, 0.0, cfg), seeds)
    poisoned = detect_leak(partial(default_env, eps, cfg), seeds)
    clean_past, _ = perturbation_check(partial(_clean_provider, cfg), 1, 2, split_t)
    poisoned_past, _ = perturbation_check(partial(_poisoned_provider, cfg, eps), 1, 2, split_t)
    return LeakDemo(
        eps=eps,
        seeds=seeds,
        clean=clean,
        poisoned=poisoned,
        clean_past_unchanged=clean_past,
        poisoned_past_unchanged=poisoned_past,
        seconds=time.perf_counter() - start,
    )


# -- the landing page --------------------------------------------------------- #


def readme_summary(readme: Path) -> tuple[str, ...]:
    """The README's opening paragraphs, before its first section heading.

    Sentences that point at the README or the project document are dropped: on the
    dashboard they would refer to a document the reader is not looking at.
    """
    if not readme.is_file():
        return ()
    paragraphs: list[str] = []
    current: list[str] = []
    for line in readme.read_text().splitlines():
        if line.startswith("## "):
            break
        if line.startswith("# "):
            continue
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    out: list[str] = []
    for para in paragraphs:
        sentences = re.split(r"(?<=\.)\s+", para)
        kept = [s for s in sentences if "README" not in s and "project document" not in s]
        if kept:
            out.append(" ".join(kept))
    return tuple(out)
