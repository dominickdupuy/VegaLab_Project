"""API payloads for the runs: the run list and its filters, one run's detail, and the run index.

Lane: Ilan Danial (machinery).

Every payload read from a run names it by its key under the runs root and its config hash.
``index`` is the run index every run route lists from. Moved from ``optspread.webapi``,
which re-exports it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from optspread.api._base import Out, _f, _inside
from optspread.artifacts import RUN_FILENAME
from optspread.dashboard.jobs.spec import EQUITY_FILENAME, JOBS_DIR
from optspread.dashboard.runindex import (
    RiskStats,
    RunDetail,
    RunRow,
    filter_runs,
    is_matrix_manifest,
    load_run_detail,
    risk_stats,
    run_facets,
    scan_runs,
)
from optspread.manifest import MANIFEST_FILENAME

MAX_EQUITY_SENT = 60  # equity paths sent to the browser; the file keeps up to 200


# -- runs ------------------------------------------------------------------------ #


class RunRowOut(Out):
    key: str  # the run directory relative to the runs root: unique, unlike run_id
    run_id: str
    config_hash: str
    seed: int | None
    policy: str
    premium: float | None
    source: str
    source_recorded: bool  # False: the manifest predates the data record (all synthetic)
    panel: str  # the id of the SPX panel a real-data run read; empty otherwise
    git_sha: str
    commit: str
    git_dirty: bool | None
    created_at: str
    n_episodes: int | None
    mean_pnl: float | None
    se_pnl: float | None
    t_stat: float | None
    mean_cost: float | None
    mean_trades: float | None
    sweep: str
    status: str
    digests_ok: bool | None
    problems: list[str]


def run_source(row: RunRow) -> str:
    """Where the run's chains came from, as its manifest records it (``RunRow.source``)."""
    return row.source


def row_out(row: RunRow, root: Path) -> RunRowOut:
    try:
        key = row.path.relative_to(root).as_posix()
    except ValueError:
        key = row.path.as_posix()
    return RunRowOut(
        key=key,
        run_id=row.run_id,
        config_hash=row.config_hash,
        seed=row.seed,
        policy=row.policy,
        premium=row.premium,
        source=run_source(row),
        source_recorded=bool(row.data_source),
        panel=row.panel,
        git_sha=row.git_sha,
        commit=row.commit,
        git_dirty=row.git_dirty,
        created_at=row.created_at,
        n_episodes=row.n_episodes,
        mean_pnl=_f(row.mean_pnl),
        se_pnl=_f(row.se_pnl),
        t_stat=_f(row.t_stat),
        mean_cost=_f(row.mean_cost),
        mean_trades=_f(row.mean_trades),
        sweep=row.sweep,
        status=row.status,
        digests_ok=row.digests_ok,
        problems=list(row.problems),
    )


class FacetsOut(Out):
    config_hashes: list[str]
    seeds: list[int]
    policies: list[str]
    premiums: list[float]
    commits: list[str]
    sweeps: list[str]
    sources: list[str]


class TrialsOut(Out):
    """The disclosure a reader needs: how many runs exist, not just the ones shown."""

    n_runs: int
    n_ok: int
    n_configs: int
    n_seeds: int


class RunListOut(Out):
    root: str
    rows: list[RunRowOut]
    facets: FacetsOut
    trials: TrialsOut
    n_shown: int


class RunFilters(Out):
    config_hash: list[str] = Field(default_factory=list)
    seed: list[int] = Field(default_factory=list)
    policy: list[str] = Field(default_factory=list)
    premium: list[float] = Field(default_factory=list)
    commit: list[str] = Field(default_factory=list)
    sweep: list[str] = Field(default_factory=list)
    source: list[str] = Field(default_factory=list)
    q: str = ""


def _filtered(rows: Sequence[RunRow], filters: RunFilters) -> tuple[RunRow, ...]:
    shown = filter_runs(
        rows,
        config_hashes=filters.config_hash,
        seeds=filters.seed,
        policies=filters.policy,
        premiums=filters.premium,
        commits=filters.commit,
        sweeps=filters.sweep,
        text=filters.q,
    )
    if filters.source:
        shown = tuple(r for r in shown if run_source(r) in filters.source)
    return tuple(shown)


def run_list(rows: Sequence[RunRow], root: Path, filters: RunFilters) -> RunListOut:
    facets = run_facets(rows)
    shown = _filtered(rows, filters)
    return RunListOut(
        root=str(root),
        rows=[row_out(r, root) for r in shown],
        facets=FacetsOut(
            config_hashes=list(facets.config_hashes),
            seeds=list(facets.seeds),
            policies=list(facets.policies),
            premiums=list(facets.premiums),
            commits=list(facets.commits),
            sweeps=list(facets.sweeps),
            sources=sorted({run_source(r) for r in rows}),
        ),
        trials=TrialsOut(
            n_runs=len(rows),
            n_ok=sum(r.status == "OK" for r in rows),
            n_configs=len(facets.config_hashes),
            n_seeds=len(facets.seeds),
        ),
        n_shown=len(shown),
    )


def resolve_run(root: Path, key: str) -> Path:
    """The run directory for ``key``, refusing anything outside the runs root and the
    pilot's matrix manifest, which describes no run (``is_matrix_manifest``)."""
    target = _inside(root, key)
    if target is None or not (target / MANIFEST_FILENAME).is_file():
        raise KeyError(key)
    if is_matrix_manifest(target):
        raise KeyError(key)
    return target


class RiskOut(Out):
    n: int
    mean: float | None
    se: float | None
    sortino: float | None
    cvar5: float | None
    max_drawdown: float | None
    hit_rate: float | None
    turnover: float | None


def risk_out(stats: RiskStats) -> RiskOut:
    return RiskOut(
        n=stats.n,
        mean=_f(stats.mean),
        se=_f(stats.se),
        sortino=_f(stats.sortino),
        cvar5=_f(stats.cvar5),
        max_drawdown=_f(stats.max_drawdown),
        hit_rate=_f(stats.hit_rate),
        turnover=_f(stats.turnover),
    )


class HistogramOut(Out):
    edges: list[float]
    counts: list[int]


def histogram(values: NDArray[np.float64], max_bins: int = 40) -> HistogramOut:
    if not len(values):
        return HistogramOut(edges=[], counts=[])
    if float(values.min()) == float(values.max()):
        x = float(values[0])
        return HistogramOut(edges=[x - 0.5, x + 0.5], counts=[len(values)])
    edges = np.histogram_bin_edges(values, bins="auto")
    if len(edges) - 1 > max_bins:
        edges = np.histogram_bin_edges(values, bins=max_bins)
    counts, edges = np.histogram(values, bins=edges)
    return HistogramOut(edges=[float(e) for e in edges], counts=[int(c) for c in counts])


class EpisodeOut(Out):
    seed: int
    pnl: float
    cost: float
    n_trades: int
    steps: int


class EquityOut(Out):
    file: str
    seeds: list[int]
    paths: list[list[float]]
    n_recorded: int


def _equity(run_dir: Path) -> EquityOut | None:
    target = run_dir / EQUITY_FILENAME
    try:
        raw: Any = json.loads(target.read_text())
        seeds, paths = list(raw["seeds"]), list(raw["paths"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return EquityOut(
        file=EQUITY_FILENAME,
        seeds=[int(s) for s in seeds[:MAX_EQUITY_SENT]],
        paths=[[float(v) for v in p] for p in paths[:MAX_EQUITY_SENT]],
        n_recorded=len(paths),
    )


class ArtifactOut(Out):
    path: str
    recorded: str
    actual: str
    ok: bool


class ProvenanceOut(Out):
    run_dir: str
    manifest: str
    results: str
    replicas: int  # runs in the index with the same config hash, this one included
    sweep_cells: int  # runs in the same sweep, this one included


class RunDetailOut(Out):
    row: RunRowOut
    manifest: dict[str, Any]
    episodes: list[EpisodeOut]
    risk: RiskOut
    histogram: HistogramOut
    equity: EquityOut | None
    artifacts: list[ArtifactOut]
    provenance: ProvenanceOut


def run_detail(detail: RunDetail, root: Path, index: Sequence[RunRow]) -> RunDetailOut:
    row = detail.row
    episodes = detail.record.episodes if detail.record else ()
    return RunDetailOut(
        row=row_out(row, root),
        manifest=detail.manifest,
        episodes=[
            EpisodeOut(seed=e.seed, pnl=e.pnl, cost=e.cost, n_trades=e.n_trades, steps=e.steps)
            for e in episodes
        ],
        risk=risk_out(risk_stats(detail.pnl, detail.trades)),
        histogram=histogram(detail.pnl),
        equity=_equity(row.path),
        artifacts=[
            ArtifactOut(path=a.path, recorded=a.recorded, actual=a.actual, ok=a.ok)
            for a in detail.artifacts
        ],
        provenance=ProvenanceOut(
            run_dir=str(row.path),
            manifest=str(row.path / MANIFEST_FILENAME),
            results=str(row.path / RUN_FILENAME),
            replicas=sum(r.config_hash == row.config_hash for r in index if row.config_hash),
            sweep_cells=sum(
                r.sweep == row.sweep and r.path.parent == row.path.parent for r in index
            )
            if row.sweep
            else 0,
        ),
    )


def load_detail(root: Path, key: str) -> RunDetail:
    return load_run_detail(resolve_run(root, key))


def index(root: Path) -> tuple[RunRow, ...]:
    """Every run under the root except the job queue's own bookkeeping."""
    return tuple(r for r in scan_runs(root) if JOBS_DIR not in r.path.parts)
