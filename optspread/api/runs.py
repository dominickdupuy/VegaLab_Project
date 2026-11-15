"""API payloads for the runs: the run list and its filters, one run's detail, and the run index.

Lane: Ilan Danial (machinery).

Every payload read from a run names it by its key under the runs root and its config hash.
``index`` is the run index every run route lists from. Moved from ``optspread.webapi``.

``policy_heatmap`` reads a launched run's ``action_log.json`` (``dashboard.jobs.worker``):
how often each action was chosen in each cell of (time through the episode, ATM implied
vol). Only a log the run's manifest hashed, and that still matches, is read.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from optspread.actions import ACTION_NAMES
from optspread.api._base import Out, _from_attrs, _inside
from optspread.artifacts import RUN_FILENAME
from optspread.config import MarketConfig
from optspread.dashboard.jobs.spec import ACTION_LOG_FILENAME, EQUITY_FILENAME, JOBS_DIR
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
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256

MAX_EQUITY_SENT = 60  # equity paths sent to the browser; the file keeps up to 200

# The four values RunRow.status returns, one per branch.
RunStatus = Literal["OK", "UNVERIFIED", "INVALID", "DIGEST MISMATCH"]


# -- runs ------------------------------------------------------------------------ #


class RunRowOut(Out):
    key: str  # the run directory relative to the runs root: unique, unlike run_id
    run_id: str
    config_hash: str
    seed: int | None
    policy: str
    premium: float | None
    alpha_bps: float | None = None  # the market's research knobs (MarketConfig)
    vol_of_vol: float | None = None
    skew_slope: float | None = None
    rate: float | None = None  # the rate its options were priced at; None on real data
    rate_default: float = MarketConfig().r  # the rate a run launched now is priced at
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
    status: RunStatus
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
    return _from_attrs(
        RunRowOut, row, key=key, source=run_source(row), source_recorded=bool(row.data_source)
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
        facets=_from_attrs(FacetsOut, facets, sources=sorted({run_source(r) for r in rows})),
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
    return RiskOut.model_validate(stats, from_attributes=True)


class HistogramOut(Out):
    edges: list[float | None]  # null where a value is not finite (an overflowing difference)
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
    paths: list[list[float | None]]  # null where the file holds a value that is not finite
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


HEATMAP_BINS = 10  # bins on each axis of the policy heatmap


class PolicyHeatmapOut(Out):
    """How often each action was chosen, by time through the episode and the ATM vol the
    policy saw. ``counts[t][v][a]``: steps in time bin ``t`` and vol bin ``v`` that chose
    action ``a`` (``action_names[a]``). One vol bin when the vol never moved (a constant
    vol world: the study's)."""

    available: bool
    reason: str = ""  # why there is no heatmap, when there is none
    file: str = ACTION_LOG_FILENAME
    time_frac_edges: list[float] = Field(default_factory=list)
    atm_iv_edges: list[float] = Field(default_factory=list)  # decimal vol
    action_names: list[str] = Field(default_factory=list)
    counts: list[list[list[int]]] = Field(default_factory=list)
    n_episodes: int = 0
    n_steps: int = 0


def _flat(raw: Any, name: str, kind: type[float] | type[int]) -> NDArray[Any]:
    return np.asarray([kind(x) for ep in raw[name] for x in ep], dtype=np.float64)


def policy_heatmap(run_dir: Path) -> PolicyHeatmapOut:
    """The policy heatmap of ``run_dir`` from its action log; ``available`` False, with the
    reason, for a run without one (not launched from the dashboard, or launched before the
    log was written), a log its manifest does not vouch for, or one that does not read."""
    target = run_dir / ACTION_LOG_FILENAME
    if not target.is_file():
        return PolicyHeatmapOut(
            available=False,
            reason="this run records no action log: only runs launched from the dashboard do",
        )
    try:
        recorded = RunManifest.read(run_dir).artifacts.get(ACTION_LOG_FILENAME)
    except (OSError, ValueError):
        recorded = None
    if recorded is None or recorded != file_sha256(target):
        return PolicyHeatmapOut(
            available=False, reason=f"{ACTION_LOG_FILENAME} does not match the run's manifest"
        )
    try:
        raw: Any = json.loads(target.read_text(encoding="utf-8"))
        actions = _flat(raw, "actions", int).astype(np.int64)
        time_frac, atm_iv = _flat(raw, "time_frac", float), _flat(raw, "atm_iv", float)
        n_episodes = len(raw["actions"])
    except (OSError, ValueError, KeyError, TypeError):
        return PolicyHeatmapOut(available=False, reason=f"{ACTION_LOG_FILENAME} does not read")
    n = len(actions)
    if n == 0 or len(time_frac) != n or len(atm_iv) != n:
        return PolicyHeatmapOut(available=False, reason=f"{ACTION_LOG_FILENAME} records no steps")
    if actions.min() < 0 or actions.max() >= len(ACTION_NAMES):
        return PolicyHeatmapOut(
            available=False, reason="the log names an action not in the library"
        )
    lo, hi = float(atm_iv.min()), float(atm_iv.max())
    n_iv = HEATMAP_BINS if hi > lo else 1
    iv_edges = np.linspace(lo, hi, n_iv + 1) if hi > lo else np.array([lo, lo])
    tf_edges = np.linspace(0.0, 1.0, HEATMAP_BINS + 1)
    # the last edge closes its bin: a step at the top lands in the last bin, not past it
    iv_bin = np.clip(np.searchsorted(iv_edges, atm_iv, side="right") - 1, 0, n_iv - 1)
    tf_bin = np.clip(np.searchsorted(tf_edges, time_frac, side="right") - 1, 0, HEATMAP_BINS - 1)
    counts = np.zeros((HEATMAP_BINS, n_iv, len(ACTION_NAMES)), dtype=np.int64)
    np.add.at(counts, (tf_bin, iv_bin, actions), 1)
    return PolicyHeatmapOut(
        available=True,
        time_frac_edges=[float(e) for e in tf_edges],
        atm_iv_edges=[float(e) for e in iv_edges],
        action_names=list(ACTION_NAMES),
        counts=counts.tolist(),
        n_episodes=n_episodes,
        n_steps=n,
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
        episodes=[EpisodeOut.model_validate(e, from_attributes=True) for e in episodes],
        risk=risk_out(risk_stats(detail.pnl, detail.trades)),
        histogram=histogram(detail.pnl),
        equity=_equity(row.path),
        artifacts=[ArtifactOut.model_validate(a, from_attributes=True) for a in detail.artifacts],
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
