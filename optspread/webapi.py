"""The dashboard's API payloads: typed responses built from what the harness wrote.

Lane: Ilan Danial (machinery).

``optspread.server`` is a thin FastAPI layer over this module: each route parses
its query, calls one builder here and returns the model it gets back. Keeping the
builders free of the web framework means they are tested and covered on the
plain ``dev`` install, and the server's own tests only have to show the wiring.

Every payload says where its numbers came from: a run key and config hash for
anything read from a run, a file path for anything read from a gate report or the
SPX panel, and the module and formula for the power lab, which computes rather
than reads. NaN is not JSON, so a figure that is undefined is sent as null.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from optspread import actionmix as am
from optspread import dashboard_panel as panel
from optspread import power as pw
from optspread.analysis import (
    ALPHA,
    NULL_PREMIUM,
    decide_contrast,
    decide_premium,
    paired_contrast,
)
from optspread.artifacts import SchemaVersionError, read_run
from optspread.config import ArmName, MarketConfig
from optspread.dashboard_data import (
    GateBoard,
    LeakDemo,
    RiskStats,
    RunDetail,
    RunRow,
    compare_runs,
    filter_runs,
    flatten_config,
    is_matrix_manifest,
    load_gate_boards,
    load_run_detail,
    readme_summary,
    risk_stats,
    run_facets,
    scan_runs,
)
from optspread.jobs import (
    EQUITY_FILENAME,
    LAUNCH_DIR,
    PRESETS,
    SOURCES,
    JobSpec,
    JobStatus,
    available_arms,
    available_sources,
)
from optspread.leaks import LeakReport
from optspread.manifest import MANIFEST_FILENAME, RunManifest
from optspread.market import SyntheticChainProvider
from optspread.orchestrate import GridRecord, read_grid, source_root
from optspread.realdata import SPLITS
from optspread.rng import SeededStepRNG
from optspread.sweep import SWEEP_FILENAME, read_sweep

MAX_EQUITY_SENT = 60  # equity paths sent to the browser; the file keeps up to 200


def _f(x: float | None) -> float | None:
    """A float for JSON: NaN and infinities become null."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fl(values: Iterable[float]) -> list[float | None]:
    return [_f(v) for v in values]


def _iso(values: NDArray[np.datetime64]) -> list[str]:
    return [str(d) for d in values.astype("datetime64[D]")]


class Out(BaseModel):
    model_config = ConfigDict(frozen=True)


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


def _inside(root: Path, key: str) -> Path | None:
    """``root / key`` resolved, or None when it leaves the root or is not a path at all."""
    try:
        target = (root / key).resolve()
        return target if target.is_relative_to(root.resolve()) else None
    except (OSError, ValueError):  # a NUL byte, or a name the filesystem refuses
        return None


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
            results=str(row.path / "run.json"),
            replicas=sum(r.config_hash == row.config_hash for r in index if row.config_hash),
            sweep_cells=sum(
                r.sweep == row.sweep and r.path.parent == row.path.parent for r in index
            )
            if row.sweep
            else 0,
        ),
    )


class DiffOut(Out):
    key: str
    a: Any
    b: Any


class SeedDiffOut(Out):
    seed: int
    a: float
    b: float
    diff: float  # b - a


class CompareOut(Out):
    a: RunDetailOut
    b: RunDetailOut
    config_diff: list[DiffOut]
    shared_seeds: int
    diff_mean: float | None
    diff_se: float | None
    diff_t: float | None
    diff_histogram: HistogramOut
    # the paired difference's two-sided 95% interval (t on the shared seeds) and each seed
    diff_df: int | None = None
    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    b_better_share: float | None = None  # share of shared seeds where B made more than A
    diffs: list[SeedDiffOut] = Field(default_factory=list)
    verdict: str = ""  # one plain sentence
    # from the runs root's catalog.json, when it lists the runs: what each one is, and the
    # question the two answer when they are in one group
    a_label: str = ""
    b_label: str = ""
    group: str = ""
    question: str = ""


def _label(row: RunRow) -> str:
    names = {
        "delta_seller": "the fixed-delta seller",
        "ppo": "PPO",
        "always_on": "the always-on condor",
        "random": "the random policy",
        "flat": "flat",
    }
    arm = names.get(row.policy, row.policy or "a run")
    where = (
        "real SPX history"
        if run_source(row) not in ("", "synthetic")
        else (
            f"{row.premium:g} vol pt{'' if row.premium == 1 else 's'} planted"
            if row.premium is not None
            else "the generator"
        )
    )
    return f"{arm} ({where})"


MINUS = "\u2212"  # the typographic minus the pages print


def _usd(x: float) -> str:
    """Dollars as the pages print them, with a true minus: "-$169" as U+2212, "$1,051"."""
    return f"{MINUS if x < 0 and round(x) != 0 else ''}${abs(x):,.0f}"


def compare_verdict(
    a: RunRow,
    b: RunRow,
    n: int,
    mean: float | None,
    lo: float | None,
    hi: float | None,
    names: tuple[str, str] = ("", ""),
) -> str:
    """One sentence a reader can act on: which run made more on the same paths, and whether
    the interval says the gap is real. ``names`` are the catalog's labels, when it has them."""
    na, nb = names[0] or _label(a), names[1] or _label(b)
    if n == 0 or mean is None:
        return "The two runs share no seeds, so they cannot be compared episode by episode."
    paths = f"the same {n} path{'s' if n != 1 else ''}"
    if lo is None or hi is None:
        return f"On {paths} {nb} less {na} is {_usd(mean)} per episode; too few to say more."
    if lo == hi == mean:
        return f"{nb} and {na} made exactly the same on all {n} shared paths."
    learned = "ppo" in (a.policy, b.policy)
    caveat = " (one trained policy each: a retrain could move this)" if learned else ""
    half = (hi - lo) / 2.0
    gap = (
        f"{nb} made {_usd(abs(mean))} {'more' if mean >= 0 else 'less'} per episode than "
        f"{na} on {paths} (± {_usd(half)}, 95%)"
    )
    if lo > 0 or hi < 0:
        return f"{gap}: a clear difference{caveat}."
    return f"{gap}: not distinguishable from noise at this sample size; run more episodes{caveat}."


def compare(
    a: RunDetail,
    b: RunDetail,
    root: Path,
    index: Sequence[RunRow],
    listed: CatalogOut | None = None,
    group: str = "",
) -> CompareOut:
    """``a`` and ``b`` paired on their shared seeds. With the runs root's catalog, the
    labels and the question come from ``group`` when both runs are in it, else from the
    first group that lists both (a baseline run can sit in several)."""
    cmp = compare_runs(a, b)
    ka, kb = row_out(a.row, root).key, row_out(b.row, root).key
    both = [
        g
        for g in (listed.groups if listed is not None else [])
        if {ka, kb} <= {r.key for r in g.runs}
    ]
    chosen = next((g for g in both if g.group == group), both[0] if both else None)
    labels = (
        {r.key: r.label for r in chosen.runs}
        if chosen is not None
        else (listed.labels if listed is not None else {})
    )
    names = (labels.get(ka, ""), labels.get(kb, ""))
    pa = {e.seed: e.pnl for e in a.record.episodes} if a.record else {}
    pb = {e.seed: e.pnl for e in b.record.episodes} if b.record else {}
    diffs = np.asarray([pb[s] - pa[s] for s in cmp.shared_seeds], dtype=np.float64)
    n = len(diffs)
    lo, hi = (
        _interval(float(cmp.diff_mean), float(cmp.diff_se), n)
        if cmp.diff_mean is not None and cmp.diff_se is not None
        else (None, None)
    )
    return CompareOut(
        a=run_detail(a, root, index),
        b=run_detail(b, root, index),
        config_diff=[DiffOut(key=k, a=va, b=vb) for k, va, vb in cmp.config_diff],
        shared_seeds=len(cmp.shared_seeds),
        diff_mean=_f(cmp.diff_mean),
        diff_se=_f(cmp.diff_se),
        diff_t=_f(cmp.diff_t),
        diff_histogram=histogram(diffs),
        diff_df=n - 1 if n > 1 else None,
        diff_ci_low=lo,
        diff_ci_high=hi,
        b_better_share=_f(float((diffs > 0).mean())) if n else None,
        diffs=[SeedDiffOut(seed=s, a=pa[s], b=pb[s], diff=pb[s] - pa[s]) for s in cmp.shared_seeds],
        verdict=compare_verdict(a.row, b.row, n, _f(cmp.diff_mean), lo, hi, names),
        a_label=names[0],
        b_label=names[1],
        group=chosen.group if chosen is not None else "",
        question=chosen.question if chosen is not None else "",
    )


def load_detail(root: Path, key: str) -> RunDetail:
    return load_run_detail(resolve_run(root, key))


# -- a designed matrix of runs: <runs root>/catalog.json, optional ------------------- #

CATALOG_FILENAME = "catalog.json"
_NOT_A_VARIANT = ("iv_premium_vol_pts", "seed")


class CatalogRunOut(Out):
    key: str
    label: str
    settings: dict[str, Any]
    present: bool  # the run is in the index under this key
    status: str
    policy: str
    premium: float | None
    source: str
    n_episodes: int | None
    mean_pnl: float | None
    se_pnl: float | None


class CatalogPairOut(Out):
    """One click to compare: a run against the group's baseline at the same premium, with
    the paired difference b - a over the seeds both ran and its two-sided 95% interval."""

    a: str  # the baseline's key
    b: str
    title: str  # "<b's label> against <a's label>"
    varies: list[str]  # the settings that differ
    premium: float | None
    n_shared: int = 0
    diff_mean: float | None = None
    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    clear: bool = False  # the interval excludes zero


class CatalogGroupOut(Out):
    group: str
    question: str
    runs: list[CatalogRunOut]
    pairs: list[CatalogPairOut]


class CatalogOut(Out):
    available: bool
    file: str
    seed: int | None = None
    groups: list[CatalogGroupOut] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)  # run key -> label
    questions: dict[str, str] = Field(default_factory=dict)  # run key -> its group's question
    groups_of: dict[str, str] = Field(default_factory=dict)  # run key -> group
    problem: str = ""  # why a catalog that is there did not read


def _distance(settings: dict[str, Any]) -> int:
    """How many settings sit away from the launch defaults (the premium and seed aside)."""
    defaults = JobSpec().model_dump(mode="json")
    return sum(
        1
        for k, v in settings.items()
        if k in defaults and k not in _NOT_A_VARIANT and defaults[k] != v
    )


def _variant(settings: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in settings.items() if k not in _NOT_A_VARIANT}


def _pairs(runs: list[CatalogRunOut]) -> list[CatalogPairOut]:
    """Within a group: at each premium, every variant against the one nearest the
    defaults; a group whose runs differ in the premium only, each premium against zero."""
    present = [r for r in runs if r.present]
    variants = {json.dumps(_variant(r.settings), sort_keys=True, default=str) for r in present}
    out: list[CatalogPairOut] = []
    if len(variants) <= 1:
        nulls = [r for r in present if (r.premium or 0.0) == 0.0]
        for r in present:
            if nulls and r is not nulls[0] and (r.premium or 0.0) != 0.0:
                out.append(
                    CatalogPairOut(
                        a=nulls[0].key,
                        b=r.key,
                        title=f"{r.label} against {nulls[0].label}",
                        varies=["iv_premium_vol_pts"],
                        premium=r.premium,
                    )
                )
        return out
    levels: dict[float | None, list[CatalogRunOut]] = {}
    for r in present:
        levels.setdefault(r.premium, []).append(r)
    for premium in sorted(levels, key=lambda p: (p is None, p or 0.0)):
        at = levels[premium]
        base = min(at, key=lambda r: (_distance(r.settings), r.key))
        for r in at:
            if r is base:
                continue
            keys = sorted(set(r.settings) | set(base.settings))
            varies = [
                k
                for k in keys
                if k not in _NOT_A_VARIANT and r.settings.get(k) != base.settings.get(k)
            ]
            out.append(
                CatalogPairOut(
                    a=base.key,
                    b=r.key,
                    title=f"{r.label} against {base.label}",
                    varies=varies,
                    premium=premium,
                )
            )
    return out


def _paired(pair: CatalogPairOut, a: dict[int, float], b: dict[int, float]) -> CatalogPairOut:
    shared = sorted(set(a) & set(b))
    d = np.asarray([b[s] - a[s] for s in shared], dtype=np.float64)
    if len(d) < 2:
        return pair.model_copy(update={"n_shared": len(d)})
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    lo, hi = _interval(mean, se, len(d))
    return pair.model_copy(
        update={
            "n_shared": len(d),
            "diff_mean": _f(mean),
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "clear": lo is not None and hi is not None and (lo > 0 or hi < 0),
        }
    )


def catalog(root: Path, rows: Sequence[RunRow]) -> CatalogOut:
    """The runs root's ``catalog.json``: the runs grouped by the question each answers,
    with their labels and the comparisons that answer it. ``available`` is False when the
    root has none, and every page then reads as it would without it."""
    path = root / CATALOG_FILENAME
    if not path.is_file():
        return CatalogOut(available=False, file=str(path))
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        entries = list(raw["runs"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return CatalogOut(available=False, file=str(path), problem=str(exc).splitlines()[0])
    by_key = {row_out(r, root).key: r for r in rows}
    groups: dict[str, CatalogGroupOut] = {}
    labels: dict[str, str] = {}
    questions: dict[str, str] = {}
    groups_of: dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict) or "key" not in e:
            continue
        key = str(e["key"]).strip("/")
        group = str(e.get("group", "other"))
        question = str(e.get("question", ""))
        label = str(e.get("label", key))
        given = e.get("settings")
        settings: dict[str, Any] = given if isinstance(given, dict) else {}
        row = by_key.get(key)
        run = CatalogRunOut(
            key=key,
            label=label,
            settings=settings,
            present=row is not None,
            status=row.status if row else "MISSING",
            policy=row.policy if row else str(settings.get("policy", "")),
            premium=row.premium if row else settings.get("iv_premium_vol_pts"),
            source=run_source(row) if row else str(settings.get("source", "synthetic")),
            n_episodes=row.n_episodes if row else settings.get("n_episodes"),
            mean_pnl=_f(row.mean_pnl) if row else None,
            se_pnl=_f(row.se_pnl) if row else None,
        )
        held = groups.setdefault(
            group, CatalogGroupOut(group=group, question=question, runs=[], pairs=[])
        )
        held.runs.append(run)
        labels.setdefault(key, label)  # a run in several groups keeps its first label
        questions.setdefault(key, question or held.question)
        groups_of.setdefault(key, group)
    episodes: dict[str, dict[int, float]] = {}

    def pnl(key: str) -> dict[int, float]:
        if key not in episodes:
            try:
                episodes[key] = {e.seed: e.pnl for e in read_run(by_key[key].path).episodes}
            except (KeyError, OSError, ValueError, SchemaVersionError):
                episodes[key] = {}
        return episodes[key]

    out = [
        g.model_copy(update={"pairs": [_paired(p, pnl(p.a), pnl(p.b)) for p in _pairs(g.runs)]})
        for g in groups.values()
    ]
    seed = raw.get("seed") if isinstance(raw, dict) else None
    return CatalogOut(
        available=True,
        file=str(path),
        seed=int(seed) if isinstance(seed, int) else None,
        groups=out,
        labels=labels,
        questions=questions,
        groups_of=groups_of,
    )


# -- the runs list: P&L by planted premium, per arm, and each arm against itself at zero -- #

# A world is a run's config less what varies inside a comparison: the arm, the seeds, the
# episode count and the premium. Runs of one arm in one world at two premiums differ only
# in the premium, so their episodes on the same seeds are paired.
_WORLD_DROP = ("policy", "seed", "n_episodes")
_PLAIN_KEYS: dict[str, Callable[[Any], str]] = {
    "cost.enabled": lambda v: "costs on" if v else "costs off",
    "market.n_days": lambda v: f"{v}-day path",
    "env.episode_days": lambda v: f"{v}-day episodes",
    "market.sigma": lambda v: f"path vol {float(v):g}",
    "env.short_delta": lambda v: f"short delta {float(v):g}",
    "env.wing_delta": lambda v: f"wing delta {float(v):g}",
    "env.roll_dte": lambda v: f"roll at {v} days",
    "market.tenor_days": lambda v: f"tenors {', '.join(str(x) for x in v)}",
    "env.initial_cash": lambda v: f"cash ${float(v):,.0f}",
    "env.expiry_idx": lambda v: "opens the nearest tenor" if v == 0 else f"opens tenor {v}",
    "cost.commission_per_contract": lambda v: f"commission ${float(v):g}",
    "cost.half_spread_frac": lambda v: f"half-spread {float(v):.1%} of mid",
    "cost.min_half_spread": lambda v: f"half-spread floor {float(v):g}",
}
P_LEVELS = (5, 25, 50, 75, 95)


class WorldOut(Out):
    id: str
    label: str  # what sets it apart from the most common world here; "" for that one
    source: str
    n_runs: int
    arms: list[str]


class PremiumGroupOut(Out):
    """Every episode of one arm in one world at one premium, pooled over its runs."""

    arm: str
    variant: str  # a learned arm's training budget; "" for a scripted arm
    world: str
    source: str
    premium: float
    n_runs: int
    n_episodes: int
    mean: float | None
    se: float | None
    ci_low: float | None  # two-sided 95%
    ci_high: float | None
    interval: str  # "runs": t on the run means; "episodes": one run, t on its episodes
    p5: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    run_keys: list[str]


class PremiumContrastOut(Out):
    """One arm at premium p against itself at zero, on the seeds both ran: the detection
    event of ``analysis.decide_premium`` when two or more runs pair up."""

    arm: str
    variant: str
    world: str
    source: str
    premium: float
    n_blocks: int  # run pairs on the same first seed
    n_pairs: int  # episodes paired, over every block
    estimate: float | None
    se: float | None
    df: int | None
    ci_low: float | None  # two-sided 95%
    ci_high: float | None
    lower_bound: float | None  # one-sided 95%, the pre-registered bound (D1)
    detected: bool
    interval: str  # "runs" (the rule) or "episodes" (one pair: no between-run term)
    reason: str = ""


class PremiumsOut(Out):
    n_runs: int  # OK runs read
    n_skipped: int  # OK runs whose files did not read
    worlds: list[WorldOut]
    groups: list[PremiumGroupOut]
    contrasts: list[PremiumContrastOut]


def _world(config: dict[str, Any], source: str) -> tuple[str, dict[str, Any]]:
    flat = flatten_config({k: v for k, v in config.items() if k not in _WORLD_DROP})
    flat.pop("market.iv_premium_vol_pts", None)
    flat = {k: v for k, v in flat.items() if not k.startswith("train.")}  # a variant, not a world
    flat["source"] = source
    blob = json.dumps(flat, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10], flat


def _world_label(flat: dict[str, Any], main: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in sorted(set(flat) | set(main)):
        if k == "source" or k.startswith("train.") or flat.get(k) == main.get(k):
            continue
        v = flat.get(k)
        parts.append(_PLAIN_KEYS[k](v) if k in _PLAIN_KEYS and v is not None else f"{k} {v}")
    return ", ".join(parts)


def _arm_variant(config: dict[str, Any]) -> str:
    """What sets a learned arm apart within a world: its training budget."""
    train = config.get("train")
    if isinstance(train, dict) and train.get("total_timesteps"):
        return f"{int(train['total_timesteps']):,} training steps"
    return ""


def _t95(n: int) -> float:
    return pw.t_critical(n, 0.025)


def _interval(mean: float, se: float, n: int) -> tuple[float | None, float | None]:
    if n < 2 or not math.isfinite(se):
        return None, None
    half = _t95(n) * se
    return _f(mean - half), _f(mean + half)


def _group(
    arm: str,
    variant: str,
    world: str,
    source: str,
    premium: float,
    runs: list[tuple[str, int, dict[int, float]]],
) -> PremiumGroupOut:
    pooled = np.asarray([p for _, _, eps in runs for p in eps.values()], dtype=np.float64)
    means = np.asarray([np.mean(list(eps.values())) for _, _, eps in runs if eps])
    n = len(pooled)
    mean = float(pooled.mean()) if n else math.nan
    if len(means) >= 2:
        se, k, kind = float(means.std(ddof=1) / math.sqrt(len(means))), len(means), "runs"
    else:
        se = float(pooled.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        k, kind = n, "episodes"
    lo, hi = _interval(mean, se, k)
    q = np.percentile(pooled, P_LEVELS) if n else [math.nan] * len(P_LEVELS)
    return PremiumGroupOut(
        arm=arm,
        variant=variant,
        world=world,
        source=source,
        premium=premium,
        n_runs=len(runs),
        n_episodes=n,
        mean=_f(mean),
        se=_f(se),
        ci_low=lo,
        ci_high=hi,
        interval=kind,
        p5=_f(q[0]),
        p25=_f(q[1]),
        p50=_f(q[2]),
        p75=_f(q[3]),
        p95=_f(q[4]),
        run_keys=[key for key, _, _ in runs],
    )


def _premium_contrast(
    arm: str,
    variant: str,
    world: str,
    source: str,
    premium: float,
    treated: list[tuple[str, int, dict[int, float]]],
    null: list[tuple[str, int, dict[int, float]]],
) -> PremiumContrastOut:
    base = {"arm": arm, "variant": variant, "world": world, "source": source, "premium": premium}
    at_null = {first: eps for _, first, eps in null}
    blocks = []
    for _, first, eps in sorted(treated, key=lambda r: r[1]):
        other = at_null.get(first)
        if other is None:
            continue
        shared = sorted(set(eps) & set(other))
        if shared:
            blocks.append([eps[s] - other[s] for s in shared])
    empty = {
        "n_blocks": len(blocks),
        "n_pairs": sum(len(b) for b in blocks),
        "estimate": None,
        "se": None,
        "df": None,
        "ci_low": None,
        "ci_high": None,
        "lower_bound": None,
        "detected": False,
    }
    if not blocks:
        reason = "no run at zero premium on the same seeds" if null else "no run at zero premium"
        return PremiumContrastOut(**base, **empty, interval="", reason=reason)
    if len(blocks) >= 2:
        n = min(len(b) for b in blocks)  # a balanced matrix: the first n shared seeds of each
        decision = decide_contrast(np.asarray([b[:n] for b in blocks], dtype=np.float64))
        lo, hi = _interval(decision.estimate, decision.se, len(blocks))
        return PremiumContrastOut(
            **base,
            n_blocks=len(blocks),
            n_pairs=n * len(blocks),
            estimate=_f(decision.estimate),
            se=_f(decision.se),
            df=decision.df,
            ci_low=lo,
            ci_high=hi,
            lower_bound=_f(decision.lower_bound),
            detected=decision.rejected,
            interval="runs",
        )
    d = np.asarray(blocks[0], dtype=np.float64)
    if len(d) < 2:
        return PremiumContrastOut(**base, **empty, interval="", reason="one paired episode")
    est, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    lo, hi = _interval(est, se, len(d))
    bound = est - pw.t_critical(len(d), ALPHA) * se
    return PremiumContrastOut(
        **base,
        n_blocks=1,
        n_pairs=len(d),
        estimate=_f(est),
        se=_f(se),
        df=len(d) - 1,
        ci_low=lo,
        ci_high=hi,
        lower_bound=_f(bound),
        detected=bool(bound > 0),
        interval="episodes",
    )


def premium_view(rows: Sequence[RunRow], root: Path, filters: RunFilters) -> PremiumsOut:
    """The OK runs ``filters`` shows, as P&L distributions per arm and premium in each
    world, and each arm against itself at zero premium on shared seeds (``#26``)."""
    read: list[tuple[str, str, str, str, float, str, int, dict[int, float], dict[str, Any]]] = []
    skipped = 0
    for row in _filtered(rows, filters):
        if row.status != "OK" or row.premium is None:
            continue
        try:
            config = RunManifest.read(row.path).config
            record = read_run(row.path)
        except (OSError, ValueError, SchemaVersionError):
            skipped += 1
            continue
        dumped = config.model_dump(mode="json")
        world, flat = _world(dumped, run_source(row))
        read.append(
            (
                row_out(row, root).key,
                str(config.policy),
                _arm_variant(dumped),
                world,
                float(config.market.iv_premium_vol_pts),
                run_source(row),
                int(config.seed),
                {e.seed: e.pnl for e in record.episodes},
                flat,
            )
        )
    counts: dict[str, int] = {}
    flats: dict[str, dict[str, Any]] = {}
    arms: dict[str, set[str]] = {}
    for _, arm, _, world, _, _, _, _, flat in read:
        counts[world] = counts.get(world, 0) + 1
        flats[world] = flat
        arms.setdefault(world, set()).add(arm)
    main = flats[max(counts, key=lambda w_: (counts[w_], w_))] if counts else {}
    cells: dict[tuple[str, str, str, str, float], list[tuple[str, int, dict[int, float]]]] = {}
    for key, arm, variant, world, premium, source, first, eps, _ in read:
        cells.setdefault((arm, variant, world, source, premium), []).append((key, first, eps))
    groups = [_group(*k, v) for k, v in sorted(cells.items())]
    contrasts = [
        _premium_contrast(
            arm,
            variant,
            world,
            source,
            premium,
            runs,
            cells.get((arm, variant, world, source, 0.0), []),
        )
        for (arm, variant, world, source, premium), runs in sorted(cells.items())
        if premium != NULL_PREMIUM and source == "synthetic"
    ]
    worlds = [
        WorldOut(
            id=w_,
            label=_world_label(flats[w_], main),
            source=str(flats[w_]["source"]),
            n_runs=counts[w_],
            arms=sorted(arms[w_]),
        )
        for w_ in sorted(counts, key=lambda w_: -counts[w_])
    ]
    return PremiumsOut(
        n_runs=len(read), n_skipped=skipped, worlds=worlds, groups=groups, contrasts=contrasts
    )


# -- gates ------------------------------------------------------------------------ #


class GateItemOut(Out):
    name: str
    owner: str
    verdict: str
    criterion: str
    evidence: str
    failed_first: str
    cited_runs: list[str]


class GateBoardOut(Out):
    milestone: str
    report_date: str
    verdict: str  # derived from the items, never taken from the file
    stated_verdict: str  # what the file says
    verdict_mismatch: bool
    counts: dict[str, int]
    items: list[GateItemOut]
    source: str
    from_json: bool


class GatesOut(Out):
    docs: str
    boards: list[GateBoardOut]


def board_out(board: GateBoard) -> GateBoardOut:
    return GateBoardOut(
        milestone=board.milestone,
        report_date=board.report_date,
        verdict=board.verdict,
        stated_verdict=board.stated_verdict,
        verdict_mismatch=board.verdict_mismatch,
        counts=board.counts,
        items=[
            GateItemOut(
                name=i.name,
                owner=i.owner,
                verdict=i.verdict,
                criterion=i.criterion,
                evidence=i.evidence,
                failed_first=i.failed_first,
                cited_runs=list(i.cited_runs),
            )
            for i in board.items
        ],
        source=str(board.source),
        from_json=board.from_json,
    )


def gates(docs: Path) -> GatesOut:
    return GatesOut(docs=str(docs), boards=[board_out(b) for b in load_gate_boards(docs)])


# -- jobs ------------------------------------------------------------------------- #


class KnobOut(Out):
    name: str
    kind: str  # "number" | "integer" | "boolean" | "choice" | "tenors"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    choices: list[str] = Field(default_factory=list)
    group: str
    help: str = ""
    label: str = ""  # what the form calls it; ``name`` is the JobSpec field
    synthetic_only: bool = False  # read by the generator only; a real chain ignores it


# What the form calls each JobSpec field.
KNOB_LABELS: dict[str, str] = {
    "policy": "Policy",
    "iv_premium_vol_pts": "Planted premium (vol pts)",
    "sigma": "Path volatility",
    "r": "Risk-free rate",
    "n_days": "Days per episode",
    "tenor_days": "Tenors (trading days)",
    "n_strikes_each_side": "Strikes each side",
    "strike_step": "Strike step (points)",
    "costs_enabled": "Costs on",
    "min_half_spread": "Minimum half-spread",
    "half_spread_frac": "Half-spread, share of mid",
    "commission_per_contract": "Commission per contract ($)",
    "short_delta": "Short delta",
    "wing_delta": "Wing delta",
    "expiry_idx": "Opening tenor (0 = nearest)",
    "roll_dte": "Roll at days to expiry",
    "initial_cash": "Initial cash ($)",
    "seed": "First seed",
    "n_episodes": "Episodes",
    "total_timesteps": "Training steps",
    "source": "Data source",
}
# The generator's knobs a real chain does not read: its surface, path and carry are the
# day's own (``realdata.provider.HistoricalChainProvider._chain``); strikes, tenors and
# the episode length still apply.
SYNTHETIC_ONLY = frozenset({"iv_premium_vol_pts", "sigma", "r"})


class PresetOut(Out):
    name: str
    label: str
    description: str
    spec: dict[str, Any]


class JobOptionsOut(Out):
    arms: list[str]
    ppo_available: bool
    sources: list[str] = Field(default_factory=lambda: ["synthetic"])  # whose data is here
    # where a job on each source lands, relative to the runs root
    landing: dict[str, str] = Field(default_factory=dict)
    # A8: the dates a real-data job trains (PPO) and is evaluated on, ISO [first, last]
    splits: dict[str, list[str]] = Field(default_factory=dict)
    knobs: list[KnobOut]
    presets: list[PresetOut]
    defaults: dict[str, Any]
    ppo_preset: PresetOut | None = None  # listed whether or not the learner is installed


_GROUPS: dict[str, tuple[str, str]] = {
    "policy": (
        "Arm",
        "Which policy runs: the scripted arms, or PPO when the learner is installed.",
    ),
    "iv_premium_vol_pts": ("Market", "Planted premium in vol points. Zero is the null world."),
    "sigma": ("Market", "Annualised vol driving the path, decimal."),
    "r": ("Market", "Risk-free rate, decimal."),
    "n_days": ("Market", "Path and episode length in trading days."),
    "tenor_days": ("Market", "Tenors listed at every step, trading days."),
    "n_strikes_each_side": ("Market", "Strikes listed either side of the money."),
    "strike_step": ("Market", "Strike spacing in index points."),
    "costs_enabled": ("Costs", "Quoted half-spread plus commission; off for the fair-world check."),
    "min_half_spread": ("Costs", "Floor on the half-spread, index points per share."),
    "half_spread_frac": ("Costs", "Half-spread as a fraction of mid."),
    "commission_per_contract": ("Costs", "Dollars per contract."),
    "short_delta": ("Structure", "Delta of the short strikes."),
    "wing_delta": ("Structure", "Delta of the long wings (further out)."),
    "expiry_idx": ("Structure", "Which listed tenor new structures open at (0 = nearest)."),
    "roll_dte": ("Structure", "The seller closes at or below this many days to expiry."),
    "initial_cash": ("Account", "Starting cash, dollars."),
    "seed": ("Run", "First episode seed; episodes use seed, seed+1, ..."),
    "n_episodes": ("Run", "Evaluation episodes."),
    "total_timesteps": (
        "Training",
        "PPO training steps; the pilot and the sweep train 1,000,000.",
    ),
    "source": (
        "Data",
        "The generator, or the real SPX panel (evaluated on 2019-01-01..2022-08-24, PPO "
        "trained on the dates before); not in the run id, recorded in the manifest.",
    ),
}


def _knob(name: str, info: Any, arms: Sequence[str]) -> KnobOut:
    group, text = _GROUPS.get(name, ("Run", ""))
    default = info.default
    annotation = str(info.annotation)
    bounds: dict[str, float | None] = {
        "minimum": None,
        "maximum": None,
        "exclusive_minimum": None,
        "exclusive_maximum": None,
    }
    for meta in info.metadata:
        for attr, key in (
            ("ge", "minimum"),
            ("le", "maximum"),
            ("gt", "exclusive_minimum"),
            ("lt", "exclusive_maximum"),
        ):
            value = getattr(meta, attr, None)
            if value is not None:
                bounds[key] = float(value)
    if name == "policy":
        kind, choices = "choice", list(arms)
    elif name == "source":
        kind, choices = "choice", list(SOURCES)
    elif name == "tenor_days":
        kind, choices, default = "tenors", [], list(default)
    elif "bool" in annotation:
        kind, choices = "boolean", []
    elif "int" in annotation:
        kind, choices = "integer", []
    else:
        kind, choices = "number", []
    return KnobOut(
        name=name,
        kind=kind,
        default=default,
        choices=choices,
        group=group,
        help=text,
        label=KNOB_LABELS.get(name, name.replace("_", " ")),
        synthetic_only=name in SYNTHETIC_ONLY,
        **bounds,
    )


def job_options() -> JobOptionsOut:
    arms = available_arms()
    knobs = [_knob(name, info, arms) for name, info in JobSpec.model_fields.items()]
    if "ppo" not in arms:
        knobs = [k for k in knobs if k.name != "total_timesteps"]
    return JobOptionsOut(
        arms=list(arms),
        ppo_available="ppo" in arms,
        sources=list(available_sources()),
        landing={s: (source_root(Path(), s) / LAUNCH_DIR).as_posix() for s in SOURCES},
        splits={k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        knobs=knobs,
        presets=[
            PresetOut(
                name=p.name,
                label=p.label,
                description=p.description,
                spec=p.spec.model_dump(mode="json"),
            )
            for p in PRESETS
        ],
        defaults=JobSpec().model_dump(mode="json"),
    )


class JobOut(Out):
    job_id: str
    state: str
    phase: str
    done: int
    total: int
    fraction: float
    run_id: str
    run_key: str
    created_at: str
    policy: str
    error: str
    log_tail: list[str]
    source: str = "synthetic"


def job_out(status: JobStatus) -> JobOut:
    return JobOut(
        job_id=status.job_id,
        state=status.state,
        phase=status.phase,
        done=status.done,
        total=status.total,
        fraction=status.fraction,
        run_id=status.run_id,
        run_key=status.run_key,
        created_at=status.created_at,
        policy=status.policy,
        error=status.error,
        log_tail=list(status.log_tail),
        source=status.source,
    )


# -- the power lab ---------------------------------------------------------------- #

# Illustrative until the variance-components pilot reports; the pilot_budget script's
# own example values, in dollars of mean P&L per episode.
POWER_DEFAULTS = {"tau2": 250_000.0, "sigma2": 1.3e8, "delta": 1500.0}
FLOOR_SEEDS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100)
FRONTIER_SEEDS = (2, 3, 5, 8, 10, 16, 20, 32, 50, 64)


class PowerParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tau2: float = Field(default=POWER_DEFAULTS["tau2"], ge=0.0)
    sigma2: float = Field(default=POWER_DEFAULTS["sigma2"], ge=0.0)
    n_seeds: int = Field(default=10, ge=2, le=10_000)  # the t test needs two seed means
    n_episodes: int = Field(default=100, ge=1, le=1_000_000)
    delta: float = Field(default=POWER_DEFAULTS["delta"], gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    target_power: float = Field(default=0.80, gt=0.5, lt=1.0)
    train_steps: int = Field(default=1_000_000, ge=0)
    episode_days: int = Field(default=63, ge=1)
    train_rate: float = Field(default=1_000.0, gt=0.0)
    eval_rate: float = Field(default=1_800.0, gt=0.0)
    cores: int = Field(default=1, ge=1, le=4096)


class CurveOut(Out):
    delta: list[float]
    power: list[float]
    power_s5: list[float]
    power_s50: list[float]


class FloorOut(Out):
    seeds: list[int]
    floor: list[float]
    floor_s5: float
    floor_s50: float
    ratio: float  # floor at S=5 over S=50: 3.8 when tau^2 dominates, sqrt(10) by the normal form


class DesignOut(Out):
    n_seeds: int
    n_episodes: int | None
    power: float | None
    train_hours: float | None
    eval_hours: float | None
    total_hours: float | None


class PowerOut(Out):
    params: PowerParams
    se: float
    var_grand_mean: float
    variance_budget: float
    power: float
    floor: float  # smallest effect detected at the target power with this design
    episodes_needed: int | None
    seeds_needed: int
    curve: CurveOut
    floor_vs_seeds: FloorOut
    frontier: list[DesignOut]
    design: DesignOut
    source: str


def _floor(p: PowerParams, n_seeds: int) -> float:
    return pw.detectable_effect_t(p.tau2, p.sigma2, n_seeds, p.n_episodes, p.alpha, p.target_power)


def _design(d: pw.Design) -> DesignOut:
    b = d.budget
    return DesignOut(
        n_seeds=d.n_seeds,
        n_episodes=d.n_episodes,
        power=_f(d.power),
        train_hours=_f(b.train_hours) if b else None,
        eval_hours=_f(b.eval_hours) if b else None,
        total_hours=_f(b.total_hours) if b else None,
    )


def power_lab(p: PowerParams) -> PowerOut:
    """Power, floor and frontier for one design, from ``optspread.power``. Instant.

    Every figure is the exact noncentral t of the committed test (#20), the frontier
    and the episodes and seeds needed included; nothing here is the normal form.
    """
    var = pw.var_grand_mean(p.tau2, p.sigma2, p.n_seeds, p.n_episodes)
    floor = _floor(p, p.n_seeds)
    top = max(2.0 * p.delta, 2.5 * max(floor, _floor(p, 5) / 2.0), 1.0)
    grid = np.linspace(0.0, top, 81)

    def curve(n_seeds: int) -> list[float]:
        return [
            pw.power_t(float(d), p.tau2, p.sigma2, n_seeds, p.n_episodes, p.alpha) for d in grid
        ]

    floors = [_floor(p, s) for s in FLOOR_SEEDS]
    designs = pw.frontier(
        p.delta,
        p.tau2,
        p.sigma2,
        FRONTIER_SEEDS,
        alpha=p.alpha,
        target_power=p.target_power,
        train_steps_per_seed=p.train_steps,
        episode_days=p.episode_days,
        train_rate=p.train_rate,
        eval_rate=p.eval_rate,
        cores=p.cores,
    )
    now = pw.budget(
        p.n_seeds,
        p.n_episodes,
        train_steps_per_seed=p.train_steps,
        episode_days=p.episode_days,
        train_rate=p.train_rate,
        eval_rate=p.eval_rate,
        cores=p.cores,
    )
    power_now = pw.power_t(p.delta, p.tau2, p.sigma2, p.n_seeds, p.n_episodes, p.alpha)
    return PowerOut(
        params=p,
        se=math.sqrt(var),
        var_grand_mean=var,
        variance_budget=pw.variance_budget_t(p.delta, p.n_seeds, p.alpha, p.target_power),
        power=power_now,
        floor=floor,
        episodes_needed=pw.episodes_for(
            p.delta, p.tau2, p.sigma2, p.n_seeds, p.alpha, p.target_power
        ),
        seeds_needed=pw.seeds_for(p.delta, p.tau2, p.sigma2, p.n_episodes, p.alpha, p.target_power),
        curve=CurveOut(
            delta=[float(d) for d in grid],
            power=curve(p.n_seeds),
            power_s5=curve(5),
            power_s50=curve(50),
        ),
        floor_vs_seeds=FloorOut(
            seeds=list(FLOOR_SEEDS),
            floor=floors,
            floor_s5=_floor(p, 5),
            floor_s50=_floor(p, 50),
            ratio=_floor(p, 5) / _floor(p, 50) if _floor(p, 50) > 0 else float("nan"),
        ),
        frontier=[_design(d) for d in designs],
        design=DesignOut(
            n_seeds=p.n_seeds,
            n_episodes=p.n_episodes,
            power=power_now,
            train_hours=now.train_hours,
            eval_hours=now.eval_hours,
            total_hours=now.total_hours,
        ),
        source=(
            "optspread.power: Var(grand mean) = tau^2/S + sigma^2/(S N), one-sided t test "
            "on the S seed means, exact noncentral t"
        ),
    )


# -- the power surface ------------------------------------------------------------ #

SURFACE_PREMIUMS = 41  # points along the premium axis


class PowerSurfaceParams(Out):
    """The power lab's variance components and N, and the premium axis to draw.

    ``delta`` is the effect, in dollars of mean P&L per episode, at one planted vol
    point (A6 takes the fixed-delta seller's contrast at 1 vol point); the surface
    scales it linearly with the premium and holds tau^2 and sigma^2 fixed.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tau2: float = Field(default=POWER_DEFAULTS["tau2"], ge=0.0)
    sigma2: float = Field(default=POWER_DEFAULTS["sigma2"], ge=0.0)
    n_episodes: int = Field(default=100, ge=1, le=1_000_000)
    delta: float = Field(default=POWER_DEFAULTS["delta"], gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    target_power: float = Field(default=0.80, gt=0.5, lt=1.0)
    max_premium: float = Field(default=4.0, gt=0.0, le=20.0)
    max_seeds: int = Field(default=100, ge=3, le=200)


class PowerSurfaceOut(Out):
    params: PowerSurfaceParams
    premium: list[float]  # (P,) vol points
    seeds: list[int]  # (S,)
    power: list[list[float | None]]  # (S, P) exact power of the one-sided t test
    contour_premium: list[float | None]  # (S,) the premium detected at the target power
    source: str


def surface_seeds(max_seeds: int) -> list[int]:
    """Every S to 30, then every fourth: the curvature is at small S."""
    seeds = list(range(2, min(30, max_seeds) + 1))
    seeds += list(range(32, max_seeds + 1, 4))
    if seeds[-1] != max_seeds:
        seeds.append(max_seeds)
    return seeds


@lru_cache(maxsize=64)  # about a second each; sliders come back to the same designs
def power_surface(p: PowerSurfaceParams) -> PowerSurfaceOut:
    """Exact detection probability over planted premium x seeds, at the page's N.

    Each cell is ``optspread.power.power_t`` at effect ``delta * premium``; the
    contour is ``detectable_effect_t`` over ``delta``, so it is exact rather than
    read off the grid.
    """
    premiums = np.linspace(0.0, p.max_premium, SURFACE_PREMIUMS)
    seeds = surface_seeds(p.max_seeds)
    grid = [
        [
            _f(pw.power_t(p.delta * float(x), p.tau2, p.sigma2, s, p.n_episodes, p.alpha))
            for x in premiums
        ]
        for s in seeds
    ]
    contour = [
        pw.detectable_effect_t(p.tau2, p.sigma2, s, p.n_episodes, p.alpha, p.target_power) / p.delta
        for s in seeds
    ]
    return PowerSurfaceOut(
        params=p,
        premium=[float(x) for x in premiums],
        seeds=seeds,
        power=grid,
        contour_premium=contour,
        source="optspread.power.power_t and detectable_effect_t: noncentral t, S - 1 df",
    )


# -- the leak demo ---------------------------------------------------------------- #


class FeatureOut(Out):
    name: str
    tested: bool
    t_stat: float | None
    p_value: float | None
    reason: str


class LeakSideOut(Out):
    flagged: bool
    worst_feature: str
    worst_t: float | None
    p_value: float | None
    threshold: float
    n_pairs: int
    n_tested: int
    past_unchanged: bool  # the perturbation property: resampling the future leaves t alone
    features: list[FeatureOut]
    t_critical: float | None = None  # the |t| a field must exceed to be flagged


class LeakDemoOut(Out):
    eps: float
    seeds: list[int]
    clean: LeakSideOut
    poisoned: LeakSideOut
    caught: bool
    seconds: float
    criterion: str
    source: str


def leak_t_critical(threshold: float, n_pairs: int) -> float | None:
    """The two-sided |t| at which a field's p falls below the Bonferroni threshold."""
    from scipy import stats

    if n_pairs <= 2 or not 0.0 < threshold < 1.0:
        return None
    return _f(float(stats.t.isf(threshold / 2.0, df=n_pairs - 2)))


def _side(report: LeakReport, past_unchanged: bool) -> LeakSideOut:
    return LeakSideOut(
        flagged=report.flagged,
        worst_feature=report.worst_feature,
        worst_t=_f(report.worst_t),
        p_value=_f(report.p_value),
        threshold=report.threshold,
        n_pairs=report.n_pairs,
        n_tested=report.n_features_tested,
        past_unchanged=past_unchanged,
        t_critical=leak_t_critical(report.threshold, report.n_pairs),
        features=[
            FeatureOut(
                name=f.name,
                tested=f.tested,
                t_stat=_f(f.t_stat) if f.tested else None,
                p_value=_f(f.p_value) if f.tested else None,
                reason=f.reason,
            )
            for f in report.features
        ],
    )


def leak_demo(demo: LeakDemo) -> LeakDemoOut:
    return LeakDemoOut(
        eps=demo.eps,
        seeds=list(demo.seeds),
        clean=_side(demo.clean, demo.clean_past_unchanged),
        poisoned=_side(demo.poisoned, demo.poisoned_past_unchanged),
        caught=demo.caught,
        seconds=demo.seconds,
        criterion=demo.criterion,
        source="optspread.leaks: detect_leak and perturbation_check, run live on default_env",
    )


class LeakDemoParams(Out):
    eps: float = Field(default=0.5, gt=0.0, le=2.0)
    n_seeds: int = Field(default=10, ge=3, le=40)


# -- sweeps ----------------------------------------------------------------------- #


class SweepSummaryOut(Out):
    key: str
    sweep_hash: str
    premiums: list[float]
    arms: list[str]
    n_cells: int
    seed: int
    n_episodes: int
    kind: str = "sweep"  # "sweep": a scripted sweep.json; "grid": an orchestrated grid-*.json
    n_seeds: int = 1  # seed blocks; a scripted sweep is one block on shared seeds
    n_present: int = 0  # cells whose run directory is under the root


class OraclePointOut(Out):
    premium: float
    mean_pnl: float
    se_pnl: float
    lift: float  # against the first premium for the same arm: a paired difference
    run_id: str


class GridCellOut(Out):
    premium: float
    arm: str
    block: int
    run_id: str
    key: str  # the cell's run key under the runs root, for /runs/?key=
    train_seed: int | None
    mean_pnl: float | None
    se_pnl: float | None
    mean_cost: float | None
    mean_trades: float | None
    present: bool


class ArmMeanOut(Out):
    """One arm at one premium over the seed blocks: the grand mean and the se of block means."""

    premium: float
    mean: float | None
    se: float | None
    n_blocks: int
    block_means: list[float | None]


class ContrastOut(Out):
    """The detection event read off the grid: the arm at p minus itself at 0, block by block.

    The pre-registered rule (``analysis.decide_premium``, A1 and A2) on the cells' episodes.
    """

    premium: float
    estimate: float | None
    se: float | None
    t: float | None
    df: int
    critical: float | None
    lower_bound: float | None
    detected: bool
    block_means: list[float | None]
    reason: str = ""  # why there is no decision, when there is none


class GridDetailOut(Out):
    grid_hash: str
    n_seeds: int
    n_episodes: int
    premiums: list[float]
    arms: list[str]
    total_timesteps: int | None
    first_path_seed: int
    costs: bool
    cells: list[GridCellOut]
    means: dict[str, list[ArmMeanOut]]
    contrasts: dict[str, list[ContrastOut]]
    missing: list[str]  # cells listed in the record whose run is not under the root
    git_shas: list[str]


class SweepOut(Out):
    summary: SweepSummaryOut
    oracle: dict[str, list[OraclePointOut]]
    source: str
    kind: str = "sweep"
    grid: GridDetailOut | None = None


class SweepListOut(Out):
    root: str
    sweeps: list[SweepSummaryOut]


GRID_PATTERN = "grid-*.json"


def _sweep_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob(SWEEP_FILENAME)) if root.is_dir() else []


def _grid_files(root: Path) -> list[Path]:
    """Every orchestrated grid record under the root, the job queue's bookkeeping aside."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob(GRID_PATTERN) if ".jobs" not in p.parts and p.is_file())


def _grid_summary(path: Path, root: Path, record: GridRecord) -> SweepSummaryOut:
    grid = record.grid
    present = sum((path.parent / c.run_id / MANIFEST_FILENAME).is_file() for c in record.cells)
    return SweepSummaryOut(
        key=path.relative_to(root).as_posix(),
        sweep_hash=record.grid_hash,
        premiums=sorted(grid.premiums),
        arms=list(grid.arms),
        n_cells=len(record.cells),
        seed=grid.base.seed,
        n_episodes=grid.base.n_episodes,
        kind="grid",
        n_seeds=grid.n_seeds,
        n_present=present,
    )


def sweeps(root: Path) -> SweepListOut:
    out: list[SweepSummaryOut] = []
    for d in _sweep_dirs(root):
        try:
            rec = read_sweep(d)
        except (OSError, ValueError, SchemaVersionError):
            continue
        out.append(
            SweepSummaryOut(
                key=d.relative_to(root).as_posix(),
                sweep_hash=rec.sweep_hash,
                premiums=list(rec.premiums),
                arms=list(rec.arms),
                n_cells=len(rec.cells),
                seed=rec.seed,
                n_episodes=rec.n_episodes,
                n_present=len(rec.cells),
            )
        )
    for path in _grid_files(root):
        try:
            record = read_grid(path)
        except (OSError, ValueError, SchemaVersionError):
            continue
        out.append(_grid_summary(path, root, record))
    return SweepListOut(root=str(root), sweeps=out)


def _arm_mean(premium: float, cells: Sequence[GridCellOut]) -> ArmMeanOut:
    means = np.asarray([c.mean_pnl for c in cells if c.mean_pnl is not None], dtype=np.float64)
    n = len(means)
    return ArmMeanOut(
        premium=premium,
        mean=_f(float(means.mean())) if n else None,
        se=_f(float(means.std(ddof=1) / math.sqrt(n))) if n > 1 else None,
        n_blocks=n,
        block_means=[c.mean_pnl for c in cells],
    )


def _contrast(root: Path, record: GridRecord, premium: float, arm: ArmName) -> ContrastOut:
    try:
        treated = record.matrix(root, premium, arm)
        null = record.matrix(root, NULL_PREMIUM, arm)
        decision = decide_premium(treated, null)
        block = paired_contrast(treated, null).mean(axis=1)
    except (OSError, ValueError, KeyError, SchemaVersionError) as exc:
        return ContrastOut(
            premium=premium,
            estimate=None,
            se=None,
            t=None,
            df=record.grid.n_seeds - 1,
            critical=None,
            lower_bound=None,
            detected=False,
            block_means=[],
            reason=str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
    return ContrastOut(
        premium=premium,
        estimate=_f(decision.estimate),
        se=_f(decision.se),
        t=_f(decision.t_stat),
        df=decision.df,
        critical=_f(decision.critical),
        lower_bound=_f(decision.lower_bound),
        detected=decision.rejected,
        block_means=_fl(block),
    )


def grid_detail(path: Path, root: Path) -> SweepOut:
    """An orchestrated grid: per-arm means by premium with the se of block means, every
    cell, and the pre-registered contrast at each premium above zero."""
    record = read_grid(path)
    grid = record.grid
    base_dir = path.parent
    cells: list[GridCellOut] = []
    missing: list[str] = []
    shas: set[str] = set()
    for c in sorted(record.cells, key=lambda c: (c.arm, c.premium, c.block)):
        run_dir = base_dir / c.run_id
        present = (run_dir / MANIFEST_FILENAME).is_file()
        if present:
            with contextlib.suppress(OSError, ValueError):
                shas.add(RunManifest.read(run_dir).git_sha)
        else:
            missing.append(f"{c.arm} at {c.premium:g}, block {c.block}: {c.run_id}")
        cells.append(
            GridCellOut(
                premium=c.premium,
                arm=c.arm,
                block=c.block,
                run_id=c.run_id,
                key=run_dir.relative_to(root).as_posix(),
                train_seed=c.train_seed,
                mean_pnl=_f(c.mean_pnl),
                se_pnl=_f(c.se_pnl),
                mean_cost=_f(c.mean_cost),
                mean_trades=_f(c.mean_trades),
                present=present,
            )
        )
    premiums = sorted(grid.premiums)
    means = {
        arm: [_arm_mean(p, [c for c in cells if c.arm == arm and c.premium == p]) for p in premiums]
        for arm in grid.arms
    }
    contrasts: dict[str, list[ContrastOut]] = {}
    if NULL_PREMIUM in grid.premiums and grid.n_seeds >= 2:
        contrasts = {
            arm: [_contrast(base_dir, record, p, arm) for p in premiums if p > NULL_PREMIUM]
            for arm in grid.arms
        }
    oracle = {
        arm: [
            OraclePointOut(
                premium=m.premium,
                mean_pnl=m.mean or 0.0,
                se_pnl=m.se or 0.0,
                lift=(m.mean or 0.0) - (pts[0].mean or 0.0),
                run_id=record.grid_hash,
            )
            for m in pts
            if m.mean is not None
        ]
        for arm, pts in means.items()
    }
    return SweepOut(
        summary=_grid_summary(path, root, record),
        oracle=oracle,
        source=str(path),
        kind="grid",
        grid=GridDetailOut(
            grid_hash=record.grid_hash,
            n_seeds=grid.n_seeds,
            n_episodes=grid.base.n_episodes,
            premiums=premiums,
            arms=list(grid.arms),
            total_timesteps=None if grid.train is None else grid.train.total_timesteps,
            first_path_seed=grid.base.seed,
            costs=grid.base.cost.enabled,
            cells=cells,
            means=means,
            contrasts=contrasts,
            missing=missing,
            git_shas=sorted(shas),
        ),
    )


def sweep_detail(root: Path, key: str) -> SweepOut:
    target = _inside(root, key)
    if target is not None and target.is_file() and target.match(GRID_PATTERN):
        try:
            return grid_detail(target, root)
        except (OSError, ValueError, SchemaVersionError) as exc:
            raise KeyError(key) from exc
    if target is None or not (target / SWEEP_FILENAME).is_file():
        raise KeyError(key)
    rec = read_sweep(target)
    oracle: dict[str, list[OraclePointOut]] = {}
    for arm in rec.arms:
        base = rec.cell(rec.premiums[0], arm).mean_pnl
        oracle[arm] = [
            OraclePointOut(
                premium=c.premium,
                mean_pnl=c.mean_pnl,
                se_pnl=c.se_pnl,
                lift=c.mean_pnl - base,
                run_id=c.run_id,
            )
            for c in (rec.cell(p, arm) for p in rec.premiums)
        ]
    return SweepOut(
        summary=SweepSummaryOut(
            key=key,
            sweep_hash=rec.sweep_hash,
            premiums=list(rec.premiums),
            arms=list(rec.arms),
            n_cells=len(rec.cells),
            seed=rec.seed,
            n_episodes=rec.n_episodes,
        ),
        oracle=oracle,
        source=str(target / SWEEP_FILENAME),
    )


# -- the action mix ---------------------------------------------------------------- #

_RUN_ID = re.compile(r"[0-9a-f]+-s\d+")
_GRID_HASH = re.compile(r"[0-9a-f]+")


class ActionMixOut(Out):
    """Which structures a policy trades (``optspread.actionmix``), read from the files
    ``scripts/action_mix.py`` wrote: one run, its episodes without their day-by-day
    lists, or one grid, per cell and pooled per arm and premium. Never computed here:
    a ppo run's re-simulation needs its learner, and a cell of 500 episodes takes
    half a minute."""

    kind: str  # "run" or "grid"
    path: str
    run: am.RunActionMix | None = None
    grid: am.GridActionMix | None = None


def action_mix(
    root: Path | None = None,
    *,
    run: str | None = None,
    grid: str | None = None,
    source: str = "synthetic",
) -> ActionMixOut:
    """The action mix of ``run`` (a run id, or a run key ending in one) or of ``grid`` (a
    grid hash) under ``root`` (``actionmix.default_root()`` when None). ``KeyError`` when
    it has not been written; ``ValueError`` for a request that names neither or both, or
    a name that is not a run id or grid hash."""
    if (run is None) == (grid is None):
        raise ValueError("ask for one run or one grid")
    if source not in ("synthetic", "optionmetrics_stub", "optionmetrics"):
        raise ValueError(f"unknown source {source!r}")
    folder = root if root is not None else am.default_root()
    suffix = "" if source == "synthetic" else f".{source}"
    if run is not None:
        rid = run.rstrip("/").rsplit("/", 1)[-1]
        if not _RUN_ID.fullmatch(rid):
            raise ValueError(f"{run!r} is not a run id")
        path = folder / f"{rid}{suffix}.json"
        if not path.is_file():
            raise KeyError(rid)
        return ActionMixOut(kind="run", path=str(path), run=am.read_mix(path).lean())
    assert grid is not None
    gh = grid.removeprefix("grid-").removesuffix(".json")
    if not _GRID_HASH.fullmatch(gh):
        raise ValueError(f"{grid!r} is not a grid hash")
    path = folder / f"grid-{gh}{suffix}.json"
    if not path.is_file():
        raise KeyError(gh)
    return ActionMixOut(kind="grid", path=str(path), grid=am.read_grid_mix(path))


# -- the SPX panel ---------------------------------------------------------------- #


class PanelFileOut(Out):
    path: str
    present: bool
    modified: str | None


class DataStatusOut(Out):
    root: str
    built: bool
    missing: list[str]
    years: list[int]
    files: list[PanelFileOut]


class SpanOut(Out):
    source: str
    first: str
    last: str
    n_days: int
    n_missing: int
    median_contracts: float


class StripOut(Out):
    source: str
    date: list[str]
    n_contracts: list[float]
    missing: list[bool]


class CoverageOut(Out):
    spans: list[SpanOut]
    missing: list[tuple[str, str]]
    overlap: tuple[str, str, int] | None
    strips: list[StripOut]
    source: str


class ExpiryOut(Out):
    expiry: str
    dte: int
    n_quotes: int
    forward: float | None
    carry_fit: str  # the build's carry-fit label, e.g. "near+r_ref"; empty when absent


class ChainIndexOut(Out):
    date: str
    sources: list[str]
    roots: list[str]
    expiries: list[ExpiryOut]
    default_expiry: str | None
    source: str


class SmileOut(Out):
    date: str
    expiry: str
    dte: int
    data_source: str
    moneyness: list[float | None]
    strike: list[float | None]
    iv: list[float | None]
    iv_vendor: list[float | None]
    cp: list[str]
    has_vendor: bool
    forward: float | None
    source: str


class SurfaceOut(Out):
    date: str
    data_source: str
    moneyness: list[float]
    dte: list[int]
    iv: list[list[float | None]]
    source: str


class TermOut(Out):
    date: str
    data_source: str
    expiry: list[str]
    dte: list[int]
    atm_iv: list[float | None]
    source: str


class SeriesOut(Out):
    data_source: str
    date: list[str]
    spot: list[float | None]
    spot_parity: list[float | None]
    iv30: list[float | None]
    rv21: list[float | None]
    rv21_fwd: list[float | None]
    vrp30: list[float | None]


class VrpStatsOut(Out):
    n_days: int
    mean_vol_pts: float | None
    median_vol_pts: float | None
    share_positive: float | None
    first: str | None
    last: str | None


class DailyOut(Out):
    series: list[SeriesOut]
    stats: dict[str, VrpStatsOut]
    source: str


class SeamOut(Out):
    available: bool
    date: list[str]
    stats: dict[str, list[float | None]]
    medians: dict[str, float | None]
    # the headline (dashboard_panel.SEAM_HEADLINE): each day's 95th-percentile |IV gap|,
    # averaged over the overlap days, and the worst day; what docs/DATA.md quotes
    p95_mean: float | None = None
    p95_worst: float | None = None
    iv30_sources: list[str]
    iv30_date: list[str]
    iv30_diff_vol_pts: list[float | None]
    source: str


class GridOut(Out):
    """One day of the gridded surface: what the 3D view draws."""

    date: str
    k: list[float]  # (K,) log-moneyness ln(K / F)
    tenor_days: list[int]  # (T,) calendar days
    iv: list[list[float | None]]  # (T, K); null where nothing was quoted
    source: str


class GridFrameOut(Out):
    date: str
    iv: list[list[float | None]]


class GridFramesOut(Out):
    """Consecutive days of the gridded surface on one set of axes, for playback."""

    k: list[float]
    tenor_days: list[int]
    frames: list[GridFrameOut]
    more: bool  # the window was cut at max_frames; ask again from the last date
    source: str


class LandscapeOut(Out):
    """ATM IV over date x tenor: the term structure through time."""

    date: list[str]  # (D,) the last trading day of each row's bucket
    tenor_days: list[int]  # (T,)
    iv: list[list[float | None]]  # (D, T)
    stride: int  # trading days averaged into each row
    n_days: int  # trading days in the window before averaging
    first: str | None
    last: str | None
    source: str


SIGMA_FROM = ("config", "rv21", "atm")


class RegionOut(Out):
    """Real less simulated over the simulator's contracts in one moneyness region."""

    name: str  # "put" (ln(K/F) < -0.03), "atm" or "call"
    label: str
    mean_diff: float | None  # vol points
    n: int


class WidestOut(Out):
    """The simulator's contract where the real surface is furthest from it."""

    strike: float
    tenor_trading_days: int
    moneyness: float  # K / F
    diff: float  # vol points, real less simulated
    real: float
    sim: float


class SimulatedOut(Out):
    """The simulator's surface on one day, priced by ``optspread.market``, beside the real one.

    The synthetic world prices every listed contract at ``sigma + premium``: one
    number over its strike and tenor grid. ``real`` is the real surface read off at
    the same contracts, so ``real - iv`` is what the synthetic world leaves out.
    """

    date: str
    spot: float
    spot_from: str  # "spx_daily.parquet" or "config"
    sigma: float
    sigma_from: str  # "config" (MarketConfig.sigma), "rv21" (the day's), "atm" (IV30 - premium)
    premium: float  # vol points
    level: float  # sigma + premium / 100: every contract's IV
    r: float
    strike: list[float]  # (K,)
    tenor_trading_days: list[int]  # (T,) as MarketConfig lists them
    tenor_days: list[float]  # (T,) the same tenors in calendar days
    k: list[list[float]]  # (T, K) ln(K / F) per tenor
    iv: list[list[float]]  # (T, K)
    real: list[list[float | None]]  # (T, K) the real surface at the same contracts
    mean_diff: float | None  # real - simulated over the contracts both price, decimals
    mean_abs_diff: float | None
    max_abs_diff: float | None
    source: str
    # Where they differ most, for the data page's question 3 (vol points).
    regions: list[RegionOut] = Field(default_factory=list)
    widest: WidestOut | None = None
    reading: str = ""


def simulated_surface(
    day: date, cfg: MarketConfig, real: panel.GridSurface, spot_from: str, sigma_from: str
) -> SimulatedOut:
    """The chain ``SyntheticChainProvider`` lists at its first step, against the real grid.

    The provider is the simulator's own: its strikes around the spot, its tenors in
    trading days, its IVs. Each contract is placed at ``ln(K / F)`` with ``F`` the
    forward at the simulator's rate, and its tenor in calendar days, so it lands on
    the axes of the real surface.
    """
    chain = SyntheticChainProvider(cfg).reset(SeededStepRNG(0))
    taus = [chain.tau(int(e)) for e in chain.expiry_days]
    ks = [np.log(chain.strikes / (chain.spot * math.exp(chain.r * t))) for t in taus]
    real_at = [real.at(k, t * 365.0) for k, t in zip(ks, taus, strict=True)]
    diff = np.asarray(real_at) - chain.ivs
    ok = np.isfinite(diff)
    k_all = np.asarray(ks)
    regions, widest = panel.sim_regions(k_all, np.asarray(real_at), np.asarray(chain.ivs))
    widest_out = (
        WidestOut(
            strike=float(chain.strikes[widest.col]),
            tenor_trading_days=int(cfg.tenor_days[widest.row]),
            moneyness=float(math.exp(k_all[widest.row, widest.col])),
            diff=widest.diff,
            real=widest.real,
            sim=widest.sim,
        )
        if widest is not None
        else None
    )
    level = cfg.sigma + cfg.iv_premium_vol_pts / 100.0
    return SimulatedOut(
        date=day.isoformat(),
        spot=chain.spot,
        spot_from=spot_from,
        sigma=cfg.sigma,
        sigma_from=sigma_from,
        premium=cfg.iv_premium_vol_pts,
        level=cfg.sigma + cfg.iv_premium_vol_pts / 100.0,
        r=chain.r,
        strike=[float(x) for x in chain.strikes],
        tenor_trading_days=[int(d) for d in cfg.tenor_days],
        tenor_days=[t * 365.0 for t in taus],
        k=[[float(x) for x in k] for k in ks],
        iv=[[float(x) for x in row] for row in chain.ivs],
        real=[_fl(row) for row in real_at],
        mean_diff=_f(float(diff[ok].mean())) if ok.any() else None,
        mean_abs_diff=_f(float(np.abs(diff[ok]).mean())) if ok.any() else None,
        max_abs_diff=_f(float(np.abs(diff[ok]).max())) if ok.any() else None,
        source=(
            "optspread.market.SyntheticChainProvider on MarketConfig("
            f"spot0={cfg.spot0:.2f}, sigma={cfg.sigma:.4f}, "
            f"iv_premium_vol_pts={cfg.iv_premium_vol_pts:g})"
        ),
        regions=[
            RegionOut(name=r.name, label=r.label, mean_diff=_f(r.mean_diff), n=r.n) for r in regions
        ],
        widest=widest_out,
        reading=panel.sim_reading(
            level,
            regions,
            widest,
            widest_out.strike if widest_out else None,
            widest_out.tenor_trading_days if widest_out else None,
        ),
    )


# -- the data page's questions: each view's numbers and its sentence --------------- #


class PremiumNowOut(Out):
    date: str  # the last day with an IV30
    iv30: float | None  # vol points
    rv_trailing: float | None  # vol points, the 21 days before, corrected
    priced: float | None  # vol points, IV30 less rv_trailing: known on the day
    measured_date: str | None  # the last day whose forward window is complete
    measured: float | None  # vol points, that day's premium
    recent: float | None  # vol points, the quarter's mean to measured_date
    recent_share_below: float | None  # share of quarters with a lower mean


class PremiumOut(Out):
    """Question 1: the VRP mapping's premium day by day, its smoothed line and a reading.

    ``premium`` is the mapping's definition (docs/VRP_MAPPING.md), so its mean over the
    days it is defined is the headline; ``smooth`` its trailing mean over
    ``smooth_days``. Every series is in vol points.
    """

    date: list[str]
    premium: list[float | None]
    smooth: list[float | None]
    iv30: list[float | None]
    rv_fwd: list[float | None]
    rv_trailing: list[float | None]
    smooth_days: int
    now: PremiumNowOut | None
    mean: float | None  # the headline: the mapping's when there is one, else this series'
    mean_low: float | None
    mean_high: float | None
    bracket: list[float | None]
    sweep_axis: list[float]
    reading: str
    source: str


class ShapeOut(Out):
    """Question 2: the surface's level, skew and term slope on one day, against every day."""

    date: str
    atm: float | None  # vol points, 30-day at the money
    skew: float | None  # vol points, the put at put_k less atm
    call: float | None  # vol points, the call at call_k less atm
    term: float | None  # vol points, ATM at long_tenor less atm
    atm_share_below: float | None
    skew_share_below: float | None
    term_share_below: float | None
    share_inverted: float | None
    skew_median: float | None
    term_median: float | None
    tenor: int
    put_k: float
    call_k: float
    long_tenor: int
    reading: str
    source: str


class FredRowOut(Out):
    year: int
    source: str
    rate_minus_bill_bp: float | None
    rate_minus_sofr_bp: float | None
    spot_vs_sp500_bp: float | None


class PanelFileCheckOut(Out):
    path: str
    recorded: str | None  # sha256 docs/DATA_MANIFEST.json records
    actual: str | None  # sha256 of the file served
    match: bool | None


class QualityOut(Out):
    """The data-quality section: the build's version, its files against the manifest, FRED."""

    panel_version: int | None
    manifest_files: int
    checked: list[PanelFileCheckOut]
    of_record: bool | None  # every checked file matches the manifest; None when unchecked
    fred: list[FredRowOut]
    fred_available: bool
    reading: str
    source: str


DATA_MANIFEST_FILE = "DATA_MANIFEST.json"


class PanelNotBuiltError(LookupError):
    """The view needs a panel file the build has not written yet."""


def _day_of(value: np.datetime64) -> date:
    out = value.astype("datetime64[D]").astype(date)
    assert isinstance(out, date)
    return out


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """docs/DATA_MANIFEST.json as a dict; empty, with the reason, when it does not read."""
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {}, str(exc)
    return (raw, "") if isinstance(raw, dict) else ({}, f"{path.name} is not a JSON object")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


# The largest windows one request may ask for: a year of gridded frames, and every
# trading day of the panel (3,939 days to 2025-08-29) with room to spare.
MAX_FRAMES = 252
MAX_DATES = 4_200


class PanelReader:
    """The SPX panel, read lazily and cached by file modification time.

    The two small tables are read whole once per rebuild; chain partitions are read
    one date at a time and the last few dates kept, which is what flicking through
    a date picker needs.
    """

    def __init__(
        self,
        root: Path,
        chain_cache: int = 8,
        read: Callable[..., panel.Columns] = panel.read_table,
    ) -> None:
        self.layout = panel.PanelLayout(root)
        self._read = read  # injected so the payloads are tested without pyarrow
        self._tables: dict[str, tuple[float, panel.Columns]] = {}
        self._chains: OrderedDict[tuple[date, float], panel.Columns] = OrderedDict()
        self._chain_cache = chain_cache
        self._computed: dict[str, tuple[float, Any]] = {}

    def _table(self, name: str, path: Path) -> panel.Columns:
        if not path.is_file():
            raise PanelNotBuiltError(str(path))
        stamp = _mtime(path)
        cached = self._tables.get(name)
        if cached is None or cached[0] != stamp:
            cached = (stamp, self._read(path))
            self._tables[name] = cached
        return cached[1]

    def daily(self) -> panel.Columns:
        return self._table("daily", self.layout.daily)

    def coverage(self) -> panel.Columns:
        return self._table("coverage", self.layout.coverage)

    def seam(self) -> panel.Columns:
        if not self.layout.seam_overlap.is_file():
            return {}
        return self._table("seam", self.layout.seam_overlap)

    def chain(self, day: date) -> panel.Columns:
        path = self.layout.chain_file(day.year)
        if not path.is_file():
            raise PanelNotBuiltError(str(path))
        key = (day, _mtime(path))
        if key not in self._chains:
            self._chains[key] = self._read(path, panel.CHAIN_COLUMNS, day)
            while len(self._chains) > self._chain_cache:
                self._chains.popitem(last=False)
        self._chains.move_to_end(key)
        return self._chains[key]

    # -- payloads ------------------------------------------------------------- #

    def status(self) -> DataStatusOut:
        layout = self.layout
        files = [layout.daily, layout.coverage, layout.seam_overlap]
        files += [layout.chain_file(y) for y in layout.years()]
        return DataStatusOut(
            root=str(layout.root),
            built=layout.built,
            missing=list(layout.missing()),
            years=list(layout.years()),
            files=[
                PanelFileOut(
                    path=str(f.relative_to(layout.root)),
                    present=f.is_file(),
                    modified=str(np.datetime64(int(_mtime(f)), "s")) if f.is_file() else None,
                )
                for f in files
            ],
        )

    def coverage_out(self) -> CoverageOut:
        cov = self.coverage()
        summary = panel.coverage_summary(cov)
        strips = []
        for src in summary.sources:
            mine = panel.take(cov, cov["source"] == src)
            order = np.argsort(mine["date"], kind="stable")
            strips.append(
                StripOut(
                    source=src,
                    date=_iso(mine["date"][order]),
                    n_contracts=[float(v) for v in mine["n_contracts"][order]],
                    missing=[bool(v) for v in mine["missing"][order]]
                    if "missing" in mine
                    else [False] * len(order),
                )
            )
        return CoverageOut(
            spans=[
                SpanOut(
                    source=s.source,
                    first=s.first.isoformat(),
                    last=s.last.isoformat(),
                    n_days=s.n_days,
                    n_missing=s.n_missing,
                    median_contracts=s.median_contracts,
                )
                for s in summary.spans
            ],
            missing=[(d.isoformat(), s) for d, s in summary.missing],
            overlap=(
                summary.overlap[0].isoformat(),
                summary.overlap[1].isoformat(),
                summary.overlap[2],
            )
            if summary.overlap
            else None,
            strips=strips,
            source=str(self.layout.coverage),
        )

    def dates(self, data_source: str | None = None) -> list[str]:
        return _iso(panel.available_dates(self.coverage(), data_source))

    def _selected(self, day: date, data_source: str | None, roots: Sequence[str]) -> panel.Columns:
        chain = self.chain(day)
        if data_source is None:
            sources = panel.chain_sources(chain)
            data_source = sources[-1] if sources else None  # optionmetrics sorts last
        return panel.select(chain, data_source, roots)

    def chain_index(
        self, day: date, data_source: str | None, roots: Sequence[str]
    ) -> ChainIndexOut:
        chain = self.chain(day)
        rows = self._selected(day, data_source, roots)
        near = panel.nearest_expiry(rows)
        return ChainIndexOut(
            date=day.isoformat(),
            sources=list(panel.chain_sources(chain)),
            roots=sorted(set(chain["root"].tolist())) if panel.n_rows(chain) else [],
            expiries=[
                ExpiryOut(
                    expiry=e.expiry.isoformat(),
                    dte=e.dte,
                    n_quotes=e.n_quotes,
                    forward=_f(e.forward),
                    carry_fit=e.carry_fit,
                )
                for e in panel.expiries(rows)
            ],
            default_expiry=near.expiry.isoformat() if near else None,
            source=str(self.layout.chain_file(day.year)),
        )

    def _source_name(self, rows: panel.Columns) -> str:
        names = panel.chain_sources(rows)
        return names[0] if len(names) == 1 else ", ".join(names)

    def smile_out(
        self, day: date, expiry: date | None, data_source: str | None, roots: Sequence[str]
    ) -> SmileOut:
        rows = self._selected(day, data_source, roots)
        if expiry is None:
            near = panel.nearest_expiry(rows)
            if near is None:
                raise PanelNotBuiltError(f"no usable expiry on {day}")
            expiry = near.expiry
        s = panel.smile(rows, expiry)
        fwd = next((e.forward for e in panel.expiries(rows) if e.expiry == expiry), None)
        return SmileOut(
            date=day.isoformat(),
            expiry=expiry.isoformat(),
            dte=s.dte,
            data_source=self._source_name(rows),
            moneyness=_fl(s.moneyness),
            strike=_fl(s.strike),
            iv=_fl(s.iv),
            iv_vendor=_fl(s.iv_vendor),
            cp=[str(c) for c in s.cp],
            has_vendor=s.has_vendor,
            forward=_f(fwd),
            source=str(self.layout.chain_file(day.year)),
        )

    def surface_out(
        self,
        day: date,
        data_source: str | None,
        roots: Sequence[str],
        min_dte: int = 1,
        max_dte: int = 365,
    ) -> SurfaceOut:
        if min_dte > max_dte:
            raise ValueError(f"min_dte {min_dte} is above max_dte {max_dte}")
        rows = self._selected(day, data_source, roots)
        surf = panel.iv_surface(rows, min_dte=min_dte, max_dte=max_dte)
        return SurfaceOut(
            date=day.isoformat(),
            data_source=self._source_name(rows),
            moneyness=[float(m) for m in surf.moneyness],
            dte=[int(d) for d in surf.dte],
            iv=[_fl(row) for row in surf.iv],
            source=str(self.layout.chain_file(day.year)),
        )

    def term_out(self, day: date, data_source: str | None, roots: Sequence[str]) -> TermOut:
        rows = self._selected(day, data_source, roots)
        ts = panel.term_structure(rows)
        return TermOut(
            date=day.isoformat(),
            data_source=self._source_name(rows),
            expiry=[e.isoformat() for e in ts.expiry],
            dte=[int(d) for d in ts.dte],
            atm_iv=_fl(ts.atm_iv),
            source=str(self.layout.chain_file(day.year)),
        )

    def daily_out(self, start: date | None = None, end: date | None = None) -> DailyOut:
        daily = self.daily()
        if panel.n_rows(daily) and (start or end):
            keep = np.ones(panel.n_rows(daily), dtype=bool)
            if start:
                keep &= daily["date"] >= np.datetime64(start, "D")
            if end:
                keep &= daily["date"] <= np.datetime64(end, "D")
            daily = panel.take(daily, keep)
        series: list[SeriesOut] = []
        stats: dict[str, VrpStatsOut] = {}
        for src in panel.daily_sources(daily):
            s = panel.vrp_series(daily, src or None)
            series.append(
                SeriesOut(
                    data_source=src,
                    date=_iso(s.date),
                    spot=_fl(s.spot),
                    spot_parity=_fl(s.spot_parity),
                    iv30=_fl(s.iv30),
                    rv21=_fl(s.rv21),
                    rv21_fwd=_fl(s.rv21_fwd),
                    vrp30=_fl(s.vrp30),
                )
            )
            st = panel.vrp_stats(s)
            stats[src] = VrpStatsOut(
                n_days=st.n_days,
                mean_vol_pts=_f(st.mean_vol_pts),
                median_vol_pts=_f(st.median_vol_pts),
                share_positive=_f(st.share_positive),
                first=st.first.isoformat() if st.first else None,
                last=st.last.isoformat() if st.last else None,
            )
        return DailyOut(series=series, stats=stats, source=str(self.layout.daily))

    def seam_out(self) -> SeamOut:
        report = panel.seam_report(self.seam())
        iv_seam = panel.seam(self.daily()) if self.layout.daily.is_file() else None
        return SeamOut(
            available=report is not None,
            date=_iso(report.date) if report else [],
            stats={k: _fl(v) for k, v in report.stats.items()} if report else {},
            medians={k: _f(report.median(k)) for k in panel.SEAM_STATS} if report else {},
            p95_mean=_f(report.mean(panel.SEAM_HEADLINE)) if report else None,
            p95_worst=_f(report.worst(panel.SEAM_HEADLINE)) if report else None,
            iv30_sources=list(iv_seam.sources) if iv_seam else [],
            iv30_date=_iso(iv_seam.date) if iv_seam else [],
            iv30_diff_vol_pts=_fl(iv_seam.diff_vol_pts) if iv_seam else [],
            source=str(self.layout.seam_overlap),
        )

    # -- the 3D views ---------------------------------------------------------- #

    def surface_grid(self) -> panel.Columns:
        return self._table("surface_grid", self.layout.surface_grid)

    def atm_term(self) -> panel.Columns:
        return self._table("atm_term", self.layout.atm_term)

    def _derived(self, name: str, path: Path, build: Callable[[], Any]) -> Any:
        """A value computed from a table, kept until the file changes."""
        stamp = _mtime(path)
        cached = self._computed.get(name)
        if cached is None or cached[0] != stamp:
            cached = (stamp, build())
            self._computed[name] = cached
        return cached[1]

    def _grid_axes(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        grid = self.surface_grid()
        axes: tuple[NDArray[np.float64], NDArray[np.int64]] = self._derived(
            "grid_axes", self.layout.surface_grid, lambda: panel.grid_axes(grid)
        )
        return axes

    def _grid_dates(self) -> NDArray[np.datetime64]:
        grid = self.surface_grid()
        dates: NDArray[np.datetime64] = self._derived(
            "grid_dates", self.layout.surface_grid, lambda: panel.grid_dates(grid)
        )
        return dates

    def grid_dates(self) -> list[str]:
        return _iso(self._grid_dates())

    def grid_day(self, day: date) -> panel.GridSurface:
        if np.datetime64(day, "D") not in self._grid_dates():
            raise PanelNotBuiltError(f"no gridded surface on {day} in {self.layout.surface_grid}")
        return panel.grid_surface(self.surface_grid(), day, self._grid_axes())

    def grid_out(self, day: date) -> GridOut:
        g = self.grid_day(day)
        return GridOut(
            date=day.isoformat(),
            k=[float(x) for x in g.k],
            tenor_days=[int(t) for t in g.tenor_days],
            iv=[_fl(row) for row in g.iv],
            source=str(self.layout.surface_grid),
        )

    def grid_frames(self, start: date, end: date | None, max_frames: int = 126) -> GridFramesOut:
        """The gridded surface on every trading day from ``start``, up to ``max_frames``."""
        if not 1 <= max_frames <= MAX_FRAMES:
            raise ValueError(f"max_frames must be between 1 and {MAX_FRAMES}, got {max_frames}")
        dates = self._grid_dates()
        keep = dates >= np.datetime64(start, "D")
        if end is not None:
            keep &= dates <= np.datetime64(end, "D")
        days = dates[keep]
        grid = self.surface_grid()
        k, tenors = self._grid_axes()
        window = (
            panel.take(grid, (grid["date"] >= days[0]) & (grid["date"] <= days[-1]))
            if len(days)
            else {}
        )
        frames = [
            GridFrameOut(
                date=str(d),
                iv=[
                    _fl(row)
                    for row in panel.grid_surface(
                        window, date.fromisoformat(str(d)), (k, tenors)
                    ).iv
                ],
            )
            for d in days[:max_frames]
        ]
        return GridFramesOut(
            k=[float(x) for x in k],
            tenor_days=[int(t) for t in tenors],
            frames=frames,
            more=len(days) > max_frames,
            source=str(self.layout.surface_grid),
        )

    def landscape_out(
        self, start: date | None = None, end: date | None = None, max_dates: int = 420
    ) -> LandscapeOut:
        if not 2 <= max_dates <= MAX_DATES:
            raise ValueError(f"max_dates must be between 2 and {MAX_DATES}, got {max_dates}")
        if start and end and start > end:
            raise ValueError(f"start {start} is after end {end}")
        term = self.atm_term()
        dates, curves = self._derived(
            "atm_curves", self.layout.atm_term, lambda: panel.atm_curves(term)
        )
        land = panel.atm_landscape(dates, curves, start=start, end=end, max_dates=max_dates)
        within = (dates >= np.datetime64(start, "D")) if start else np.ones(len(dates), bool)
        if end:
            within &= dates <= np.datetime64(end, "D")
        return LandscapeOut(
            date=_iso(land.date),
            tenor_days=[int(t) for t in land.tenor_days],
            iv=[_fl(row) for row in land.iv],
            stride=land.stride,
            n_days=int(within.sum()),
            first=str(dates[within][0]) if within.any() else None,
            last=str(dates[within][-1]) if within.any() else None,
            source=str(self.layout.atm_term),
        )

    # -- the data page's questions ------------------------------------------------ #

    def premium_out(self, mapping: dict[str, Any] | None = None) -> PremiumOut:
        """The premium day by day; the headline and interval from ``mapping`` when given."""
        daily = self.daily()
        s: panel.PremiumSeries = self._derived(
            "premium", self.layout.daily, lambda: panel.premium_series(daily)
        )
        now = panel.premium_now(s)
        ok = np.isfinite(s.premium)
        head = (mapping or {}).get("headline") or {}
        mean = head.get("mean", float(s.premium[ok].mean()) if ok.any() else None)
        low, high = head.get("mean_low"), head.get("mean_high")
        axis = [float(a) for a in (mapping or {}).get("sweep_axis", [])]
        from optspread import vrp

        bracket = list((mapping or {}).get("bracket") or vrp.bracket(mean or 0.0, axis))
        first = head.get("first") or (str(s.date[ok][0]) if ok.any() else "")
        last = head.get("last") or (str(s.date[ok][-1]) if ok.any() else "")
        reading = (
            panel.premium_reading(mean, low, high, first, last, bracket, axis, now)
            if mean is not None
            else "The daily table has no day with a complete forward window yet."
        )

        def pts(x: NDArray[np.float64]) -> list[float | None]:
            # a thousandth of a vol point is below anything the chart can show
            return [None if v is None else round(v, 3) for v in _fl(x)]

        return PremiumOut(
            date=_iso(s.date),
            premium=pts(s.premium),
            smooth=pts(s.smooth),
            iv30=pts(s.iv30 * 100.0),
            rv_fwd=pts(s.rv_fwd * 100.0),
            rv_trailing=pts(s.rv_trailing * 100.0),
            smooth_days=panel.SMOOTH_DAYS,
            now=PremiumNowOut(
                date=now.day.isoformat(),
                iv30=_f(now.iv30 * 100.0),
                rv_trailing=_f(now.rv_trailing * 100.0),
                priced=_f(now.priced),
                measured_date=now.measured_date.isoformat() if now.measured_date else None,
                measured=_f(now.measured),
                recent=_f(now.recent),
                recent_share_below=_f(now.recent_share_below),
            )
            if now
            else None,
            mean=_f(mean),
            mean_low=_f(low),
            mean_high=_f(high),
            bracket=[_f(b) for b in bracket],
            sweep_axis=axis,
            reading=reading,
            source=str(self.layout.daily),
        )

    def shape_out(self, day: date) -> ShapeOut:
        grid = self.surface_grid()
        h: panel.ShapeHistory = self._derived(
            "shape", self.layout.surface_grid, lambda: panel.shape_history(grid)
        )
        sd = panel.shape_on(h, day)
        if sd is None:
            raise PanelNotBuiltError(f"no gridded surface on {day} in {self.layout.surface_grid}")
        return ShapeOut(
            date=day.isoformat(),
            atm=_f(sd.atm),
            skew=_f(sd.skew),
            call=_f(sd.call),
            term=_f(sd.term),
            atm_share_below=_f(sd.atm_share_below),
            skew_share_below=_f(sd.skew_share_below),
            term_share_below=_f(sd.term_share_below),
            share_inverted=_f(sd.share_inverted),
            skew_median=_f(sd.skew_median),
            term_median=_f(sd.term_median),
            tenor=sd.tenor,
            put_k=sd.put_k,
            call_k=sd.call_k,
            long_tenor=sd.long_tenor,
            reading=panel.shape_reading(sd),
            source=str(self.layout.surface_grid),
        )

    def quality_out(self, docs: Path) -> QualityOut:
        """The build's version and its files against docs/DATA_MANIFEST.json, and FRED."""
        from optspread.realdata import panel_digests

        manifest, _ = _read_manifest(docs / DATA_MANIFEST_FILE)
        files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
        version = manifest.get("panel_version")
        actual = panel_digests(self.layout.panel)
        checked = []
        for name, digest in sorted(actual.items()):
            rec = files.get(name, {})
            recorded = rec.get("sha256") if isinstance(rec, dict) else None
            checked.append(
                PanelFileCheckOut(
                    path=name,
                    recorded=recorded,
                    actual=digest,
                    match=(recorded == digest) if recorded else None,
                )
            )
        verdicts = [c.match for c in checked]
        of_record = (
            None
            if not verdicts or any(v is None for v in verdicts)
            else all(bool(v) for v in verdicts)
        )
        fred_path = self.layout.panel / "fred_check.parquet"
        fred: list[FredRowOut] = []
        if fred_path.is_file():
            cols = self._table("fred", fred_path)
            order: Sequence[int] = (
                np.lexsort((cols["source"], cols["year"])).tolist() if panel.n_rows(cols) else []
            )
            for i in order:

                def col(name: str, i: int = int(i)) -> float | None:
                    return _f(float(cols[name][i])) if name in cols else None

                fred.append(
                    FredRowOut(
                        year=int(cols["year"][i]),
                        source=str(cols["source"][i]),
                        rate_minus_bill_bp=col("rate_minus_bill_bp"),
                        rate_minus_sofr_bp=col("rate_minus_sofr_bp"),
                        spot_vs_sp500_bp=col("spot_vs_sp500_bp"),
                    )
                )
        reading = ""
        if self.layout.coverage.is_file():
            cov = panel.coverage_summary(self.coverage())
            days = panel.available_dates(self.coverage())
            report = panel.seam_report(self.seam())
            reading = panel.quality_reading(
                n_days=len(days),
                first=str(days[0]) if len(days) else "",
                last=str(days[-1]) if len(days) else "",
                # a day one vendor misses and the other quotes is not missing from the panel
                n_missing=len({d for d, _ in cov.missing} - {_day_of(x) for x in days}),
                seam_days=report.n_days if report else 0,
                seam_p95=report.mean(panel.SEAM_HEADLINE) if report else None,
                seam_worst=report.worst(panel.SEAM_HEADLINE) if report else None,
                panel_version=int(version) if isinstance(version, int) else None,
                of_record=of_record,
            )
        return QualityOut(
            panel_version=int(version) if isinstance(version, int) else None,
            manifest_files=len(files),
            checked=checked,
            of_record=of_record,
            fred=fred,
            fred_available=fred_path.is_file(),
            reading=reading,
            source=str(docs / DATA_MANIFEST_FILE),
        )

    def _daily_row(self, day: date) -> dict[str, float]:
        """The day's spot, RV21 and IV30, OptionMetrics first; empty when not in the table."""
        if not self.layout.daily.is_file():
            return {}
        daily = self.daily()
        if not panel.n_rows(daily):
            return {}
        mine = daily["date"] == np.datetime64(day, "D")
        if not mine.any():
            return {}
        idx = np.flatnonzero(mine)
        if "source" in daily:
            om = idx[daily["source"][idx] == "optionmetrics"]
            idx = om if len(om) else idx
        i = int(idx[0])
        return {
            name: float(daily[col][i])
            for name, col in (("spot", "spot"), ("rv21", "rv21"), ("iv30", "iv30_atm"))
            if col in daily
        }

    def simulated_out(self, day: date, premium: float, sigma_from: str = "config") -> SimulatedOut:
        if sigma_from not in SIGMA_FROM:
            raise ValueError(
                f"sigma_from must be one of {', '.join(SIGMA_FROM)}, got {sigma_from!r}"
            )
        real = self.grid_day(day)
        row = self._daily_row(day)
        base = MarketConfig()
        spot = row.get("spot", math.nan)
        spot_from = "spx_daily.parquet" if math.isfinite(spot) and spot > 0 else "config"
        if spot_from == "config":
            spot = base.spot0
        if sigma_from == "config":
            sigma = base.sigma
        else:
            name = "rv21" if sigma_from == "rv21" else "iv30"
            value = row.get(name, math.nan)
            if not math.isfinite(value):
                raise ValueError(f"no {name} on {day} in {self.layout.daily}")
            sigma = value if sigma_from == "rv21" else value - premium / 100.0
        if not sigma > 0.0:
            raise ValueError(f"sigma {sigma:.4f} is not positive; lower the premium")
        cfg = base.model_copy(update={"spot0": spot, "sigma": sigma, "iv_premium_vol_pts": premium})
        return simulated_surface(day, cfg, real, spot_from, sigma_from)


# -- the overview ------------------------------------------------------------------ #


class GateTileOut(Out):
    milestone: str
    verdict: str
    verdict_mismatch: bool
    report_date: str
    n_items: int
    n_pass: int
    source: str


class OverviewOut(Out):
    summary: list[str]
    gates: list[GateTileOut]
    data: DataStatusOut
    coverage: list[SpanOut]
    overlap: tuple[str, str, int] | None
    n_runs: int
    n_ok: int
    latest: list[RunRowOut]
    sweeps: int
    ppo_available: bool


def overview(
    readme: Path, docs: Path, root: Path, reader: PanelReader, rows: Sequence[RunRow]
) -> OverviewOut:
    boards = load_gate_boards(docs)
    spans: list[SpanOut] = []
    overlap = None
    if reader.layout.coverage.is_file():
        cov = reader.coverage_out()
        spans, overlap = cov.spans, cov.overlap
    return OverviewOut(
        summary=list(readme_summary(readme)),
        gates=[
            GateTileOut(
                milestone=b.milestone,
                verdict=b.verdict,
                verdict_mismatch=b.verdict_mismatch,
                report_date=b.report_date,
                n_items=len(b.items),
                n_pass=b.counts.get("PASS", 0),
                source=str(b.source),
            )
            for b in boards
        ],
        data=reader.status(),
        coverage=spans,
        overlap=overlap,
        n_runs=len(rows),
        n_ok=sum(r.status == "OK" for r in rows),
        latest=[row_out(r, root) for r in rows[:8]],
        sweeps=len(_sweep_dirs(root)) + len(_grid_files(root)),
        ppo_available="ppo" in available_arms(),
    )


def index(root: Path) -> tuple[RunRow, ...]:
    """Every run under the root except the job queue's own bookkeeping."""
    return tuple(r for r in scan_runs(root) if ".jobs" not in r.path.parts)
