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

import json
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm

from optspread import dashboard_panel as panel
from optspread import power as pw
from optspread.actions import ACTION_NAMES
from optspread.artifacts import SchemaVersionError
from optspread.dashboard_data import (
    GateBoard,
    LeakDemo,
    RiskStats,
    RunDetail,
    RunRow,
    compare_runs,
    filter_runs,
    load_gate_boards,
    load_run_detail,
    readme_summary,
    risk_stats,
    run_facets,
    scan_runs,
)
from optspread.jobs import (
    ACTION_LOG_FILENAME,
    EQUITY_FILENAME,
    PRESETS,
    JobSpec,
    JobStatus,
    available_arms,
)
from optspread.config import MarketConfig
from optspread.leaks import LeakReport
from optspread.manifest import MANIFEST_FILENAME
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
    alpha_bps: float | None
    vol_of_vol: float | None
    skew_slope: float | None
    source: str
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
    """Where the run's chains came from. Every run is synthetic until a real
    ``ChainSource`` is wired into the harness; a config that names one says so."""
    return "synthetic"


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
        alpha_bps=row.alpha_bps,
        vol_of_vol=row.vol_of_vol,
        skew_slope=row.skew_slope,
        source=run_source(row),
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


def run_list(rows: Sequence[RunRow], root: Path, filters: RunFilters) -> RunListOut:
    facets = run_facets(rows)
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
    """The run directory for ``key``, refusing anything outside the runs root."""
    target = _inside(root, key)
    if target is None or not (target / MANIFEST_FILENAME).is_file():
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


_N_BINS = 10


class PolicyHeatmapOut(Out):
    """Action frequency binned by (time_frac, atm_iv). Shape: [n_time][n_iv][n_actions]."""

    atm_iv_edges: list[float]
    time_frac_edges: list[float]
    action_names: list[str]
    counts: list[list[list[int]]]  # [time_bin][iv_bin][action_id]
    n_steps: int
    source: str


def policy_heatmap(run_dir: Path) -> PolicyHeatmapOut | None:
    """Where the policy acted, by time through the episode and the ATM vol it saw.

    Reads the action log the harness wrote beside ``equity.json``. None when the
    run has no log (runs from before it) or the log is empty or malformed.
    """
    target = run_dir / ACTION_LOG_FILENAME
    try:
        raw: Any = json.loads(target.read_text())
        actions = [list(map(int, ep)) for ep in raw["actions"]]
        atm_iv = [list(map(float, ep)) for ep in raw["atm_iv"]]
        time_frac = [list(map(float, ep)) for ep in raw["time_frac"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None

    flat_actions = np.asarray([a for ep in actions for a in ep], dtype=np.int64)
    flat_iv = np.asarray([v for ep in atm_iv for v in ep], dtype=np.float64)
    flat_tf = np.asarray([v for ep in time_frac for v in ep], dtype=np.float64)
    n_steps = len(flat_actions)
    if n_steps == 0 or not (len(flat_iv) == len(flat_tf) == n_steps):
        return None
    if int(flat_actions.min()) < 0:
        return None

    n_actions = int(flat_actions.max()) + 1
    iv_edges = np.linspace(float(flat_iv.min()), float(flat_iv.max()) + 1e-9, _N_BINS + 1)
    tf_edges = np.linspace(0.0, 1.0 + 1e-9, _N_BINS + 1)
    iv_bins = np.clip(np.searchsorted(iv_edges, flat_iv, side="right") - 1, 0, _N_BINS - 1)
    tf_bins = np.clip(np.searchsorted(tf_edges, flat_tf, side="right") - 1, 0, _N_BINS - 1)

    counts = np.zeros((_N_BINS, _N_BINS, n_actions), dtype=np.int64)
    np.add.at(counts, (tf_bins, iv_bins, flat_actions), 1)

    names = [ACTION_NAMES[i] if i < len(ACTION_NAMES) else str(i) for i in range(n_actions)]
    return PolicyHeatmapOut(
        atm_iv_edges=[float(e) for e in iv_edges],
        time_frac_edges=[float(e) for e in tf_edges],
        action_names=names,
        counts=counts.tolist(),
        n_steps=n_steps,
        source=str(target),
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


class CompareOut(Out):
    a: RunDetailOut
    b: RunDetailOut
    config_diff: list[DiffOut]
    shared_seeds: int
    diff_mean: float | None
    diff_se: float | None
    diff_t: float | None
    diff_histogram: HistogramOut


def compare(a: RunDetail, b: RunDetail, root: Path, index: Sequence[RunRow]) -> CompareOut:
    cmp = compare_runs(a, b)
    pa = {e.seed: e.pnl for e in a.record.episodes} if a.record else {}
    pb = {e.seed: e.pnl for e in b.record.episodes} if b.record else {}
    diffs = np.asarray([pb[s] - pa[s] for s in cmp.shared_seeds], dtype=np.float64)
    return CompareOut(
        a=run_detail(a, root, index),
        b=run_detail(b, root, index),
        config_diff=[DiffOut(key=k, a=va, b=vb) for k, va, vb in cmp.config_diff],
        shared_seeds=len(cmp.shared_seeds),
        diff_mean=_f(cmp.diff_mean),
        diff_se=_f(cmp.diff_se),
        diff_t=_f(cmp.diff_t),
        diff_histogram=histogram(diffs),
    )


def load_detail(root: Path, key: str) -> RunDetail:
    return load_run_detail(resolve_run(root, key))


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


class PresetOut(Out):
    name: str
    label: str
    description: str
    spec: dict[str, Any]


class JobOptionsOut(Out):
    arms: list[str]
    ppo_available: bool
    knobs: list[KnobOut]
    presets: list[PresetOut]
    defaults: dict[str, Any]


_GROUPS: dict[str, tuple[str, str]] = {
    "policy": (
        "Arm",
        "Which policy runs: the scripted arms, or PPO when the learner is installed.",
    ),
    "iv_premium_vol_pts": ("Market", "Planted premium in vol points (D2). Zero is the null world."),
    "vol_of_vol": (
        "Market",
        "Annualised vol-of-vol (0 = constant vol, GBM). Adds path-dependency to the smile level.",
    ),
    "skew_slope": (
        "Market",
        "IV tilt in vol pts per unit log(S/K). Positive = equity left skew "
        "(higher IV at lower strikes). 0 = flat smile.",
    ),
    "alpha_bps": (
        "Market",
        "Per-day drift in basis points (0 = zero-drift, 25 = long-biased SPX-like edge).",
    ),
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
    "total_timesteps": ("Training", "PPO training steps; not pre-registered (D3)."),
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
    elif name == "tenor_days":
        kind, choices, default = "tenors", [], list(default)
    elif "bool" in annotation:
        kind, choices = "boolean", []
    elif "int" in annotation:
        kind, choices = "integer", []
    else:
        kind, choices = "number", []
    return KnobOut(
        name=name, kind=kind, default=default, choices=choices, group=group, help=text, **bounds
    )


def job_options() -> JobOptionsOut:
    arms = available_arms()
    knobs = [_knob(name, info, arms) for name, info in JobSpec.model_fields.items()]
    if "ppo" not in arms:
        knobs = [k for k in knobs if k.name != "total_timesteps"]
    return JobOptionsOut(
        arms=list(arms),
        ppo_available="ppo" in arms,
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
    )


# -- the power lab ---------------------------------------------------------------- #

# Illustrative until the variance-components pilot reports; the pilot_budget script's
# own example values, in dollars of mean P&L per episode.
POWER_DEFAULTS = {"tau2": 250_000.0, "sigma2": 1.3e8, "delta": 1500.0}
FLOOR_SEEDS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100)
FRONTIER_SEEDS = (2, 3, 5, 8, 10, 16, 20, 32, 50, 64)


class PowerParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tau2: float = Field(default=POWER_DEFAULTS["tau2"], ge=0.0)
    sigma2: float = Field(default=POWER_DEFAULTS["sigma2"], ge=0.0)
    n_seeds: int = Field(default=10, ge=1, le=10_000)
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
    ratio: float  # floor at S=5 over floor at S=50: about sqrt(10) when tau^2 dominates


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
    z = float(norm.ppf(1.0 - p.alpha) + norm.ppf(p.target_power))
    return z * math.sqrt(pw.var_grand_mean(p.tau2, p.sigma2, n_seeds, p.n_episodes))


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
    """Power, floor and frontier for one design, from ``optspread.power``. Instant."""
    var = pw.var_grand_mean(p.tau2, p.sigma2, p.n_seeds, p.n_episodes)
    floor = _floor(p, p.n_seeds)
    top = max(2.0 * p.delta, 2.5 * max(floor, _floor(p, 5) / 2.0), 1.0)
    grid = np.linspace(0.0, top, 81)

    def curve(n_seeds: int) -> list[float]:
        return [pw.power(float(d), p.tau2, p.sigma2, n_seeds, p.n_episodes, p.alpha) for d in grid]

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
    power_now = pw.power(p.delta, p.tau2, p.sigma2, p.n_seeds, p.n_episodes, p.alpha)
    return PowerOut(
        params=p,
        se=math.sqrt(var),
        var_grand_mean=var,
        variance_budget=pw.variance_budget(p.delta, p.alpha, p.target_power),
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
        source="optspread.power: Var(grand mean) = tau^2/S + sigma^2/(S N), one-sided normal test",
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


class LeakDemoOut(Out):
    eps: float
    seeds: list[int]
    clean: LeakSideOut
    poisoned: LeakSideOut
    caught: bool
    seconds: float
    criterion: str
    source: str


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


class OraclePointOut(Out):
    premium: float
    mean_pnl: float
    se_pnl: float
    lift: float  # against the first premium for the same arm: a paired difference
    run_id: str


class PendingOut(Out):
    """A view whose input does not exist yet, and what will produce it."""

    status: str
    waiting_for: str


class SweepOut(Out):
    summary: SweepSummaryOut
    oracle: dict[str, list[OraclePointOut]]
    power_curve: PendingOut
    attribution: PendingOut
    source: str


class SweepListOut(Out):
    root: str
    sweeps: list[SweepSummaryOut]
    power_curve: PendingOut
    attribution: PendingOut


POWER_PENDING = PendingOut(
    status="waiting for sweep",
    waiting_for="the confirmatory sweep's PPO cells and the power-curve analysis",
)
ATTRIBUTION_PENDING = PendingOut(
    status="waiting for sweep",
    waiting_for="P&L attribution output: P&L by greek plus the residual",
)


def _sweep_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob(SWEEP_FILENAME)) if root.is_dir() else []


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
            )
        )
    return SweepListOut(
        root=str(root), sweeps=out, power_curve=POWER_PENDING, attribution=ATTRIBUTION_PENDING
    )


def sweep_detail(root: Path, key: str) -> SweepOut:
    target = _inside(root, key)
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
        power_curve=POWER_PENDING,
        attribution=ATTRIBUTION_PENDING,
        source=str(target / SWEEP_FILENAME),
    )


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
    iv30_sources: list[str]
    iv30_date: list[str]
    iv30_diff_vol_pts: list[float | None]
    source: str


class PanelNotBuiltError(LookupError):
    """The view needs a panel file the build has not written yet."""


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


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
            iv30_sources=list(iv_seam.sources) if iv_seam else [],
            iv30_date=_iso(iv_seam.date) if iv_seam else [],
            iv30_diff_vol_pts=_fl(iv_seam.diff_vol_pts) if iv_seam else [],
            source=str(self.layout.seam_overlap),
        )


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
        sweeps=len(_sweep_dirs(root)),
        ppo_available="ppo" in available_arms(),
    )


def index(root: Path) -> tuple[RunRow, ...]:
    """Every run under the root except the job queue's own bookkeeping."""
    return tuple(r for r in scan_runs(root) if ".jobs" not in r.path.parts)


# -- synthetic data preview ------------------------------------------------------- #


class SynthPathsOut(BaseModel):
    model_config = ConfigDict(frozen=True)
    n_days: int
    n_paths: int
    seed: int
    # shape: (n_paths, n_days+1) — index 0 is the initial value before any step
    spots: list[list[float]]           # spot price paths
    atm_ivs: list[list[float]]         # ATM IV paths (same shape)
    # smile snapshot at day mid-point: strikes relative to spot, one row per path
    smile_log_moneyness: list[float]   # log(S/K) values, shared across paths (n_strikes,)
    smile_ivs: list[list[float]]       # shape: (n_paths, n_strikes)


def synth_paths(config: MarketConfig, n_paths: int = 5, seed: int = 0) -> SynthPathsOut:
    """Generate n_paths sample price and vol paths from the given MarketConfig."""
    from optspread.market import SyntheticChainProvider
    from optspread.rng import SeededStepRNG

    spots_out: list[list[float]] = []
    atm_ivs_out: list[list[float]] = []
    smile_ivs_out: list[list[float]] = []
    smile_lm: list[float] = []

    mid_day = config.n_days // 2

    for i in range(n_paths):
        prov = SyntheticChainProvider(config)
        chain = prov.reset(SeededStepRNG(seed + i))
        spots_i = [chain.spot]
        atm_iv_i = [float(chain.ivs[0, chain.ivs.shape[1] // 2])]
        smile_snap_lm: list[float] = []
        smile_snap_iv: list[float] = []

        for d in range(config.n_days):
            chain = prov.step()
            spots_i.append(chain.spot)
            atm_iv_i.append(float(chain.ivs[0, chain.ivs.shape[1] // 2]))
            if prov.day == mid_day:
                # log-moneyness: log(spot/K), positive = OTM puts (low K)
                lm = np.log(chain.spot / chain.strikes).tolist()
                smile_snap_lm = [float(x) for x in lm]
                smile_snap_iv = [float(chain.ivs[0, j]) for j in range(len(chain.strikes))]

        spots_out.append(spots_i)
        atm_ivs_out.append([v * 100.0 for v in atm_iv_i])  # convert to pct for display
        smile_ivs_out.append([v * 100.0 for v in smile_snap_iv])
        smile_lm = smile_snap_lm  # capture from this path (same strikes each path)

    return SynthPathsOut(
        n_days=config.n_days,
        n_paths=n_paths,
        seed=seed,
        spots=spots_out,
        atm_ivs=atm_ivs_out,
        smile_log_moneyness=smile_lm,
        smile_ivs=smile_ivs_out,
    )
