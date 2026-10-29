"""The dashboard's study-level payloads: the design, the checks, the analyses, the work it starts.

Lane: Ilan Danial (machinery).

``optspread.webapi`` builds the payloads read straight off runs and the panel; this module
builds the ones that sit over them, under the same rules: every payload names the file
or module its numbers came from, and an undefined figure is null.

- **The study design** as ``docs/PRE_REGISTRATION.md`` fixes it (D1, A1, A2, A6, A7), with
  its power and floor recomputed by ``optspread.power`` rather than copied.
- **Checks**: every gate item, one row each, with what failed first and its report file.
- **The real premium** (``docs/VRP_MAPPING.json``) and **the M4 analysis**
  (``docs/M4_ANALYSIS.json``, read as written), the first read off the second's power
  curves with ``optspread.vrp.power_at``.
- **Background work** the run page starts and polls: attribution (``optspread.attribution``,
  cached under the attribution root, never in a run directory), drawdown and return on
  posted capital (a re-simulation of the run's episodes), and replay (``optspread.replay``).
- **The leak detector's floor** and its **per-field controls** (``optspread.leaks``).
- **What the fixed-delta seller opens on a real day**: an illustration on the real chain.
- **The edge of a grid** (``optspread.edge``): per arm and premium, each component's and
  each structure's share of the paired contrast, read from ``<attribution root>/edge``
  as ``scripts/edge_attribution.py`` wrote it, with each cell's run under the runs root.
- **Launch**: a PPO preset and the wall-time estimate at the rates A7 measured.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from optspread import actionmix as am
from optspread import dashboard_panel as panel
from optspread import power as pw
from optspread.config import MarketConfig
from optspread.dashboard_data import RunRow, load_gate_boards
from optspread.jobs import JobSpec, available_arms
from optspread.manifest import MANIFEST_FILENAME, RunManifest
from optspread.webapi import (
    FeatureOut,
    JobOptionsOut,
    Out,
    PanelNotBuiltError,
    PanelReader,
    PresetOut,
    _f,
    _side,
    job_options,
)

# -- the study design (docs/PRE_REGISTRATION.md) ------------------------------------ #

PREREG_FILE = "PRE_REGISTRATION.md"
# A7: the pilot's PPO contrast components, and the fixed-delta seller's (the control).
A7_TAU2, A7_SIGMA2 = 51_838.0, 105_996_836.0
A7_SELLER_TAU2, A7_SELLER_SIGMA2 = 1_026.0, 828_609.0
# A6: the effect to detect, the seller's contrast at 1 vol point, in $ per episode. It is
# also the dashboard's conversion between dollars and vol points of planted premium.
A6_DELTA = 824.3
# A7: steps/s/core measured on the pilot machine (train: PPO; eval: environment steps).
A7_TRAIN_RATE, A7_EVAL_RATE = 804.0, 1_538.0


class MeasuredDesignOut(Out):
    """The design's power and floor at the sweep's own variance components (M4's S3).

    ``docs/M4_ANALYSIS.json`` ``design_check.arms.<arm>`` at its reference premium: the
    same S, N, alpha and effect as A7's plan, with the contrast's measured tau^2 and
    sigma^2 in place of the pilot's. This, not the plan, is the power the design had.
    """

    arm: str
    premium: float
    tau2: float
    sigma2: float
    power: float
    floor: float  # $ per episode
    floor_vol_pts: float
    file: str


class StudyOut(Out):
    """The confirmatory design as the pre-registration fixes it, with its power recomputed.

    ``power`` and ``floor`` are A7's plan (the pilot's PPO variances); ``measured`` is
    the same design read with the sweep's own variances, when the analysis is written.
    """

    test: str
    alpha: float
    event: str
    n_seeds: int
    n_episodes: int
    train_steps: int
    episode_days: int
    premiums: list[float]
    arms: list[str]
    grid_hash: str
    delta_per_vol_pt: float
    tau2: float
    sigma2: float
    seller_tau2: float
    seller_sigma2: float
    target_power: float
    power: float
    floor: float  # $ per episode, exact t at S and N
    floor_vol_pts: float
    train_rate: float
    eval_rate: float
    core_hours: float
    doc: str
    source: str
    measured: MeasuredDesignOut | None = None


def measured_design(docs: Path, arm: str = "ppo") -> MeasuredDesignOut | None:
    """The design check of ``docs/M4_ANALYSIS.json`` for ``arm`` at its reference premium,
    or None when the analysis is not there or does not carry one."""
    raw, _ = _read_json(docs / M4_FILE)
    if raw is None:
        return None
    try:
        check = raw["design_check"]
        ref = float(check["reference_premium"])
        rows = check["arms"]
        pick = arm if arm in rows else next(iter(rows))
        row = next(r for r in rows[pick] if float(r["premium"]) == ref)
        floor = float(row["floor"])
        return MeasuredDesignOut(
            arm=str(pick),
            premium=ref,
            tau2=float(row["tau2"]),
            sigma2=float(row["sigma2"]),
            power=float(row["power"]),
            floor=floor,
            floor_vol_pts=floor / A6_DELTA,
            file=M4_FILE,
        )
    except (KeyError, StopIteration, TypeError, ValueError):
        return None


def sweep_core_hours(
    n_seeds: int,
    n_episodes: int,
    train_steps: int,
    episode_days: int,
    premiums: int,
    arms: Sequence[str],
    train_rate: float = A7_TRAIN_RATE,
    eval_rate: float = A7_EVAL_RATE,
) -> float:
    """Core-hours for a premium x arm x S grid: evaluation for every arm, training for ppo."""
    per_block = sum(
        n_episodes * episode_days / eval_rate + (train_steps / train_rate if a == "ppo" else 0.0)
        for a in arms
    )
    return premiums * n_seeds * per_block / 3600.0


def study(docs: Path | None = None) -> StudyOut:
    """The design; with ``docs``, also its power as the analysis measured it."""
    from optspread.m4analysis import CONFIRMATORY_GRID, CONFIRMATORY_GRID_HASH, TARGET_POWER

    g = CONFIRMATORY_GRID
    s, n = g.n_seeds, g.base.n_episodes
    steps = g.train.total_timesteps if g.train is not None else 0
    floor = pw.detectable_effect_t(A7_TAU2, A7_SIGMA2, s, n, 0.05, TARGET_POWER)
    return StudyOut(
        test="one-sided Student t on the S seed means, S \u2212 1 df",  # a true minus
        alpha=0.05,
        event="each arm at premium p against itself at 0, paired by seed block",
        n_seeds=s,
        n_episodes=n,
        train_steps=steps,
        episode_days=g.base.env.episode_days,
        premiums=sorted(g.premiums),
        arms=list(g.arms),
        grid_hash=CONFIRMATORY_GRID_HASH,
        delta_per_vol_pt=A6_DELTA,
        tau2=A7_TAU2,
        sigma2=A7_SIGMA2,
        seller_tau2=A7_SELLER_TAU2,
        seller_sigma2=A7_SELLER_SIGMA2,
        target_power=TARGET_POWER,
        power=pw.power_t(A6_DELTA, A7_TAU2, A7_SIGMA2, s, n, 0.05),
        floor=floor,
        floor_vol_pts=floor / A6_DELTA,
        train_rate=A7_TRAIN_RATE,
        eval_rate=A7_EVAL_RATE,
        core_hours=sweep_core_hours(s, n, steps, g.base.env.episode_days, len(g.premiums), g.arms),
        doc=PREREG_FILE,
        source=(
            "docs/PRE_REGISTRATION.md D1, A1, A2, A6, A7; optspread.m4analysis.CONFIRMATORY_GRID"
            + ("; measured: docs/M4_ANALYSIS.json design_check" if docs is not None else "")
        ),
        measured=measured_design(docs) if docs is not None else None,
    )


# -- checks: every gate item, one row each ----------------------------------------- #


class CheckOut(Out):
    name: str  # as the report names the item
    label: str  # as the page names it: what is checked, in plain words
    verdict: str
    failed_first: str
    criterion: str
    evidence: str
    owner: str
    report: str  # the readable report (its .md when there is one), at /api/docs/<report>
    sidecar: str  # the file the row was read from
    report_date: str


# Report items named after the process that produced them, named on the page for what
# they check. Every other item's name already says that.
CHECK_LABELS: dict[str, str] = {
    "Wave 0 no-edge gate": "No edge in the fair world",
    "Dashboard run browser and gate board": "Dashboard run browser and check board",
}


def check_label(name: str) -> str:
    if name in CHECK_LABELS:
        return CHECK_LABELS[name]
    return re.sub(r"\bgate\b", "check", name)


def checks(docs: Path) -> list[CheckOut]:
    """One row per check across the gate reports, the latest report's row where two share a name."""
    boards = sorted(
        load_gate_boards(docs), key=lambda b: (b.report_date, b.milestone), reverse=True
    )
    seen: set[str] = set()
    out: list[CheckOut] = []
    for b in boards:
        for i in b.items:
            if i.name in seen:
                continue
            seen.add(i.name)
            readable = b.source.with_suffix(".md")
            out.append(
                CheckOut(
                    name=i.name,
                    label=check_label(i.name),
                    verdict=i.verdict,
                    failed_first=i.failed_first,
                    criterion=i.criterion,
                    evidence=i.evidence,
                    owner=i.owner,
                    report=readable.name if readable.is_file() else b.source.name,
                    sidecar=b.source.name,
                    report_date=b.report_date,
                )
            )
    return out


DOC_NAME = re.compile(r"^[A-Za-z0-9_]+\.(md|json)$")


def doc_file(docs: Path, name: str) -> Path:
    """A report or record in ``docs/`` by its file name; nothing else is served."""
    target = docs / name
    if not DOC_NAME.match(name) or not target.is_file():
        raise KeyError(name)
    return target


# -- the real premium on the sweep's axis (docs/VRP_MAPPING.json) ------------------- #

VRP_FILE = "VRP_MAPPING.json"
M4_FILE = "M4_ANALYSIS.json"


class VrpOut(Out):
    available: bool
    file: str
    mapping: dict[str, Any] | None
    error: str = ""


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, ""
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"{path.name} does not read: {exc}"
    if not isinstance(raw, dict):
        return None, f"{path.name} is not a JSON object"
    return raw, ""


def vrp_mapping(docs: Path) -> VrpOut:
    raw, error = _read_json(docs / VRP_FILE)
    if raw is not None and "headline" not in raw:
        raw, error = None, f"{VRP_FILE} has no headline"
    return VrpOut(available=raw is not None, file=VRP_FILE, mapping=raw, error=error)


# -- the M4 analysis (docs/M4_ANALYSIS.json) --------------------------------------- #


class MappedPowerOut(Out):
    """One arm's power curve read at the real premium, at its interval's ends and over it.

    Read as the M4 gate reads it (``m4gate.read_curve``): from the curve's knots through
    ``vrp.power_at_noncentrality``, a curve that dips by sampling noise read as it is;
    a sidecar without knots is read linearly off its ``interpolated`` list
    (``vrp.power_at``).
    """

    arm: str
    premium: float
    low: float
    high: float
    power: float | None
    power_low: float | None
    power_high: float | None
    clipped: bool
    reason: str = ""  # why the curve was not read, when it was not
    power_min: float | None = None  # the lowest power over the interval
    power_max: float | None = None  # the highest
    band_min: float | None = None  # the curve's band over the interval, from its knots
    band_max: float | None = None


class M4Out(Out):
    available: bool
    file: str
    analysis: dict[str, Any] | None
    vrp: dict[str, Any] | None  # {estimate, interval, label, file}
    mapped: list[MappedPowerOut]
    waiting_for: str
    error: str = ""


M4_WAITING = "the analysis of the confirmatory sweep: `make m4` writes docs/M4_ANALYSIS.json"


def _mapped(
    arm: str,
    curve: Any,
    estimate: float,
    interval: tuple[float, float],
    n_seeds: int | None = None,
    alpha: float | None = None,
) -> MappedPowerOut:
    from optspread.m4gate import read_curve
    from optspread.vrp import ALPHA, power_at

    lo, hi = sorted(interval)
    band: tuple[float | None, float | None] = (None, None)
    try:
        points = list(curve.get("points") or ())
        if points and n_seeds is not None:
            m, band = read_curve(
                points, n_seeds, ALPHA if alpha is None else alpha, lo, hi, estimate
            )
        else:
            pts = [(float(p["premium"]), float(p["power"])) for p in curve["interpolated"]]
            m = power_at([x for x, _ in pts], [y for _, y in pts], estimate, (lo, hi))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return MappedPowerOut(
            arm=arm,
            premium=estimate,
            low=lo,
            high=hi,
            power=None,
            power_low=None,
            power_high=None,
            clipped=False,
            reason=str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
    return MappedPowerOut(
        arm=arm,
        premium=m.premium,
        low=m.low,
        high=m.high,
        power=m.power,
        power_low=m.power_low,
        power_high=m.power_high,
        clipped=m.clipped,
        power_min=m.power_min,
        power_max=m.power_max,
        band_min=band[0],
        band_max=band[1],
    )


def m4_analysis(docs: Path) -> M4Out:
    """The sidecar as ``scripts/m4_analysis.py`` wrote it, with the real premium read off it."""
    from optspread.m4analysis import SCHEMA_VERSION as M4_SCHEMA

    raw, error = _read_json(docs / M4_FILE)
    version = raw.get("schema_version") if raw is not None else None
    if raw is not None and version != M4_SCHEMA:
        raw, error = None, f"{M4_FILE} is schema v{version}, not v{M4_SCHEMA}"
    vrp = vrp_mapping(docs).mapping
    real: dict[str, Any] | None = None
    if vrp is not None:
        try:
            inputs = vrp["power_inputs"]
            real = {
                "estimate": float(inputs["estimate"]),
                "interval": [float(x) for x in inputs["interval"]],
                "label": str(vrp.get("headline", {}).get("label", "")),
                "file": VRP_FILE,
            }
        except (KeyError, TypeError, ValueError):
            real = None
    mapped: list[MappedPowerOut] = []
    if raw is not None and real is not None:
        lo, hi = real["interval"]
        try:
            n_seeds: int | None = int(raw["design"]["n_seeds"])
        except (KeyError, TypeError, ValueError):
            n_seeds = None
        settings = raw.get("settings") or {}
        alpha = settings.get("alpha") if isinstance(settings, dict) else None
        for arm in raw.get("arm_order", []):
            curve = raw.get("arms", {}).get(arm, {}).get("curve", {})
            mapped.append(
                _mapped(
                    str(arm),
                    curve,
                    real["estimate"],
                    (lo, hi),
                    n_seeds,
                    float(alpha) if isinstance(alpha, int | float) else None,
                )
            )
    return M4Out(
        available=raw is not None,
        file=M4_FILE,
        analysis=raw,
        vrp=real,
        mapped=mapped,
        waiting_for=M4_WAITING,
        error=error,
    )


# -- background work: attribution, capital and replay ------------------------------- #


class TaskState(Out):
    state: str  # "idle" | "running" | "done" | "failed"
    error: str = ""
    seconds: float | None = None


class Tasks:
    """Work the API starts and the page polls: one thread, keyed by kind and run key.

    Not ``optspread.jobs``: these read a run and write nowhere but the attribution root,
    so they need no process pool and no run directory of their own.
    """

    def __init__(self, workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="optspread-task")
        self._futures: dict[tuple[str, str], tuple[Future[Any], float]] = {}
        self._finished: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()  # a done future runs its callback inside submit

    def submit(self, kind: str, key: str, work: Callable[[], Any]) -> None:
        """Start ``work`` unless the same kind is already running for this key."""
        with self._lock:
            held = self._futures.get((kind, key))
            if held is not None and not held[0].done():
                return
            start = time.perf_counter()
            future = self._pool.submit(work)
            future.add_done_callback(lambda _: self._finish((kind, key)))
            self._futures[(kind, key)] = (future, start)

    def _finish(self, task: tuple[str, str]) -> None:
        with self._lock:
            self._finished[task] = time.perf_counter()

    def state(self, kind: str, key: str) -> tuple[TaskState, Any]:
        with self._lock:
            held = self._futures.get((kind, key))
            end = self._finished.get((kind, key))
        if held is None:
            return TaskState(state="idle"), None
        future, start = held
        if not future.done():
            return TaskState(state="running", seconds=time.perf_counter() - start), None
        seconds = (end or time.perf_counter()) - start
        exc = future.exception()
        if exc is not None:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return TaskState(state="failed", error=text, seconds=seconds), None
        return TaskState(state="done", seconds=seconds), future.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _scripted(policy: str) -> bool:
    return policy in available_arms() and policy != "ppo"


# attribution


class AttributionOut(Out):
    state: str  # "cached" | "computing" | "absent" | "failed"
    can_compute: bool
    reason: str
    file: str
    result: dict[str, Any] | None  # the cached RunAttribution, every episode, no steps
    seed: int | None  # the episode whose steps are sent
    steps: list[dict[str, Any]]
    seconds: float | None = None


def attribution_file(root: Path, row: RunRow) -> Path:
    """Where ``scripts/attribution.py`` writes this run: ``<run_id>[.<source>].json``."""
    suffix = "" if row.source in ("", "synthetic") else f".{row.source}"
    return root / f"{row.run_id}{suffix}.json"


def real_data_reason(row: RunRow) -> str:
    """Why a real-data run cannot be re-simulated here: its panel is absent, or is not the
    panel it read (rebuilt since), or it ran on dates other than A8's eval split. Empty
    for a synthetic run and for one whose recorded source and split this server serves."""
    if row.source in ("", "synthetic"):
        return ""
    from optspread.attribution import recorded_source
    from optspread.data import DataUnavailableError, make_source

    if not make_source(row.source).available:  # type: ignore[arg-type]
        return f"the {row.source} panel this run read is not on this server"
    try:
        recorded_source(RunManifest.read(row.path))
    except DataUnavailableError as exc:
        return f"the {row.source} panel this run read cannot be read here: {exc}"
    except ValueError as exc:
        return str(exc).splitlines()[0]
    return ""


def _ppo_reason(row: RunRow) -> str:
    if row.policy != "ppo":
        return ""
    if "ppo" not in available_arms():
        return "a PPO run is re-simulated with its policy.zip, which needs the train extra"
    if not (row.path / "policy.zip").is_file():
        return "this PPO run saved no policy.zip to re-simulate with"
    return ""


def attribution_reason(row: RunRow) -> str:
    """Why this run cannot be attributed from the dashboard; empty when it can. A real-data
    run is attributed on the panel and split it recorded, as the action mix and replay are."""
    if row.status != "OK":
        return f"the run is {row.status}; attribution needs files that match the manifest"
    return _ppo_reason(row) or real_data_reason(row)


def compute_attribution(run_dir: Path, out: Path) -> Path:
    """``attribution.attribute_run`` on the run, written under ``out``, never into the run."""
    from optspread.attribution import attribute_run, write_attribution

    result = attribute_run(run_dir)
    return write_attribution(result, out)


def _stamp(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


@lru_cache(maxsize=16)
def _read_attribution(path: str, stamp: tuple[int, int]) -> Any:
    """A cached attribution, read once while the file stands (``stamp``): a big one is
    megabytes of JSON, and a run page asks for it again with every episode it opens."""
    from optspread.attribution import read_attribution

    return read_attribution(Path(path))


def attribution_view(
    root: Path, row: RunRow, task: TaskState, seed: int | None = None
) -> AttributionOut:
    target = attribution_file(root, row)
    reason = attribution_reason(row)
    base = {"can_compute": not reason, "reason": reason, "file": str(target)}
    if target.is_file():
        try:
            result = _read_attribution(str(target), _stamp(target))
        except (OSError, ValueError) as exc:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return AttributionOut(
                state="failed", **{**base, "reason": text}, result=None, seed=None, steps=[]
            )
        episodes = {e.seed: e for e in result.episodes}
        pick = seed if seed in episodes else (result.episodes[0].seed if result.episodes else None)
        steps = (
            [s.model_dump(mode="json") for s in episodes[pick].steps] if pick is not None else []
        )
        return AttributionOut(
            state="cached",
            **base,
            result=_finite_json(result.without_steps().model_dump(mode="json")),
            seed=pick,
            steps=_finite_json(steps),
        )
    if task.state == "running":
        return AttributionOut(
            state="computing", **base, result=None, seed=None, steps=[], seconds=task.seconds
        )
    if task.state == "failed":
        return AttributionOut(
            state="failed", **{**base, "reason": task.error}, result=None, seed=None, steps=[]
        )
    return AttributionOut(state="absent", **base, result=None, seed=None, steps=[])


def _finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    return value


# capital: drawdown and return on what the book posts


class BandOut(Out):
    """Quantiles across episodes at each day of the episode (day 0 is the reset)."""

    mean: list[float | None]
    p10: list[float | None]
    p50: list[float | None]
    p90: list[float | None]


def band(paths: Sequence[Sequence[float]]) -> BandOut | None:
    """Mean and 10/50/90th percentiles by day over paths of any lengths."""
    if not paths:
        return None
    n = max(len(p) for p in paths)
    grid = np.full((len(paths), n), np.nan)
    for i, p in enumerate(paths):
        grid[i, : len(p)] = p
    with np.errstate(all="ignore"):
        q = np.nanpercentile(grid, (10, 50, 90), axis=0)
        mean = np.nanmean(grid, axis=0)
    return BandOut(
        mean=[_f(x) for x in mean],
        p10=[_f(x) for x in q[0]],
        p50=[_f(x) for x in q[1]],
        p90=[_f(x) for x in q[2]],
    )


class CapitalEpisodeOut(Out):
    seed: int
    pnl: float
    max_drawdown: float  # $, peak to trough of the equity path within the episode
    peak_margin: float  # $, the most margin the book held at any step
    return_on_cash: float  # pnl / initial cash
    return_on_margin: float | None  # pnl / peak margin; None when nothing was held


class CapitalOut(Out):
    state: str  # "done" | "computing" | "absent" | "failed"
    can_compute: bool
    reason: str
    initial_cash: float | None = None
    n: int = 0
    reproduces: bool | None = None  # every re-simulated P&L equals its run.json row
    agrees: bool | None = None  # every P&L within ``rtol`` of its row (relative, floor $1)
    rtol: float | None = None
    n_differ: int = 0  # episodes beyond ``rtol``
    max_abs_diff: float | None = None  # largest |re-simulated - saved| episode P&L, $
    mean_return_on_cash: float | None = None
    mean_return_on_margin: float | None = None  # total P&L over total peak margin
    mean_peak_margin: float | None = None
    max_peak_margin: float | None = None
    median_drawdown: float | None = None
    worst_drawdown: float | None = None
    worst_drawdown_on_cash: float | None = None
    method: str = ""
    episodes: list[CapitalEpisodeOut] = Field(default_factory=list)
    seconds: float | None = None
    # The equity path of every episode, re-simulated: the first ``MAX_PATHS_SENT`` as they
    # ran, and the band of all of them by day; the margin the book posted, by day.
    path_seeds: list[int] = Field(default_factory=list)
    paths: list[list[float]] = Field(default_factory=list)
    equity_band: BandOut | None = None
    margin_band: BandOut | None = None
    file: str = ""  # the cache under the attribution root; empty when not cached


MAX_PATHS_SENT = 60  # equity paths sent per run, as webapi.MAX_EQUITY_SENT
CAPITAL_DIR = "capital"  # under the attribution root: the cached re-simulations


def capital_file(root: Path, row: RunRow) -> Path:
    """``<attribution root>/capital/<run_id>[.<source>].json``: never in a run directory."""
    suffix = "" if row.source in ("", "synthetic") else f".{row.source}"
    return root / CAPITAL_DIR / f"{row.run_id}{suffix}.json"


CAPITAL_METHOD = (
    "episodes re-simulated from the manifest's config on the data it recorded (the synthetic "
    "world, or the real panel's eval split, refused when the panel was rebuilt since), as the "
    "harness runs them (a PPO run with the policy.zip it saved, not retrained); equity and margin "
    "read at every step from the env (margin as "
    "optspread.margin books it: max loss for defined risk, the Cboe rule otherwise); "
    "return on cash is P&L over initial cash, return on margin is P&L over the peak margin "
    "the episode held, drawdown is peak to trough within each episode"
)


def compute_capital(run_dir: Path, out: Path | None = None) -> CapitalOut:
    """Re-run the episodes and read the equity and the margin the book carried at each step.

    With ``out`` the result is also written there (the cache under the attribution root);
    a run directory, or a folder holding runs, is refused as a place to write."""
    from optspread.artifacts import read_run
    from optspread.env import OBS_NAMES
    from optspread.harness import build_env
    from optspread.policies import make_policy

    manifest = RunManifest.read(run_dir)
    config = manifest.config
    record = read_run(run_dir)
    saved = {e.seed: e.pnl for e in record.episodes}
    if manifest.data_record.source == "synthetic":
        env = build_env(config)
    else:  # the panel and split it recorded, refused when the panel was rebuilt since
        from optspread.attribution import recorded_provider
        from optspread.sweep import changed_artifacts

        changed = changed_artifacts(run_dir, manifest)
        if changed:
            raise ValueError(f"{', '.join(changed)} no longer match the manifest in {run_dir}")
        source, provider = recorded_provider(manifest)
        env = build_env(config, source, provider=provider)
    policy: Any
    if config.policy == "ppo":
        from optspread.attribution import load_sb3_policy
        from optspread.train import POLICY_FILENAME

        saved_policy = run_dir / POLICY_FILENAME
        if not saved_policy.is_file():
            raise FileNotFoundError(f"a ppo run is re-simulated with its {POLICY_FILENAME}")
        policy = load_sb3_policy(saved_policy, config)
    else:
        policy = make_policy(config.policy, roll_dte=config.env.roll_dte)
    cash = float(config.env.initial_cash)
    at = OBS_NAMES.index("margin_frac")
    rows: list[CapitalEpisodeOut] = []
    equity_paths: list[list[float]] = []
    margin_paths: list[list[float]] = []
    same = True
    diffs: list[float] = []
    for seed in range(config.seed, config.seed + config.n_episodes):
        obs, info = env.reset(seed=seed)
        policy.reset(seed)
        equity0 = float(info["equity"])
        path = [0.0]
        margin = [float(obs[at]) * cash]
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy.act(obs, info))
            path.append(float(info["equity"]) - equity0)
            margin.append(float(obs[at]) * cash)
        equity_paths.append(path)
        margin_paths.append(margin)
        eq = np.asarray(path)
        pnl = float(info["equity"]) - equity0
        same &= seed in saved and saved[seed] == pnl
        diffs.append(abs(pnl - saved[seed]) if seed in saved else math.inf)
        peak = max(margin)
        rows.append(
            CapitalEpisodeOut(
                seed=seed,
                pnl=pnl,
                max_drawdown=float((np.maximum.accumulate(eq) - eq).max()),
                peak_margin=peak,
                return_on_cash=pnl / cash,
                return_on_margin=pnl / peak if peak > 0 else None,
            )
        )
    n_differ = sum(
        1
        for r, d in zip(rows, diffs, strict=True)
        if not d <= SCRIPTED_RTOL * max(1.0, abs(saved.get(r.seed, 0.0)))
    )
    dd = np.asarray([r.max_drawdown for r in rows])
    peaks = np.asarray([r.peak_margin for r in rows])
    total_peak = float(peaks.sum())
    result = CapitalOut(
        state="done",
        can_compute=True,
        reason="",
        initial_cash=cash,
        n=len(rows),
        reproduces=same,
        agrees=n_differ == 0,
        rtol=SCRIPTED_RTOL,
        n_differ=n_differ,
        max_abs_diff=_f(max(diffs)) if diffs else None,
        mean_return_on_cash=_f(float(np.mean([r.return_on_cash for r in rows]))) if rows else None,
        mean_return_on_margin=_f(sum(r.pnl for r in rows) / total_peak) if total_peak else None,
        mean_peak_margin=_f(float(peaks.mean())) if rows else None,
        max_peak_margin=_f(float(peaks.max())) if rows else None,
        median_drawdown=_f(float(np.median(dd))) if rows else None,
        worst_drawdown=_f(float(dd.max())) if rows else None,
        worst_drawdown_on_cash=_f(float(dd.max()) / cash) if rows else None,
        method=CAPITAL_METHOD,
        episodes=rows,
        path_seeds=[r.seed for r in rows[:MAX_PATHS_SENT]],
        paths=equity_paths[:MAX_PATHS_SENT],
        equity_band=band(equity_paths),
        margin_band=band(margin_paths),
    )
    if out is not None:
        _refuse_run_folder(out.parent)
        out.parent.mkdir(parents=True, exist_ok=True)
        result = result.model_copy(update={"file": str(out)})
        out.write_text(result.model_dump_json(), encoding="utf-8", newline="\n")
    return result


def _refuse_run_folder(folder: Path) -> None:
    """A cache is written beside runs, never into one or into a folder of them."""
    if (folder / MANIFEST_FILENAME).exists() or (
        folder.is_dir() and any((d / MANIFEST_FILENAME).is_file() for d in folder.iterdir())
    ):
        raise ValueError(f"{folder} holds runs; a re-simulation is cached elsewhere")


def read_capital(path: Path) -> CapitalOut | None:
    """The cached re-simulation at ``path``, or None when absent or unreadable."""
    try:
        return _read_capital(str(path), _stamp(path))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=32)
def _read_capital(path: str, stamp: tuple[int, int]) -> CapitalOut:
    return CapitalOut.model_validate_json(Path(path).read_text(encoding="utf-8"))


def capital_reason(row: RunRow) -> str:
    """Why this run's capital cannot be re-simulated here; empty when it can. A real-data
    run is re-simulated on the panel and split it recorded, as the action mix and replay are."""
    if row.status != "OK":
        return f"the run is {row.status}"
    if row.policy != "ppo" and not _scripted(row.policy):
        return f"no policy named {row.policy!r} to re-simulate with"
    return _ppo_reason(row) or real_data_reason(row)


def capital_view(row: RunRow, task: TaskState, result: Any, root: Path | None = None) -> CapitalOut:
    """The re-simulation: cached under ``root`` (the attribution root) if it is there,
    else the task's result, its progress, its failure or why it cannot run here."""
    reason = capital_reason(row)
    if root is not None and task.state != "running":
        cached = read_capital(capital_file(root, row))
        if cached is not None:
            return cached
    if task.state == "done" and isinstance(result, CapitalOut):
        return result.model_copy(update={"seconds": task.seconds})
    if task.state == "running":
        return CapitalOut(state="computing", can_compute=True, reason="", seconds=task.seconds)
    if task.state == "failed":
        return CapitalOut(state="failed", can_compute=not reason, reason=task.error)
    return CapitalOut(state="absent", can_compute=not reason, reason=reason, method=CAPITAL_METHOD)


# the action mix of one run, re-simulated on demand (optspread.actionmix)

MIX_DIR = "actionmix"  # under the attribution root: full mixes, each holding kept
MAX_HOLDINGS_SENT = 400


class HoldingOut(Out):
    """One structure from its open to the day the book was flat again."""

    seed: int
    structure: str
    open_day: int
    exit_day: int
    days: int
    exit: str  # "policy" | "expiry" | "margin" | "horizon"
    tenor_days: int
    short_delta: float | None  # mean |delta| of the short legs at the open
    long_delta: float | None  # of the long wings; None without any
    credit: float
    margin: float
    strikes: list[float]


class EpisodeTimelineOut(Out):
    """One episode day by day: what was held over each move and what the trade did."""

    seed: int
    pnl: float
    held: list[str]
    events: list[str]
    holdings: list[HoldingOut]


class MixDetailOut(Out):
    n_holdings: int
    days_held: dict[str, list[int]]  # per structure: holdings by days held (index = days)
    holdings: list[HoldingOut]  # the first MAX_HOLDINGS_SENT, in episode order
    example: EpisodeTimelineOut | None  # the episode of median P&L


class MixViewOut(Out):
    state: str  # "cached" (full) | "lean" (summary only) | "computing" | "absent" | "failed"
    can_compute: bool
    reason: str
    file: str
    mix: am.RunActionMix | None  # lean: the summary and each episode's counts
    detail: MixDetailOut | None
    seconds: float | None = None


def mix_file(root: Path, row: RunRow) -> Path:
    """``<attribution root>/actionmix/<run_id>[.<source>].json``."""
    suffix = "" if row.source in ("", "synthetic") else f".{row.source}"
    return root / MIX_DIR / f"{row.run_id}{suffix}.json"


def mix_reason(row: RunRow) -> str:
    """Why this run's action mix cannot be re-simulated here; empty when it can."""
    if row.status != "OK":
        return f"the run is {row.status}; the mix is re-simulated from files that match"
    return _ppo_reason(row) or real_data_reason(row)


def compute_mix(run_dir: Path, out: Path) -> Path:
    """``actionmix.mix_run`` on the run, every holding kept, written to ``out``."""
    return am.write_mix(am.mix_run(run_dir), out, episodes=True)


def _holding(seed: int, h: am.Holding) -> HoldingOut:
    shorts, longs = h.abs_deltas(short=True), h.abs_deltas(short=False)
    return HoldingOut(
        seed=seed,
        structure=h.structure,
        open_day=h.open_day,
        exit_day=h.exit_day,
        days=h.days,
        exit=h.exit,
        tenor_days=h.tenor_days,
        short_delta=_f(sum(shorts) / len(shorts)) if shorts else None,
        long_delta=_f(sum(longs) / len(longs)) if longs else None,
        credit=h.credit,
        margin=h.margin,
        strikes=sorted(leg.strike for leg in h.legs),
    )


def mix_detail(mix: am.RunActionMix) -> MixDetailOut:
    """What the summary pools away: holding periods, each holding, and one episode."""
    names = mix.action_names
    held = [_holding(e.seed, h) for e in mix.episodes for h in e.holdings]
    longest = max((h.days for h in held), default=0)
    days_held = {s: [0] * (longest + 1) for s in am.STRUCTURES}
    for h in held:
        days_held.setdefault(h.structure, [0] * (longest + 1))[h.days] += 1
    example = None
    if mix.episodes:
        ranked = sorted(mix.episodes, key=lambda e: (e.pnl, e.seed))
        e = ranked[(len(ranked) - 1) // 2]
        example = EpisodeTimelineOut(
            seed=e.seed,
            pnl=e.pnl,
            held=[names[i] if 0 <= i < len(names) else str(i) for i in e.held],
            events=list(e.events),
            holdings=[_holding(e.seed, h) for h in e.holdings],
        )
    return MixDetailOut(
        n_holdings=len(held),
        days_held=days_held,
        holdings=held[:MAX_HOLDINGS_SENT],
        example=example,
    )


@lru_cache(maxsize=16)
def _read_full_mix(path: str, stamp: int) -> tuple[am.RunActionMix, MixDetailOut]:
    mix = am.read_mix(Path(path))
    return mix.lean(), mix_detail(mix)


def mix_view(root: Path, row: RunRow, task: TaskState, lean_root: Path | None = None) -> MixViewOut:
    """The run's mix: the full one cached under ``root`` (the attribution root); else the
    lean one ``scripts/action_mix.py`` wrote under ``lean_root``, its summary without the
    holdings, while the full one is computed or when it cannot be."""
    reason = mix_reason(row)
    full = mix_file(root, row)
    base = {"can_compute": not reason, "reason": reason, "file": str(full)}
    if full.is_file() and task.state != "running":
        try:
            mix, detail = _read_full_mix(str(full), full.stat().st_mtime_ns)
        except (OSError, ValueError) as exc:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return MixViewOut(state="failed", **{**base, "reason": text}, mix=None, detail=None)
        return MixViewOut(state="cached", **base, mix=mix, detail=detail)
    lean = None
    folder = lean_root if lean_root is not None else am.default_root()
    suffix = "" if row.source in ("", "synthetic") else f".{row.source}"
    lean_path = folder / f"{row.run_id}{suffix}.json"
    if lean_path.is_file():
        with contextlib.suppress(OSError, ValueError):
            lean = am.read_mix(lean_path).lean()
    if task.state == "running":
        return MixViewOut(state="computing", **base, mix=lean, detail=None, seconds=task.seconds)
    if task.state == "failed":
        return MixViewOut(state="failed", **{**base, "reason": task.error}, mix=lean, detail=None)
    if lean is not None:
        return MixViewOut(state="lean", **{**base, "file": str(lean_path)}, mix=lean, detail=None)
    return MixViewOut(state="absent", **base, mix=None, detail=None)


# replay

# The seller's tolerance for a scripted run replayed elsewhere: every number of run.json
# within a billionth, relative (``optspread.attribution.PNL_RTOL``, the M4 gate's).
SCRIPTED_RTOL = 1e-9


class ReplayOut(Out):
    state: str  # "idle" | "running" | "done" | "failed"
    can_replay: bool
    reason: str  # why it cannot run here; the failure; or, when done, what replay said
    verdict: str = ""  # "IDENTICAL" | "AGREES" (within ``rtol``) | "DIFFERS"
    rtol: float | None = None  # the tolerance asked for; None for bytes only (a PPO retrain)
    max_rel_diff: float | None = None  # the largest relative difference, with ``rtol``
    comparable: bool = True  # False: a retrained PPO run on another platform than its own
    note: str = ""  # what the verdict means here
    retrains: bool = False  # the replay retrains a policy before it evaluates
    train_steps: int | None = None
    estimate_hours: float | None = None  # one core at A7's rates, training included
    seconds: float | None = None


class ReplayDone(Out):
    """What ``run_replay`` hands back to the task: replay's verdict and its numbers."""

    verdict: str  # "IDENTICAL" | "WITHIN" | "DIFFERS", as optspread.replay names them
    reason: str
    seconds: float
    rtol: float | None
    max_rel_diff: float | None
    comparable: bool


PPO_NOT_COMPARABLE = (
    "A retrained PPO run is not comparable across platforms: training compounds a "
    "last-bit difference into another policy, so it is checked statistically "
    "(optspread.reproduction), not by replay, off the machine it was trained on."
)


def replay_reason(row: RunRow) -> str:
    if row.policy == "ppo" and "ppo" not in available_arms():
        return "a PPO run retrains to replay, which needs the train extra"
    if row.status in ("INVALID",):
        return "the manifest does not read; there is nothing to replay from"
    return ""


def _replay_cost(row: RunRow) -> tuple[bool, int | None, float | None]:
    """``(retrains, train steps, hours)`` of replaying ``row`` on one core."""
    try:
        config = RunManifest.read(row.path).config
    except (OSError, ValueError):
        return row.policy == "ppo", None, None
    steps = config.train.total_timesteps if config.train is not None else None
    est = launch_estimate(max(1, config.n_episodes), max(1, config.env.episode_days), steps)
    return config.train is not None, steps, est.total_hours


def run_replay(run_dir: Path) -> ReplayDone:
    """A scripted run within ``SCRIPTED_RTOL``; a PPO run byte for byte, where a tolerance
    means nothing off the platform it was trained on."""
    from optspread.replay import platform_differs, replay_run

    learned = RunManifest.read(run_dir).config.train is not None
    out = replay_run(run_dir, rtol=None if learned else SCRIPTED_RTOL)
    return ReplayDone(
        verdict=out.verdict,
        reason=out.reason,
        seconds=out.seconds,
        rtol=out.rtol,
        max_rel_diff=out.max_rel_diff,
        comparable=not (learned and platform_differs(run_dir)),
    )


def replay_note(done: ReplayDone) -> str:
    if done.verdict == "IDENTICAL":
        return "run.json reproduced bit for bit"
    if done.verdict == "WITHIN":
        return (
            f"every number of run.json within {done.rtol:g} relative "
            f"(largest {done.max_rel_diff:.2g}): the last bits a platform rounds its own way"
        )
    if not done.comparable:
        return PPO_NOT_COMPARABLE
    if done.rtol is not None:
        return f"beyond {done.rtol:g} relative"
    return "run.json differs"


def replay_view(row: RunRow, task: TaskState, result: Any) -> ReplayOut:
    reason = replay_reason(row)
    retrains, steps, hours = _replay_cost(row)
    cost = {"retrains": retrains, "train_steps": steps, "estimate_hours": hours}
    if task.state == "done" and isinstance(result, ReplayDone):
        return ReplayOut(
            state="done",
            can_replay=not reason,
            reason=str(result.reason),
            verdict={"IDENTICAL": "IDENTICAL", "WITHIN": "AGREES"}.get(result.verdict, "DIFFERS"),
            rtol=result.rtol,
            max_rel_diff=result.max_rel_diff,
            comparable=result.comparable,
            note=replay_note(result),
            seconds=float(result.seconds),
            **cost,
        )
    if task.state == "running":
        return ReplayOut(state="running", can_replay=True, reason="", seconds=task.seconds, **cost)
    if task.state == "failed":
        return ReplayOut(state="failed", can_replay=not reason, reason=task.error, **cost)
    note = PPO_NOT_COMPARABLE if retrains else ""
    return ReplayOut(state="idle", can_replay=not reason, reason=reason, note=note, **cost)


# -- the leak detector's floor and its per-field controls --------------------------- #

FLOOR_EPS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025)


class LeakFloorParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    n_seeds: int = Field(default=10, ge=3, le=40)
    eps_grid: list[float] = Field(
        default_factory=lambda: list(FLOOR_EPS), min_length=2, max_length=10
    )


class LeakFloorOut(Out):
    eps: list[float]  # descending, as leaks.leak_floor plants them
    flagged: list[bool]
    floor: float | None  # smallest eps still caught
    seeds: list[int]
    seconds: float
    source: str


def leak_floor_out(p: LeakFloorParams) -> LeakFloorOut:
    from optspread.leaks import default_env, leak_floor

    if any(not 0.0 < e <= 2.0 for e in p.eps_grid):
        raise ValueError("every eps must be in (0, 2]")
    seeds = list(range(p.n_seeds))
    start = time.perf_counter()
    report = leak_floor(default_env, p.eps_grid, seeds)
    return LeakFloorOut(
        eps=list(report.eps_grid),
        flagged=list(report.flagged),
        floor=report.floor,
        seeds=seeds,
        seconds=time.perf_counter() - start,
        source="optspread.leaks.leak_floor on default_env: spot look-ahead at each eps",
    )


class LeakFieldParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    field: str = "ret_1d"
    eps: float = Field(default=0.25, gt=0.0, le=2.0)
    n_seeds: int = Field(default=10, ge=3, le=40)


class LeakFieldOut(Out):
    field: str
    eps: float
    scale: float  # the field's own standard deviation under the probe
    seeds: list[int]
    flagged: bool
    named_correctly: bool  # the worst feature is the field that was poisoned
    worst_feature: str
    p_value: float | None
    threshold: float
    features: list[FeatureOut]
    seconds: float
    source: str
    t_critical: float | None = None  # the |t| a field must exceed to be flagged


def leak_fields() -> list[str]:
    from optspread.env import OBS_NAMES

    return list(OBS_NAMES)


def _poisoned_env(field_name: str, eps: float, scale: float) -> Any:
    from optspread.leaks import default_env, poison_field

    return poison_field(default_env(), field_name, eps, scale)


def leak_field_out(p: LeakFieldParams) -> LeakFieldOut:
    """Look-ahead planted in one field, eps of its own sd: does the detector fire on it?"""
    from functools import partial

    from optspread.leaks import default_env, detect_leak, field_scales

    if p.field not in leak_fields():
        raise ValueError(f"{p.field!r} is not an observation field")
    seeds = list(range(p.n_seeds))
    start = time.perf_counter()
    scale = field_scales(default_env, seeds)[p.field]
    report = detect_leak(partial(_poisoned_env, p.field, p.eps, scale), seeds)
    side = _side(report, past_unchanged=True)
    return LeakFieldOut(
        field=p.field,
        eps=p.eps,
        scale=scale,
        seeds=seeds,
        flagged=report.flagged,
        named_correctly=report.flagged and report.worst_feature == p.field,
        worst_feature=report.worst_feature,
        p_value=_f(report.p_value),
        threshold=report.threshold,
        features=side.features,
        seconds=time.perf_counter() - start,
        source="optspread.leaks.poison_field and detect_leak on default_env",
        t_critical=side.t_critical,
    )


class FieldControlOut(Out):
    field: str
    flagged: bool
    named_correctly: bool
    p_value: float | None


class LeakCoverageOut(Out):
    eps: float
    seeds: list[int]
    clean_flagged: bool
    passed: bool
    uncovered: list[str]
    misattributed: list[str]
    controls: list[FieldControlOut]
    seconds: float
    source: str


def leak_coverage_out(eps: float, n_seeds: int) -> LeakCoverageOut:
    from optspread.leaks import default_env, field_coverage

    seeds = list(range(n_seeds))
    start = time.perf_counter()
    rep = field_coverage(default_env, seeds, eps)
    return LeakCoverageOut(
        eps=eps,
        seeds=seeds,
        clean_flagged=rep.clean.flagged,
        passed=rep.passed,
        uncovered=list(rep.uncovered),
        misattributed=list(rep.misattributed),
        controls=[
            FieldControlOut(
                field=c.field,
                flagged=c.flagged,
                named_correctly=c.named_correctly,
                p_value=_f(c.p_value),
            )
            for c in rep.controls
        ],
        seconds=time.perf_counter() - start,
        source="optspread.leaks.field_coverage on default_env",
    )


# -- what the fixed-delta seller opens on a real day -------------------------------- #


class LegOut(Out):
    right: str
    strike: float
    qty: int
    price: float  # per share, the chain's mid
    iv: float
    delta: float
    moneyness: float  # K / spot


class SellerOut(Out):
    """The fixed-delta seller's structure on one real close: an illustration, not a backtest."""

    date: str
    structure: str
    spot: float
    r: float
    tenor_trading_days: int
    tenor_calendar_days: float
    short_delta: float
    wing_delta: float
    legs: list[LegOut]
    credit: float  # $ received, one structure
    max_loss: float | None  # $, defined risk only, as optspread.margin decides it
    # $, the worst expiry loss computed from the legs whenever both sides have a wing,
    # whatever the margin check decides; equal to max_loss when the check agrees
    capped_loss: float | None = None
    # the legs cap the loss but the frozen margin check margins them as naked (#39)
    margined_as_naked: bool = False
    margin: float  # $, optspread.margin at the open
    breakevens: list[float]
    note: str
    source: str
    reading: str = ""  # the data page's question 4, in one plain sentence


SELLER_NOTE = (
    "Illustration on the real chain at the close; the experiment runs on the synthetic world."
)


def real_chain(
    day: date,
    k: NDArray[np.float64],
    tenors: NDArray[np.int64],
    iv: NDArray[np.float64],
    spot: float,
    carry: float,
    cfg: MarketConfig,
) -> Any:
    """The chain ``realdata.provider.HistoricalChainProvider`` builds for this close.

    The day's gridded surface is laid on the provider's own grid and made free of
    calendar arbitrage as its store makes it (``calendar_free``), then its ``_chain`` is
    run on a one-day store, so strikes, tenors and vols are the ones a real-data episode
    would see on day 0.
    """
    from types import SimpleNamespace

    from optspread.realdata.panel import K_GRID, TENOR_GRID
    from optspread.realdata.provider import HistoricalChainProvider, SurfaceStore, calendar_free

    cube = np.full((len(TENOR_GRID), len(K_GRID)), np.nan)
    ki = {round(float(x), 4): j for j, x in enumerate(K_GRID)}
    ti = {int(t): i for i, t in enumerate(TENOR_GRID)}
    for r, t in enumerate(tenors):
        for c, x in enumerate(k):
            if int(t) in ti and round(float(x), 4) in ki and np.isfinite(iv[r, c]):
                cube[ti[int(t)], ki[round(float(x), 4)]] = iv[r, c]
    if not np.isfinite(cube).any():
        raise PanelNotBuiltError(f"no gridded surface on {day} to build a chain from")
    store = SurfaceStore(
        dates=(day.isoformat(),),
        spot=np.asarray([spot]),
        carry=np.asarray([carry]),
        ivs=calendar_free(cube)[None],
    )
    shim: Any = SimpleNamespace(config=cfg, store=store, _start=0, _day=0)
    shim._smile = lambda *a: HistoricalChainProvider._smile(shim, *a)
    return HistoricalChainProvider._chain(shim)


def capped_loss(legs: Sequence[Any]) -> tuple[float | None, bool]:
    """The worst expiry loss the legs cap, and whether the frozen margin check still
    margins them as naked shorts.

    A long wing beyond every short caps the loss whatever ``margin.is_defined_risk``
    decides: its absolute tolerance misreads some high-spot condors as naked (#39), and
    the harness, frozen, margins them so. Without a wing on each side the loss is not
    capped: (None, False)."""
    from optspread.margin import is_defined_risk, max_loss

    shorts = [leg for leg in legs if leg.qty < 0]
    winged = bool(shorts) and all(
        any(
            w.qty > 0
            and w.right == sh.right
            and (w.strike < sh.strike if sh.right == "P" else w.strike > sh.strike)
            for w in legs
        )
        for sh in shorts
    )
    if not winged:
        return None, False
    return float(max_loss(tuple(legs))), not is_defined_risk(tuple(legs))


def seller_on(chain: Any, day: date, cfg: MarketConfig) -> SellerOut:
    from optspread.actions import ACTION_NAMES, IRON_CONDOR, build_position
    from optspread.config import EnvConfig
    from optspread.margin import is_defined_risk, margin_requirement, max_loss
    from optspread.types import MULTIPLIER

    env = EnvConfig()
    pos = build_position(IRON_CONDOR, chain, env)
    assert pos is not None
    legs = pos.legs
    exp = legs[0].expiry_day
    tenor = int(exp - chain.t)
    credit = -sum(leg.qty * leg.entry_price for leg in legs) * MULTIPLIER
    shorts = {leg.right: leg.strike for leg in legs if leg.qty < 0}
    per_share = credit / MULTIPLIER
    breakevens = [shorts["P"] - per_share] if "P" in shorts else []
    breakevens += [shorts["C"] + per_share] if "C" in shorts else []
    longs = {leg.right: leg.strike for leg in legs if leg.qty > 0}
    defined = is_defined_risk(legs)
    loss = float(max_loss(legs)) if defined else None
    capped, naked = capped_loss(legs)
    margin = float(margin_requirement(legs, chain))
    reading = (
        panel.seller_reading(
            day.isoformat(),
            float(chain.spot),
            tenor,
            shorts["P"],
            shorts["C"],
            longs["P"],
            longs["C"],
            float(credit),
            capped if capped is not None else loss,
            margin,
            breakevens,
            env.short_delta,
            naked=naked,
        )
        if {"P", "C"} <= shorts.keys() and {"P", "C"} <= longs.keys()
        else ""
    )
    return SellerOut(
        date=day.isoformat(),
        structure=ACTION_NAMES[IRON_CONDOR],
        spot=float(chain.spot),
        r=float(chain.r),
        tenor_trading_days=tenor,
        tenor_calendar_days=tenor * 365.0 / cfg.trading_days_per_year,
        short_delta=env.short_delta,
        wing_delta=env.wing_delta,
        legs=[
            LegOut(
                right=leg.right,
                strike=leg.strike,
                qty=leg.qty,
                price=leg.entry_price,
                iv=chain.iv_at(exp, leg.strike),
                delta=chain.greeks(leg.right, leg.strike, exp).delta,
                moneyness=leg.strike / float(chain.spot),
            )
            for leg in sorted(legs, key=lambda x: x.strike)
        ],
        credit=float(credit),
        max_loss=loss,
        capped_loss=capped,
        margined_as_naked=naked,
        margin=margin,
        breakevens=[float(b) for b in breakevens],
        note=SELLER_NOTE,
        reading=reading,
        source=(
            "optspread.actions.build_position(IRON_CONDOR) at EnvConfig() deltas on the chain "
            "realdata.provider builds from surface_grid.parquet and spx_daily.parquet"
        ),
    )


def _close(reader: PanelReader, day: date) -> tuple[float, float]:
    """The spot and 30-day carry the provider prices ``day`` on, OptionMetrics first.

    As ``realdata.provider`` reads them: the parity spot and the carry against it
    (``spot_parity``, ``carry30_parity``) where the panel has them, else the close and
    ``carry30``; ``rate30`` where the carry is missing, else zero.
    """
    daily = reader.daily()
    if "date" not in daily:
        raise PanelNotBuiltError(f"no daily table at {reader.layout.daily}")
    idx = np.flatnonzero(daily["date"] == np.datetime64(day, "D"))
    if "source" in daily and len(idx):
        om = idx[daily["source"][idx] == "optionmetrics"]
        idx = om if len(om) else idx
    if not len(idx):
        raise PanelNotBuiltError(f"no daily row on {day} in {reader.layout.daily}")
    i = int(idx[0])
    spot = float(daily["spot"][i])
    carry_cols: tuple[str, ...] = ("carry30", "rate30")
    if "spot_parity" in daily and "carry30_parity" in daily:
        parity = float(daily["spot_parity"][i])
        if math.isfinite(parity) and parity > 0:
            spot = parity
        carry_cols = ("carry30_parity", "rate30")
    carry = math.nan
    for col in carry_cols:
        if col in daily and math.isfinite(float(daily[col][i])):
            carry = float(daily[col][i])
            break
    if not (math.isfinite(spot) and spot > 0):
        raise PanelNotBuiltError(f"no spot on {day} in {reader.layout.daily}")
    return spot, carry if math.isfinite(carry) else 0.0


def seller_out(reader: PanelReader, day: date) -> SellerOut:
    """What the fixed-delta seller would open at ``day``'s close on the real chain."""
    g = reader.grid_day(day)
    spot, carry = _close(reader, day)
    cfg = MarketConfig()
    chain = real_chain(day, g.k, g.tenor_days, g.iv, spot, carry, cfg)
    return seller_on(chain, day, cfg)


# -- launch: a PPO preset, and the estimate with training in it ---------------------- #

PPO_PRESET = PresetOut(
    name="ppo_short",
    label="PPO, short",
    description=(
        "PPO at 4 vol pts, 100 episodes, 100,000 training steps: a short illustration; "
        "the sweep trains 1,000,000."
    ),
    spec={
        **JobSpec().model_dump(mode="json"),
        "policy": "ppo",
        "iv_premium_vol_pts": 4.0,
        "n_episodes": 100,
        "total_timesteps": 100_000,
    },
)


class EstimateOut(Out):
    eval_steps: int
    train_steps: int
    eval_hours: float
    train_hours: float
    total_hours: float
    train_rate: float
    eval_rate: float
    source: str


def launch_options() -> JobOptionsOut:
    """``webapi.job_options`` with the PPO preset beside the scripted ones."""
    return job_options().model_copy(update={"ppo_preset": PPO_PRESET})


def launch_estimate(n_episodes: int, n_days: int, total_timesteps: int | None) -> EstimateOut:
    """Wall time of one run on one core at the rates A7 measured, training included."""
    steps = n_episodes * n_days
    train = int(total_timesteps or 0)
    eval_h = steps / A7_EVAL_RATE / 3600.0
    train_h = train / A7_TRAIN_RATE / 3600.0
    return EstimateOut(
        eval_steps=steps,
        train_steps=train,
        eval_hours=eval_h,
        train_hours=train_h,
        total_hours=eval_h + train_h,
        train_rate=A7_TRAIN_RATE,
        eval_rate=A7_EVAL_RATE,
        source="docs/PRE_REGISTRATION.md A7: 804 train and 1,538 eval steps/s/core",
    )


# -- launch: a live preview of the run a spec stands for ---------------------------- #

STRUCTURE_PLAIN = {
    "IRON_CONDOR": (
        "an iron condor (a short put and a short call, each with a long wing further out)"
    ),
    "SHORT_STRANGLE": "a short strangle (a short put and a short call, no wings)",
    "PUT_CREDIT_SPREAD": "a put credit spread (a short put with a long put further out)",
}


class PreviewLegOut(Out):
    right: str
    strike: float
    qty: int
    price: float  # per share, the chain's mid
    iv: float
    delta: float  # per-share delta of one long contract
    moneyness: float  # K / spot


class DayZeroOut(Out):
    """What the policy opens on day 0 of the first episode: the chain it sees, its legs."""

    policy: str
    reference: bool  # True for PPO: the fixed-delta seller's choice, shown for scale
    structure: str  # "FLAT" when nothing is opened
    plain: str  # one sentence
    spot: float
    tenor_days: int | None
    legs: list[PreviewLegOut]
    credit: float | None  # $ received for one structure (negative: paid)
    margin: float | None
    max_loss: float | None  # $, defined risk only
    breakevens: list[float]


class PeerOut(Out):
    key: str
    policy: str
    premium: float | None
    source: str
    n_episodes: int | None
    mean_pnl: float | None
    relation: str  # "same run" | "control" | "same arm" | "other arm"
    paired: bool  # starts on the same first seed: its episodes pair with this run's
    note: str


class PreviewOut(Out):
    ok: bool
    errors: list[str]
    run_id: str = ""
    config_hash: str = ""
    exists: bool = False  # this exact run is on disk: launching it reuses it
    run_key: str = ""
    world: str = ""  # the market the run sees, in one sentence
    implied_vol: float | None = None  # sigma plus the premium, decimal; None on real data
    day0: DayZeroOut | None = None
    day0_note: str = ""  # why there is no day 0, when there is none
    estimate: EstimateOut | None = None
    peers: list[PeerOut] = Field(default_factory=list)


def _clean(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _day_zero(config: Any, source: str) -> DayZeroOut:
    from optspread.actions import ACTION_NAMES, build_position
    from optspread.harness import build_env
    from optspread.margin import is_defined_risk, margin_requirement, max_loss
    from optspread.policies import DeltaSellerPolicy, make_policy
    from optspread.types import MULTIPLIER

    env = build_env(config, source)  # type: ignore[arg-type]
    obs, info = env.reset(seed=config.seed)
    learned = config.policy == "ppo"
    policy: Any = (
        DeltaSellerPolicy(roll_dte=config.env.roll_dte)
        if learned
        else make_policy(config.policy, roll_dte=config.env.roll_dte)
    )
    policy.reset(config.seed)
    action = int(policy.act(obs, info))
    chain = info["chain"]
    pos = build_position(action, chain, config.env)
    who = (
        "The fixed-delta seller"
        if learned
        else {
            "flat": "Flat",
            "always_on": "The always-on condor",
            "random": "The random policy",
            "delta_seller": "The fixed-delta seller",
        }.get(config.policy, config.policy)
    )
    if pos is None:
        return DayZeroOut(
            policy=config.policy,
            reference=learned,
            structure=ACTION_NAMES[action],
            plain=f"{who} stays flat on day 0: it opens nothing.",
            spot=float(chain.spot),
            tenor_days=None,
            legs=[],
            credit=None,
            margin=None,
            max_loss=None,
            breakevens=[],
        )
    legs = pos.legs
    exp = legs[0].expiry_day
    credit = -sum(leg.qty * leg.entry_price for leg in legs) * MULTIPLIER
    shorts = {leg.right: leg.strike for leg in legs if leg.qty < 0}
    per_share = credit / MULTIPLIER
    breakevens = [shorts["P"] - per_share] if "P" in shorts else []
    breakevens += [shorts["C"] + per_share] if "C" in shorts else []
    name = ACTION_NAMES[action]
    tenor = int(exp - chain.t)
    plain = (
        f"{who} opens {STRUCTURE_PLAIN.get(name, name)} {tenor} trading days to expiry, "
        f"short strikes at {config.env.short_delta:g} delta, taking in ${credit:,.0f}."
    )
    return DayZeroOut(
        policy=config.policy,
        reference=learned,
        structure=name,
        plain=plain,
        spot=float(chain.spot),
        tenor_days=tenor,
        legs=[
            PreviewLegOut(
                right=leg.right,
                strike=leg.strike,
                qty=leg.qty,
                price=leg.entry_price,
                iv=chain.iv_at(exp, leg.strike),
                delta=chain.greeks(leg.right, leg.strike, exp).delta,
                moneyness=leg.strike / float(chain.spot),
            )
            for leg in sorted(legs, key=lambda x: x.strike)
        ],
        credit=float(credit),
        margin=float(margin_requirement(legs, chain)),
        max_loss=float(max_loss(legs)) if is_defined_risk(legs) else None,
        breakevens=[float(b) for b in breakevens],
    )


def _peers(spec: JobSpec, config: Any, runs_root: Path, rows: Sequence[RunRow]) -> list[PeerOut]:
    from optspread.webapi import _world, row_out, run_source

    world, _ = _world(config.model_dump(mode="json"), spec.source)
    run_id = f"{config_hash_of(config)}-s{config.seed}"
    found: list[tuple[int, PeerOut]] = []
    for row in rows:
        if row.status != "OK":
            continue
        try:
            other = RunManifest.read(row.path).config
        except (OSError, ValueError):
            continue
        if _world(other.model_dump(mode="json"), run_source(row))[0] != world:
            continue
        paired = other.seed == config.seed
        same_arm = other.policy == config.policy
        premium = float(other.market.iv_premium_vol_pts)
        mine = float(config.market.iv_premium_vol_pts)
        if row.run_id == run_id and same_arm:
            rank, relation, note = 0, "same run", "this exact run is on disk: launching reuses it"
        elif same_arm and premium == 0.0 and mine != 0.0:
            rank, relation = 1, "control"
            note = "the same arm at zero premium: the paired control" + (
                "" if paired else " (other seeds: not paired)"
            )
        elif same_arm:
            rank, relation, note = 2, "same arm", f"the same arm at {premium:g} vol pts"
        elif premium == mine:
            rank, relation, note = 3, "other arm", "another arm in the same world"
        else:
            continue
        out = row_out(row, runs_root)
        found.append(
            (
                rank * 2 + (0 if paired else 1),
                PeerOut(
                    key=out.key,
                    policy=row.policy,
                    premium=row.premium,
                    source=run_source(row),
                    n_episodes=row.n_episodes,
                    mean_pnl=_f(row.mean_pnl),
                    relation=relation,
                    paired=paired,
                    note=note,
                ),
            )
        )
    found.sort(key=lambda x: (x[0], x[1].premium or 0.0, x[1].key))
    # a relation met on the same seeds needs no unpaired runs beside it
    paired = {p.relation for _, p in found if p.paired}
    kept = [p for _, p in found if p.paired or p.relation not in paired]
    return kept[:MAX_PEERS]


MAX_PEERS = 12


def config_hash_of(config: Any) -> str:
    from optspread.config import config_hash

    return config_hash(config)


def launch_preview(raw: dict[str, Any], runs_root: Path, rows: Sequence[RunRow]) -> PreviewOut:
    """What the launch form's spec would run: refused with every reason, or its run id, the
    world in a sentence, what it opens on day 0, its wall time and the runs it meets."""
    from pydantic import ValidationError

    from optspread.jobs import LAUNCH_DIR, to_run_config
    from optspread.orchestrate import source_root

    try:
        spec = JobSpec.model_validate(raw)
    except ValidationError as exc:
        return PreviewOut(ok=False, errors=[_clean(str(e["msg"])) for e in exc.errors()])
    config = to_run_config(spec)
    h = config_hash_of(config)
    run_id = f"{h}-s{config.seed}"
    run_dir = source_root(runs_root, spec.source) / LAUNCH_DIR / run_id
    real = spec.source != "synthetic"
    if real:
        world = (
            "Real SPX closes on A8's evaluation dates: each episode starts on a date drawn "
            "from the seed and trades the day's real surface; no premium is planted."
        )
        iv = None
    else:
        iv = spec.sigma + spec.iv_premium_vol_pts / 100.0
        world = (
            f"A simulated index whose path moves at {spec.sigma:.0%} vol while options are "
            f"priced at {iv:.0%}: a planted premium of {spec.iv_premium_vol_pts:g} vol "
            f"point{'' if spec.iv_premium_vol_pts == 1 else 's'} for a seller to earn."
            if spec.iv_premium_vol_pts
            else f"A simulated index at {spec.sigma:.0%} vol with options priced at the same "
            "vol: the null world, where selling earns nothing before costs."
        )
    day0, note = None, ""
    try:
        day0 = _day_zero(config, spec.source)
    except Exception as exc:  # a missing panel, a chain the deltas cannot reach
        note = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    steps = config.train.total_timesteps if config.train is not None else None
    return PreviewOut(
        ok=True,
        errors=[],
        run_id=run_id,
        config_hash=h,
        exists=(run_dir / MANIFEST_FILENAME).is_file(),
        run_key=run_dir.relative_to(runs_root).as_posix(),
        world=world,
        implied_vol=iv,
        day0=day0,
        day0_note=note,
        estimate=launch_estimate(spec.n_episodes, spec.n_days, steps),
        peers=_peers(spec, config, runs_root, rows),
    )


# -- the edge: which components and structures carry it (optspread.edge) ------------ #

EDGE_DIR = "edge"  # under the attribution root, as scripts/edge_attribution.py writes it
_EDGE_FILE = re.compile(r"edge-([0-9a-f]+)(?:-first(\d+))?\.json")
_HASH = re.compile(r"[0-9a-f]+")
EDGE_COMMAND = 'make edge ARGS="--grid <runs root>/{grid} --workers 2"'
ACTIONMIX_COMMAND = 'make actionmix ARGS="--grid <runs root>/{grid} --workers 2"'


class EdgeFileOut(Out):
    """One ``edge-<grid hash>[-first<N>].json``, from its header; ``error`` when unreadable."""

    name: str
    grid_hash: str
    first: int | None
    source: str = ""
    arms: list[str] = Field(default_factory=list)
    premiums: list[float] = Field(default_factory=list)
    n_seeds: int = 0
    n_episodes: int = 0
    reproduction: str = ""
    error: str = ""


class EdgeGridOut(Out):
    """A grid under the runs root, with the commands that write its edge and action mix."""

    grid_hash: str
    key: str  # the grid file relative to the runs root, as the Sweep page keys it
    has_edge: bool
    edge_command: str
    actionmix_command: str


class EdgeListOut(Out):
    files: list[EdgeFileOut]
    grids: list[EdgeGridOut]
    folder: str  # where the files are read, relative to the attribution root
    command: str  # the command with a placeholder grid, for a root that has none


class EdgeOut(Out):
    """One grid's edge as ``optspread.edge`` wrote it, with each cell's run under the root."""

    name: str
    folder: str
    edge: dict[str, Any]  # GridEdge, field for field (the schema of optspread.edge)
    run_keys: dict[str, str | None]  # run id -> run key under the runs root, None if absent
    grid_key: str | None
    actionmix_command: str
    # run id -> {"n_episodes", "n_differ"} for a cell that differs, read from the run's own
    # attribution when one is cached: how many of its episodes are beyond PNL_RTOL
    episode_counts: dict[str, dict[str, int]] = Field(default_factory=dict)


def _episode_counts(attribution_root: Path, source: str, run_id: str) -> dict[str, int] | None:
    """Episodes and those beyond tolerance in a run's cached attribution, if it is there."""
    from optspread.attribution import read_attribution

    suffix = "" if source in ("", "synthetic") else f".{source}"
    path = attribution_root / f"{run_id}{suffix}.json"
    if not path.is_file():
        return None
    try:
        att = read_attribution(path)
    except (OSError, ValueError):
        return None
    return {
        "n_episodes": len(att.episodes),
        "n_differ": sum(1 for e in att.episodes if not e.agrees),
    }


def _edge_folder(attribution_root: Path) -> Path:
    return attribution_root / EDGE_DIR


def _edge_file_name(grid_hash: str, first: int | None) -> str:
    return f"edge-{grid_hash}{'' if first is None else f'-first{first}'}.json"


def _edge_header(path: Path) -> EdgeFileOut:
    m = _EDGE_FILE.fullmatch(path.name)
    assert m is not None
    base = {"name": path.name, "grid_hash": m[1], "first": int(m[2]) if m[2] else None}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EdgeFileOut(
            **base,
            source=str(raw["source"]),
            arms=[str(a) for a in raw["arms"]],
            premiums=[float(p) for p in raw["premiums"]],
            n_seeds=int(raw["n_seeds"]),
            n_episodes=int(raw["n_episodes"]),
            reproduction=str(raw["reproduction"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return EdgeFileOut(**base, error=f"unreadable: {type(exc).__name__}")


def _grid_files(runs_root: Path) -> dict[str, list[Path]]:
    """Every ``grid-<hash>.json`` under the runs root by hash, the job queue's aside."""
    out: dict[str, list[Path]] = {}
    if not runs_root.is_dir():
        return out
    for p in sorted(runs_root.rglob("grid-*.json")):
        gh = p.stem.removeprefix("grid-")
        if ".jobs" not in p.parts and p.is_file() and _HASH.fullmatch(gh):
            out.setdefault(gh, []).append(p)
    return out


def _grid_path(runs_root: Path, grid_hash: str, source: str) -> Path | None:
    """The grid's record: one off the synthetic world sits under its source's folder."""
    found = _grid_files(runs_root).get(grid_hash, [])
    real = [p for p in found if source in p.relative_to(runs_root).parts]
    pool = real if source not in ("", "synthetic") and real else found
    return pool[0] if pool else None


def edge_list(attribution_root: Path, runs_root: Path) -> EdgeListOut:
    """The edge files under ``<attribution root>/edge``, newest grid first, and the grids
    under the runs root with the commands that write theirs."""
    folder = _edge_folder(attribution_root)
    paths = (
        [p for p in folder.glob("edge-*.json") if _EDGE_FILE.fullmatch(p.name)]
        if folder.is_dir()
        else []
    )
    paths.sort(key=lambda p: (-p.stat().st_mtime, p.name))
    files = [_edge_header(p) for p in paths]
    have = {f.grid_hash for f in files}
    grids = [
        EdgeGridOut(
            grid_hash=gh,
            key=(key := ps[0].relative_to(runs_root).as_posix()),
            has_edge=gh in have,
            edge_command=EDGE_COMMAND.format(grid=key),
            actionmix_command=ACTIONMIX_COMMAND.format(grid=key),
        )
        for gh, ps in _grid_files(runs_root).items()
    ]
    return EdgeListOut(
        files=files,
        grids=grids,
        folder=f"{EDGE_DIR}/",
        command=EDGE_COMMAND.format(grid="grid-<hash>.json"),
    )


def edge_detail(
    attribution_root: Path, runs_root: Path, grid: str, first: int | None = None
) -> EdgeOut:
    """``edge-<grid>.json`` (or ``-first<N>``) as written. With no ``first``, the full
    read when there is one, else the partial read of the most episodes. ``ValueError``
    for a name that is not a grid hash; ``KeyError`` when no such file is written."""
    from optspread.edge import read_edge

    gh = grid.removeprefix("grid-").removeprefix("edge-").removesuffix(".json")
    if not _HASH.fullmatch(gh):
        raise ValueError(f"{grid!r} is not a grid hash")
    if first is not None and first < 2:
        raise ValueError(f"first must be at least 2, got {first}")
    folder = _edge_folder(attribution_root)
    if first is None and not (folder / _edge_file_name(gh, None)).is_file():
        partial = [
            int(m[2])
            for p in (folder.glob(f"edge-{gh}-first*.json") if folder.is_dir() else [])
            if (m := _EDGE_FILE.fullmatch(p.name)) and m[1] == gh and m[2]
        ]
        first = max(partial) if partial else None
    path = folder / _edge_file_name(gh, first)
    if not path.is_file():
        raise KeyError(gh)
    result = read_edge(path)
    grid_path = _grid_path(runs_root, gh, result.source)
    keys: dict[str, str | None] = {}
    for cell in result.cells:
        run_dir = grid_path.parent / cell.run_id if grid_path is not None else None
        ok = run_dir is not None and (run_dir / "manifest.json").is_file()
        keys[cell.run_id] = run_dir.relative_to(runs_root).as_posix() if ok and run_dir else None
    grid_key = grid_path.relative_to(runs_root).as_posix() if grid_path is not None else None
    counts = {
        c.run_id: n
        for c in result.cells
        if c.reproduction == "differs"
        and (n := _episode_counts(attribution_root, result.source, c.run_id)) is not None
    }
    return EdgeOut(
        name=path.name,
        folder=f"{EDGE_DIR}/",
        edge=_finite_json(result.model_dump(mode="json")),
        run_keys=keys,
        grid_key=grid_key,
        actionmix_command=ACTIONMIX_COMMAND.format(grid=grid_key or f"grid-{gh}.json"),
        episode_counts=counts,
    )
