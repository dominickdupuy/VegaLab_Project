"""The run page's re-simulations, which it starts and polls: who may run them, attribution,
drawdown and return on posted capital, and replay.

Lane: Ilan Danial (machinery).

Attribution (``optspread.attribution``) and capital are cached under the attribution root,
never in a run directory; replay is ``optspread.replay``. The server runs the work on one
thread with ``optspread.dashboard.tasks.Tasks``. Moved from ``optspread.webviews``, which
re-exports it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from optspread.api._base import Out, _from_attrs, _source_suffix
from optspread.api.launch import launch_estimate
from optspread.dashboard.jobs.spec import available_arms
from optspread.dashboard.runindex import RunRow
from optspread.dashboard.tasks import TaskState
from optspread.manifest import MANIFEST_FILENAME, RunManifest
from optspread.train import POLICY_FILENAME

# -- background work: attribution, capital and replay ------------------------------- #


def _scripted(policy: str) -> bool:
    return policy in available_arms() and policy != "ppo"


# attribution


class AttributionOut(Out):
    state: Literal["cached", "computing", "absent", "failed"]  # attribution_view's branches
    can_compute: bool
    reason: str
    file: str
    result: dict[str, Any] | None  # the cached RunAttribution, every episode, no steps
    seed: int | None  # the episode whose steps are sent
    steps: list[dict[str, Any]]
    seconds: float | None = None


def attribution_file(root: Path, row: RunRow) -> Path:
    """Where ``scripts/attribution.py`` writes this run: ``<run_id>[.<source>].json``."""
    return root / f"{row.run_id}{_source_suffix(row.source)}.json"


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
    if not (row.path / POLICY_FILENAME).is_file():
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
            result=result.without_steps().model_dump(mode="json"),
            seed=pick,
            steps=steps,
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
    return BandOut(mean=mean.tolist(), p10=q[0].tolist(), p50=q[1].tolist(), p90=q[2].tolist())


class CapitalEpisodeOut(Out):
    seed: int
    pnl: float
    max_drawdown: float  # $, peak to trough of the equity path within the episode
    peak_margin: float  # $, the most margin the book held at any step
    return_on_cash: float  # pnl / initial cash
    return_on_margin: float | None  # pnl / peak margin; None when nothing was held


class CapitalOut(Out):
    # The server writes "done" (compute_capital) or "computing", "absent", "failed"
    # (capital_view), but a cached file is served as it reads (read_capital), whatever
    # state it holds: a hand-edited or foreign file's state reaches the page as written,
    # so the schema says only str.
    state: str
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
    return root / CAPITAL_DIR / f"{row.run_id}{_source_suffix(row.source)}.json"


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
        from optspread.runs.rundir import changed_artifacts

        changed = changed_artifacts(run_dir, manifest)
        if changed:
            raise ValueError(f"{', '.join(changed)} no longer match the manifest in {run_dir}")
        source, provider = recorded_provider(manifest)
        env = build_env(config, source, provider=provider)
    policy: Any
    if config.policy == "ppo":
        from optspread.attribution import load_sb3_policy

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
    top = max(diffs) if diffs else math.inf
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
        # a seed missing from run.json (an infinite difference) leaves no largest one: None
        max_abs_diff=top if math.isfinite(top) else None,
        mean_return_on_cash=float(np.mean([r.return_on_cash for r in rows])) if rows else None,
        mean_return_on_margin=sum(r.pnl for r in rows) / total_peak if total_peak else None,
        mean_peak_margin=float(peaks.mean()) if rows else None,
        max_peak_margin=float(peaks.max()) if rows else None,
        median_drawdown=float(np.median(dd)) if rows else None,
        worst_drawdown=float(dd.max()) if rows else None,
        worst_drawdown_on_cash=float(dd.max()) / cash if rows else None,
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


# replay

# The seller's tolerance for a scripted run replayed elsewhere: every number of run.json
# within a billionth, relative (``optspread.attribution.PNL_RTOL``, the M4 gate's).
SCRIPTED_RTOL = 1e-9


class ReplayOut(Out):
    state: Literal["idle", "running", "done", "failed"]  # replay_view's branches
    can_replay: bool
    reason: str  # why it cannot run here; the failure; or, when done, what replay said
    # "" until done, then "IDENTICAL" | "AGREES" (within ``rtol``) | "DIFFERS", as
    # replay_view maps replay's own verdict
    verdict: Literal["", "IDENTICAL", "AGREES", "DIFFERS"] = ""
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
    return _from_attrs(ReplayDone, out, comparable=not (learned and platform_differs(run_dir)))


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
