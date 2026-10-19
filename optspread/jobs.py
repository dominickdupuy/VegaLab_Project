"""Runs launched from the dashboard: the knobs, the job queue, and the worker.

Lane: Ilan Danial (machinery).

A run started from a browser is an ordinary run. The knobs map onto a
``RunConfig`` field for field, the worker drives the same environment and policy
the harness does, and the result is written through ``write_run``, so it lands as a
hashed run directory keyed by its config hash and seed and is listed by the run
browser like any other. Nothing about a launched run is special except that it
also records each episode's equity path (``equity.json``), hashed into the same
manifest, which a command-line run does not need.

Runs are deterministic in their config and seed, so a launch whose run directory
already exists and still matches its manifest is not recomputed: the job reports
the run that is already on disk.

Jobs run in a bounded process pool so a long run never blocks the server. Each
worker writes its progress and log to two small files under the jobs directory,
which is how the server reads progress across the process boundary without a
shared queue.

PPO is offered only when ``optspread.train`` (W6-5) is importable with its
learner installed. It is detected, not assumed, so the dashboard never offers an
arm this checkout cannot run.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import secrets
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optspread.artifacts import RUN_FILENAME, read_run, write_run
from optspread.config import (
    CostConfig,
    EnvConfig,
    MarketConfig,
    PolicyName,
    RunConfig,
    config_hash,
)
from optspread.harness import EpisodeResult, build_env, summarize
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256
from optspread.policies import Policy, make_policy

EQUITY_FILENAME = "equity.json"
LAUNCH_DIR = "launched"
JOBS_DIR = ".jobs"
MAX_EQUITY_PATHS = 200  # episodes whose equity path is kept; the rest are summarised only

Arm = Literal["flat", "always_on", "random", "delta_seller", "ppo"]
SCRIPTED_ARMS: tuple[Arm, ...] = ("flat", "always_on", "random", "delta_seller")

_MARKET, _COST, _ENV, _RUN = MarketConfig(), CostConfig(), EnvConfig(), RunConfig()


def ppo_available() -> bool:
    """``optspread.train`` exposes ``train_ppo``, ``RunConfig`` takes a ``train`` block,
    and the learner is installed. All three, or the arm is hidden."""
    if importlib.util.find_spec("stable_baselines3") is None:
        return False
    if importlib.util.find_spec("optspread.train") is None:
        return False
    if "train" not in RunConfig.model_fields:
        return False
    return callable(getattr(importlib.import_module("optspread.train"), "train_ppo", None))


def available_arms() -> tuple[Arm, ...]:
    return (*SCRIPTED_ARMS, "ppo") if ppo_available() else SCRIPTED_ARMS


class JobSpec(BaseModel):
    """Every knob the launch panel exposes, with the defaults of ``optspread.config``.

    Bounds are there to keep a browser from asking for a run that would take a day
    or cannot be built, not to encode a research choice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: Arm = "delta_seller"
    iv_premium_vol_pts: float = Field(default=_MARKET.iv_premium_vol_pts, ge=-10.0, le=20.0)
    sigma: float = Field(default=_MARKET.sigma, gt=0.0, le=1.5)
    r: float = Field(default=_MARKET.r, ge=-0.05, le=0.25)
    n_days: int = Field(default=_MARKET.n_days, ge=5, le=504)
    tenor_days: tuple[int, ...] = Field(default=_MARKET.tenor_days, min_length=1, max_length=6)
    n_strikes_each_side: int = Field(default=_MARKET.n_strikes_each_side, ge=2, le=64)
    strike_step: float = Field(default=_MARKET.strike_step, gt=0.0, le=500.0)
    costs_enabled: bool = _COST.enabled
    min_half_spread: float = Field(default=_COST.min_half_spread, ge=0.0, le=10.0)
    half_spread_frac: float = Field(default=_COST.half_spread_frac, ge=0.0, le=0.5)
    commission_per_contract: float = Field(default=_COST.commission_per_contract, ge=0.0, le=20.0)
    short_delta: float = Field(default=_ENV.short_delta, gt=0.0, lt=0.5)
    wing_delta: float = Field(default=_ENV.wing_delta, gt=0.0, lt=0.5)
    expiry_idx: int = Field(default=_ENV.expiry_idx, ge=0)
    roll_dte: int = Field(default=_ENV.roll_dte, ge=0, le=60)
    initial_cash: float = Field(default=_ENV.initial_cash, gt=0.0, le=1e9)
    seed: int = Field(default=_RUN.seed, ge=0, lt=2**31)
    n_episodes: int = Field(default=_RUN.n_episodes, ge=1, le=2000)
    total_timesteps: int | None = Field(default=None, gt=0, le=5_000_000)  # PPO only

    @model_validator(mode="after")
    def _consistent(self) -> JobSpec:
        tenors = self.tenor_days
        if any(t < 1 for t in tenors) or list(tenors) != sorted(set(tenors)):
            raise ValueError("tenor_days must be distinct positive days in increasing order")
        if self.expiry_idx >= len(tenors):
            raise ValueError(f"expiry_idx {self.expiry_idx} has no tenor (there are {len(tenors)})")
        if self.wing_delta >= self.short_delta:
            raise ValueError("wing_delta must sit further out than short_delta (be smaller)")
        if self.roll_dte >= tenors[self.expiry_idx]:
            raise ValueError("roll_dte must be shorter than the tenor new structures open at")
        if (self.policy == "ppo") != (self.total_timesteps is not None):
            raise ValueError("total_timesteps is set exactly when the arm is ppo")
        if self.sigma + self.iv_premium_vol_pts / 100.0 <= 0.0:
            raise ValueError(
                f"the implied vol, sigma {self.sigma} plus the premium "
                f"{self.iv_premium_vol_pts} vol points, must be above zero"
            )
        return self

    @property
    def steps(self) -> int:
        """Environment steps the evaluation takes, for progress and for a cost estimate."""
        return self.n_episodes * self.n_days


def to_run_config(spec: JobSpec) -> RunConfig:
    """The ``RunConfig`` a spec stands for. Episodes span the whole path (``episode_days``
    equals ``n_days``), as every gate and the sweep run them."""
    payload: dict[str, Any] = {
        "market": {
            "sigma": spec.sigma,
            "r": spec.r,
            "n_days": spec.n_days,
            "tenor_days": list(spec.tenor_days),
            "n_strikes_each_side": spec.n_strikes_each_side,
            "strike_step": spec.strike_step,
            "iv_premium_vol_pts": spec.iv_premium_vol_pts,
        },
        "cost": {
            "enabled": spec.costs_enabled,
            "min_half_spread": spec.min_half_spread,
            "half_spread_frac": spec.half_spread_frac,
            "commission_per_contract": spec.commission_per_contract,
        },
        "env": {
            "initial_cash": spec.initial_cash,
            "episode_days": spec.n_days,
            "short_delta": spec.short_delta,
            "wing_delta": spec.wing_delta,
            "expiry_idx": spec.expiry_idx,
            "roll_dte": spec.roll_dte,
        },
        "policy": spec.policy,
        "seed": spec.seed,
        "n_episodes": spec.n_episodes,
    }
    if spec.policy == "ppo":
        payload["train"] = {"total_timesteps": spec.total_timesteps, "seed": spec.seed}
    return RunConfig.model_validate(payload)


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    label: str
    description: str
    spec: JobSpec


PRESETS: tuple[Preset, ...] = (
    Preset(
        "null_world",
        "Null world",
        "The fixed-delta seller with nothing planted: zero expectancy before costs, "
        "negative after. The control every other run is read against.",
        JobSpec(),
    ),
    Preset(
        "planted_1",
        "Planted 1 vol pt",
        "The same seller with a 1 vol point premium planted on the surface: the smallest "
        "step on the sweep's effect-size axis (decision D2).",
        JobSpec(iv_premium_vol_pts=1.0),
    ),
    Preset(
        "planted_4",
        "Planted 4 vol pts",
        "A premium large enough that the seller's mean P&L clears its noise on a hundred "
        "episodes: what a detected effect looks like.",
        JobSpec(iv_premium_vol_pts=4.0),
    ),
    Preset(
        "costs_off",
        "Costs off",
        "Always-on condor with spread and commission switched off: the Wave 0 check that "
        "the null world is fair before costs.",
        JobSpec(policy="always_on", costs_enabled=False),
    ),
    Preset(
        "churn",
        "Random churn",
        "The random policy with costs on: it trades most and bleeds fastest, the third "
        "ordering of the Wave 0 gate.",
        JobSpec(policy="random"),
    ),
)


# -- the worker ----------------------------------------------------------------- #


def run_episode_traced(
    config: RunConfig, policy: Policy, seed: int
) -> tuple[EpisodeResult, list[float]]:
    """``harness.run_episode`` plus the equity path. The same loop, so the same numbers."""
    env = build_env(config)
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    equity0, total_cost, steps, done = info["equity"], 0.0, 0, False
    path = [0.0]
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, info))
        total_cost += info["cost"]
        steps += 1
        path.append(float(info["equity"] - equity0))
    return EpisodeResult(seed, info["equity"] - equity0, total_cost, info["n_trades"], steps), path


_SCRIPTED: dict[str, PolicyName] = {name: name for name in get_args(PolicyName)}


def _make_policy(config: RunConfig, log: Callable[[str], None]) -> Policy:
    name = str(config.policy)
    if name in _SCRIPTED:
        return make_policy(_SCRIPTED[name], roll_dte=config.env.roll_dte)
    train = importlib.import_module("optspread.train")
    log("training PPO at the frozen D3 hyperparameters")
    policy: Policy = train.train_ppo(config)
    log("training finished")
    return policy


def _existing(run_dir: Path, config: RunConfig) -> bool:
    if not (run_dir / RUN_FILENAME).is_file() or not (run_dir / MANIFEST_FILENAME).is_file():
        return False
    try:
        manifest = RunManifest.read(run_dir)
        read_run(run_dir)
    except (OSError, ValueError):
        return False
    return manifest.config_hash == config_hash(config) and manifest.artifacts.get(
        RUN_FILENAME
    ) == file_sha256(run_dir / RUN_FILENAME)


class _Progress:
    """The two files a worker writes and the server reads."""

    def __init__(self, jobs_dir: Path, job_id: str) -> None:
        self.state = jobs_dir / f"{job_id}.json"
        self.log_path = jobs_dir / f"{job_id}.log"

    def write(self, **fields: Any) -> None:
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(fields))
        tmp.replace(self.state)  # atomic, so a reader never sees half a file

    def log(self, line: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        with self.log_path.open("a") as fh:
            fh.write(f"{stamp}  {line}\n")


def execute_job(job_id: str, spec_json: str, runs_root: str) -> str:
    """Run one job to a run directory and return its path relative to ``runs_root``.

    Module-level and string-typed so the process pool can pickle the call.
    """
    root = Path(runs_root)
    progress = _Progress(root / JOBS_DIR, job_id)
    spec = JobSpec.model_validate_json(spec_json)
    config = to_run_config(spec)
    run_id = f"{config_hash(config)}-s{config.seed}"
    run_dir = root / LAUNCH_DIR / run_id
    rel = str(run_dir.relative_to(root).as_posix())
    total = config.n_episodes
    progress.write(done=0, total=total, phase="starting", run_id=run_id, key=rel)
    progress.log(f"config {config_hash(config)} seed {config.seed}: {total} episodes")
    if _existing(run_dir, config):
        progress.log("this run is already on disk and matches its manifest; not recomputed")
        progress.write(done=total, total=total, phase="reused", run_id=run_id, key=rel)
        return rel
    policy = _make_policy(config, progress.log)
    results: list[EpisodeResult] = []
    paths: list[list[float]] = []
    start, last = time.perf_counter(), 0.0
    for i, seed in enumerate(range(config.seed, config.seed + total)):
        result, path = run_episode_traced(config, policy, seed)
        results.append(result)
        if i < MAX_EQUITY_PATHS:
            paths.append(path)
        now = time.perf_counter()
        if now - last > 0.25 or i + 1 == total:
            progress.write(done=i + 1, total=total, phase="evaluating", run_id=run_id, key=rel)
            last = now
    elapsed = time.perf_counter() - start
    episodes = tuple(results)
    summary = summarize(episodes)
    manifest = write_run(run_dir, config, summary, episodes)
    equity = run_dir / EQUITY_FILENAME
    equity.write_text(
        json.dumps({"seeds": [r.seed for r in results[: len(paths)]], "paths": paths})
    )
    manifest.add_artifact(run_dir, equity)
    manifest.write(run_dir)
    steps = sum(r.steps for r in results)
    progress.log(
        f"mean P&L {summary.mean_pnl:,.0f} (se {summary.se_pnl:,.0f}); "
        f"{steps:,} steps in {elapsed:.1f}s"
    )
    progress.write(done=total, total=total, phase="written", run_id=run_id, key=rel)
    return rel


# -- the queue -------------------------------------------------------------------- #

JobState = Literal["queued", "running", "done", "failed"]


@dataclass(slots=True)
class Job:
    job_id: str
    spec: JobSpec
    created_at: str
    future: Future[str] = field(repr=False)
    error: str = ""


@dataclass(frozen=True, slots=True)
class JobStatus:
    job_id: str
    state: JobState
    phase: str
    done: int
    total: int
    run_id: str
    run_key: str  # the run directory relative to the runs root, once written
    created_at: str
    policy: str
    error: str
    log_tail: tuple[str, ...]

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


class JobManager:
    """A bounded pool of workers and the jobs it has been given, newest first."""

    def __init__(self, runs_root: Path, workers: int = 2, executor: Executor | None = None) -> None:
        self.runs_root = runs_root
        self.jobs_dir = runs_root / JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor if executor is not None else ProcessPoolExecutor(workers)
        self.jobs: dict[str, Job] = {}

    def submit(self, spec: JobSpec) -> str:
        if spec.policy == "ppo" and not ppo_available():
            raise ValueError("the PPO arm is not available in this checkout")
        to_run_config(spec)  # refuse here, synchronously, anything the worker would refuse
        job_id = secrets.token_hex(6)
        future = self.executor.submit(
            execute_job, job_id, spec.model_dump_json(), str(self.runs_root)
        )
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        self.jobs[job_id] = Job(job_id, spec, stamp, future)
        return job_id

    def status(self, job_id: str, tail: int = 20) -> JobStatus:
        job = self.jobs[job_id]
        progress = _Progress(self.jobs_dir, job_id)
        try:
            state: dict[str, Any] = json.loads(progress.state.read_text())
        except (OSError, ValueError):
            state = {}
        try:
            lines = progress.log_path.read_text().splitlines()[-tail:]
        except OSError:
            lines = []
        future = job.future
        if future.done():
            exc = future.exception()
            if exc is not None:
                job.error = job.error or "".join(traceback.format_exception_only(exc)).strip()
                status: JobState = "failed"
            else:
                status = "done"
        else:
            status = "running" if state else "queued"
        return JobStatus(
            job_id=job_id,
            state=status,
            phase=str(state.get("phase", "queued")) if status != "failed" else "failed",
            done=int(state.get("done", 0)),
            total=int(state.get("total", job.spec.n_episodes)),
            run_id=str(state.get("run_id", "")),
            run_key=future.result() if status == "done" else "",
            created_at=job.created_at,
            policy=job.spec.policy,
            error=job.error,
            log_tail=tuple(lines),
        )

    def all(self) -> tuple[JobStatus, ...]:
        return tuple(self.status(j) for j in reversed(list(self.jobs)))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
