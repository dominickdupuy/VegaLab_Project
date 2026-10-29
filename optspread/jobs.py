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

The pool outlives a dead worker (#30). A worker killed outright (out of memory, say)
breaks a ``ProcessPoolExecutor`` for good: every job on it fails with
``BrokenProcessPool``, and so would every job submitted after. The manager notices
on its next call, builds a fresh pool, fails the jobs that had started (each says
its worker died) and puts the ones that had not back on the new pool. The queue is
capped (``max_pending`` jobs queued or running; ``QueueFullError`` past it, which
the server turns into a 429), and a job can be cancelled: a queued one never runs,
and a running one is asked to stop through a flag file, read between episodes and,
while a PPO job trains, every few hundred training steps (``train.STOP_CHECK_STEPS``), so
a cancelled training run gives its worker back at once. One that finishes anyway has its
result discarded. A run directory it had already written stays: it is an ordinary run,
possibly another job's too.

The pool's workers are spawned, as the orchestrator's are: the server is a multithreaded
process, and forking one is how a child inherits a lock some other thread held.

A PPO job saves the policy it trained beside ``run.json`` and hashes it into the
manifest, exactly as ``orchestrate.execute_run`` does, so the capital, attribution and
action-mix views re-simulate a launched PPO run with the policy it was scored with.

A run directory is reported as already on disk only when ``sweep.valid_run`` accepts it
(the real learner for a PPO run, the default components, the panel the run read) and its
manifest hashes its equity paths too, and for a PPO run its ``policy.zip``. Both are
written before ``run.json`` and the manifest, so a job killed part way never leaves a
valid manifest without them.

PPO is offered only when ``optspread.train`` (W6-5) is importable with its
learner installed. It is detected, not assumed, so the dashboard never offers an
arm this checkout cannot run.

A job runs on the generator unless its spec names the real panel (``source``). A
real-data job evaluates on the ``eval`` split and, for PPO, trains on ``train``
(``optspread.train``; the dates of pre-registration A8), and lands under
``optionmetrics/launched`` of the runs root rather than ``launched``: the source is not
in the run id, so the two would otherwise share a directory. A source whose data is
absent is refused at submission, as an arm that is not installed is.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import multiprocessing as mp
import secrets
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optspread.artifacts import write_run
from optspread.config import (
    CostConfig,
    EnvConfig,
    MarketConfig,
    PolicyName,
    RunConfig,
    config_hash,
)
from optspread.data import SourceName, make_source
from optspread.harness import EpisodeResult, build_env, summarize
from optspread.manifest import RunManifest, callable_name
from optspread.orchestrate import source_root
from optspread.policies import Policy, make_policy
from optspread.sweep import valid_run
from optspread.train import POLICY_FILENAME, TrainedPolicy, TrainingStoppedError, check_disjoint

EQUITY_FILENAME = "equity.json"
LAUNCH_DIR = "launched"
JOBS_DIR = ".jobs"
MAX_EQUITY_PATHS = 200  # episodes whose equity path is kept; the rest are summarised only
MAX_PENDING = 8  # jobs queued or running at once; a launch past it is refused, not queued
MAX_TRAIN_SEED = 2**15  # TrainConfig.seed's bound: a PPO job trains on its first seed

Arm = Literal["flat", "always_on", "random", "delta_seller", "ppo"]
SCRIPTED_ARMS: tuple[Arm, ...] = ("flat", "always_on", "random", "delta_seller")
# The data a job can run on: the generator, or the real SPX panel (optspread.data).
JobSource = Literal["synthetic", "optionmetrics"]
SOURCES: tuple[JobSource, ...] = get_args(JobSource)

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


def available_sources() -> tuple[JobSource, ...]:
    """The sources a job can run on here: the generator always, the panel when present."""
    return tuple(s for s in SOURCES if make_source(s).available)


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
    # Not part of the config, so not of the run id; the manifest records it.
    source: JobSource = "synthetic"

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
        if self.policy == "ppo" and self.seed >= MAX_TRAIN_SEED:
            # to_run_config trains on the first seed, and TrainConfig.seed is below 2**15
            raise ValueError(
                f"a PPO run trains on its first seed, which must be below {MAX_TRAIN_SEED:,} "
                f"(the training seed's range); got {self.seed:,}"
            )
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
        "The fixed-delta seller, nothing planted: it loses its costs. The control.",
        JobSpec(),
    ),
    Preset(
        "planted_1",
        "Planted 1 vol pt",
        "The seller with 1 vol pt planted: the smallest step on the sweep's axis.",
        JobSpec(iv_premium_vol_pts=1.0),
    ),
    Preset(
        "planted_4",
        "Planted 4 vol pts",
        "A premium large enough to clear the noise in 100 episodes: a detected effect.",
        JobSpec(iv_premium_vol_pts=4.0),
    ),
    Preset(
        "costs_off",
        "Costs off",
        "The always-on condor without costs: the fair world breaks even before costs.",
        JobSpec(policy="always_on", costs_enabled=False),
    ),
    Preset(
        "churn",
        "Random churn",
        "The random policy with costs: it trades most and loses fastest.",
        JobSpec(policy="random"),
    ),
)


# -- the worker ----------------------------------------------------------------- #


def run_episode_traced(
    config: RunConfig, policy: Policy, seed: int, source: SourceName = "synthetic"
) -> tuple[EpisodeResult, list[float]]:
    """``harness.run_episode`` plus the equity path. The same loop, so the same numbers.
    On the real source the episode is drawn from the ``eval`` split."""
    env = build_env(config, source)
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


def _make_policy(
    config: RunConfig,
    log: Callable[[str], None],
    source: SourceName = "synthetic",
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Policy, TrainedPolicy | None, str | None]:
    """The policy, the same policy again when it is learned (to be saved), and the
    trainer that produced it. ``should_stop`` is asked while it trains
    (``train.train_ppo``), the cancel flag of a job."""
    name = str(config.policy)
    if name in _SCRIPTED:
        return make_policy(_SCRIPTED[name], roll_dte=config.env.roll_dte), None, None
    train = importlib.import_module("optspread.train")
    check_disjoint(config)  # as orchestrate.execute_run: never scored on a path it trained on
    log("training PPO at the frozen D3 hyperparameters")
    trainer = (
        train.train_ppo
        if should_stop is None
        else partial(train.train_ppo, should_stop=should_stop)
    )
    policy: TrainedPolicy = train.train_on(trainer, config, source)
    log("training finished")
    return policy, policy, callable_name(train.train_ppo)


def _existing(run_dir: Path, config: RunConfig, source: SourceName = "synthetic") -> bool:
    """A job's run already on disk: valid by ``sweep.valid_run`` with its equity paths."""
    if valid_run(run_dir, config, source) is None:
        return False
    try:
        artifacts = RunManifest.read(run_dir).artifacts
    except (OSError, ValueError):
        return False
    # a learned run without its policy cannot be re-simulated: trained again, not reused
    needed = {EQUITY_FILENAME} | ({POLICY_FILENAME} if config.train is not None else set())
    return needed <= set(artifacts)


class JobCancelledError(RuntimeError):
    """Raised in a worker that finds its job's cancel flag; nothing is written after it."""


class _Progress:
    """The files a worker writes and the server reads, and the one the server writes."""

    def __init__(self, jobs_dir: Path, job_id: str) -> None:
        self.state = jobs_dir / f"{job_id}.json"
        self.log_path = jobs_dir / f"{job_id}.log"
        self.cancel_flag = jobs_dir / f"{job_id}.cancel"

    def check_cancelled(self) -> None:
        if self.cancel_flag.exists():
            self.log("cancelled; stopping before anything else is written")
            raise JobCancelledError("cancelled from the dashboard")

    def write(self, **fields: Any) -> None:
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(fields), encoding="utf-8")
        tmp.replace(self.state)  # atomic, so a reader never sees half a file

    def log(self, line: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as fh:
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
    run_dir = source_root(root, spec.source) / LAUNCH_DIR / run_id
    rel = str(run_dir.relative_to(root).as_posix())
    total = config.n_episodes
    progress.write(done=0, total=total, phase="starting", run_id=run_id, key=rel)
    progress.log(
        f"config {config_hash(config)} seed {config.seed}: {total} episodes on {spec.source}"
    )
    if _existing(run_dir, config, spec.source):
        progress.log("this run is already on disk and matches its manifest; not recomputed")
        progress.write(done=total, total=total, phase="reused", run_id=run_id, key=rel)
        return rel
    progress.check_cancelled()
    try:
        policy, trained, trainer = _make_policy(
            config, progress.log, spec.source, progress.cancel_flag.exists
        )
    except TrainingStoppedError:
        progress.check_cancelled()  # raises: the cancel flag is what stopped it
        raise
    results: list[EpisodeResult] = []
    paths: list[list[float]] = []
    start, last = time.perf_counter(), 0.0
    for i, seed in enumerate(range(config.seed, config.seed + total)):
        progress.check_cancelled()
        result, path = run_episode_traced(config, policy, seed, spec.source)
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
    # The policy and the equity paths first: the manifest is written once, last, with
    # them hashed in. The policy is saved as orchestrate.execute_run saves it.
    run_dir.mkdir(parents=True, exist_ok=True)
    extra: list[Path] = []
    if trained is not None:
        trained.save(run_dir / POLICY_FILENAME)
        extra.append(run_dir / POLICY_FILENAME)
    equity = run_dir / EQUITY_FILENAME
    equity.write_text(
        json.dumps({"seeds": [r.seed for r in results[: len(paths)]], "paths": paths}),
        encoding="utf-8",
        newline="\n",
    )
    extra.append(equity)
    write_run(run_dir, config, summary, episodes, extra, source=spec.source, trainer=trainer)
    steps = sum(r.steps for r in results)
    progress.log(
        f"mean P&L {summary.mean_pnl:,.0f} (se {summary.se_pnl:,.0f}); "
        f"{steps:,} steps in {elapsed:.1f}s"
    )
    progress.write(done=total, total=total, phase="written", run_id=run_id, key=rel)
    return rel


# -- the queue -------------------------------------------------------------------- #

JobState = Literal["queued", "running", "done", "failed", "cancelled"]

# What a job that was on the pool when a worker died says, unless it never started.
WORKER_DIED = (
    "BrokenProcessPool: a worker process died while this job was on it (killed, for "
    "example out of memory); the pool was rebuilt and later jobs run on the new one"
)


class QueueFullError(RuntimeError):
    """``max_pending`` jobs are already queued or running. The server answers 429."""


class JobNotCancellableError(RuntimeError):
    """The job has already finished, failed or been cancelled."""


@dataclass(slots=True)
class Job:
    job_id: str
    spec: JobSpec
    created_at: str
    future: Future[str] = field(repr=False)
    error: str = ""
    cancelled: bool = False  # asked to stop; whatever it returns is discarded
    pool: int = 0  # which pool it was handed to: JobManager.rebuilds at the time
    relaunched: bool = False


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
    source: str = "synthetic"

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


JobFunction = Callable[[str, str, str], str]


class JobManager:
    """A bounded pool of workers and the jobs it has been given, newest first.

    ``executor`` is the pool to start on; a broken one is replaced by ``factory()``,
    by default a process pool of ``workers``. ``target`` is the function a job runs,
    ``execute_job``; a test swaps in one whose worker dies.
    """

    def __init__(
        self,
        runs_root: Path,
        workers: int = 2,
        executor: Executor | None = None,
        *,
        max_pending: int = MAX_PENDING,
        factory: Callable[[], Executor] | None = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError(f"max_pending must be at least 1, got {max_pending}")
        self.runs_root = runs_root
        self.jobs_dir = runs_root / JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.factory: Callable[[], Executor] = (
            factory
            if factory is not None
            else lambda: ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        )
        self.executor = executor if executor is not None else self.factory()
        self.max_pending = max_pending
        self.target: JobFunction = execute_job
        self.jobs: dict[str, Job] = {}
        self.rebuilds = 0  # pools replaced after a dead worker
        self._lock = threading.RLock()  # the server calls in from a thread pool

    def _launch(self, job_id: str, spec: JobSpec) -> tuple[Future[str], int]:
        """Hand the job to the pool, rebuilding it first if it is already broken."""
        args = (job_id, spec.model_dump_json(), str(self.runs_root))
        try:
            return self.executor.submit(self.target, *args), self.rebuilds
        except BrokenProcessPool:
            self._rebuild()
            return self.executor.submit(self.target, *args), self.rebuilds

    def _rebuild(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self.factory()
        self.rebuilds += 1

    def _recover(self) -> None:
        """After a dead worker: a new pool, the started jobs failed, the rest relaunched.

        A job counts as started once its worker has written its progress file. One
        that had not is relaunched once; if its pool dies again it is failed too, so
        a job that kills its worker on start cannot take pool after pool with it.
        """
        hit = [
            j
            for j in self.jobs.values()
            if not j.error
            and not j.cancelled
            and j.future.done()
            and not j.future.cancelled()
            and isinstance(j.future.exception(), BrokenProcessPool)
        ]
        if any(j.pool == self.rebuilds for j in hit):  # not a pool already replaced
            self._rebuild()
        for job in hit:
            started = _Progress(self.jobs_dir, job.job_id).state.exists()
            if started or job.relaunched:
                job.error = WORKER_DIED
                continue
            job.future, job.pool = self._launch(job.job_id, job.spec)
            job.relaunched = True

    def pending(self) -> int:
        """Jobs queued or running: what the cap counts."""
        return sum(not j.future.done() for j in self.jobs.values())

    def submit(self, spec: JobSpec) -> str:
        if spec.policy == "ppo" and not ppo_available():
            raise ValueError("the PPO arm is not available in this checkout")
        if spec.source not in available_sources():
            raise ValueError(f"the {spec.source} data is not available here (set VEGALAB_DATA)")
        to_run_config(spec)  # refuse here, synchronously, anything the worker would refuse
        with self._lock:
            self._recover()
            if self.pending() >= self.max_pending:
                raise QueueFullError(
                    f"the job queue is full ({self.max_pending} jobs queued or running); "
                    "launch again when one finishes, or cancel one"
                )
            job_id = secrets.token_hex(6)
            future, pool = self._launch(job_id, spec)
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            self.jobs[job_id] = Job(job_id, spec, stamp, future, pool=pool)
        return job_id

    def cancel(self, job_id: str) -> JobStatus:
        """Cancel a queued job, or ask a running one to stop and discard what it returns."""
        with self._lock:
            self._recover()
            job = self.jobs[job_id]
            if job.cancelled or job.future.done():
                raise JobNotCancellableError(f"job {job_id} is already {self.status(job_id).state}")
            job.cancelled = True
            if not job.future.cancel():
                _Progress(self.jobs_dir, job_id).cancel_flag.touch()
            return self.status(job_id)

    def status(self, job_id: str, tail: int = 20) -> JobStatus:
        with self._lock:
            self._recover()
            job = self.jobs[job_id]
            future = job.future
        progress = _Progress(self.jobs_dir, job_id)
        try:
            state: dict[str, Any] = json.loads(progress.state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        try:
            lines = progress.log_path.read_text(encoding="utf-8").splitlines()[-tail:]
        except OSError:
            lines = []
        run_key = ""
        if job.cancelled:
            status: JobState = "cancelled"
        elif job.error:
            status = "failed"
        elif future.done():
            exc = future.exception()
            if exc is not None:
                job.error = "".join(traceback.format_exception_only(exc)).strip()
                status = "failed"
            else:
                status, run_key = "done", future.result()
        else:
            status = "running" if state else "queued"
        phase = str(state.get("phase", "queued"))
        if status in ("failed", "cancelled"):
            phase = status
        return JobStatus(
            job_id=job_id,
            state=status,
            phase=phase,
            done=int(state.get("done", 0)),
            total=int(state.get("total", job.spec.n_episodes)),
            run_id=str(state.get("run_id", "")),
            run_key=run_key,
            created_at=job.created_at,
            policy=job.spec.policy,
            error=job.error,
            log_tail=tuple(lines),
            source=job.spec.source,
        )

    def all(self) -> tuple[JobStatus, ...]:
        with self._lock:
            self._recover()
            ids = list(reversed(list(self.jobs)))
        return tuple(self.status(j) for j in ids)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
