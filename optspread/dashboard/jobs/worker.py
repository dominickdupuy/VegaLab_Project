"""Runs launched from the dashboard, the worker: one job to one run directory.

Lane: Ilan Danial (machinery).

A run started from a browser is an ordinary run. The knobs map onto a
``RunConfig`` field for field, the worker drives the same environment and policy
the harness does, and the result is written through ``write_run``, so it lands as a
hashed run directory keyed by its config hash and seed and is listed by the run
browser like any other. Nothing about a launched run is special except that it
also records each episode's equity path (``equity.json``) and, for every step, the
action chosen and the time through the episode and ATM vol it was chosen on
(``action_log.json``, the run page's policy heatmap), both hashed into the same
manifest, which a command-line run does not need.

Runs are deterministic in their config and seed, so a launch whose run directory
already exists and still matches its manifest is not recomputed: the job reports
the run that is already on disk.

Each worker writes its progress and log to two small files under the jobs directory
(``_Progress``), which is how the server reads progress across the process boundary
without a shared queue. A running job is asked to stop through a flag file, read between
episodes and, while a PPO job trains, every few hundred training steps.

A PPO job saves the policy it trained beside ``run.json`` and hashes it into the
manifest, exactly as ``schedule.execute_run`` does, so the capital, attribution and
action-mix views re-simulate a launched PPO run with the policy it was scored with. An
A2C or DQN job (the research arms) is trained by its own trainer (``train.trainer_for``),
which its manifest names, and is saved and re-simulated the same way.

A run directory is reported as already on disk only when ``rundir.valid_run`` accepts it
(the real learner for a PPO run, the default components, the panel the run read) and its
manifest hashes its equity paths and action log too, and for a learned run its
``policy.zip``. All are written before ``run.json`` and the manifest, so a job killed part
way never leaves a valid manifest without them. A run launched before the action log
existed is computed again once, with the same numbers, and then has one.

A job runs on the generator unless its spec names the real panel (``source``). A
real-data job evaluates on the ``eval`` split and, for PPO, trains on ``train``
(``optspread.train``; the dates of pre-registration A8), and lands under
``optionmetrics/launched`` of the runs root rather than ``launched``: the source is not
in the run id, so the two would otherwise share a directory. A source whose data is
absent is refused at submission, as an arm that is not installed is.

The process pool pickles ``execute_job`` by module path, so it stays module-level here.
Moved from ``optspread.jobs``.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, get_args

from optspread.artifacts import write_run
from optspread.config import PolicyName, RunConfig, config_hash
from optspread.dashboard.jobs.spec import (
    ACTION_LOG_FILENAME,
    EQUITY_FILENAME,
    JOBS_DIR,
    MAX_EQUITY_PATHS,
    JobSpec,
    launched_run_dir,
    to_run_config,
)

# The ``name as name`` imports: what this module used before the launched run's directory
# came from spec.launched_run_dir, still importable from here.
from optspread.dashboard.jobs.spec import LAUNCH_DIR as LAUNCH_DIR
from optspread.data import SourceName
from optspread.env import OBS_NAMES
from optspread.fileio import atomic_write_text
from optspread.harness import EpisodeResult, build_env, summarize
from optspread.manifest import RunManifest, callable_name
from optspread.policies import Observation, Policy, make_policy
from optspread.runs.rundir import run_id, valid_run
from optspread.runs.rundir import source_root as source_root
from optspread.train import POLICY_FILENAME, TrainedPolicy, TrainingStoppedError, check_disjoint

# -- the worker ----------------------------------------------------------------- #

# Where the action log's two observation fields sit, looked up by name (OBS_NAMES is
# append-only, so these never move).
TIME_FRAC = OBS_NAMES.index("time_frac")
ATM_IV = OBS_NAMES.index("atm_iv")
ACTION_LOG_SCHEMA = 1


@dataclass
class ActionLog:
    """Each kept episode's steps: the action chosen, and the observation's time through
    the episode and near-tenor ATM implied vol when it was chosen (``action_log.json``)."""

    seeds: list[int] = field(default_factory=list)
    actions: list[list[int]] = field(default_factory=list)
    time_frac: list[list[float]] = field(default_factory=list)
    atm_iv: list[list[float]] = field(default_factory=list)

    def start(self, seed: int) -> None:
        self.seeds.append(seed)
        self.actions.append([])
        self.time_frac.append([])
        self.atm_iv.append([])

    def record(self, action: int, obs: Observation) -> None:
        self.actions[-1].append(int(action))
        self.time_frac[-1].append(round(float(obs[TIME_FRAC]), 6))
        self.atm_iv[-1].append(round(float(obs[ATM_IV]), 6))

    def dumps(self) -> str:
        return json.dumps(
            {
                "schema_version": ACTION_LOG_SCHEMA,
                "seeds": self.seeds,
                "actions": self.actions,
                "time_frac": self.time_frac,
                "atm_iv": self.atm_iv,
            },
            separators=(",", ":"),
        )


def run_episode_traced(
    config: RunConfig,
    policy: Policy,
    seed: int,
    source: SourceName = "synthetic",
    log: ActionLog | None = None,
) -> tuple[EpisodeResult, list[float]]:
    """``harness.run_episode`` plus the equity path, and each step into ``log`` when one
    is given. The same loop, so the same numbers. On the real source the episode is drawn
    from the ``eval`` split."""
    env = build_env(config, source)
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    equity0, total_cost, steps, done = info["equity"], 0.0, 0, False
    path = [0.0]
    if log is not None:
        log.start(seed)
    while not done:
        action = policy.act(obs, info)
        if log is not None:
            log.record(action, obs)
        obs, _, done, _, info = env.step(action)
        total_cost += info["cost"]
        steps += 1
        path.append(float(info["equity"] - equity0))
    return EpisodeResult(seed, info["equity"] - equity0, total_cost, info["n_trades"], steps), path


_SCRIPTED: dict[str, PolicyName] = {name: name for name in get_args(PolicyName)}


def _trainer_name(config: RunConfig) -> str | None:
    """The trainer a learned job's run is made by, as its manifest names it."""
    if config.train is None:
        return None
    train = importlib.import_module("optspread.train")
    return callable_name(train.trainer_for(config.train.algorithm))


def _make_policy(
    config: RunConfig,
    log: Callable[[str], None],
    source: SourceName = "synthetic",
    should_stop: Callable[[], bool] | None = None,
) -> tuple[Policy, TrainedPolicy | None, str | None]:
    """The policy, the same policy again when it is learned (to be saved), and the
    trainer that produced it (``train.trainer_for`` the arm). ``should_stop`` is asked
    while it trains, the cancel flag of a job."""
    name = str(config.policy)
    if config.train is None:  # a scripted arm (RunConfig: no train exactly then)
        return make_policy(_SCRIPTED[name], roll_dte=config.env.roll_dte), None, None
    train = importlib.import_module("optspread.train")
    check_disjoint(config)  # as schedule.execute_run: never scored on a path it trained on
    if config.train.algorithm == "ppo":
        log("training PPO at the frozen D3 hyperparameters")
    else:
        log(f"training {name.upper()} at the Stable-Baselines3 defaults (a research arm)")
    chosen = train.trainer_for(config.train.algorithm)
    trainer = chosen if should_stop is None else partial(chosen, should_stop=should_stop)
    policy: TrainedPolicy = train.train_on(trainer, config, source)
    log("training finished")
    return policy, policy, callable_name(chosen)


def _existing(run_dir: Path, config: RunConfig, source: SourceName = "synthetic") -> bool:
    """A job's run already on disk: valid by ``rundir.valid_run`` with its equity paths,
    and for a learned run made by its arm's trainer."""
    trainer = _trainer_name(config)
    found = (
        valid_run(run_dir, config, source)
        if trainer is None
        else valid_run(run_dir, config, source, trainer=trainer)
    )
    if found is None:
        return False
    try:
        artifacts = RunManifest.read(run_dir).artifacts
    except (OSError, ValueError):
        return False
    # a learned run without its policy cannot be re-simulated: trained again, not reused
    needed = {EQUITY_FILENAME, ACTION_LOG_FILENAME}
    needed |= {POLICY_FILENAME} if config.train is not None else set()
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
        # Atomic, so a reader never sees half a file; the helper also waits out Windows
        # refusing the replace (PermissionError) while another handle has the file open,
        # as the server does for a moment on every status poll.
        atomic_write_text(self.state, json.dumps(fields))

    def log(self, line: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S UTC")
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
    rid = run_id(config)
    run_dir = launched_run_dir(root, config, spec.source)
    rel = str(run_dir.relative_to(root).as_posix())
    total = config.n_episodes

    def report(done: int, phase: str) -> None:
        progress.write(done=done, total=total, phase=phase, run_id=rid, key=rel)

    report(0, "starting")
    progress.log(
        f"config {config_hash(config)} seed {config.seed}: {total} episodes on {spec.source}"
    )
    if _existing(run_dir, config, spec.source):
        progress.log("this run is already on disk and matches its manifest; not recomputed")
        report(total, "reused")
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
    log = ActionLog()
    start, last = time.perf_counter(), 0.0
    for i, seed in enumerate(range(config.seed, config.seed + total)):
        progress.check_cancelled()
        kept = log if i < MAX_EQUITY_PATHS else None
        result, path = run_episode_traced(config, policy, seed, spec.source, kept)
        results.append(result)
        if i < MAX_EQUITY_PATHS:
            paths.append(path)
        now = time.perf_counter()
        if now - last > 0.25 or i + 1 == total:
            report(i + 1, "evaluating")
            last = now
    elapsed = time.perf_counter() - start
    episodes = tuple(results)
    summary = summarize(episodes)
    # The policy and the equity paths first: the manifest is written once, last, with
    # them hashed in. The policy is saved as schedule.execute_run saves it.
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
    action_log = run_dir / ACTION_LOG_FILENAME
    action_log.write_text(log.dumps(), encoding="utf-8", newline="\n")
    extra.append(action_log)
    write_run(run_dir, config, summary, episodes, extra, source=spec.source, trainer=trainer)
    steps = sum(r.steps for r in results)
    progress.log(
        f"mean P&L {summary.mean_pnl:,.0f} (se {summary.se_pnl:,.0f}); "
        f"{steps:,} steps in {elapsed:.1f}s"
    )
    report(total, "written")
    return rel
