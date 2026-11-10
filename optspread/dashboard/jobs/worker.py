"""Runs launched from the dashboard, the worker: one job to one run directory.

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

Each worker writes its progress and log to two small files under the jobs directory
(``_Progress``), which is how the server reads progress across the process boundary
without a shared queue. A running job is asked to stop through a flag file, read between
episodes and, while a PPO job trains, every few hundred training steps.

A PPO job saves the policy it trained beside ``run.json`` and hashes it into the
manifest, exactly as ``schedule.execute_run`` does, so the capital, attribution and
action-mix views re-simulate a launched PPO run with the policy it was scored with.

A run directory is reported as already on disk only when ``rundir.valid_run`` accepts it
(the real learner for a PPO run, the default components, the panel the run read) and its
manifest hashes its equity paths too, and for a PPO run its ``policy.zip``. Both are
written before ``run.json`` and the manifest, so a job killed part way never leaves a
valid manifest without them.

A job runs on the generator unless its spec names the real panel (``source``). A
real-data job evaluates on the ``eval`` split and, for PPO, trains on ``train``
(``optspread.train``; the dates of pre-registration A8), and lands under
``optionmetrics/launched`` of the runs root rather than ``launched``: the source is not
in the run id, so the two would otherwise share a directory. A source whose data is
absent is refused at submission, as an arm that is not installed is.

The process pool pickles ``execute_job`` by module path, so it stays module-level here.
Moved from ``optspread.jobs``, which re-exports it.
"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, get_args

from optspread.artifacts import write_run
from optspread.config import PolicyName, RunConfig, config_hash
from optspread.dashboard.jobs.spec import (
    EQUITY_FILENAME,
    JOBS_DIR,
    LAUNCH_DIR,
    MAX_EQUITY_PATHS,
    JobSpec,
    to_run_config,
)
from optspread.data import SourceName
from optspread.harness import EpisodeResult, build_env, summarize
from optspread.manifest import RunManifest, callable_name
from optspread.policies import Policy, make_policy
from optspread.runs.rundir import source_root, valid_run
from optspread.train import POLICY_FILENAME, TrainedPolicy, TrainingStoppedError, check_disjoint

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
        # The replace is atomic, so a reader never sees half a file. Windows refuses it
        # (PermissionError) while another handle has the file open, as the server does for
        # a moment on every status poll, so wait that out briefly before giving up.
        attempts = 20
        for attempt in range(attempts):
            try:
                tmp.replace(self.state)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.05)

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
