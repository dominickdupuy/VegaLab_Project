"""Ilan Danial: a run launched from the dashboard is an ordinary, hashed run (week 6).

The launch panel is only worth having if what it starts is indistinguishable from
a run started at the command line: the same config hash for the same knobs, the
same numbers for the same seeds, and a run directory the browser lists as OK. The
equity path it also records must be hashed into the manifest like any artifact.

The manager tests wait on the job's own future, not on a clock: a busy machine makes
a job slower, never a failure. The timeout only bounds a hang.

The queue must outlive a dead worker (#30): a job whose worker is killed outright
(``tests/dying_jobs.py``) fails and says why, the jobs queued behind it run on a
rebuilt pool, and every job after it runs as if nothing had happened. The cap refuses
a launch rather than queueing it, and a cancelled job never reports a run.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest
from dying_jobs import DIES_ON_START, DIES_RUNNING, dying_job
from pydantic import ValidationError

from optspread.config import RunConfig, config_hash
from optspread.dashboard_data import load_run_detail, load_run_row
from optspread.harness import build_env, run_episode
from optspread.jobs import (
    EQUITY_FILENAME,
    LAUNCH_DIR,
    PRESETS,
    WORKER_DIED,
    JobCancelledError,
    JobManager,
    JobNotCancellableError,
    JobSpec,
    JobStatus,
    QueueFullError,
    _Progress,
    available_arms,
    execute_job,
    ppo_available,
    run_episode_traced,
    to_run_config,
)
from optspread.policies import make_policy

SMALL = JobSpec(n_days=21, tenor_days=(7, 14), n_episodes=3, seed=11)


def test_default_knobs_are_the_config_defaults() -> None:
    config = to_run_config(JobSpec(policy="flat"))
    assert config == RunConfig(policy="flat")
    assert config_hash(config) == config_hash(RunConfig(policy="flat"))


def test_knobs_map_onto_every_config_field() -> None:
    spec = JobSpec(
        policy="always_on",
        iv_premium_vol_pts=2.0,
        sigma=0.3,
        r=0.01,
        n_days=42,
        tenor_days=(14, 28),
        n_strikes_each_side=8,
        strike_step=25.0,
        costs_enabled=False,
        min_half_spread=0.1,
        half_spread_frac=0.03,
        commission_per_contract=1.0,
        short_delta=0.2,
        wing_delta=0.1,
        expiry_idx=0,
        roll_dte=3,
        initial_cash=500_000.0,
        seed=9,
        n_episodes=5,
    )
    c = to_run_config(spec)
    assert (c.market.sigma, c.market.r, c.market.n_days) == (0.3, 0.01, 42)
    assert c.market.tenor_days == (14, 28) and c.market.iv_premium_vol_pts == 2.0
    assert (c.market.n_strikes_each_side, c.market.strike_step) == (8, 25.0)
    assert not c.cost.enabled and c.cost.min_half_spread == 0.1
    assert (c.cost.half_spread_frac, c.cost.commission_per_contract) == (0.03, 1.0)
    assert (c.env.short_delta, c.env.wing_delta, c.env.expiry_idx) == (0.2, 0.1, 0)
    assert (c.env.roll_dte, c.env.initial_cash, c.env.episode_days) == (3, 500_000.0, 42)
    assert (c.policy, c.seed, c.n_episodes) == ("always_on", 9, 5)
    assert spec.steps == 5 * 42


@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        ({"tenor_days": (42, 21)}, "increasing"),
        ({"tenor_days": (21, 21)}, "increasing"),
        ({"expiry_idx": 2}, "no tenor"),
        ({"wing_delta": 0.2, "short_delta": 0.16}, "further out"),
        ({"roll_dte": 42}, "roll_dte"),
        ({"total_timesteps": 1000}, "exactly when"),
        ({"policy": "ppo"}, "exactly when"),
        ({"n_episodes": 0}, "greater than or equal"),
        ({"bogus": 1}, "Extra inputs"),
        ({"policy": "ppo", "total_timesteps": 1000, "seed": 32768}, "below 32,768"),
    ],
)
def test_inconsistent_knobs_are_refused(knobs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        JobSpec.model_validate(knobs)


def test_a_ppo_job_trains_on_a_seed_its_config_accepts() -> None:
    """The training seed is the first seed and must be below 2**15: the largest one builds,
    one past it is refused at the spec with the reason, not by the worker part way in."""
    ok = JobSpec(policy="ppo", total_timesteps=1000, seed=32767)
    config = to_run_config(ok)
    assert config.train is not None and config.train.seed == 32767
    with pytest.raises(ValidationError, match="PPO run trains on its first seed"):
        JobSpec(policy="ppo", total_timesteps=1000, seed=32768)
    assert JobSpec(seed=200_000).seed == 200_000  # the scripted arms keep the whole range


def test_every_preset_builds_a_config() -> None:
    names = [p.name for p in PRESETS]
    assert len(set(names)) == len(names) and "null_world" in names
    for preset in PRESETS:
        to_run_config(preset.spec)
    null = next(p for p in PRESETS if p.name == "null_world").spec
    assert null.iv_premium_vol_pts == 0.0 and null.costs_enabled


def test_ppo_is_hidden_without_the_learner() -> None:
    assert ppo_available() is (("ppo") in available_arms())
    if not ppo_available():
        assert available_arms() == ("flat", "always_on", "random", "delta_seller")


def test_the_traced_episode_is_the_harness_episode() -> None:
    config = to_run_config(SMALL)
    policy = make_policy("delta_seller")
    traced, path = run_episode_traced(config, policy, 11)
    plain = run_episode(build_env(config), make_policy("delta_seller"), 11)
    assert traced == plain
    assert path[0] == 0.0 and path[-1] == pytest.approx(plain.pnl)
    assert len(path) == plain.steps + 1


def test_a_launched_run_is_an_ordinary_hashed_run(tmp_path: Path) -> None:
    (tmp_path / ".jobs").mkdir()
    rel = execute_job("j1", SMALL.model_dump_json(), str(tmp_path))
    config = to_run_config(SMALL)
    assert rel == f"{LAUNCH_DIR}/{config_hash(config)}-s11"
    row = load_run_row(tmp_path / rel)
    assert row.status == "OK" and row.digests_ok and row.n_episodes == 3
    detail = load_run_detail(tmp_path / rel)
    assert {a.path for a in detail.artifacts} == {"run.json", EQUITY_FILENAME}
    equity = json.loads((tmp_path / rel / EQUITY_FILENAME).read_text())
    assert equity["seeds"] == [11, 12, 13] and len(equity["paths"]) == 3
    state = json.loads((tmp_path / ".jobs" / "j1.json").read_text())
    assert state["phase"] == "written" and state["done"] == state["total"] == 3
    log = (tmp_path / ".jobs" / "j1.log").read_text()
    assert "mean P&L" in log
    # the same knobs again: deterministic, so the run on disk is reported, not recomputed
    assert execute_job("j2", SMALL.model_dump_json(), str(tmp_path)) == rel
    assert json.loads((tmp_path / ".jobs" / "j2.json").read_text())["phase"] == "reused"
    # an edited results file no longer matches: recomputed and rewritten
    (tmp_path / rel / "run.json").write_text("{}")
    assert execute_job("j3", SMALL.model_dump_json(), str(tmp_path)) == rel
    assert load_run_row(tmp_path / rel).status == "OK"


HANG = 600.0  # seconds: a job here takes about one idle; this only catches one that never ends


def _finished(manager: JobManager, job_id: str) -> JobStatus:
    """Block until the job's future resolves, then read its status as the server would."""
    done, _ = wait([manager.jobs[job_id].future], timeout=HANG)
    assert done, f"job {job_id} still unfinished after {HANG:.0f}s: a hang, not a slow machine"
    return manager.status(job_id)


def _settled(manager: JobManager, job_id: str) -> JobStatus:
    """``_finished``, following a job the manager relaunched on a rebuilt pool."""
    for _ in range(4):
        future = manager.jobs[job_id].future
        status = _finished(manager, job_id)  # reading the status is what recovers
        if manager.jobs[job_id].future is future:
            return status
    raise AssertionError(f"job {job_id} was relaunched more than once")


def test_the_manager_reports_progress_and_failures(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, executor=ThreadPoolExecutor(1))
    try:
        job = manager.submit(SMALL)
        status = _finished(manager, job)
        assert status.state == "done" and status.fraction == 1.0
        assert status.run_key.startswith(LAUNCH_DIR) and status.run_id.endswith("-s11")
        assert status.policy == "delta_seller" and status.log_tail
        assert [s.job_id for s in manager.all()] == [job]

        def boom(*_: object) -> str:
            raise RuntimeError("worker fell over")

        manager.executor.shutdown()
        manager.executor = ThreadPoolExecutor(1)
        original = manager.executor.submit
        manager.executor.submit = lambda fn, *a: original(boom, *a)  # type: ignore[method-assign,assignment]
        bad = manager.submit(SMALL)
        failed = _finished(manager, bad)
        assert failed.state == "failed" and "worker fell over" in failed.error
        assert failed.phase == "failed" and failed.fraction == 0.0
        if not ppo_available():
            with pytest.raises(ValueError, match="PPO"):
                manager.submit(JobSpec(policy="ppo", total_timesteps=100))
    finally:
        manager.shutdown()


def test_jobs_run_in_a_separate_process(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, executor=ProcessPoolExecutor(1))
    try:
        job = manager.submit(SMALL.model_copy(update={"n_episodes": 1}))
        assert _finished(manager, job).state == "done"
    finally:
        manager.shutdown()


def test_a_dead_worker_fails_its_job_and_the_pool_is_rebuilt(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, workers=1)
    manager.target = dying_job
    try:
        doomed = manager.submit(SMALL.model_copy(update={"seed": DIES_RUNNING}))
        behind = manager.submit(SMALL.model_copy(update={"n_episodes": 1}))  # queued on it
        first = manager.jobs[behind].future
        failed = _settled(manager, doomed)
        assert failed.state == "failed" and failed.error == WORKER_DIED
        assert manager.rebuilds == 1
        # The job queued behind it had not started: it runs on the new pool.
        assert manager.jobs[behind].future is not first
        assert _settled(manager, behind).state == "done"
        # And every later job runs as if nothing had happened (it was a 500 before #30).
        later = manager.submit(SMALL.model_copy(update={"n_episodes": 2}))
        done = _settled(manager, later)
        assert done.state == "done" and done.run_key.startswith(LAUNCH_DIR)
        assert manager.rebuilds == 1
        assert [s.state for s in manager.all()] == ["done", "done", "failed"]
    finally:
        manager.shutdown()


def test_a_pool_found_broken_at_launch_is_rebuilt_from_the_factory(tmp_path: Path) -> None:
    class Broken(ThreadPoolExecutor):
        def submit(self, *args: object, **kwargs: object) -> Future[str]:
            raise BrokenProcessPool("a worker died while the queue was idle")

    manager = JobManager(tmp_path, executor=Broken(1), factory=lambda: ThreadPoolExecutor(1))
    try:
        job = manager.submit(SMALL)
        assert manager.rebuilds == 1 and _finished(manager, job).state == "done"
    finally:
        manager.shutdown()


def test_a_job_that_kills_its_worker_on_start_is_relaunched_once(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, workers=1)
    manager.target = dying_job
    try:
        job = manager.submit(SMALL.model_copy(update={"seed": DIES_ON_START}))
        status = _settled(manager, job)
        assert status.state == "failed" and status.error == WORKER_DIED
        assert manager.jobs[job].relaunched and manager.rebuilds == 2
        assert _settled(manager, manager.submit(SMALL)).state == "done"
    finally:
        manager.shutdown()


def test_the_queue_is_capped_and_a_job_can_be_cancelled(tmp_path: Path) -> None:
    STUBBORN = 21  # a job that finishes whatever its cancel flag says
    started = {11: threading.Event(), STUBBORN: threading.Event()}
    gate = threading.Event()

    def gated(job_id: str, spec_json: str, runs_root: str) -> str:
        seed = JobSpec.model_validate_json(spec_json).seed
        if seed in started:
            started[seed].set()
        assert gate.wait(HANG)
        if seed == STUBBORN:
            return f"{LAUNCH_DIR}/stubborn"
        return execute_job(job_id, spec_json, runs_root)

    manager = JobManager(tmp_path, executor=ThreadPoolExecutor(2), max_pending=3)
    manager.target = gated
    try:
        running = manager.submit(SMALL)
        stubborn = manager.submit(SMALL.model_copy(update={"seed": STUBBORN}))
        assert all(event.wait(HANG) for event in started.values())
        queued = manager.submit(SMALL.model_copy(update={"seed": 12}))
        with pytest.raises(QueueFullError, match="full"):
            manager.submit(SMALL.model_copy(update={"seed": 13}))
        assert manager.status(queued).state == "queued"

        cancelled = manager.cancel(queued)
        assert (cancelled.state, cancelled.phase) == ("cancelled", "cancelled")
        assert manager.jobs[queued].future.cancelled()  # it will never run
        last = manager.submit(SMALL.model_copy(update={"seed": 13}))  # a slot came free
        assert manager.cancel(running).state == "cancelled"  # flagged while it runs
        assert manager.cancel(stubborn).state == "cancelled"
        with pytest.raises(JobNotCancellableError, match="cancelled"):
            manager.cancel(running)
        gate.set()

        stopped = _finished(manager, running)
        assert isinstance(manager.jobs[running].future.exception(), JobCancelledError)
        assert stopped.state == "cancelled" and stopped.run_key == ""
        config = to_run_config(SMALL)
        assert not (tmp_path / LAUNCH_DIR / f"{config_hash(config)}-s11").exists()
        discarded = _finished(manager, stubborn)
        assert manager.jobs[stubborn].future.result() == f"{LAUNCH_DIR}/stubborn"
        assert discarded.state == "cancelled" and discarded.run_key == ""  # discarded
        assert _finished(manager, last).state == "done"
        with pytest.raises(JobNotCancellableError, match="done"):
            manager.cancel(last)
        with pytest.raises(ValueError, match="max_pending"):
            JobManager(tmp_path, executor=ThreadPoolExecutor(1), max_pending=0)
    finally:
        gate.set()
        manager.shutdown()


# -- the machinery audit: the pool's start method, a killed write, a cancelled training -- #


def test_the_job_pool_spawns_its_workers_as_the_orchestrator_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing.context

    # The platform default as Linux has it: fork, from a multithreaded server process.
    # Windows has no fork, so there a start method that is not spawn stands in for it.
    if "fork" in multiprocessing.get_all_start_methods():
        default: multiprocessing.context.BaseContext = multiprocessing.get_context("fork")
    else:

        class NotSpawn(multiprocessing.context.BaseContext):
            _name = "not-spawn"

        default = NotSpawn()
    monkeypatch.setattr(multiprocessing.context._default_context, "_actual_context", default)
    manager = JobManager(tmp_path, workers=1)
    try:
        from multiprocessing.context import SpawnContext

        executor = manager.executor
        assert isinstance(executor, ProcessPoolExecutor)
        # Asked for by name, not the platform's default (fork on Linux, spawn on macOS).
        assert isinstance(executor._mp_context, SpawnContext)  # type: ignore[attr-defined]
        fresh = manager.factory()
        assert isinstance(fresh._mp_context, SpawnContext)  # type: ignore[attr-defined]
        fresh.shutdown()
    finally:
        manager.shutdown()


def test_a_job_killed_before_its_equity_paths_is_recomputed_not_reused(tmp_path: Path) -> None:
    from optspread.artifacts import write_run
    from optspread.harness import run_episodes, summarize
    from optspread.manifest import RunManifest

    (tmp_path / ".jobs").mkdir()
    config = to_run_config(SMALL)
    run_dir = tmp_path / LAUNCH_DIR / f"{config_hash(config)}-s{config.seed}"
    # What a worker killed between the results and the equity paths used to leave: a
    # valid manifest of run.json alone.
    episodes = run_episodes(config)
    write_run(run_dir, config, summarize(episodes), episodes)
    rel = execute_job("j1", SMALL.model_dump_json(), str(tmp_path))
    state = json.loads((tmp_path / ".jobs" / "j1.json").read_text())
    assert state["phase"] == "written", "a run without its equity paths is not the job's run"
    assert EQUITY_FILENAME in RunManifest.read(tmp_path / rel).artifacts
    execute_job("j2", SMALL.model_dump_json(), str(tmp_path))
    assert json.loads((tmp_path / ".jobs" / "j2.json").read_text())["phase"] == "reused"


def test_cancelling_a_training_job_stops_the_training(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    import time

    (tmp_path / ".jobs").mkdir()
    # A budget that takes minutes to train; cancelled, it must stop within seconds.
    spec = SMALL.model_copy(update={"policy": "ppo", "total_timesteps": 200_000})
    log = tmp_path / ".jobs" / "j1.log"
    flag = tmp_path / ".jobs" / "j1.cancel"

    def cancel_once_training() -> None:
        while not (log.is_file() and "training PPO" in log.read_text()):
            time.sleep(0.05)
        time.sleep(0.5)
        flag.touch()

    threading.Thread(target=cancel_once_training, daemon=True).start()
    start = time.perf_counter()
    with pytest.raises(JobCancelledError):
        execute_job("j1", spec.model_dump_json(), str(tmp_path))
    assert time.perf_counter() - start < 60, "the training ran on after the cancel"
    assert not (tmp_path / LAUNCH_DIR).exists(), "nothing is written after a cancel"


def _fake_ppo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train the jobs' PPO arm with the stand-in learner: no torch, still a Trainer."""
    from fake_learner import fake_trainer

    import optspread.train as train

    def trainer(config: RunConfig, should_stop: object = None) -> object:
        return fake_trainer(config)

    monkeypatch.setattr(train, "train_ppo", trainer)


def test_a_launched_ppo_run_saves_its_policy_as_the_orchestrator_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_learner import fake_trainer

    from optspread.manifest import RunManifest, file_sha256
    from optspread.orchestrate import execute_run
    from optspread.train import POLICY_FILENAME

    _fake_ppo(monkeypatch)
    (tmp_path / ".jobs").mkdir()
    spec = SMALL.model_copy(update={"policy": "ppo", "total_timesteps": 100})
    run_dir = tmp_path / execute_job("j1", spec.model_dump_json(), str(tmp_path))
    manifest = RunManifest.read(run_dir)
    assert set(manifest.artifacts) == {"run.json", EQUITY_FILENAME, POLICY_FILENAME}
    assert manifest.artifacts[POLICY_FILENAME] == file_sha256(run_dir / POLICY_FILENAME)
    # the orchestrator's run of the same config: the same results and the same policy
    cli = tmp_path / "cli"
    execute_run(to_run_config(spec), cli, trainer=fake_trainer)
    for name in ("run.json", POLICY_FILENAME):
        assert (run_dir / name).read_bytes() == (cli / name).read_bytes(), name
    assert RunManifest.read(cli).artifacts[POLICY_FILENAME] == manifest.artifacts[POLICY_FILENAME]


def test_a_scripted_job_saves_no_policy(tmp_path: Path) -> None:
    from optspread.train import POLICY_FILENAME

    (tmp_path / ".jobs").mkdir()
    run_dir = tmp_path / execute_job("j1", SMALL.model_dump_json(), str(tmp_path))
    assert not (run_dir / POLICY_FILENAME).exists()


def test_a_launched_ppo_run_is_resimulated_with_the_policy_it_saved(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from optspread.actionmix import mix_run
    from optspread.attribution import attribute_run
    from optspread.manifest import RunManifest
    from optspread.train import POLICY_FILENAME

    (tmp_path / ".jobs").mkdir()
    spec = SMALL.model_copy(update={"policy": "ppo", "total_timesteps": 64})
    rel = execute_job("j1", spec.model_dump_json(), str(tmp_path))
    run_dir = tmp_path / rel
    assert (run_dir / POLICY_FILENAME).is_file()
    # re-simulated with the saved policy, not retrained: the run's own numbers again
    assert mix_run(run_dir).reproduces is True
    assert attribute_run(run_dir).reproduces is True
    execute_job("j2", spec.model_dump_json(), str(tmp_path))
    assert json.loads((tmp_path / ".jobs" / "j2.json").read_text())["phase"] == "reused"
    # what a job wrote before it saved its policy: a learned run without policy.zip is
    # trained again rather than reported as on disk
    manifest = RunManifest.read(run_dir)
    del manifest.artifacts[POLICY_FILENAME]
    manifest.write(run_dir)
    (run_dir / POLICY_FILENAME).unlink()
    assert execute_job("j3", spec.model_dump_json(), str(tmp_path)) == rel
    assert json.loads((tmp_path / ".jobs" / "j3.json").read_text())["phase"] == "written"
    assert POLICY_FILENAME in RunManifest.read(run_dir).artifacts


def test_a_progress_write_waits_out_a_reader_holding_the_file(tmp_path: Path) -> None:
    """The server reads a worker's progress file to answer a status poll. Windows refuses
    to replace a file another handle has open (PermissionError), so a poll landing on a
    write used to fail the job although the run was fine. Here a reader holds the file
    open, for a bounded time, across the whole write: the write must wait it out and land
    whole. Linux allows the replace, so there this passes trivially."""
    import time

    progress = _Progress(tmp_path, "j1")
    progress.write(done=0, total=3, phase="starting")
    opened, hold = threading.Event(), 0.2

    def reader() -> None:
        # as JobManager.status opens it (read_text): no delete sharing on Windows
        with progress.state.open(encoding="utf-8") as fh:
            fh.read()
            opened.set()
            time.sleep(hold)

    holder = threading.Thread(target=reader)
    holder.start()
    try:
        assert opened.wait(10.0), "the reader never opened the progress file"
        progress.write(done=1, total=3, phase="evaluating")
    finally:
        holder.join()
    state = json.loads(progress.state.read_text(encoding="utf-8"))
    assert state == {"done": 1, "total": 3, "phase": "evaluating"}
    assert not progress.state.with_suffix(".tmp").exists()
