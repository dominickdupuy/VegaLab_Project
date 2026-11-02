"""Ilan Danial: a run launched from the dashboard is an ordinary, hashed run (week 6).

The launch panel is only worth having if what it starts is indistinguishable from
a run started at the command line: the same config hash for the same knobs, the
same numbers for the same seeds, and a run directory the browser lists as OK. The
equity path it also records must be hashed into the manifest like any artifact.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from optspread.config import RunConfig, config_hash
from optspread.dashboard_data import load_run_detail, load_run_row
from optspread.harness import build_env, run_episode
from optspread.jobs import (
    ACTION_LOG_FILENAME,
    EQUITY_FILENAME,
    LAUNCH_DIR,
    PRESETS,
    JobManager,
    JobSpec,
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
    ],
)
def test_inconsistent_knobs_are_refused(knobs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        JobSpec.model_validate(knobs)


def test_vol_of_vol_and_skew_in_jobspec_pass_through() -> None:
    spec = JobSpec(vol_of_vol=0.3, skew_slope=-0.05, n_days=21, tenor_days=(7, 14), n_episodes=3)
    cfg = to_run_config(spec)
    assert cfg.market.vol_of_vol == pytest.approx(0.3)
    assert cfg.market.skew_slope == pytest.approx(-0.05)


def test_jobspec_defaults_give_unchanged_config_hash() -> None:
    spec = JobSpec(policy="flat")
    assert config_hash(to_run_config(spec)) == config_hash(RunConfig(policy="flat"))


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


def test_a2c_arm_in_available_arms_when_sb3_present() -> None:
    from optspread.jobs import available_arms, ppo_available

    if not ppo_available():
        pytest.skip("stable_baselines3 not installed")
    arms = available_arms()
    assert "a2c" in arms
    assert "dqn" in arms


def test_a2c_jobspec_to_run_config() -> None:
    from optspread.jobs import JobSpec, to_run_config

    # expiry_idx defaults to 1, so the spec needs two tenors (as SMALL does), not one
    spec = JobSpec(policy="a2c", total_timesteps=200, n_days=21, tenor_days=(7, 14), n_episodes=2)
    cfg = to_run_config(spec)
    assert cfg.policy == "a2c"
    assert cfg.train is not None
    assert cfg.train.algorithm == "a2c"
    assert cfg.train.total_timesteps == 200


def test_learned_arms_need_total_timesteps_and_scripted_arms_refuse_it() -> None:
    from optspread.jobs import LEARNED_ARMS

    assert LEARNED_ARMS == frozenset({"ppo", "a2c", "dqn"})
    for arm in LEARNED_ARMS:
        with pytest.raises(ValidationError, match="exactly when"):
            JobSpec(policy=arm)  # no total_timesteps
    with pytest.raises(ValidationError, match="exactly when"):
        JobSpec(policy="flat", total_timesteps=100)


def test_action_log_is_written_alongside_equity(tmp_path: Path) -> None:
    (tmp_path / ".jobs").mkdir()
    rel =execute_job("test-alog", SMALL.model_dump_json(), str(tmp_path))
    run_dir = tmp_path / rel
    log_path = run_dir / ACTION_LOG_FILENAME
    assert log_path.is_file(), "action_log.json must be written"
    data = json.loads(log_path.read_text())
    assert set(data) == {"actions", "atm_iv", "time_frac"}
    n_eps = SMALL.n_episodes
    assert len(data["actions"]) == n_eps
    assert len(data["atm_iv"]) == n_eps
    assert len(data["time_frac"]) == n_eps
    # each episode has one entry per step, and the steps are the episode's days
    assert len(data["actions"][0]) == SMALL.n_days
    assert len(data["atm_iv"][0]) == SMALL.n_days
    assert len(data["time_frac"][0]) == SMALL.n_days
    assert all(isinstance(a, int) for a in data["actions"][0])
    assert data["time_frac"][0][0] == pytest.approx(0.0)
    assert all(v > 0.0 for v in data["atm_iv"][0])
    assert ACTION_LOG_FILENAME in {a.path for a in load_run_detail(run_dir).artifacts}


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
    assert {a.path for a in detail.artifacts} == {"run.json", EQUITY_FILENAME, ACTION_LOG_FILENAME}
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


def _wait(manager: JobManager, job_id: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while manager.status(job_id).state in ("queued", "running"):
        assert time.monotonic() < deadline, "job did not finish"
        time.sleep(0.05)


def test_the_manager_reports_progress_and_failures(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, executor=ThreadPoolExecutor(1))
    try:
        job = manager.submit(SMALL)
        _wait(manager, job)
        status = manager.status(job)
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
        _wait(manager, bad)
        failed = manager.status(bad)
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
        _wait(manager, job, timeout=120.0)
        assert manager.status(job).state == "done"
    finally:
        manager.shutdown()

def test_alpha_bps_jobspec_passes_through_to_market_config() -> None:
    from optspread.jobs import JobSpec, to_run_config
    spec = JobSpec(policy="flat", n_days=10, tenor_days=(5, 10), n_episodes=2, alpha_bps=30.0)
    cfg = to_run_config(spec)
    assert cfg.market.alpha_bps == 30.0
