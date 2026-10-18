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
