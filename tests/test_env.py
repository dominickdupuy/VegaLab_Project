"""Ilan Danial: environment schema, determinism, and reproducibility from config."""

import numpy as np

from optspread.actions import IRON_CONDOR
from optspread.config import RunConfig
from optspread.env import OBS_DIM, OBS_NAMES
from optspread.harness import build_env, evaluate, run_episode
from optspread.policies import AlwaysOnPolicy, FlatPolicy


def test_observation_schema_is_fixed() -> None:
    env = build_env(RunConfig())
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert len(set(OBS_NAMES)) == OBS_DIM
    assert info["day"] == 0 and info["position"] is None
    obs, _, done, truncated, info = env.step(IRON_CONDOR)
    assert obs[OBS_NAMES.index("has_position")] == 1.0
    assert obs[OBS_NAMES.index("margin_frac")] > 0.0
    assert not done and not truncated


def test_flat_episode_is_exactly_zero() -> None:
    result = run_episode(build_env(RunConfig(cost=RunConfig().cost)), FlatPolicy(), seed=3)
    assert result.pnl == 0.0 and result.cost == 0.0 and result.n_trades == 0
    assert result.steps == 63


def test_same_seed_is_bit_identical() -> None:
    env = build_env(RunConfig())
    a = run_episode(env, AlwaysOnPolicy(), seed=11)
    b = run_episode(env, AlwaysOnPolicy(), seed=11)
    c = run_episode(env, AlwaysOnPolicy(), seed=12)
    assert a == b
    assert a.n_trades >= 2 and a.pnl != c.pnl


def test_evaluate_reproduces_from_config_alone() -> None:
    cfg = RunConfig(policy="delta_seller", n_episodes=5, seed=100)
    assert evaluate(cfg) == evaluate(cfg)
    assert evaluate(cfg).n == 5
