"""Ilan Danial: the real learner, at the frozen D3 hyperparameters (week 6).

Skipped without the ``train`` extra, so ``make check`` on a dev-only install does
not pull torch. With it installed, these are the claims the pilot and the sweep
rest on:

    the frozen set is still the installed library's defaults, and is what the
        model was actually built with;
    the learner trains on its own paths, one per episode, never path 0 again;
    two trainings of one config are the same policy, and a different training
        seed is a different one;
    a PPO run directory replays bit for bit from its manifest.

Training budgets are one rollout (2,048 steps), seconds each.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

import torch

from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.harness import run_episodes
from optspread.orchestrate import execute_run
from optspread.replay import replay_matches
from optspread.train import TrainingPaths, frozen_hyperparameters, train_ppo

RUN = RunConfig(
    policy="ppo",
    seed=9000,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21), iv_premium_vol_pts=1.0),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
    train=TrainConfig(total_timesteps=2048, seed=0),
)


def test_the_frozen_set_is_the_installed_librarys_defaults() -> None:
    record = frozen_hyperparameters()
    assert record.library_version is not None
    assert record.matches_library, record.render()


def test_the_model_is_built_with_the_frozen_set_and_trains_on_its_own_paths() -> None:
    model = train_ppo(RUN).model
    hp = RUN.train.hyperparameters if RUN.train else None
    assert hp is not None
    assert (model.n_steps, model.batch_size, model.n_epochs) == (
        hp.n_steps,
        hp.batch_size,
        hp.n_epochs,
    )
    assert (model.gamma, model.gae_lambda, model.ent_coef) == (hp.gamma, hp.gae_lambda, 0.0)
    assert model.policy.net_arch == {"pi": [64, 64], "vf": [64, 64]}
    assert model.policy.activation_fn is torch.nn.Tanh
    assert str(model.device) == "cpu"
    assert torch.get_num_threads() == 1

    wrapper = model.get_env().envs[0]
    while not isinstance(wrapper, TrainingPaths):
        wrapper = wrapper.env
    assert wrapper.train_seed == 0
    assert wrapper.episodes >= 2048 // 21  # a fresh path every episode


def test_one_config_trains_one_policy_and_another_seed_trains_another() -> None:
    a, b = train_ppo(RUN), train_ppo(RUN)
    assert run_episodes(RUN, a) == run_episodes(RUN, b)
    for pa, pb in zip(a.model.policy.parameters(), b.model.policy.parameters(), strict=True):
        assert torch.equal(pa, pb)

    other = RUN.model_copy(update={"train": TrainConfig(total_timesteps=2048, seed=1)})
    c = train_ppo(other)
    assert not all(
        torch.equal(pa, pc)
        for pa, pc in zip(a.model.policy.parameters(), c.model.policy.parameters(), strict=True)
    )


def test_a_ppo_run_replays_bit_for_bit(tmp_path: Path) -> None:
    execute_run(RUN, tmp_path)
    assert (tmp_path / "policy.zip").stat().st_size > 0
    matches, reason = replay_matches(tmp_path)
    assert matches, reason
