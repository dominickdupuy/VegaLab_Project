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
from optspread.experiment.schedule import execute_run
from optspread.harness import run_episodes
from optspread.manifest import RunManifest
from optspread.replay import replay_matches
from optspread.train import TrainingPaths, frozen_hyperparameters, train_ppo

pytestmark = pytest.mark.needs_torch

RUN = RunConfig(
    policy="ppo",
    seed=9000,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21), iv_premium_vol_pts=1.0),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
    train=TrainConfig(total_timesteps=2048, seed=0),
)


@pytest.mark.invariant
def test_the_frozen_set_is_the_installed_librarys_defaults() -> None:
    record = frozen_hyperparameters()
    assert record.library_version is not None
    assert record.matches_library, record.render()


@pytest.mark.invariant
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


@pytest.mark.invariant
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


@pytest.mark.invariant
def test_a_ppo_run_replays_bit_for_bit(tmp_path: Path) -> None:
    execute_run(RUN, tmp_path)
    assert (tmp_path / "policy.zip").stat().st_size > 0
    # The manifest names the real learner, the trainer the M2 items require (#31).
    assert RunManifest.read(tmp_path).trainer == "optspread.train.train_ppo"
    matches, reason = replay_matches(tmp_path)
    assert matches, reason


@pytest.mark.invariant
def test_a_checkpoint_is_the_policy_a_run_of_that_budget_trains() -> None:
    # The learning-curve check (#4) reads its curve from one training run per seed;
    # that is only the curve of the budget question if each checkpoint is the policy a
    # run trained to that many steps would have returned.
    seen: dict[int, tuple[float, ...]] = {}

    def record(steps: int, policy: object) -> None:
        seen[steps] = tuple(e.pnl for e in run_episodes(RUN, policy))  # type: ignore[arg-type]

    assert RUN.train is not None
    longer = RUN.model_copy(
        update={"train": RUN.train.model_copy(update={"total_timesteps": 4096})}
    )
    train_ppo(longer, every=2048, on_checkpoint=record)
    assert sorted(seen) == [2048, 4096]
    assert seen[2048] == tuple(e.pnl for e in run_episodes(RUN, train_ppo(RUN)))
    assert seen[4096] == tuple(e.pnl for e in run_episodes(longer, train_ppo(longer)))


@pytest.mark.invariant
def test_asking_whether_to_stop_trains_the_same_policy_and_stopping_ends_it() -> None:
    from optspread.train import STOP_CHECK_STEPS, TrainingStoppedError

    asks: list[int] = []

    def never() -> bool:
        asks.append(1)
        return False

    plain, asked = train_ppo(RUN), train_ppo(RUN, should_stop=never)
    assert asks, "the training was never asked"
    episodes = run_episodes(RUN, plain)
    assert run_episodes(RUN, asked) == episodes, "asking draws on no random state"
    for a, b in zip(plain.model.policy.parameters(), asked.model.policy.parameters(), strict=True):
        assert torch.equal(a, b)
    long = RUN.model_copy(
        update={"train": TrainConfig(total_timesteps=1_000_000, seed=0)}  # minutes to train
    )
    with pytest.raises(TrainingStoppedError, match=f"stopped at {STOP_CHECK_STEPS:,} steps"):
        train_ppo(long, should_stop=lambda: True)


@pytest.mark.invariant
@pytest.mark.parametrize("arm", ["a2c", "dqn"])
def test_a_research_arm_trains_its_own_learner_and_resimulates_from_its_policy(
    tmp_path: Path, arm: str
) -> None:
    """A2C and DQN at the library defaults: trained on their own paths, deterministic in
    the training seed, saved as the orchestrator saves PPO and re-simulated from it."""
    from optspread.actionmix import mix_run
    from optspread.train import POLICY_FILENAME, train, trainer_for

    run = RUN.model_copy(
        update={"policy": arm, "train": TrainConfig(algorithm=arm, total_timesteps=512, seed=0)}
    )
    run = RunConfig.model_validate(run.model_dump())
    a, b = train(run), train(run)
    assert type(a.model).__name__ == arm.upper()
    assert run_episodes(run, a) == run_episodes(run, b)
    wrapper = a.model.get_env().envs[0]
    while not isinstance(wrapper, TrainingPaths):
        wrapper = wrapper.env
    assert wrapper.episodes >= 512 // 21
    execute_run(run, tmp_path, trainer=trainer_for(arm))
    assert (tmp_path / POLICY_FILENAME).is_file()
    assert RunManifest.read(tmp_path).trainer == f"optspread.train.train_{arm}"
    assert mix_run(tmp_path).reproduces is True  # loaded as its own algorithm, not as PPO
    assert replay_matches(tmp_path, trainer_for(arm))[0]


@pytest.mark.behaviour
def test_mypy_strict_passes_with_the_learner_installed() -> None:
    # The type: ignore on the checkpoint callback is needed only without Stable-Baselines3
    # (BaseCallback is Any there); with it installed it must not be an unused ignore.
    pytest.importorskip("mypy")
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    done = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "optspread/train.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.behaviour
def test_ci_type_checks_and_runs_the_ppo_tests_with_the_learner_installed() -> None:
    ci = (
        Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "ci.yml"
    ).read_text()
    job = ci.split("check-train:", 1)[1].split("\n  web:", 1)[0]
    assert 'pip install -e ".[dev,train]"' in job and "download.pytorch.org/whl/cpu" in job
    assert "mypy --strict optspread scripts" in job and "tests/experiment/test_ppo.py" in job
