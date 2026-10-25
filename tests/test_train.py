"""Ilan Danial: the PPO arm's config and the parts of training that need no torch (week 6).

Decision D3 is enforced here as a rule rather than recorded as an intention: the
hyperparameters are a frozen set in the config, any other set is refused, and they
enter the hash of every PPO run. Adding the learner re-keyed no existing run: the
hashes below were taken before ``RunConfig.train`` existed.

Training paths are disjoint from evaluation paths and from each other, whatever
seed the learner resets with. The tests that need Stable-Baselines3 itself are in
``tests/test_ppo.py`` and skip without the ``train`` extra.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest

from optspread.config import (
    FROZEN_PPO,
    PPOHyperparameters,
    RunConfig,
    TrainConfig,
    canonical_json,
    config_hash,
)
from optspread.harness import build_env, run_episodes
from optspread.manifest import RunManifest
from optspread.sweep import SweepConfig
from optspread.train import (
    TRAIN_PATH_BASE,
    TRAIN_PATH_STRIDE,
    HyperparameterRecord,
    SB3Policy,
    TrainingPaths,
    check_disjoint,
    frozen_hyperparameters,
    train_ppo,
    training_path_seed,
)

PPO_RUN = RunConfig(policy="ppo", train=TrainConfig(total_timesteps=10_000, seed=3))


# -- the config ------------------------------------------------------------- #


def test_adding_the_learner_re_keyed_no_existing_run() -> None:
    # Re-pinned once, on purpose: adopting the fitted cost parameters for the Nov 1
    # freeze (#8, docs/COSTS.md) re-keys every costed config. Adding the learner
    # did not.
    assert config_hash(RunConfig()) == "17c15aa74ce473a9"
    assert config_hash(RunConfig(policy="delta_seller", seed=1)) == "377f82c6d51a937e"
    assert config_hash(SweepConfig()) == "1c7a87a7c73c3636"
    assert "train" not in canonical_json(RunConfig())


def test_a_ppo_run_needs_its_training_config_and_a_scripted_run_refuses_one() -> None:
    with pytest.raises(ValueError, match="ppo run needs its TrainConfig"):
        RunConfig(policy="ppo")
    with pytest.raises(ValueError, match="scripted run must not carry one"):
        RunConfig(policy="delta_seller", train=TrainConfig(total_timesteps=1))
    with pytest.raises(ValueError, match="total_timesteps"):
        TrainConfig()  # not pre-registered, so no default


def test_training_enters_the_hash() -> None:
    hashes = {
        config_hash(PPO_RUN),
        config_hash(PPO_RUN.model_copy(update={"train": TrainConfig(total_timesteps=20_000)})),
        config_hash(
            PPO_RUN.model_copy(update={"train": TrainConfig(total_timesteps=10_000, seed=4)})
        ),
        config_hash(
            PPO_RUN.model_copy(
                update={"train": TrainConfig(total_timesteps=10_000, seed=3, device="cuda")}
            )
        ),
    }
    assert len(hashes) == 4
    assert '"learning_rate":0.0003' in canonical_json(PPO_RUN)


def test_a_ppo_config_round_trips_through_its_manifest(tmp_path: Path) -> None:
    manifest = RunManifest.create(PPO_RUN)
    manifest.write(tmp_path)
    back = RunManifest.read(tmp_path)
    assert back.config == PPO_RUN
    assert back.config_hash == config_hash(PPO_RUN)


def test_the_hyperparameters_are_frozen_by_d3() -> None:
    assert TrainConfig(total_timesteps=1).hyperparameters == FROZEN_PPO
    with pytest.raises(ValueError, match=r"frozen by decision D3; learning_rate differ"):
        TrainConfig(total_timesteps=1, hyperparameters=PPOHyperparameters(learning_rate=1e-3))
    with pytest.raises(ValueError, match="n_epochs, n_steps"):
        TrainConfig(total_timesteps=1, hyperparameters=PPOHyperparameters(n_steps=128, n_epochs=4))


def test_a_copy_that_skips_validation_cannot_train_off_the_frozen_set() -> None:
    tuned = TrainConfig(total_timesteps=1).model_copy(
        update={"hyperparameters": PPOHyperparameters(learning_rate=1e-2)}
    )
    config = PPO_RUN.model_copy(update={"train": tuned})  # model_copy does not validate
    assert config.train is not None and config.train.hyperparameters != FROZEN_PPO
    with pytest.raises(ValueError, match=r"frozen by decision D3; learning_rate differ"):
        train_ppo(config)  # refused before the learner is imported, so no torch needed


def test_a_ppo_config_is_never_evaluated_with_a_scripted_stand_in() -> None:
    with pytest.raises(ValueError, match="trained policy"):
        run_episodes(PPO_RUN)
    with pytest.raises(ValueError, match="no training config"):
        train_ppo(RunConfig(policy="delta_seller"))


# -- training paths ---------------------------------------------------------- #


def test_training_paths_are_disjoint_from_evaluation_and_from_each_other() -> None:
    assert training_path_seed(0, 0) == TRAIN_PATH_BASE
    assert training_path_seed(1, 0) == training_path_seed(0, 0) + TRAIN_PATH_STRIDE
    last_of_seed_0 = training_path_seed(0, TRAIN_PATH_STRIDE - 1)
    assert last_of_seed_0 < training_path_seed(1, 0)
    with pytest.raises(ValueError):
        training_path_seed(0, TRAIN_PATH_STRIDE)
    with pytest.raises(ValueError):
        training_path_seed(-1, 0)

    check_disjoint(RunConfig(seed=TRAIN_PATH_BASE - 10, n_episodes=10))
    with pytest.raises(ValueError, match="training paths"):
        check_disjoint(RunConfig(seed=TRAIN_PATH_BASE - 10, n_episodes=11))
    with pytest.raises(ValueError):
        check_disjoint(RunConfig(seed=-1))


def test_every_training_episode_gets_its_own_path_whatever_seed_it_resets_with() -> None:
    # Stable-Baselines3 resets with a seed once and then with None; SpreadEnv reads None as
    # path 0. The wrapper ignores both and walks the training seed's own path sequence.
    config = RunConfig(policy="ppo", train=TrainConfig(total_timesteps=1, seed=5))
    wrapped = TrainingPaths(build_env(config), train_seed=5)
    direct = build_env(config)
    spots = []
    for k, seed in enumerate((123, None, None, 0)):
        _, info = wrapped.reset(seed=seed)
        _, _, _, _, info = wrapped.step(0)
        _, ref = direct.reset(seed=training_path_seed(5, k))
        _, _, _, _, ref = direct.step(0)
        assert info["spot"] == ref["spot"], k
        spots.append(info["spot"])
    assert len(set(spots)) == 4
    assert wrapped.episodes == 4


# -- the trained policy and the D3 record ------------------------------------- #


class _Model:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def predict(self, obs: object, deterministic: bool = False) -> tuple[object, None]:
        assert deterministic, "evaluation takes the greedy action"
        return np.array(2), None

    def save(self, path: Path) -> None:
        self.saved.append(path)


def test_the_sb3_adapter_acts_greedily_and_saves_through_the_model(tmp_path: Path) -> None:
    model = _Model()
    policy = SB3Policy(model)
    policy.reset(0)
    assert policy.act(np.zeros(14, dtype=np.float32), {}) == 2
    policy.save(tmp_path / "policy.zip")
    assert model.saved == [tmp_path / "policy.zip"]


def test_the_frozen_set_is_hashed_and_reports_when_the_library_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    record = frozen_hyperparameters()
    assert record.hash == config_hash(FROZEN_PPO)
    assert record.values == FROZEN_PPO.model_dump(mode="json")
    assert record.library_version is None and record.matches_library is None
    assert "not installed" in record.render()
    assert record.render().startswith(f"{len(record.values)} PPO hyperparameters")


def test_the_record_names_what_drifted_from_the_library() -> None:
    drifted = HyperparameterRecord({}, "abc", "9.9.9", False, ("ent_coef", "n_steps"))
    assert "differ from the stable-baselines3 9.9.9 defaults in ent_coef, n_steps" in (
        drifted.render()
    )
    same = HyperparameterRecord({}, "abc", "9.9.9", True)
    assert "equal to the stable-baselines3 9.9.9 defaults" in same.render()
