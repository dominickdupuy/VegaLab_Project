"""The PPO arm: training at the frozen D3 hyperparameters, and the trained policy.

Lane: Ilan Danial (machinery).

Decision D3 fixes the learner as Stable-Baselines3 PPO at the library defaults,
frozen. ``optspread.config.PPOHyperparameters`` holds those values written out and
``TrainConfig`` refuses any other set, so the hyperparameters of every PPO run are
in its config and therefore in its hash. ``total_timesteps`` is the one training
number that is not pre-registered, and it has no default.

Two things here exist because a learner would otherwise quietly break the design:

**Training paths are disjoint from evaluation paths.** Stable-Baselines3 seeds the
first reset and then resets with no seed, and ``SpreadEnv`` reads a missing seed as
path 0, so an unwrapped training run would see one path over and over. The
``TrainingPaths`` wrapper hands episode ``k`` of training seed ``s`` the path seed
``TRAIN_PATH_BASE + s * TRAIN_PATH_STRIDE + k``. Evaluation seeds must sit below
``TRAIN_PATH_BASE``, which ``check_disjoint`` enforces, so a policy is never scored
on a path it was trained on, and two training seeds never share a path.

**A run is deterministic on CPU.** Torch runs one thread per process
(``torch.set_num_threads(1)``), the learner and its action sampling are seeded
from ``TrainConfig.seed``, and evaluation takes the greedy action. With those three
a PPO run replays bit for bit on the same machine and library versions, which is
what ``optspread.replay`` checks. One thread is also the efficient choice: the
sweep parallelises over processes, and a 64x64 MLP gains nothing from more.

Stable-Baselines3 and torch are the optional ``train`` extra. Nothing here imports
them at module import; ``train_ppo`` and ``frozen_hyperparameters`` import them
inside the call. Everything that orchestrates training takes a ``Trainer``
callable, so the orchestration is tested without the learner installed.
"""

from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from optspread.config import FROZEN_PPO, PPOHyperparameters, RunConfig, config_hash
from optspread.env import Action
from optspread.harness import build_env
from optspread.policies import Info, Observation, Policy

POLICY_FILENAME = "policy.zip"
TORCH_THREADS = 1

# Training path seeds live at and above TRAIN_PATH_BASE; evaluation seeds below it.
# The stride bounds one training run at 2**32 episodes, far past any total_timesteps
# TrainConfig admits, and TrainConfig.seed < 2**15 keeps every block below 2**49.
TRAIN_PATH_BASE = 2**48
TRAIN_PATH_STRIDE = 2**32


def training_path_seed(train_seed: int, episode: int) -> int:
    """The path seed of training episode ``episode`` under training seed ``train_seed``."""
    if train_seed < 0 or not 0 <= episode < TRAIN_PATH_STRIDE:
        raise ValueError(f"no training path for seed {train_seed}, episode {episode}")
    return TRAIN_PATH_BASE + train_seed * TRAIN_PATH_STRIDE + episode


def check_disjoint(config: RunConfig) -> None:
    """Refuse a config whose evaluation seeds could reach into the training paths."""
    last = config.seed + config.n_episodes - 1
    if config.seed < 0 or last >= TRAIN_PATH_BASE:
        raise ValueError(
            f"evaluation seeds {config.seed}..{last} must lie in [0, 2**48): "
            "seeds at and above 2**48 are training paths"
        )


class TrainingPaths(gym.Wrapper[NDArray[np.float32], Action, NDArray[np.float32], Action]):
    """Give every training episode its own path, whatever seed the learner resets with."""

    def __init__(self, env: gym.Env[NDArray[np.float32], Action], train_seed: int) -> None:
        super().__init__(env)
        self.train_seed = train_seed
        self.episodes = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        path = training_path_seed(self.train_seed, self.episodes)
        self.episodes += 1
        return self.env.reset(seed=path, options=options)


# -- the trained policy and the trainer contract ---------------------------- #


class TrainedPolicy(Policy, Protocol):
    """A ``Policy`` that came out of training and can write itself into a run directory."""

    def save(self, path: Path) -> None: ...


# Config in, trained policy out. ``train_ppo`` is the production one; the
# orchestration and replay take any, so their logic is tested without torch.
Trainer = Callable[[RunConfig], TrainedPolicy]


class SB3Policy:
    """A trained Stable-Baselines3 model behind the ``Policy`` interface: greedy actions."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: Observation, info: Info) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def save(self, path: Path) -> None:
        self.model.save(path)


def sb3_kwargs(hp: PPOHyperparameters) -> dict[str, Any]:
    """The frozen set as ``PPO(...)`` keyword arguments, every one passed explicitly."""
    import torch

    activations: dict[str, type[torch.nn.Module]] = {"tanh": torch.nn.Tanh}
    return {
        "learning_rate": hp.learning_rate,
        "n_steps": hp.n_steps,
        "batch_size": hp.batch_size,
        "n_epochs": hp.n_epochs,
        "gamma": hp.gamma,
        "gae_lambda": hp.gae_lambda,
        "clip_range": hp.clip_range,
        "clip_range_vf": hp.clip_range_vf,
        "normalize_advantage": hp.normalize_advantage,
        "ent_coef": hp.ent_coef,
        "vf_coef": hp.vf_coef,
        "max_grad_norm": hp.max_grad_norm,
        "use_sde": hp.use_sde,
        "sde_sample_freq": hp.sde_sample_freq,
        "target_kl": hp.target_kl,
        "policy_kwargs": {
            "net_arch": {"pi": list(hp.net_arch_pi), "vf": list(hp.net_arch_vf)},
            "activation_fn": activations[hp.activation_fn],
            "ortho_init": hp.ortho_init,
            "share_features_extractor": hp.share_features_extractor,
        },
    }


def train_ppo(config: RunConfig) -> SB3Policy:
    """Train the PPO arm of ``config`` and return it. Needs the ``train`` extra."""
    if config.train is None:
        raise ValueError(f"policy {config.policy!r} has no training config")
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(TORCH_THREADS)
    train = config.train
    env = TrainingPaths(build_env(config), train.seed)
    model = PPO(
        train.hyperparameters.policy,
        env,
        seed=train.seed,
        device=train.device,
        verbose=0,
        **sb3_kwargs(train.hyperparameters),
    )
    model.learn(total_timesteps=train.total_timesteps)
    return SB3Policy(model)


# -- what the M2 gate report states about D3 -------------------------------- #


@dataclass(frozen=True, slots=True)
class HyperparameterRecord:
    """The frozen PPO set, its hash, and whether it still equals the installed library's
    defaults (None when Stable-Baselines3 is not installed, so nothing was compared)."""

    values: dict[str, Any]
    hash: str
    library_version: str | None
    matches_library: bool | None
    differences: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        if self.matches_library is None:
            check = "stable-baselines3 not installed; defaults not compared"
        elif self.matches_library:
            check = f"equal to the stable-baselines3 {self.library_version} defaults"
        else:
            check = (
                f"differ from the stable-baselines3 {self.library_version} defaults in "
                + ", ".join(self.differences)
            )
        return f"{len(self.values)} PPO hyperparameters, hash {self.hash}: {check}"


def _library_defaults() -> dict[str, Any]:
    """The installed library's defaults, in ``PPOHyperparameters`` terms."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy

    algo = {
        k: p.default
        for k, p in inspect.signature(PPO.__init__).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    pol = {
        k: p.default
        for k, p in inspect.signature(ActorCriticPolicy.__init__).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    names = {torch.nn.Tanh: "tanh", torch.nn.ReLU: "relu"}
    defaults = {k: algo[k] for k in PPOHyperparameters.model_fields if k in algo}
    defaults["activation_fn"] = names.get(pol["activation_fn"], repr(pol["activation_fn"]))
    defaults["ortho_init"] = pol["ortho_init"]
    defaults["share_features_extractor"] = pol["share_features_extractor"]
    # ActorCriticPolicy resolves net_arch=None to 64x64 for both heads in its body, not its
    # signature; the signature default is checked to still be None, the body is pinned by
    # a test that builds the model.
    if pol["net_arch"] is None:
        defaults["net_arch_pi"] = defaults["net_arch_vf"] = (64, 64)
    defaults["policy"] = "MlpPolicy"
    return defaults


def frozen_hyperparameters() -> HyperparameterRecord:
    """The D3 set every PPO run trains with, hashed, and compared to the library when present."""
    values = FROZEN_PPO.model_dump(mode="json")
    try:
        version: str | None = importlib.metadata.version("stable-baselines3")
    except importlib.metadata.PackageNotFoundError:
        version = None
    matches: bool | None = None
    differences: tuple[str, ...] = ()
    if version is not None:
        defaults = _library_defaults()
        frozen = FROZEN_PPO.model_dump()
        differences = tuple(
            sorted(k for k in frozen if k not in defaults or defaults[k] != frozen[k])
        )
        matches = not differences
    return HyperparameterRecord(values, config_hash(FROZEN_PPO), version, matches, differences)
