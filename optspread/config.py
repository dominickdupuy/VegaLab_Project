"""Typed configuration for every component, plus a content hash.

Lane: Ilan Danial (machinery). Depends on nothing; everything else depends on it.

Every model is frozen. A ``RunConfig`` fully determines a run's numerical output
together with its seed, so ``config_hash(run)`` is the identity every manifest,
figure and dashboard row is keyed by.

A learned arm carries its training in the config too (``RunConfig.train``), so the
hash of a PPO run covers the hyperparameters, the training budget and the training
seed, not just the world it was evaluated in. A scripted run has no ``train`` and
serialises without the key, so adding the learner re-keyed no existing run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

# The scripted policies ``optspread.policies.make_policy`` builds from a name alone.
PolicyName = Literal["flat", "always_on", "random", "delta_seller"]
# Every arm a run can have: the scripted ones plus the learner, which needs training.
ArmName = Literal[PolicyName, "ppo"]


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    spot0: float = 5000.0
    sigma: float = 0.20  # annualised vol driving the path
    r: float = 0.04
    n_days: int = 63  # path horizon in trading days
    tenor_days: tuple[int, ...] = (21, 42)  # tenors listed at every step
    n_strikes_each_side: int = 16
    strike_step: float = 50.0
    iv_premium_vol_pts: float = 0.0  # planted premium; 0.0 is the null world
    trading_days_per_year: int = 252


class CostConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_half_spread: float = 0.05  # per share, in index points
    half_spread_frac: float = 0.02  # fraction of mid
    commission_per_contract: float = 0.65


class EnvConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_cash: float = 1_000_000.0
    episode_days: int = 63
    short_delta: float = 0.16
    wing_delta: float = 0.05
    expiry_idx: int = 1  # which listed tenor new structures open at
    roll_dte: int = 5  # delta_seller closes at or below this many days to expiry


class PPOHyperparameters(BaseModel):
    """Decision D3: Stable-Baselines3 PPO at the library defaults, frozen.

    Every value is written out rather than left to the library, so an upgrade that
    moves a default cannot move an experiment, and so the values enter the config
    hash of every PPO run. They are the defaults of ``PPO`` and ``ActorCriticPolicy``
    in stable-baselines3 2.x (checked against the installed library by
    ``optspread.train.frozen_hyperparameters``). ``TrainConfig`` refuses any other
    set: tuning is the rejected alternative of D3, so it is not a parameter.
    """

    model_config = ConfigDict(frozen=True)

    policy: Literal["MlpPolicy"] = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: float | None = None
    net_arch_pi: tuple[int, ...] = (64, 64)
    net_arch_vf: tuple[int, ...] = (64, 64)
    activation_fn: Literal["tanh"] = "tanh"
    ortho_init: bool = True
    share_features_extractor: bool = True


FROZEN_PPO = PPOHyperparameters()


class TrainConfig(BaseModel):
    """How a learned arm was trained. Part of the run's identity, so part of its hash.

    ``total_timesteps`` is not pre-registered (D3 fixes the hyperparameters, not the
    budget), so it has no default: every caller states it. ``seed`` seeds the
    learner and selects the training paths, which ``optspread.train`` keeps disjoint
    from every evaluation path. ``device`` is recorded because it can change bits;
    CPU is the default because a 64x64 MLP is faster there than on a GPU.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: Literal["ppo"] = "ppo"
    total_timesteps: int = Field(gt=0, lt=2**32)
    seed: int = Field(default=0, ge=0, lt=2**15)
    device: str = "cpu"
    hyperparameters: PPOHyperparameters = Field(default_factory=PPOHyperparameters)

    @field_validator("hyperparameters")
    @classmethod
    def _frozen(cls, value: PPOHyperparameters) -> PPOHyperparameters:
        if value != FROZEN_PPO:
            changed = sorted(
                k for k, v in value.model_dump().items() if v != FROZEN_PPO.model_dump()[k]
            )
            raise ValueError(
                f"PPO hyperparameters are frozen by decision D3; {', '.join(changed)} differ"
            )
        return value


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    market: MarketConfig = Field(default_factory=MarketConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    policy: ArmName = "flat"
    seed: int = 0
    n_episodes: int = 100
    train: TrainConfig | None = None  # set exactly when the arm is learned

    @model_validator(mode="after")
    def _train_iff_learned(self) -> RunConfig:
        if (self.policy == "ppo") != (self.train is not None):
            raise ValueError(
                f"policy {self.policy!r} with train={self.train!r}: a ppo run needs its "
                "TrainConfig and a scripted run must not carry one"
            )
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_train(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # A scripted run serialises exactly as it did before ``train`` existed, so its
        # config hash, its run id and every run directory already on disk are unchanged.
        data: dict[str, Any] = handler(self)
        if self.train is None:
            data.pop("train", None)
        return data


def canonical_json(model: BaseModel) -> str:
    """Key-sorted, whitespace-free JSON: the same config always serialises identically."""
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def config_hash(model: BaseModel) -> str:
    """First 16 hex chars of the SHA-256 of the canonical JSON."""
    return hashlib.sha256(canonical_json(model).encode()).hexdigest()[:16]
