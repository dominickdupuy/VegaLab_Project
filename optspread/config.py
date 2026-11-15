"""Typed configuration for every component, plus a content hash.

Lane: Ilan Danial (machinery). Depends on nothing; everything else depends on it.

Every model is frozen. A ``RunConfig`` fully determines a run's numerical output
together with its seed, so ``config_hash(run)`` is the identity every manifest,
figure and dashboard row is keyed by.

A learned arm carries its training in the config too (``RunConfig.train``), so the
hash of a PPO run covers the hyperparameters, the training budget and the training
seed, not just the world it was evaluated in. A scripted run has no ``train`` and
serialises without the key, so adding the learner re-keyed no existing run.

The market's research knobs (``alpha_bps``, ``vol_of_vol``, ``skew_slope``) follow the
same rule: at their defaults they are left out of the serialised form, so the
confirmatory world, every run directory already on disk and ``config_hash(RunConfig())``
are unchanged. The same holds for the research arms (A2C and DQN): a run of either
carries a ``TrainConfig`` naming its algorithm, and drops the PPO hyperparameters it
never reads.

The rate is the one default that moved. The path is a zero-drift GBM and the book earns
nothing on cash, so only ``r = 0`` prices every contract at its expected payoff under
the path; the study (the pilot, the A7 sweep and every record made from them) ran at
``r = 0.04``, whose null world was not fair by the carry docs/DRIFT_SENSITIVITY.md
measures (pre-registration amendment A11). The default is now ``r = 0.0``, the fair
world, for runs made from here on; ``STUDY_R`` and ``STUDY_MARKET`` are the study's
world, pinned wherever code rebuilds the study's configs, so the confirmatory grid, its
hashes and every run on disk read as they did. A manifest records its full config, so
a run on disk never depended on the default.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

# The scripted policies ``optspread.policies.make_policy`` builds from a name alone.
PolicyName = Literal["flat", "always_on", "random", "delta_seller"]
# The learned arms, each trained before it is evaluated. PPO is the confirmatory learner
# (D3); A2C and DQN are research arms at the Stable-Baselines3 defaults, not pre-registered.
LearnedArm = Literal["ppo", "a2c", "dqn"]
LEARNED_ARMS: tuple[LearnedArm, ...] = ("ppo", "a2c", "dqn")
# Every arm a run can have: the scripted ones plus the learners, which need training.
ArmName = Literal[PolicyName, LearnedArm]


def scripted_name(policy: ArmName) -> PolicyName | None:
    """The scripted policy ``policy`` names, or None for a learned arm (it needs training)."""
    return None if policy in LEARNED_ARMS else cast(PolicyName, policy)


# The market knobs a research run may move, each left out of the hash at its default.
RESEARCH_KNOBS: tuple[str, ...] = ("alpha_bps", "vol_of_vol", "skew_slope")


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    spot0: float = 5000.0
    sigma: float = 0.20  # annualised vol driving the path
    # 0.0: forward = spot under the zero-drift path, the fair world. The study ran at
    # STUDY_R (module docstring); a real source reads its own rate and ignores this.
    r: float = 0.0
    n_days: int = 63  # path horizon in trading days
    tenor_days: tuple[int, ...] = (21, 42)  # tenors listed at every step
    n_strikes_each_side: int = 16
    strike_step: float = 50.0
    iv_premium_vol_pts: float = 0.0  # planted premium; 0.0 is the null world
    trading_days_per_year: int = 252
    # Research knobs, not part of the confirmatory design (optspread.market says how each
    # moves the world). 0.0 everywhere is the world the study ran in.
    alpha_bps: float = Field(default=0.0, ge=0.0, le=200.0)  # drift per day, basis points
    vol_of_vol: float = Field(default=0.0, ge=0.0)  # annualised vol of the path vol
    skew_slope: float = 0.0  # IV change in vol (decimal) per unit of ln(S / K)

    @model_serializer(mode="wrap")
    def _omit_default_knobs(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # A world with every research knob at zero serialises exactly as it did before the
        # knobs existed, so its config hash, its run id and its run directory are unchanged.
        data: dict[str, Any] = handler(self)
        for name in RESEARCH_KNOBS:
            if getattr(self, name) == 0.0:
                data.pop(name, None)
        return data


# The study's world: the rate the pilot and the A7 sweep ran at (module docstring, A11).
STUDY_R = 0.04
STUDY_MARKET = MarketConfig(r=STUDY_R)


class CostConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_half_spread: float = 0.15  # per share, in index points; fitted, docs/COSTS.md
    half_spread_frac: float = 0.0075  # fraction of mid; fitted, docs/COSTS.md
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


def require_frozen(value: PPOHyperparameters) -> PPOHyperparameters:
    """``value`` if it is the D3 set, else a ValueError naming what differs."""
    if value != FROZEN_PPO:
        frozen = FROZEN_PPO.model_dump()
        changed = sorted(k for k, v in value.model_dump().items() if v != frozen[k])
        raise ValueError(
            f"PPO hyperparameters are frozen by decision D3; {', '.join(changed)} differ"
        )
    return value


class TrainConfig(BaseModel):
    """How a learned arm was trained. Part of the run's identity, so part of its hash.

    ``total_timesteps`` is not pre-registered (D3 fixes the hyperparameters, not the
    budget), so it has no default: every caller states it. ``seed`` seeds the
    learner and selects the training paths, which ``optspread.train`` keeps disjoint
    from every evaluation path. ``device`` is recorded because it can change bits;
    CPU is the default because a 64x64 MLP is faster there than on a GPU.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: LearnedArm = "ppo"
    total_timesteps: int = Field(gt=0, lt=2**32)
    seed: int = Field(default=0, ge=0, lt=2**15)
    device: str = "cpu"
    hyperparameters: PPOHyperparameters = Field(default_factory=PPOHyperparameters)

    @model_validator(mode="after")
    def _frozen(self) -> TrainConfig:
        # D3 binds PPO only; A2C and DQN train at their own library defaults and never
        # read this field.
        if self.algorithm == "ppo":
            require_frozen(self.hyperparameters)
        return self

    @model_serializer(mode="wrap")
    def _omit_unread_hyperparameters(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # A2C and DQN read no PPO hyperparameter, so their hash does not depend on one.
        data: dict[str, Any] = handler(self)
        if self.algorithm != "ppo":
            data.pop("hyperparameters", None)
        return data


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
        learned = self.policy in LEARNED_ARMS
        if learned != (self.train is not None):
            arm = self.policy if learned else "learned"
            raise ValueError(
                f"policy {self.policy!r} with train={self.train!r}: a {arm} run needs its "
                "TrainConfig and a scripted run must not carry one"
            )
        if self.train is not None and self.train.algorithm != self.policy:
            raise ValueError(
                f"train.algorithm {self.train.algorithm!r} must match policy {self.policy!r}"
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
