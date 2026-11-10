"""Runs launched from the dashboard, the knobs: what a launch can ask for, as which run.

Lane: Ilan Danial (machinery).

A run started from a browser is an ordinary run. The knobs (``JobSpec``) map onto a
``RunConfig`` field for field (``to_run_config``); the presets the launch panel offers and
the on-disk names the worker, the queue and the API share are here too.

PPO is offered only when ``optspread.train`` (W6-5) is importable with its
learner installed. It is detected, not assumed, so the dashboard never offers an
arm this checkout cannot run. A source whose data is absent is not offered either
(``available_sources``).

Moved from ``optspread.jobs``, which re-exports it.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.data import SourceName, make_source
from optspread.runs.rundir import SOURCES, JobSource, run_id, source_root

EQUITY_FILENAME = "equity.json"
LAUNCH_DIR = "launched"
JOBS_DIR = ".jobs"
MAX_EQUITY_PATHS = 200  # episodes whose equity path is kept; the rest are summarised only
MAX_TRAIN_SEED = 2**15  # TrainConfig.seed's bound: a PPO job trains on its first seed

Arm = Literal["flat", "always_on", "random", "delta_seller", "ppo"]
SCRIPTED_ARMS: tuple[Arm, ...] = ("flat", "always_on", "random", "delta_seller")

_MARKET, _COST, _ENV, _RUN = MarketConfig(), CostConfig(), EnvConfig(), RunConfig()


def ppo_available() -> bool:
    """``optspread.train`` exposes ``train_ppo``, ``RunConfig`` takes a ``train`` block,
    and the learner is installed. All three, or the arm is hidden."""
    if importlib.util.find_spec("stable_baselines3") is None:
        return False
    if importlib.util.find_spec("optspread.train") is None:
        return False
    if "train" not in RunConfig.model_fields:
        return False
    return callable(getattr(importlib.import_module("optspread.train"), "train_ppo", None))


def available_arms() -> tuple[Arm, ...]:
    return (*SCRIPTED_ARMS, "ppo") if ppo_available() else SCRIPTED_ARMS


def available_sources() -> tuple[JobSource, ...]:
    """The sources a job can run on here: the generator always, the panel when present."""
    return tuple(s for s in SOURCES if make_source(s).available)


class JobSpec(BaseModel):
    """Every knob the launch panel exposes, with the defaults of ``optspread.config``.

    Bounds are there to keep a browser from asking for a run that would take a day
    or cannot be built, not to encode a research choice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: Arm = "delta_seller"
    iv_premium_vol_pts: float = Field(default=_MARKET.iv_premium_vol_pts, ge=-10.0, le=20.0)
    sigma: float = Field(default=_MARKET.sigma, gt=0.0, le=1.5)
    r: float = Field(default=_MARKET.r, ge=-0.05, le=0.25)
    n_days: int = Field(default=_MARKET.n_days, ge=5, le=504)
    tenor_days: tuple[int, ...] = Field(default=_MARKET.tenor_days, min_length=1, max_length=6)
    n_strikes_each_side: int = Field(default=_MARKET.n_strikes_each_side, ge=2, le=64)
    strike_step: float = Field(default=_MARKET.strike_step, gt=0.0, le=500.0)
    costs_enabled: bool = _COST.enabled
    min_half_spread: float = Field(default=_COST.min_half_spread, ge=0.0, le=10.0)
    half_spread_frac: float = Field(default=_COST.half_spread_frac, ge=0.0, le=0.5)
    commission_per_contract: float = Field(default=_COST.commission_per_contract, ge=0.0, le=20.0)
    short_delta: float = Field(default=_ENV.short_delta, gt=0.0, lt=0.5)
    wing_delta: float = Field(default=_ENV.wing_delta, gt=0.0, lt=0.5)
    expiry_idx: int = Field(default=_ENV.expiry_idx, ge=0)
    roll_dte: int = Field(default=_ENV.roll_dte, ge=0, le=60)
    initial_cash: float = Field(default=_ENV.initial_cash, gt=0.0, le=1e9)
    seed: int = Field(default=_RUN.seed, ge=0, lt=2**31)
    n_episodes: int = Field(default=_RUN.n_episodes, ge=1, le=2000)
    total_timesteps: int | None = Field(default=None, gt=0, le=5_000_000)  # PPO only
    # Not part of the config, so not of the run id; the manifest records it.
    source: JobSource = "synthetic"

    @model_validator(mode="after")
    def _consistent(self) -> JobSpec:
        tenors = self.tenor_days
        if any(t < 1 for t in tenors) or list(tenors) != sorted(set(tenors)):
            raise ValueError("tenor_days must be distinct positive days in increasing order")
        if self.expiry_idx >= len(tenors):
            raise ValueError(f"expiry_idx {self.expiry_idx} has no tenor (there are {len(tenors)})")
        if self.wing_delta >= self.short_delta:
            raise ValueError("wing_delta must sit further out than short_delta (be smaller)")
        if self.roll_dte >= tenors[self.expiry_idx]:
            raise ValueError("roll_dte must be shorter than the tenor new structures open at")
        if (self.policy == "ppo") != (self.total_timesteps is not None):
            raise ValueError("total_timesteps is set exactly when the arm is ppo")
        if self.policy == "ppo" and self.seed >= MAX_TRAIN_SEED:
            # to_run_config trains on the first seed, and TrainConfig.seed is below 2**15
            raise ValueError(
                f"a PPO run trains on its first seed, which must be below {MAX_TRAIN_SEED:,} "
                f"(the training seed's range); got {self.seed:,}"
            )
        if self.sigma + self.iv_premium_vol_pts / 100.0 <= 0.0:
            raise ValueError(
                f"the implied vol, sigma {self.sigma} plus the premium "
                f"{self.iv_premium_vol_pts} vol points, must be above zero"
            )
        return self

    @property
    def steps(self) -> int:
        """Environment steps the evaluation takes, for progress and for a cost estimate."""
        return self.n_episodes * self.n_days


def to_run_config(spec: JobSpec) -> RunConfig:
    """The ``RunConfig`` a spec stands for. Episodes span the whole path (``episode_days``
    equals ``n_days``), as every gate and the sweep run them."""
    payload: dict[str, Any] = {
        "market": {
            "sigma": spec.sigma,
            "r": spec.r,
            "n_days": spec.n_days,
            "tenor_days": list(spec.tenor_days),
            "n_strikes_each_side": spec.n_strikes_each_side,
            "strike_step": spec.strike_step,
            "iv_premium_vol_pts": spec.iv_premium_vol_pts,
        },
        "cost": {
            "enabled": spec.costs_enabled,
            "min_half_spread": spec.min_half_spread,
            "half_spread_frac": spec.half_spread_frac,
            "commission_per_contract": spec.commission_per_contract,
        },
        "env": {
            "initial_cash": spec.initial_cash,
            "episode_days": spec.n_days,
            "short_delta": spec.short_delta,
            "wing_delta": spec.wing_delta,
            "expiry_idx": spec.expiry_idx,
            "roll_dte": spec.roll_dte,
        },
        "policy": spec.policy,
        "seed": spec.seed,
        "n_episodes": spec.n_episodes,
    }
    if spec.policy == "ppo":
        payload["train"] = {"total_timesteps": spec.total_timesteps, "seed": spec.seed}
    return RunConfig.model_validate(payload)


def launched_run_dir(runs_root: Path, config: RunConfig, source: SourceName) -> Path:
    """Where a launched run of ``config`` on ``source`` lives under the runs root:
    ``launched/<run id>`` under the source's folder (``rundir.source_root``), since the
    source is not in a run id."""
    return source_root(runs_root, source) / LAUNCH_DIR / run_id(config)


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    label: str
    description: str
    spec: JobSpec


PRESETS: tuple[Preset, ...] = (
    Preset(
        "null_world",
        "Null world",
        "The fixed-delta seller, nothing planted: it loses its costs. The control.",
        JobSpec(),
    ),
    Preset(
        "planted_1",
        "Planted 1 vol pt",
        "The seller with 1 vol pt planted: the smallest step on the sweep's axis.",
        JobSpec(iv_premium_vol_pts=1.0),
    ),
    Preset(
        "planted_4",
        "Planted 4 vol pts",
        "A premium large enough to clear the noise in 100 episodes: a detected effect.",
        JobSpec(iv_premium_vol_pts=4.0),
    ),
    Preset(
        "costs_off",
        "Costs off",
        "The always-on condor without costs: the fair world breaks even before costs.",
        JobSpec(policy="always_on", costs_enabled=False),
    ),
    Preset(
        "churn",
        "Random churn",
        "The random policy with costs: it trades most and loses fastest.",
        JobSpec(policy="random"),
    ),
)
