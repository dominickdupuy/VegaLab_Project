"""Typed configuration for every component, plus a content hash.

Lane: Ilan Danial (machinery). Depends on nothing; everything else depends on it.

Every model is frozen. A ``RunConfig`` fully determines a run's numerical output
together with its seed, so ``config_hash(run)`` is the identity every manifest,
figure and dashboard row is keyed by.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PolicyName = Literal["flat", "always_on", "random", "delta_seller"]


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


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    market: MarketConfig = Field(default_factory=MarketConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    policy: PolicyName = "flat"
    seed: int = 0
    n_episodes: int = 100


def canonical_json(model: BaseModel) -> str:
    """Key-sorted, whitespace-free JSON: the same config always serialises identically."""
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def config_hash(model: BaseModel) -> str:
    """First 16 hex chars of the SHA-256 of the canonical JSON."""
    return hashlib.sha256(canonical_json(model).encode()).hexdigest()[:16]
