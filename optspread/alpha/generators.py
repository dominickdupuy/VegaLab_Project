"""Synthetic worlds with a planted, known edge, and the null world each is compared to.

Every generator is a pure function of ``(params, market, rng)``. Day ``t``'s values are
computed from ``rng.stream(s, PATH_CHANNEL)`` for ``s <= t`` only, so the no-look-ahead
contract of ``optspread.rng`` holds and the leak suite's splice test applies unchanged.

Each day ``t >= 1`` draws ``z`` (the spot shock) first and, where needed, ``u`` (an
auxiliary shock) second. ``z`` is the very draw ``SyntheticChainProvider.log_return``
makes, so the ``noise`` generator is the existing null world bit for bit, and every
generator's **twin** (``twin_of``: ``noise`` on the same seed) shares its spot shocks.
Whatever a policy earns on the signal world beyond what it earns on the twin came from
the planted signal, because nothing else differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from optspread.config import MarketConfig
from optspread.rng import PATH_CHANNEL, StepRNG

FloatArray = NDArray[np.float64]
IV_FLOOR = 0.01


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NoiseParams(_Params):
    """Zero-drift GBM priced at its own vol: nothing to earn, costs to pay."""

    kind: Literal["noise"] = "noise"


class ConstVrpParams(_Params):
    """Every option priced ``premium_pts`` vol points over the vol that drives the path."""

    kind: Literal["const_vrp"] = "const_vrp"
    premium_pts: float = Field(default=3.0, ge=-10.0, le=20.0)


class MrVrpParams(_Params):
    """The implied-vol premium is an OU process in vol points: rich some days, cheap others."""

    kind: Literal["mr_vrp"] = "mr_vrp"
    mean_pts: float = Field(default=0.0, ge=-10.0, le=20.0)
    vol_pts: float = Field(default=4.0, ge=0.0, le=20.0)
    half_life_days: float = Field(default=10.0, gt=0.0, le=1e9)


class SpotMrParams(_Params):
    """Log spot mean-reverts to its start; options still priced at the GBM vol."""

    kind: Literal["spot_mr"] = "spot_mr"
    half_life_days: float = Field(default=10.0, gt=0.0, le=1e9)


class DriftParams(_Params):
    """The underlying drifts ``bps_per_day`` basis points a day."""

    kind: Literal["drift"] = "drift"
    bps_per_day: float = Field(default=5.0, ge=-50.0, le=50.0)


GeneratorParams = Annotated[
    NoiseParams | ConstVrpParams | MrVrpParams | SpotMrParams | DriftParams,
    Field(discriminator="kind"),
]
GeneratorKind = Literal["noise", "const_vrp", "mr_vrp", "spot_mr", "drift"]


@dataclass(frozen=True, slots=True)
class GeneratorInfo:
    title: str
    edge: str  # what was planted
    taker: str  # which policy can take it, and whether the agent can see it
    signal_label: str  # what the signal series is, with units


GENERATORS: dict[str, GeneratorInfo] = {
    "noise": GeneratorInfo(
        "Pure noise",
        "Nothing. Options are priced at exactly the vol that drives the path.",
        "No one. Every policy should lose its trading costs, and a learner that "
        "earns here in sample has memorised noise.",
        "none",
    ),
    "const_vrp": GeneratorInfo(
        "Constant variance premium",
        "Implied vol sits a fixed number of vol points above realised vol.",
        "Any short-vol seller, all the time. Unconditional: no observation needed.",
        "premium, vol points",
    ),
    "mr_vrp": GeneratorInfo(
        "Mean-reverting variance premium",
        "The implied-vol premium wanders around its mean and reverts with the stated half-life.",
        "A seller that sells when implied vol is rich and stays flat when it is "
        "cheap. The agent sees it through atm_iv.",
        "premium, vol points",
    ),
    "spot_mr": GeneratorInfo(
        "Mean-reverting spot",
        "Log spot reverts to its start, so realised vol over a multi-day hold is "
        "below the implied vol the options are priced at.",
        "A short-vol seller that holds. Unconditional: the reverting state is not "
        "in the observation.",
        "log deviation from start, %",
    ),
    "drift": GeneratorInfo(
        "Drift",
        "The underlying drifts a fixed number of basis points a day.",
        "The put credit spread, all the time. Unconditional.",
        "cumulative drift, bps",
    ),
}


@dataclass(frozen=True, slots=True)
class SynthPath:
    """One path, days ``0..n_days``: spot, the flat ATM implied vol, and the planted signal."""

    spot: FloatArray
    iv: FloatArray
    signal: FloatArray


def twin_of(params: GeneratorParams) -> NoiseParams:
    """The null twin: same seeds, same spot shocks, nothing planted."""
    return NoiseParams()


def _phi(half_life_days: float) -> float:
    return math.exp(-math.log(2.0) / half_life_days)


def generate(params: GeneratorParams, market: MarketConfig, rng: StepRNG) -> SynthPath:
    n, sigma = market.n_days, market.sigma
    dt = 1.0 / market.trading_days_per_year
    sqdt = float(np.sqrt(dt))
    spot = np.empty(n + 1, dtype=np.float64)
    iv = np.full(n + 1, sigma, dtype=np.float64)
    signal = np.zeros(n + 1, dtype=np.float64)
    spot[0] = market.spot0

    if isinstance(params, MrVrpParams):
        phi = _phi(params.half_life_days)
        g0 = rng.stream(0, PATH_CHANNEL)
        g0.standard_normal()  # day 0's z: no return into day 0, kept so u0 is a fixed draw
        prem = params.mean_pts + params.vol_pts * float(g0.standard_normal())
        signal[0] = prem
        innov = params.vol_pts * math.sqrt(1.0 - phi * phi)
    else:
        phi, prem, innov = 1.0, 0.0, 0.0

    s, x = market.spot0, 0.0
    for t in range(1, n + 1):
        g = rng.stream(t, PATH_CHANNEL)
        z = float(g.standard_normal())
        lr = -0.5 * sigma * sigma * dt + sigma * sqdt * z  # SyntheticChainProvider.log_return
        if isinstance(params, SpotMrParams):
            x = _phi(params.half_life_days) * x + lr
            s = market.spot0 * float(np.exp(x))
            signal[t] = 100.0 * x
        elif isinstance(params, DriftParams):
            s *= float(np.exp(lr + params.bps_per_day / 10_000.0))
            signal[t] = params.bps_per_day * t
        else:
            s *= float(np.exp(lr))
        if isinstance(params, MrVrpParams):
            u = float(g.standard_normal())
            prem = params.mean_pts + phi * (prem - params.mean_pts) + innov * u
            signal[t] = prem
        spot[t] = s

    if isinstance(params, ConstVrpParams):
        signal[:] = params.premium_pts
        iv[:] = sigma + params.premium_pts / 100.0
    elif isinstance(params, MrVrpParams):
        iv[:] = np.maximum(sigma + signal / 100.0, IV_FLOOR)
    if isinstance(params, ConstVrpParams) and sigma + params.premium_pts / 100.0 <= 0.0:
        raise ValueError(f"sigma {sigma} plus premium {params.premium_pts} pts is not a vol")
    return SynthPath(spot=spot, iv=iv, signal=signal)
