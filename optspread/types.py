"""Shared value types and the ``ChainProvider`` protocol (week-one interface session).

Everyone builds against these. Changing a field here is an interface change and
needs the whole team, not a lane owner.

SPX options are European and cash-settled: there is no early assignment, and a
leg held to ``expiry_day`` settles at intrinsic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from optspread.pricing import bs_greeks, bs_price
from optspread.rng import StepRNG

Right = Literal["C", "P"]
MULTIPLIER = 100.0  # dollars per point per contract
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float  # per unit vol (not per vol point)
    theta: float  # per year


@dataclass(frozen=True, slots=True)
class OptionLeg:
    """One option within a structure. ``qty > 0`` long, ``qty < 0`` short."""

    right: Right
    strike: float
    expiry_day: int  # absolute day index; time to expiry decays as the path advances
    qty: int
    entry_price: float  # per-share premium at open

    def __post_init__(self) -> None:
        if self.qty == 0:
            raise ValueError("qty must be non-zero")
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")

    def intrinsic(self, spot: float) -> float:
        return max(spot - self.strike, 0.0) if self.right == "C" else max(self.strike - spot, 0.0)


@dataclass(frozen=True, slots=True)
class Chain:
    """The option chain visible at the close of day ``t``. Immutable.

    ``ivs`` has shape ``(n_expiries, n_strikes)``. ``expiry_days`` are absolute day
    indices, ascending. In the null world every entry of ``ivs`` equals the vol that
    drives the path; a planted premium shifts the whole surface by a constant.
    """

    t: int
    spot: float
    r: float
    strikes: FloatArray
    expiry_days: NDArray[np.int64]
    ivs: FloatArray
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.ivs.shape != (self.expiry_days.shape[0], self.strikes.shape[0]):
            raise ValueError(f"ivs shape {self.ivs.shape} != (n_expiries, n_strikes)")

    @property
    def n_expiries(self) -> int:
        return int(self.expiry_days.shape[0])

    def tau(self, expiry_day: int) -> float:
        """Years to expiry as seen from this chain's ``t`` (clamped at zero)."""
        return max(expiry_day - self.t, 0) / self.trading_days_per_year

    def expiry_row(self, expiry_day: int) -> int:
        """Row of the nearest listed expiry (a held leg may outlive the listed grid)."""
        return int(np.argmin(np.abs(self.expiry_days - expiry_day)))

    def strike_col(self, strike: float) -> int:
        return int(np.argmin(np.abs(self.strikes - strike)))

    def iv_at(self, expiry_day: int, strike: float) -> float:
        return float(self.ivs[self.expiry_row(expiry_day), self.strike_col(strike)])

    def price(self, right: Right, strike: float, expiry_day: int) -> float:
        """Per-share mid price of one option on this chain."""
        sigma = self.iv_at(expiry_day, strike)
        return float(bs_price(right, self.spot, strike, self.r, sigma, self.tau(expiry_day)))

    def greeks(self, right: Right, strike: float, expiry_day: int) -> Greeks:
        sigma = self.iv_at(expiry_day, strike)
        d, g, v, th = bs_greeks(right, self.spot, strike, self.r, sigma, self.tau(expiry_day))
        return Greeks(float(d), float(g), float(v), float(th))

    def mark(self, leg: OptionLeg) -> float:
        """Per-share mark of ``leg`` on this chain (sign-agnostic)."""
        return self.price(leg.right, leg.strike, leg.expiry_day)


@dataclass(frozen=True, slots=True)
class Position:
    """A held structure: the action that opened it, its legs, and the open day."""

    action_id: int
    legs: tuple[OptionLeg, ...]
    open_day: int

    def value(self, chain: Chain) -> float:
        """Signed dollar liquidation value (long legs positive, short negative)."""
        return sum(leg.qty * chain.mark(leg) for leg in self.legs) * MULTIPLIER

    def entry_value(self) -> float:
        return sum(leg.qty * leg.entry_price for leg in self.legs) * MULTIPLIER

    @property
    def first_expiry(self) -> int:
        return min(leg.expiry_day for leg in self.legs)


@runtime_checkable
class ChainProvider(Protocol):
    """The market engine. Synthetic today; an OptionMetrics implementation later.

    No-look-ahead contract: ``step`` is the only method that advances the path,
    and the chain returned at step ``t`` depends only on ``rng.stream(s)`` for
    ``s <= t``.
    """

    @property
    def n_days(self) -> int: ...

    @property
    def done(self) -> bool: ...

    def reset(self, rng: StepRNG) -> Chain: ...

    def step(self) -> Chain: ...
