"""Margin: max-loss for defined-risk structures, Reg-T style for undefined risk.

Lane: Hemdutt Rao (market layer).

Build order from the project document: defined-risk first (margin is max-loss
arithmetic, verified in tests), undefined-risk second. The undefined-risk rule
below is the standard broker approximation for naked short index options. The
per-step re-mark with breach-driven liquidation lives in ``optspread.env``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from optspread.types import MULTIPLIER, Chain, OptionLeg

NAKED_PCT = 0.20  # 20% of underlying, less OTM amount, plus premium
NAKED_FLOOR_PCT = 0.10  # never below 10% of the strike


def expiry_payoff(legs: tuple[OptionLeg, ...], spots: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dollar payoff of the structure at expiry across ``spots``, net of entry premium."""
    total = np.zeros_like(spots, dtype=np.float64)
    for leg in legs:
        if leg.right == "C":
            intrinsic = np.maximum(spots - leg.strike, 0.0)
        else:
            intrinsic = np.maximum(leg.strike - spots, 0.0)
        total += leg.qty * (intrinsic - leg.entry_price)
    return np.asarray(total * MULTIPLIER, dtype=np.float64)


def _strike_grid(legs: tuple[OptionLeg, ...]) -> NDArray[np.float64]:
    return np.array(sorted({leg.strike for leg in legs}), dtype=np.float64)


def is_defined_risk(legs: tuple[OptionLeg, ...]) -> bool:
    """True when the expiry payoff is bounded below beyond both ends of the strike grid."""
    strikes = _strike_grid(legs)
    grid = np.concatenate(([0.0], strikes, [10.0 * strikes.max()]))
    payoff = expiry_payoff(legs, grid)
    return bool(payoff[0] >= payoff[1] - 1e-9 and payoff[-1] >= payoff[-2] - 1e-9)


def max_loss(legs: tuple[OptionLeg, ...]) -> float:
    """Largest dollar loss at expiry of a defined-risk structure (>= 0)."""
    return float(max(0.0, -expiry_payoff(legs, _strike_grid(legs)).min()))


def margin_requirement(legs: tuple[OptionLeg, ...], chain: Chain) -> float:
    """Dollars of buying power the structure consumes at this chain."""
    if not legs:
        return 0.0
    if is_defined_risk(legs):
        return max_loss(legs)
    # Undefined risk: each naked short leg margined on its own; a strangle carries the
    # larger side plus the other side's premium (Reg-T style).
    shorts = [leg for leg in legs if leg.qty < 0]
    reqs: list[float] = []
    for leg in shorts:
        if leg.right == "C":
            otm = max(leg.strike - chain.spot, 0.0)
        else:
            otm = max(chain.spot - leg.strike, 0.0)
        req = max(NAKED_PCT * chain.spot - otm, NAKED_FLOOR_PCT * leg.strike) + chain.mark(leg)
        reqs.append(abs(leg.qty) * req * MULTIPLIER)
    if len(shorts) == 2 and shorts[0].right != shorts[1].right:
        lesser = shorts[int(np.argmin(reqs))]
        return max(reqs) + abs(lesser.qty) * chain.mark(lesser) * MULTIPLIER
    return float(sum(reqs))
