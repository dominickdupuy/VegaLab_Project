"""Margin: max-loss for defined risk, the Cboe naked index rule for undefined risk.

Lane: Hemdutt Rao (market layer).

Build order from the project document: defined-risk first (margin is max-loss
arithmetic, verified in tests), undefined-risk second. The per-step re-mark with
breach-driven liquidation lives in ``optspread.env``; the pre-trade check that
refuses an open the book cannot carry lives in ``optspread.portfolio``.

Undefined risk follows Cboe Rule 10.3(c)(5)(A) for a short uncovered option on a
broad-based index, which SPX is. Per contract, times the multiplier:

    100% of the option's current market value
    plus 15% of the underlying index value, less any out-of-the-money amount,
    never below the current market value plus
        10% of the underlying index value for a call,
        10% of the exercise price for a put.

The out-of-the-money amount is ``K - S`` for a call and ``S - K`` for a put, when
positive. FINRA Rule 4210(f)(2)(E)(i) carries the same 15% and 10% for a broad
index stock group, and spells out the asymmetric floor: the call's is a share of
the underlying, the put's a share of its aggregate exercise price. The rule this
replaced charged the equity-option 20% and floored both rights at 10% of the
strike, so SPX was over-margined and a far call was floored on the wrong
quantity. The rule text was read from the SEC's approval order for SR-CBOE-2022-058
(Release 34-97019, 2023), which restates 10.3(c)(5), and from FINRA 4210 itself.

A short put and a short call on the same index are margined as one: the greater
of the two requirements plus the current market value of the other side (FINRA
Rule 4210(f)(2)(G)(i); Cboe's strategy-based margin table for broad-based index
straddles and strangles states the same). That is the strangle treatment this
module already had; only the per-leg arithmetic under it changed. Two naked legs
of the same right get no such offset and are summed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from optspread.types import MULTIPLIER, Chain, OptionLeg

NAKED_PCT = 0.15  # broad-based index: 15% of the underlying, less OTM, plus the premium
NAKED_FLOOR_PCT = 0.10  # floor: 10% of the underlying (call) or of the strike (put)


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
    # Undefined risk: each naked short leg margined on its own; a short put and a short
    # call together carry the greater side plus the other side's current value.
    shorts = [leg for leg in legs if leg.qty < 0]
    reqs = [abs(leg.qty) * naked_requirement(leg, chain) * MULTIPLIER for leg in shorts]
    if len(shorts) == 2 and shorts[0].right != shorts[1].right:
        lesser = shorts[int(np.argmin(reqs))]
        return max(reqs) + abs(lesser.qty) * chain.mark(lesser) * MULTIPLIER
    return float(sum(reqs))


def naked_requirement(leg: OptionLeg, chain: Chain) -> float:
    """Per-share requirement of one short uncovered leg, Cboe Rule 10.3(c)(5)(A)."""
    if leg.right == "C":
        otm = max(leg.strike - chain.spot, 0.0)
        floor = NAKED_FLOOR_PCT * chain.spot
    else:
        otm = max(chain.spot - leg.strike, 0.0)
        floor = NAKED_FLOOR_PCT * leg.strike
    return max(NAKED_PCT * chain.spot - otm, floor) + chain.mark(leg)
