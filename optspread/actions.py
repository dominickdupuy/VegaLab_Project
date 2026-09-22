"""The action library: the structures a policy may hold. Locked at the Nov 1 cutoff.

Lane: Hemdutt Rao (market layer).

    0  FLAT               nothing held
    1  IRON_CONDOR        short strangle at ``short_delta`` with long wings at ``wing_delta``
    2  SHORT_STRANGLE     short strangle at ``short_delta``, undefined risk
    3  PUT_CREDIT_SPREAD  short put at ``short_delta``, long put at ``wing_delta``

The library is append-only until the 2026-11-01 cutoff, so an id never changes
meaning and a saved policy keeps its action semantics. Appending does move
``action_frac`` in the observation, because that slot is the held id over
``N_ACTIONS``; the Wave 0 table is re-recorded whenever the library grows.

Strikes are chosen on the listed grid by nearest delta, so a "16-delta" leg is
the grid strike whose delta is closest to 0.16. Every leg is one contract.
"""

from __future__ import annotations

import numpy as np

from optspread.config import EnvConfig
from optspread.pricing import bs_delta
from optspread.types import Chain, OptionLeg, Position, Right

FLAT, IRON_CONDOR, SHORT_STRANGLE, PUT_CREDIT_SPREAD = 0, 1, 2, 3
N_ACTIONS = 4
ACTION_NAMES = ("FLAT", "IRON_CONDOR", "SHORT_STRANGLE", "PUT_CREDIT_SPREAD")


def strike_for_delta(chain: Chain, right: Right, target: float, expiry_day: int) -> float:
    """Grid strike whose |delta| is closest to ``target`` at ``expiry_day``."""
    row = chain.expiry_row(expiry_day)
    deltas = bs_delta(
        right, chain.spot, chain.strikes, chain.r, chain.ivs[row], chain.tau(expiry_day)
    )
    return float(chain.strikes[int(np.argmin(np.abs(np.abs(deltas) - target)))])


def _leg(chain: Chain, right: Right, delta: float, expiry_day: int, qty: int) -> OptionLeg:
    K = strike_for_delta(chain, right, delta, expiry_day)
    return OptionLeg(right, K, expiry_day, qty, chain.price(right, K, expiry_day))


def build_position(action_id: int, chain: Chain, cfg: EnvConfig) -> Position | None:
    """Legs for ``action_id`` at this chain's mid prices; ``None`` for FLAT."""
    if action_id == FLAT:
        return None
    if not 0 <= action_id < N_ACTIONS:
        raise ValueError(f"action_id {action_id} out of range")
    exp = int(chain.expiry_days[min(cfg.expiry_idx, chain.n_expiries - 1)])
    if action_id == PUT_CREDIT_SPREAD:
        legs = [
            _leg(chain, "P", cfg.short_delta, exp, -1),
            _leg(chain, "P", cfg.wing_delta, exp, +1),
        ]
        return Position(action_id=action_id, legs=tuple(legs), open_day=chain.t)
    legs = [_leg(chain, "P", cfg.short_delta, exp, -1), _leg(chain, "C", cfg.short_delta, exp, -1)]
    if action_id == IRON_CONDOR:
        legs.append(_leg(chain, "P", cfg.wing_delta, exp, +1))
        legs.append(_leg(chain, "C", cfg.wing_delta, exp, +1))
    return Position(action_id=action_id, legs=tuple(legs), open_day=chain.t)
