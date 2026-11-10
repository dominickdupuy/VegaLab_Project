"""API payload for what the fixed-delta seller opens on a real day: an illustration on the
real chain.

Lane: Ilan Danial (machinery). Moved from ``optspread.webviews``, which re-exports it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optspread import dashboard_panel as panel
from optspread.api._base import Out
from optspread.api.panel import PanelNotBuiltError, PanelReader
from optspread.config import MarketConfig

# -- what the fixed-delta seller opens on a real day -------------------------------- #


class LegOut(Out):
    right: str
    strike: float
    qty: int
    price: float  # per share, the chain's mid
    iv: float
    delta: float
    moneyness: float  # K / spot


class SellerOut(Out):
    """The fixed-delta seller's structure on one real close: an illustration, not a backtest."""

    date: str
    structure: str
    spot: float
    r: float
    tenor_trading_days: int
    tenor_calendar_days: float
    short_delta: float
    wing_delta: float
    legs: list[LegOut]
    credit: float  # $ received, one structure
    max_loss: float | None  # $, defined risk only, as optspread.margin decides it
    # $, the worst expiry loss computed from the legs whenever both sides have a wing,
    # whatever the margin check decides; equal to max_loss when the check agrees
    capped_loss: float | None = None
    # the legs cap the loss but the frozen margin check margins them as naked (#39)
    margined_as_naked: bool = False
    margin: float  # $, optspread.margin at the open
    breakevens: list[float]
    note: str
    source: str
    reading: str = ""  # the data page's question 4, in one plain sentence


SELLER_NOTE = (
    "Illustration on the real chain at the close; the experiment runs on the synthetic world."
)


def real_chain(
    day: date,
    k: NDArray[np.float64],
    tenors: NDArray[np.int64],
    iv: NDArray[np.float64],
    spot: float,
    carry: float,
    cfg: MarketConfig,
) -> Any:
    """The chain ``realdata.provider.HistoricalChainProvider`` builds for this close.

    The day's gridded surface is laid on the provider's own grid and made free of
    calendar arbitrage as its store makes it (``calendar_free``), then its ``_chain`` is
    run on a one-day store, so strikes, tenors and vols are the ones a real-data episode
    would see on day 0.
    """
    from types import SimpleNamespace

    from optspread.realdata.panel import K_GRID, TENOR_GRID
    from optspread.realdata.provider import HistoricalChainProvider, SurfaceStore, calendar_free

    cube = np.full((len(TENOR_GRID), len(K_GRID)), np.nan)
    ki = {round(float(x), 4): j for j, x in enumerate(K_GRID)}
    ti = {int(t): i for i, t in enumerate(TENOR_GRID)}
    for r, t in enumerate(tenors):
        for c, x in enumerate(k):
            if int(t) in ti and round(float(x), 4) in ki and np.isfinite(iv[r, c]):
                cube[ti[int(t)], ki[round(float(x), 4)]] = iv[r, c]
    if not np.isfinite(cube).any():
        raise PanelNotBuiltError(f"no gridded surface on {day} to build a chain from")
    store = SurfaceStore(
        dates=(day.isoformat(),),
        spot=np.asarray([spot]),
        carry=np.asarray([carry]),
        ivs=calendar_free(cube)[None],
    )
    shim: Any = SimpleNamespace(config=cfg, store=store, _start=0, _day=0)
    shim._smile = lambda *a: HistoricalChainProvider._smile(shim, *a)
    return HistoricalChainProvider._chain(shim)


def capped_loss(legs: Sequence[Any]) -> tuple[float | None, bool]:
    """The worst expiry loss the legs cap, and whether the frozen margin check still
    margins them as naked shorts.

    A long wing beyond every short caps the loss whatever ``margin.is_defined_risk``
    decides: its absolute tolerance misreads some high-spot condors as naked (#39), and
    the harness, frozen, margins them so. Without a wing on each side the loss is not
    capped: (None, False)."""
    from optspread.margin import is_defined_risk, max_loss

    shorts = [leg for leg in legs if leg.qty < 0]
    winged = bool(shorts) and all(
        any(
            w.qty > 0
            and w.right == sh.right
            and (w.strike < sh.strike if sh.right == "P" else w.strike > sh.strike)
            for w in legs
        )
        for sh in shorts
    )
    if not winged:
        return None, False
    return float(max_loss(tuple(legs))), not is_defined_risk(tuple(legs))


def seller_on(chain: Any, day: date, cfg: MarketConfig) -> SellerOut:
    from optspread.actions import ACTION_NAMES, IRON_CONDOR, build_position
    from optspread.config import EnvConfig
    from optspread.margin import is_defined_risk, margin_requirement, max_loss
    from optspread.types import MULTIPLIER

    env = EnvConfig()
    pos = build_position(IRON_CONDOR, chain, env)
    assert pos is not None
    legs = pos.legs
    exp = legs[0].expiry_day
    tenor = int(exp - chain.t)
    credit = -sum(leg.qty * leg.entry_price for leg in legs) * MULTIPLIER
    shorts = {leg.right: leg.strike for leg in legs if leg.qty < 0}
    per_share = credit / MULTIPLIER
    breakevens = [shorts["P"] - per_share] if "P" in shorts else []
    breakevens += [shorts["C"] + per_share] if "C" in shorts else []
    longs = {leg.right: leg.strike for leg in legs if leg.qty > 0}
    defined = is_defined_risk(legs)
    loss = float(max_loss(legs)) if defined else None
    capped, naked = capped_loss(legs)
    margin = float(margin_requirement(legs, chain))
    reading = (
        panel.seller_reading(
            day.isoformat(),
            float(chain.spot),
            tenor,
            shorts["P"],
            shorts["C"],
            longs["P"],
            longs["C"],
            float(credit),
            capped if capped is not None else loss,
            margin,
            breakevens,
            env.short_delta,
            naked=naked,
        )
        if {"P", "C"} <= shorts.keys() and {"P", "C"} <= longs.keys()
        else ""
    )
    return SellerOut(
        date=day.isoformat(),
        structure=ACTION_NAMES[IRON_CONDOR],
        spot=float(chain.spot),
        r=float(chain.r),
        tenor_trading_days=tenor,
        tenor_calendar_days=tenor * 365.0 / cfg.trading_days_per_year,
        short_delta=env.short_delta,
        wing_delta=env.wing_delta,
        legs=[
            LegOut(
                right=leg.right,
                strike=leg.strike,
                qty=leg.qty,
                price=leg.entry_price,
                iv=chain.iv_at(exp, leg.strike),
                delta=chain.greeks(leg.right, leg.strike, exp).delta,
                moneyness=leg.strike / float(chain.spot),
            )
            for leg in sorted(legs, key=lambda x: x.strike)
        ],
        credit=float(credit),
        max_loss=loss,
        capped_loss=capped,
        margined_as_naked=naked,
        margin=margin,
        breakevens=[float(b) for b in breakevens],
        note=SELLER_NOTE,
        reading=reading,
        source=(
            "optspread.actions.build_position(IRON_CONDOR) at EnvConfig() deltas on the chain "
            "realdata.provider builds from surface_grid.parquet and spx_daily.parquet"
        ),
    )


def _close(reader: PanelReader, day: date) -> tuple[float, float]:
    """The spot and 30-day carry the provider prices ``day`` on, OptionMetrics first.

    As ``realdata.provider`` reads them: the parity spot and the carry against it
    (``spot_parity``, ``carry30_parity``) where the panel has them, else the close and
    ``carry30``; ``rate30`` where the carry is missing, else zero.
    """
    daily = reader.daily()
    if "date" not in daily:
        raise PanelNotBuiltError(f"no daily table at {reader.layout.daily}")
    idx = np.flatnonzero(daily["date"] == np.datetime64(day, "D"))
    if "source" in daily and len(idx):
        om = idx[daily["source"][idx] == "optionmetrics"]
        idx = om if len(om) else idx
    if not len(idx):
        raise PanelNotBuiltError(f"no daily row on {day} in {reader.layout.daily}")
    i = int(idx[0])
    spot = float(daily["spot"][i])
    carry_cols: tuple[str, ...] = ("carry30", "rate30")
    if "spot_parity" in daily and "carry30_parity" in daily:
        parity = float(daily["spot_parity"][i])
        if math.isfinite(parity) and parity > 0:
            spot = parity
        carry_cols = ("carry30_parity", "rate30")
    carry = math.nan
    for col in carry_cols:
        if col in daily and math.isfinite(float(daily[col][i])):
            carry = float(daily[col][i])
            break
    if not (math.isfinite(spot) and spot > 0):
        raise PanelNotBuiltError(f"no spot on {day} in {reader.layout.daily}")
    return spot, carry if math.isfinite(carry) else 0.0


def seller_out(reader: PanelReader, day: date) -> SellerOut:
    """What the fixed-delta seller would open at ``day``'s close on the real chain."""
    g = reader.grid_day(day)
    spot, carry = _close(reader, day)
    cfg = MarketConfig()
    chain = real_chain(day, g.k, g.tenor_days, g.iv, spot, carry, cfg)
    return seller_on(chain, day, cfg)
