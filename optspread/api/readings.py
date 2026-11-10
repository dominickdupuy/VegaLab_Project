"""API payloads for the data page's questions and the simulated surface beside a real one.

Lane: Ilan Danial (machinery).

``PanelReader`` (``optspread.api.panel``) builds these. Moved from ``optspread.webapi``,
which re-exports it.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from pydantic import Field

from optspread import dashboard_panel as panel
from optspread.api._base import Out, _f, _fl
from optspread.config import MarketConfig
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG

SIGMA_FROM = ("config", "rv21", "atm")


class RegionOut(Out):
    """Real less simulated over the simulator's contracts in one moneyness region."""

    name: str  # "put" (ln(K/F) < -0.03), "atm" or "call"
    label: str
    mean_diff: float | None  # vol points
    n: int


class WidestOut(Out):
    """The simulator's contract where the real surface is furthest from it."""

    strike: float
    tenor_trading_days: int
    moneyness: float  # K / F
    diff: float  # vol points, real less simulated
    real: float
    sim: float


class SimulatedOut(Out):
    """The simulator's surface on one day, priced by ``optspread.market``, beside the real one.

    The synthetic world prices every listed contract at ``sigma + premium``: one
    number over its strike and tenor grid. ``real`` is the real surface read off at
    the same contracts, so ``real - iv`` is what the synthetic world leaves out.
    """

    date: str
    spot: float
    spot_from: str  # "spx_daily.parquet" or "config"
    sigma: float
    sigma_from: str  # "config" (MarketConfig.sigma), "rv21" (the day's), "atm" (IV30 - premium)
    premium: float  # vol points
    level: float  # sigma + premium / 100: every contract's IV
    r: float
    strike: list[float]  # (K,)
    tenor_trading_days: list[int]  # (T,) as MarketConfig lists them
    tenor_days: list[float]  # (T,) the same tenors in calendar days
    k: list[list[float]]  # (T, K) ln(K / F) per tenor
    iv: list[list[float]]  # (T, K)
    real: list[list[float | None]]  # (T, K) the real surface at the same contracts
    mean_diff: float | None  # real - simulated over the contracts both price, decimals
    mean_abs_diff: float | None
    max_abs_diff: float | None
    source: str
    # Where they differ most, for the data page's question 3 (vol points).
    regions: list[RegionOut] = Field(default_factory=list)
    widest: WidestOut | None = None
    reading: str = ""


def simulated_surface(
    day: date, cfg: MarketConfig, real: panel.GridSurface, spot_from: str, sigma_from: str
) -> SimulatedOut:
    """The chain ``SyntheticChainProvider`` lists at its first step, against the real grid.

    The provider is the simulator's own: its strikes around the spot, its tenors in
    trading days, its IVs. Each contract is placed at ``ln(K / F)`` with ``F`` the
    forward at the simulator's rate, and its tenor in calendar days, so it lands on
    the axes of the real surface.
    """
    chain = SyntheticChainProvider(cfg).reset(SeededStepRNG(0))
    taus = [chain.tau(int(e)) for e in chain.expiry_days]
    ks = [np.log(chain.strikes / (chain.spot * math.exp(chain.r * t))) for t in taus]
    real_at = [real.at(k, t * 365.0) for k, t in zip(ks, taus, strict=True)]
    diff = np.asarray(real_at) - chain.ivs
    ok = np.isfinite(diff)
    k_all = np.asarray(ks)
    regions, widest = panel.sim_regions(k_all, np.asarray(real_at), np.asarray(chain.ivs))
    widest_out = (
        WidestOut(
            strike=float(chain.strikes[widest.col]),
            tenor_trading_days=int(cfg.tenor_days[widest.row]),
            moneyness=float(math.exp(k_all[widest.row, widest.col])),
            diff=widest.diff,
            real=widest.real,
            sim=widest.sim,
        )
        if widest is not None
        else None
    )
    level = cfg.sigma + cfg.iv_premium_vol_pts / 100.0
    return SimulatedOut(
        date=day.isoformat(),
        spot=chain.spot,
        spot_from=spot_from,
        sigma=cfg.sigma,
        sigma_from=sigma_from,
        premium=cfg.iv_premium_vol_pts,
        level=cfg.sigma + cfg.iv_premium_vol_pts / 100.0,
        r=chain.r,
        strike=[float(x) for x in chain.strikes],
        tenor_trading_days=[int(d) for d in cfg.tenor_days],
        tenor_days=[t * 365.0 for t in taus],
        k=[[float(x) for x in k] for k in ks],
        iv=[[float(x) for x in row] for row in chain.ivs],
        real=[_fl(row) for row in real_at],
        mean_diff=_f(float(diff[ok].mean())) if ok.any() else None,
        mean_abs_diff=_f(float(np.abs(diff[ok]).mean())) if ok.any() else None,
        max_abs_diff=_f(float(np.abs(diff[ok]).max())) if ok.any() else None,
        source=(
            "optspread.market.SyntheticChainProvider on MarketConfig("
            f"spot0={cfg.spot0:.2f}, sigma={cfg.sigma:.4f}, "
            f"iv_premium_vol_pts={cfg.iv_premium_vol_pts:g})"
        ),
        regions=[
            RegionOut(name=r.name, label=r.label, mean_diff=_f(r.mean_diff), n=r.n) for r in regions
        ],
        widest=widest_out,
        reading=panel.sim_reading(
            level,
            regions,
            widest,
            widest_out.strike if widest_out else None,
            widest_out.tenor_trading_days if widest_out else None,
        ),
    )


# -- the data page's questions: each view's numbers and its sentence --------------- #


class PremiumNowOut(Out):
    date: str  # the last day with an IV30
    iv30: float | None  # vol points
    rv_trailing: float | None  # vol points, the 21 days before, corrected
    priced: float | None  # vol points, IV30 less rv_trailing: known on the day
    measured_date: str | None  # the last day whose forward window is complete
    measured: float | None  # vol points, that day's premium
    recent: float | None  # vol points, the quarter's mean to measured_date
    recent_share_below: float | None  # share of quarters with a lower mean


class PremiumOut(Out):
    """Question 1: the VRP mapping's premium day by day, its smoothed line and a reading.

    ``premium`` is the mapping's definition (docs/VRP_MAPPING.md), so its mean over the
    days it is defined is the headline; ``smooth`` its trailing mean over
    ``smooth_days``. Every series is in vol points.
    """

    date: list[str]
    premium: list[float | None]
    smooth: list[float | None]
    iv30: list[float | None]
    rv_fwd: list[float | None]
    rv_trailing: list[float | None]
    smooth_days: int
    now: PremiumNowOut | None
    mean: float | None  # the headline: the mapping's when there is one, else this series'
    mean_low: float | None
    mean_high: float | None
    bracket: list[float | None]
    sweep_axis: list[float]
    reading: str
    source: str


class ShapeOut(Out):
    """Question 2: the surface's level, skew and term slope on one day, against every day."""

    date: str
    atm: float | None  # vol points, 30-day at the money
    skew: float | None  # vol points, the put at put_k less atm
    call: float | None  # vol points, the call at call_k less atm
    term: float | None  # vol points, ATM at long_tenor less atm
    atm_share_below: float | None
    skew_share_below: float | None
    term_share_below: float | None
    share_inverted: float | None
    skew_median: float | None
    term_median: float | None
    tenor: int
    put_k: float
    call_k: float
    long_tenor: int
    reading: str
    source: str


class FredRowOut(Out):
    year: int
    source: str
    rate_minus_bill_bp: float | None
    rate_minus_sofr_bp: float | None
    spot_vs_sp500_bp: float | None


class PanelFileCheckOut(Out):
    path: str
    recorded: str | None  # sha256 docs/DATA_MANIFEST.json records
    actual: str | None  # sha256 of the file served
    match: bool | None


class QualityOut(Out):
    """The data-quality section: the build's version, its files against the manifest, FRED."""

    panel_version: int | None
    manifest_files: int
    checked: list[PanelFileCheckOut]
    of_record: bool | None  # every checked file matches the manifest; None when unchecked
    fred: list[FredRowOut]
    fred_available: bool
    reading: str
    source: str
