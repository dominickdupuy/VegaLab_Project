"""From quotes to a surface: parity-implied carry, implied volatility, greeks.

Lane: Hemdutt Rao (data adapter).

Both vendors go through exactly this code, which is what keeps the seam honest.
OptionMetrics computes its own IVs from its own zero curve and dividend
projections; the Databento quotes come with neither. Rather than borrow one
vendor's carry for the other, the carry is read off the quotes themselves:

    C(K) - P(K) = D * (F - K)

so a line through call-minus-put against strike, fitted per (date, root, expiry)
on the strikes nearest the money, gives the discount factor ``D`` as minus its
slope and the forward ``F`` as its intercept over ``D``. Implied volatility then
comes from ``optspread.pricing.bs_price`` on the prepaid forward ``S = F * D`` with
``r = -ln(D) / T``, which is Black-76 written in the engine's own terms, and the
greeks from ``bs_greeks`` the same way. The vendor's IV is kept beside ours as a
check, never in place of it.

Time to expiry is calendar ACT/365 from the 16:00 close. An AM-settled contract
(the SPX root) stops at the open of its expiry day, 6.5 hours earlier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optspread.pricing import bs_greeks, bs_price

AM_CUT_DAYS = 6.5 / 24.0
NEAR_BAND = 0.05  # strikes within 5% of the parity-ATM strike fit the carry
WIDE_BAND = 0.15  # fallback band when the near band has fewer than MIN_PAIRS
MIN_PAIRS = 3
SHORT_TAU = 60.0 / 365.0  # below this the slope barely identifies D; borrow the day's rate
REF_TAU_MAX = 1.5  # expiries in [SHORT_TAU, REF_TAU_MAX] set the day's reference rate
IV_LO, IV_HI, IV_ITER = 1e-4, 5.0, 60

F = NDArray[np.float64]


def time_to_expiry(date: pd.Series, expiry: pd.Series, am_settled: pd.Series) -> F:
    days = (pd.to_datetime(expiry) - pd.to_datetime(date)).dt.days.to_numpy(dtype=np.float64)
    return np.asarray((days - AM_CUT_DAYS * am_settled.to_numpy(dtype=np.float64)) / 365.0)


def fit_carry(quotes: pd.DataFrame) -> pd.DataFrame:
    """Forward and discount factor per (date, root, expiry) from put-call parity.

    ``quotes`` needs date, root, expiry, cp, strike, mid. Returns one row per group
    with ``forward``, ``discount``, ``n_pairs`` and ``carry_fit`` ("near", "wide" or
    "none"). A group whose fit is not economically sane (``D`` outside (0.5, 1.05],
    a non-positive forward) is returned with NaN carry rather than a guess.
    """
    keys = ["date", "root", "expiry"]
    calls = quotes[quotes["cp"] == "C"][[*keys, "strike", "mid"]]
    puts = quotes[quotes["cp"] == "P"][[*keys, "strike", "mid"]]
    pairs = calls.merge(puts, on=[*keys, "strike"], suffixes=("_c", "_p"))
    pairs = pairs.dropna(subset=["mid_c", "mid_p"])
    pairs["y"] = pairs["mid_c"] - pairs["mid_p"]
    pairs["abs_y"] = pairs["y"].abs()
    atm = pairs.loc[pairs.groupby(keys)["abs_y"].idxmin(), [*keys, "strike"]]
    pairs = pairs.merge(atm.rename(columns={"strike": "k_atm"}), on=keys)
    dist = (pairs["strike"] / pairs["k_atm"] - 1.0).abs()

    near = _ols(pairs[dist <= NEAR_BAND], keys).assign(carry_fit="near")
    wide = _ols(pairs[dist <= WIDE_BAND], keys).assign(carry_fit="wide")
    fits = near[near["n_pairs"] >= MIN_PAIRS]
    rest = wide.merge(fits[keys], on=keys, how="left", indicator=True)
    rest = rest[(rest["_merge"] == "left_only") & (rest["n_pairs"] >= MIN_PAIRS)]
    fits = pd.concat([fits, rest.drop(columns="_merge")], ignore_index=True)

    sane = (fits["discount"] > 0.5) & (fits["discount"] <= 1.05) & (fits["forward"] > 0.0)
    fits.loc[~sane, ["forward", "discount"]] = np.nan
    fits.loc[~sane, "carry_fit"] = "none"
    fits = _short_dated(fits, quotes)
    groups = quotes[keys].drop_duplicates()
    out = groups.merge(fits, on=keys, how="left")
    out["carry_fit"] = out["carry_fit"].fillna("none")
    out["n_pairs"] = out["n_pairs"].fillna(0).astype(np.int64)
    return out.reset_index(drop=True)


def _short_dated(fits: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Short expiries take the day's reference rate and refit only the forward.

    For a few days to expiry ``D`` differs from 1 by less than the quotes' noise, so
    the fitted slope says little about the rate even though the intercept pins the
    forward well. With ``D`` fixed at ``exp(-r_ref * T)``, the forward is the mean of
    ``K + (C - P) / D`` over the same pairs. A day with no reference expiry keeps its
    own fit.
    """
    tau = (
        fits[["date", "root", "expiry"]]
        .merge(
            quotes[["date", "root", "expiry", "tau"]].drop_duplicates(["date", "root", "expiry"]),
            on=["date", "root", "expiry"],
            how="left",
        )["tau"]
        .to_numpy(dtype=np.float64)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = -np.log(fits["discount"].to_numpy(dtype=np.float64)) / tau
    ref_mask = (tau >= SHORT_TAU) & (tau <= REF_TAU_MAX) & np.isfinite(rate)
    r_ref = pd.Series(rate[ref_mask]).groupby(fits["date"].to_numpy()[ref_mask]).median()
    day_ref = fits["date"].map(r_ref).to_numpy(dtype=np.float64)
    short = (tau < SHORT_TAU) & np.isfinite(day_ref) & (fits["n_pairs"].to_numpy() > 0)
    D = np.exp(-day_ref * tau)
    forward = (fits["sum_y"].to_numpy() / D + fits["sum_x"].to_numpy()) / fits["n_pairs"].to_numpy()
    out = fits.copy()
    out.loc[short, "discount"] = D[short]
    out.loc[short, "forward"] = forward[short]
    out.loc[short, "carry_fit"] = out.loc[short, "carry_fit"] + "+r_ref"
    return out.drop(columns=["sum_x", "sum_y"])


def _ols(pairs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Grouped least squares of y on strike: y = a + b K, so D = -b and F = a / D."""
    g = (
        pairs.assign(
            x=pairs["strike"], xx=pairs["strike"] ** 2, xy=pairs["strike"] * pairs["y"], n=1.0
        )
        .groupby(keys)[["n", "x", "y", "xx", "xy"]]
        .sum()
    )
    det = g["n"] * g["xx"] - g["x"] ** 2
    slope = (g["n"] * g["xy"] - g["x"] * g["y"]) / det
    intercept = (g["y"] - slope * g["x"]) / g["n"]
    discount = -slope
    out = pd.DataFrame(
        {
            "forward": intercept / discount,
            "discount": discount,
            "n_pairs": g["n"].astype(np.int64),
            "sum_x": g["x"],
            "sum_y": g["y"],
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


def implied_vol(is_call: NDArray[np.bool_], S: F, K: F, r: F, T: F, price: F) -> F:
    """Vectorised bisection on ``bs_price``. NaN where the price is outside no-arbitrage bounds."""
    lo = np.full(price.shape, IV_LO)
    hi = np.full(price.shape, IV_HI)
    p_lo = _price(is_call, S, K, r, lo, T)
    p_hi = _price(is_call, S, K, r, hi, T)
    ok = np.isfinite(price) & (T > 0.0) & (price > p_lo) & (price < p_hi)
    for _ in range(IV_ITER):
        mid = 0.5 * (lo + hi)
        above = _price(is_call, S, K, r, mid, T) > price
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def _price(is_call: NDArray[np.bool_], S: F, K: F, r: F, sigma: F, T: F) -> F:
    return np.where(is_call, bs_price("C", S, K, r, sigma, T), bs_price("P", S, K, r, sigma, T))


def price_surface(quotes: pd.DataFrame) -> pd.DataFrame:
    """Attach carry, our IV and greeks to normalised quotes.

    ``quotes`` needs date, root, expiry, cp, strike, bid, ask and ``am_settled``.
    Adds mid, tau, forward, discount, carry_fit, iv, delta, gamma, vega, theta.
    """
    q = quotes.copy()
    q["mid"] = np.where(
        (q["bid"] > 0.0) & (q["ask"] >= q["bid"]), 0.5 * (q["bid"] + q["ask"]), np.nan
    )
    q["tau"] = time_to_expiry(q["date"], q["expiry"], q["am_settled"])
    q = q[q["tau"] > 0.0]
    carry = fit_carry(q)
    q = q.merge(carry, on=["date", "root", "expiry"], how="left")
    D = q["discount"].to_numpy(dtype=np.float64)
    T = q["tau"].to_numpy(dtype=np.float64)
    S = q["forward"].to_numpy(dtype=np.float64) * D
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -np.log(D) / T
    K = q["strike"].to_numpy(dtype=np.float64)
    is_call = (q["cp"] == "C").to_numpy()
    iv = implied_vol(is_call, S, K, r, T, q["mid"].to_numpy(dtype=np.float64))
    q["iv"] = iv
    for name, value in zip(
        ("delta", "gamma", "vega", "theta"), _greeks(is_call, S, K, r, iv, T), strict=True
    ):
        q[name] = np.where(np.isfinite(iv), value, np.nan)
    return q.reset_index(drop=True)


def _greeks(is_call: NDArray[np.bool_], S: F, K: F, r: F, sigma: F, T: F) -> tuple[F, F, F, F]:
    sig = np.where(np.isfinite(sigma), sigma, 0.2)
    c = bs_greeks("C", S, K, r, sig, T)
    p = bs_greeks("P", S, K, r, sig, T)
    d, g, v, th = (np.where(is_call, cg, pg) for cg, pg in zip(c, p, strict=True))
    return d, g, v, th
