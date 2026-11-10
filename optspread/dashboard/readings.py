"""The data page's readings: each question the page asks, answered in one sentence.

Lane: Ilan Danial (machinery).

Built on the panel views in ``optspread.dashboard.panel_views``, whose private
``_day`` they share by its name; nothing there depends on this module. Moved from
``optspread.dashboard_panel``, which re-exports it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from optspread.dashboard.panel_views import Columns, _day, grid_dates, n_rows

# -- the readings: each question the data page asks, answered in one sentence -------- #
#
# The page leads with four questions (the premium, the surface on a day, the simulator
# against the market, the seller on a close) and answers each in a plain sentence
# computed from the panel. The numbers and the sentence are both made here, so a test
# reads exactly what the page shows.

MINUS = "\u2212"
SMOOTH_DAYS = 63  # a quarter of trading days: the premium's smoothed line


def signed(x: float, digits: int = 2) -> str:
    """``+0.64`` or ``-0.34`` with the typographic minus (U+2212): a sign always."""
    return f"{'+' if x >= 0 else MINUS}{abs(x):.{digits}f}"


def trailing_mean(
    x: NDArray[np.float64], window: int = SMOOTH_DAYS, min_count: int | None = None
) -> NDArray[np.float64]:
    """The mean of the last ``window`` values at each point, NaNs skipped.

    NaN where fewer than ``min_count`` (two thirds of the window by default) are finite,
    and where ``x`` itself is NaN, so the line ends where the data do.
    """
    x = np.asarray(x, dtype=np.float64)
    need = max(1, window * 2 // 3 if min_count is None else min_count)
    ok = np.isfinite(x)
    sums = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    counts = np.concatenate([[0], np.cumsum(ok)])
    hi = np.arange(1, len(x) + 1)
    lo = np.maximum(0, hi - window)
    n = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n >= need, (sums[hi] - sums[lo]) / np.maximum(n, 1), np.nan)
    out[~ok] = np.nan
    return np.asarray(out, dtype=np.float64)


def share_below(values: NDArray[np.float64], x: float) -> float:
    """The share of the finite ``values`` strictly below ``x``; NaN when there are none."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v) or not math.isfinite(x):
        return float("nan")
    return float((v < x).mean())


@dataclass(frozen=True, slots=True)
class PremiumSeries:
    """The VRP mapping's premium per day (docs/VRP_MAPPING.md), in vol points.

    ``premium = 100 * (IV30 - RV / c21)``, RV the zero-mean vol of the next 21 trading
    days: the series the mapping's headline averages, here day by day with a
    quarter's trailing mean beside it. ``priced`` is the same against the 21 days
    before: what the day's IV30 charges over what had just been realised, known on
    the day, where ``premium`` is known only 21 trading days later.
    """

    date: NDArray[np.datetime64]
    iv30: NDArray[np.float64]  # decimals
    rv_fwd: NDArray[np.float64]  # decimals, corrected
    rv_trailing: NDArray[np.float64]  # decimals, corrected
    premium: NDArray[np.float64]  # vol points
    smooth: NDArray[np.float64]  # vol points, the trailing mean of ``premium``
    priced: NDArray[np.float64]  # vol points, IV30 against the trailing RV


def premium_series(daily: Columns, smooth: int = SMOOTH_DAYS) -> PremiumSeries:
    """One row per date (OptionMetrics where both sources quote it), the mapping's premium."""
    from optspread import vrp

    if not n_rows(daily):
        empty = np.asarray([], np.float64)
        return PremiumSeries(
            np.asarray([], "datetime64[D]"), empty, empty, empty, empty, empty, empty
        )
    d = np.asarray(daily["date"], dtype="datetime64[D]")
    other = (
        (daily["source"] != "optionmetrics").astype(np.int64)
        if "source" in daily
        else np.zeros(len(d), np.int64)
    )
    order = np.lexsort((other, d))
    d = d[order]
    first = np.ones(len(d), dtype=bool)
    first[1:] = d[1:] != d[:-1]
    keep = order[first]
    d = d[first]
    spot = np.asarray(daily["spot"][keep], dtype=np.float64)
    iv = np.asarray(daily["iv30_atm"][keep], dtype=np.float64)
    bias = vrp.vol_bias(vrp.WINDOW)
    rv = vrp.forward_realised(spot, d, unbiased=True)
    trail = vrp.trailing_realised(spot) / bias
    prem = vrp.premium_vol_pts(iv, rv)
    return PremiumSeries(
        date=d,
        iv30=iv,
        rv_fwd=rv,
        rv_trailing=trail,
        premium=prem,
        smooth=trailing_mean(prem, smooth),
        priced=vrp.premium_vol_pts(iv, trail),
    )


@dataclass(frozen=True, slots=True)
class PremiumNow:
    """Where the premium stands at the end of the panel."""

    day: date  # the last day with an IV30
    iv30: float
    rv_trailing: float
    priced: float  # vol points: IV30 less the 21 days before, known on the day
    measured_date: date | None  # the last day whose forward window is complete
    measured: float  # vol points, that day's premium
    recent: float  # vol points, the trailing quarter's mean at ``measured_date``
    recent_share_below: float  # share of all quarters with a lower mean


def premium_now(s: PremiumSeries) -> PremiumNow | None:
    has_iv = np.flatnonzero(np.isfinite(s.iv30))
    if not len(has_iv):
        return None
    i = int(has_iv[-1])
    done = np.flatnonzero(np.isfinite(s.premium))
    j = int(done[-1]) if len(done) else None
    recent = float(s.smooth[j]) if j is not None else float("nan")
    return PremiumNow(
        day=_day(s.date[i]),
        iv30=float(s.iv30[i]),
        rv_trailing=float(s.rv_trailing[i]),
        priced=float(s.priced[i]),
        measured_date=_day(s.date[j]) if j is not None else None,
        measured=float(s.premium[j]) if j is not None else float("nan"),
        recent=recent,
        recent_share_below=share_below(s.smooth, recent),
    )


def axis_place(value: float, bracket: Sequence[float | None], axis: Sequence[float]) -> str:
    """Where ``value`` sits on the sweep's premium axis, in words."""
    lo, hi = [*bracket, None, None][:2]
    if lo is not None and lo == value:
        return f"at the sweep's {lo:g} vol-point level"
    if lo is not None and hi is not None:
        return f"between the sweep's {lo:g} and {hi:g} vol-point levels"
    if hi is not None:
        return f"below the sweep's lowest level ({hi:g})"
    if lo is not None:
        return f"above the sweep's highest level ({lo:g})"
    return f"off the sweep's axis ({', '.join(f'{a:g}' for a in axis)})"


def premium_reading(
    mean: float,
    low: float | None,
    high: float | None,
    first: str,
    last: str,
    bracket: Sequence[float | None],
    axis: Sequence[float],
    now: PremiumNow | None = None,
) -> str:
    """Question 1, "what premium does the market pay for volatility?", in one sentence."""
    span = f"{first[:4]} to {last[:4]}" if first[:4] != last[:4] else first[:4]
    interval = ""
    if low is not None and high is not None:
        zero = "so it could be zero" if low <= 0.0 <= high else "clear of zero"
        interval = f" (95% interval {signed(low)} to {signed(high)}, {zero})"
    out = (
        f"From {span} the market priced 30-day volatility {signed(mean)} vol points above "
        f"the volatility that followed, on average{interval}: "
        f"{axis_place(mean, bracket, axis)}"
    )
    if now is not None and math.isfinite(now.recent) and now.measured_date is not None:
        rank = now.recent_share_below
        side = "above" if rank >= 0.5 else "below"
        share = rank if rank >= 0.5 else 1.0 - rank
        out += (
            f". The latest quarter, to {now.measured_date.isoformat()}, averaged "
            f"{signed(now.recent)}, {'higher' if side == 'above' else 'lower'} than "
            f"{math.floor(share * 100)}% of quarters"
        )
    return out + "."


# The surface's shape on one day, read off the build's grid at fixed points.
SHAPE_TENOR = 30  # calendar days: the IV30 horizon
SHAPE_PUT_K = -0.10  # ln(K / F): a put about 10% below the forward
SHAPE_CALL_K = 0.05  # a call about 5% above it
SHAPE_LONG_TENOR = 365  # the term structure's far end


@dataclass(frozen=True, slots=True)
class ShapeHistory:
    """Per gridded date, the surface's level, skew and term slope, in vol points.

    ``atm`` is the 30-day at-the-money IV; ``skew`` the 30-day put about 10% below
    the forward less ``atm`` (how much more downside protection costs); ``call`` the
    30-day call about 5% above less ``atm``; ``term`` the year-out ATM IV less ``atm``
    (positive: upward sloping, the calm shape; negative: inverted, as in a sell-off).
    NaN where the grid has no quote at the point.
    """

    date: NDArray[np.datetime64]
    atm: NDArray[np.float64]
    skew: NDArray[np.float64]
    call: NDArray[np.float64]
    term: NDArray[np.float64]
    tenor: int
    put_k: float
    call_k: float
    long_tenor: int


def _grid_point(
    grid: Columns, dates: NDArray[np.datetime64], tenor: int, k: float
) -> NDArray[np.float64]:
    out = np.full(len(dates), np.nan)
    if not n_rows(grid):
        return out
    mine = (grid["tenor_days"].astype(np.int64) == tenor) & (
        np.abs(grid["k"].astype(np.float64) - k) < 1e-6
    )
    d = grid["date"][mine].astype("datetime64[D]")
    idx = np.searchsorted(dates, d)
    ok = (idx < len(dates)) & (dates[np.minimum(idx, len(dates) - 1)] == d)
    out[idx[ok]] = grid["iv"][mine].astype(np.float64)[ok] * 100.0
    return out


def shape_history(
    grid: Columns,
    tenor: int = SHAPE_TENOR,
    put_k: float = SHAPE_PUT_K,
    call_k: float = SHAPE_CALL_K,
    long_tenor: int = SHAPE_LONG_TENOR,
) -> ShapeHistory:
    dates = grid_dates(grid)
    atm = _grid_point(grid, dates, tenor, 0.0)
    return ShapeHistory(
        date=dates,
        atm=atm,
        skew=_grid_point(grid, dates, tenor, put_k) - atm,
        call=_grid_point(grid, dates, tenor, call_k) - atm,
        term=_grid_point(grid, dates, long_tenor, 0.0) - atm,
        tenor=tenor,
        put_k=put_k,
        call_k=call_k,
        long_tenor=long_tenor,
    )


@dataclass(frozen=True, slots=True)
class ShapeDay:
    date: date
    atm: float  # vol points
    skew: float
    call: float
    term: float
    atm_share_below: float  # the share of days with a lower level
    skew_share_below: float  # ... a flatter skew
    term_share_below: float  # ... a lower (more inverted) term slope
    share_inverted: float  # the share of days with an inverted term structure
    skew_median: float  # the typical day's skew, vol points
    term_median: float  # the typical day's term slope
    tenor: int
    put_k: float
    call_k: float
    long_tenor: int


def shape_on(h: ShapeHistory, day: date) -> ShapeDay | None:
    hit = np.flatnonzero(h.date == np.datetime64(day, "D"))
    if not len(hit):
        return None
    i = int(hit[0])
    term = h.term[np.isfinite(h.term)]
    return ShapeDay(
        date=day,
        atm=float(h.atm[i]),
        skew=float(h.skew[i]),
        call=float(h.call[i]),
        term=float(h.term[i]),
        atm_share_below=share_below(h.atm, float(h.atm[i])),
        skew_share_below=share_below(h.skew, float(h.skew[i])),
        term_share_below=share_below(h.term, float(h.term[i])),
        share_inverted=float((term < 0).mean()) if len(term) else float("nan"),
        skew_median=_median(h.skew),
        term_median=_median(h.term),
        tenor=h.tenor,
        put_k=h.put_k,
        call_k=h.call_k,
        long_tenor=h.long_tenor,
    )


def _median(x: NDArray[np.float64]) -> float:
    v = x[np.isfinite(x)]
    return float(np.median(v)) if len(v) else float("nan")


def _rank_words(share: float, more: str, less: str) -> str:
    """``steeper than on 80% of days`` or ``flatter than on 70% of days``."""
    if share >= 0.5:
        return f"{more} than on {math.floor(share * 100)}% of days"
    return f"{less} than on {math.floor((1.0 - share) * 100)}% of days"


def shape_reading(s: ShapeDay) -> str:
    """Question 2, "what does the surface look like on a day?", in one sentence."""
    if not math.isfinite(s.atm):
        return f"On {s.date.isoformat()} the grid has no {s.tenor}-day at-the-money quote."
    put_pct = round((1.0 - math.exp(s.put_k)) * 100)
    out = (
        f"On {s.date.isoformat()} {s.tenor}-day at-the-money vol was {s.atm:.1f} "
        f"({_rank_words(s.atm_share_below, 'higher', 'lower')})"
    )
    if math.isfinite(s.skew):
        out += (
            f"; a put {put_pct}% below the forward cost {s.skew:.1f} vol points more than "
            f"at the money ({s.skew_median:.1f} on the typical day)"
        )
    if math.isfinite(s.term):
        far = "a year out" if s.long_tenor == 365 else f"at {s.long_tenor} days"
        if s.term < 0:
            out += (
                f"; and the term structure was inverted, {far} {abs(s.term):.1f} below the "
                f"{s.tenor}-day, as on only {s.share_inverted:.0%} of days (a stressed market)"
            )
        else:
            out += (
                f"; and the term structure sloped up, {far} {s.term:.1f} above the "
                f"{s.tenor}-day ({signed(s.term_median, 1)} typical)"
            )
    return out + "."


# Where the simulator and the market differ: real less simulated over the simulator's
# own contracts, by moneyness.
NEAR_K = 0.03  # |ln(K / F)| within this is "near the money"
REGIONS = (("put", "the put wing"), ("atm", "near the money"), ("call", "the call wing"))


@dataclass(frozen=True, slots=True)
class Region:
    name: str  # "put", "atm" or "call"
    label: str
    mean_diff: float  # vol points, real less simulated; NaN with no quoted contract
    n: int  # contracts the real surface prices


@dataclass(frozen=True, slots=True)
class Widest:
    """The contract where real and simulated differ most."""

    row: int  # tenor index
    col: int  # strike index
    diff: float  # vol points, real less simulated
    real: float  # vol points
    sim: float


def sim_regions(
    k: NDArray[np.float64],
    real: NDArray[np.float64],
    sim: NDArray[np.float64],
    near: float = NEAR_K,
) -> tuple[tuple[Region, ...], Widest | None]:
    """Mean real-less-simulated per moneyness region, and the widest gap, in vol points."""
    k = np.asarray(k, dtype=np.float64)
    diff = (np.asarray(real, dtype=np.float64) - np.asarray(sim, dtype=np.float64)) * 100.0
    ok = np.isfinite(diff)
    masks = {"put": k < -near, "atm": np.abs(k) <= near, "call": k > near}
    regions = []
    for name, label in REGIONS:
        m = masks[name] & ok
        regions.append(
            Region(name, label, float(diff[m].mean()) if m.any() else float("nan"), int(m.sum()))
        )
    widest = None
    if ok.any():
        flat = int(np.argmax(np.where(ok, np.abs(diff), -np.inf)))
        r, c = divmod(flat, diff.shape[1])
        widest = Widest(
            r, c, float(diff[r, c]), float(real[r, c]) * 100.0, float(sim[r, c]) * 100.0
        )
    return tuple(regions), widest


def sim_reading(
    level: float,
    regions: Sequence[Region],
    widest: Widest | None,
    strike: float | None = None,
    tenor_trading_days: int | None = None,
) -> str:
    """Question 3, "how does the simulator differ from the market?", in one sentence."""
    head = f"The simulator prices every contract at one level, {level * 100:.1f} vol points"
    known = [r for r in regions if math.isfinite(r.mean_diff)]
    if not known or widest is None:
        return head + "; the real surface quotes none of its contracts on this day."
    most = max(known, key=lambda r: abs(r.mean_diff))
    word = "above" if most.mean_diff >= 0 else "below"
    out = (
        f"{head}; the market is furthest from it in {most.label}, "
        f"{abs(most.mean_diff):.1f} vol points {word} on average"
    )
    rest = [r for r in known if r is not most]
    if rest:
        out += " (" + ", ".join(f"{r.label} {signed(r.mean_diff, 1)}" for r in rest) + ")"
    if strike is not None:
        where = f"the {strike:,.0f} strike" + (
            f" at {tenor_trading_days} trading days" if tenor_trading_days else ""
        )
        out += f"; the widest gap is {where}, real {widest.real:.1f} against {widest.sim:.1f}"
    put = next((r for r in known if r.name == "put"), None)
    call = next((r for r in known if r.name == "call"), None)
    if put is not None and call is not None and put.mean_diff - call.mean_diff > 1.0:
        out += (
            ". The gap falls from puts to calls: that slope is the market's skew, which "
            "the simulator does not have"
        )
    return out + "."


def seller_reading(
    day: str,
    spot: float,
    tenor_trading_days: int,
    short_put: float,
    short_call: float,
    long_put: float,
    long_call: float,
    credit: float,
    max_loss: float | None,
    margin: float,
    breakevens: Sequence[float],
    short_delta: float,
    naked: bool = False,
) -> str:
    """Question 4, "what would the seller have opened that day?", in one sentence.

    ``max_loss`` is the loss the legs cap; ``naked`` says the frozen margin check still
    margins them as naked shorts (#39), so the margin posted is above that cap."""
    below = (1.0 - short_put / spot) * 100.0
    above = (short_call / spot - 1.0) * 100.0
    out = (
        f"At the close of {day}, with SPX at {spot:,.0f}, the seller would sell the "
        f"{short_put:,.0f} put and the {short_call:,.0f} call (about {below:.0f}% below and "
        f"{above:.0f}% above, the {short_delta * 100:.0f}-delta strikes), buy the "
        f"{long_put:,.0f} put and {long_call:,.0f} call as wings, and collect "
        f"${credit:,.0f} for {tenor_trading_days} trading days"
    )
    if len(breakevens) == 2:
        out += f"; it keeps money if SPX ends between {breakevens[0]:,.0f} and {breakevens[1]:,.0f}"
    if max_loss is not None:
        out += f", can lose at most ${max_loss:,.0f}"
    if naked and max_loss is not None:
        out += (
            f", and posts ${margin:,.0f} of margin: the wings cap the loss, but the frozen "
            "harness margins this condor as naked shorts (#39)."
        )
    else:
        out += f", and posts ${margin:,.0f} of margin."
    return out


def quality_reading(
    n_days: int,
    first: str,
    last: str,
    n_missing: int,
    seam_days: int,
    seam_p95: float | None,
    panel_version: int | None,
    of_record: bool | None,
    seam_worst: float | None = None,
) -> str:
    """The data-quality section's one line: coverage, the seam and the build.

    ``seam_p95`` is the seam's headline (``SEAM_HEADLINE``): each overlap day's 95th
    percentile IV gap, averaged over the days; ``seam_worst`` the largest day's."""
    missing = "none missing" if n_missing == 0 else f"{n_missing:,} missing"
    out = f"{n_days:,} trading days from {first} to {last}, {missing}"
    if seam_days and seam_p95 is not None and math.isfinite(seam_p95):
        out += (
            f"; on the {seam_days} days both vendors quote, their IVs differ by "
            f"{seam_p95:.3f} vol points at the 95th percentile on the average day"
        )
        if seam_worst is not None and math.isfinite(seam_worst):
            out += f" ({seam_worst:.3f} on the worst)"
    if panel_version is not None:
        out += f"; panel version {panel_version}"
        if of_record is True:
            out += ", the panel of record"
        elif of_record is False:
            out += ", but its files are not the ones docs/DATA_MANIFEST.json records"
    return out + "."
