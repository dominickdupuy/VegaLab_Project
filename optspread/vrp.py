"""The real market's volatility risk premium, measured on the sweep's axis (#15).

Lane: Hemdutt Rao (market layer: the VRP mapping).

The simulator plants its premium as ``iv = sigma + iv_premium_vol_pts / 100`` on
every contract while the path is driven by ``sigma`` (``optspread.market``). The
planted premium is therefore implied volatility minus the volatility the path then
realises, in vol points, and the real analogue is the same difference with the
realised side measured over the window that follows the quote:

    premium_t = 100 * (IV30_t - RV_fwd_t / c_21)
    RV_fwd_t  = sqrt(252 / 21 * sum_{i=1..21} r_{t+i}^2),   r = close-to-close log return

21 trading days is the simulator's front tenor (``MarketConfig().tenor_days[0]``)
and 252 its year (``trading_days_per_year``). The mean is zero rather than the
sample mean because the simulator's path has zero drift and a delta-hedged seller's
P&L accrues on squared returns, not on squared deviations. ``c_21 = vol_bias(21)``
removes the Jensen gap: a vol measured over 21 Gaussian returns reads low on
average, and the simulator's premium is against the path's true sigma, so without
it the same estimator run on the simulator would overstate the planted premium.
The sensitivities: uncorrected; the panel's own ``rv21_fwd`` (sample standard
deviation, ddof 1, ``demean=True``); a calendar-matched window (the 30 calendar
days IV30 covers, annualised over 365 like the IV); and the premium taken in
variance units and converted to vol points at a stated sigma, which needs no
correction because the mean of a squared return is unbiased.

Uncertainty is a stationary bootstrap (Politis and Romano 1994). The daily premium
is serially dependent twice over: consecutive days share 20 of their 21 forward
returns, which makes it at least an MA(20) mechanically, and volatility clusters on
top of that. The mean block length is the Politis and White (2004) automatic choice
(with the Patton, Politis and White 2009 correction), floored at the window, so no
block is shorter than the overlap it has to keep together. Percentile intervals.

The power curve is read at the mapped premium and at its interval ends in the way the
M4 analysis builds it (``optspread.m4analysis``): ``power_at_noncentrality`` takes the
curve's knots (the anchor at zero and one per sweep premium, with the noncentrality
each implies), holds the noncentrality linear in premium between them and returns the
exact noncentral-t power there, which is the number the analysis's own interpolated
curve holds at every one of its points. ``power_at`` reads a curve given as sampled
powers (the sidecar's ``interpolated`` list, drawn every 0.05 vol points), linearly
between its samples as the dashboard draws it. Neither asks the curve to rise with the
premium: at S = 5 a curve estimated from one t statistic per premium dips by sampling
noise (the real PPO curve falls from 0.475 at 0.25 vol points to 0.346 at 0.5, and from
1.000 at 2 to 0.991 at 4), and a dip is the curve, not a defect in it. So each also
reports the lowest and the highest power over the interval (``power_min``,
``power_max``), which is the interval of the power; on a curve that rises they are the
powers at the interval's ends. Only a sampled power outside [0, 1], or knots that are
not a curve, are refused.

Every function here works on numpy arrays; reading the panel is the script's job
(``scripts/vrp_mapping.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nct, norm

from optspread.config import MarketConfig
from optspread.power import t_critical
from optspread.realdata import HOLDOUT_START

_MARKET = MarketConfig()
WINDOW = _MARKET.tenor_days[0]  # 21 trading days: the front tenor
TRADING_DAYS = _MARKET.trading_days_per_year  # 252
SIGMA = _MARKET.sigma  # the simulator's path vol, for the variance-unit conversion
IV_CALENDAR_DAYS = 30  # IV30 is ACT/365 over 30 calendar days (optspread.realdata.panel)
MAX_WINDOW_SPAN_DAYS = 35  # 21 trading days span at most ~32 calendar days with holidays
N_BOOT = 10_000
SEED = 15  # the issue number; any fixed seed makes the report reproducible
ALPHA = 0.05
HOLDOUT_AFTER = HOLDOUT_START - timedelta(days=1)  # the last evaluation date (A8)

Dates = NDArray[np.datetime64]
Floats = NDArray[np.float64]


# -- the forward-realised series ---------------------------------------------- #


def log_returns(spot: Floats) -> Floats:
    """``r[t] = ln(S[t] / S[t-1])``; ``r[0]`` is NaN."""
    s = np.asarray(spot, dtype=np.float64)
    out = np.full(s.shape, np.nan)
    out[1:] = np.diff(np.log(s))
    return out


def forward_realised(
    spot: Floats,
    dates: Dates | None = None,
    window: int = WINDOW,
    trading_days: int = TRADING_DAYS,
    demean: bool = False,
    unbiased: bool = False,
) -> Floats:
    """Annualised close-to-close vol over the ``window`` returns after each date.

    ``out[t]`` uses ``r[t+1] .. r[t+window]``: the moves after the close at ``t``,
    which is when IV30 is quoted. NaN where the window runs past the data, holds a
    NaN return, or (given ``dates``) spans more than ``MAX_WINDOW_SPAN_DAYS``
    calendar days, which only a hole in the data can do.

    ``demean=False`` is the zero-mean root mean square, the simulator's definition.
    ``demean=True`` is the sample standard deviation (ddof 1), the panel's
    ``rv21_fwd``. ``unbiased=True`` divides the zero-mean figure by
    ``vol_bias(window)``, so that on a path of constant vol it is that vol in
    expectation (see ``vol_bias``).
    """
    r = log_returns(spot)
    n = r.size
    out = np.full(n, np.nan)
    if n <= window:
        return out
    w = np.lib.stride_tricks.sliding_window_view(r[1:], window)  # row t: r[t+1 .. t+window]
    var = w.var(axis=1, ddof=1) if demean else (w * w).mean(axis=1)
    out[: n - window] = np.sqrt(var * trading_days)
    if unbiased and not demean:
        out /= vol_bias(window)
    if dates is not None:
        d = np.asarray(dates, dtype="datetime64[D]")
        span = (d[window:] - d[: n - window]).astype(np.int64)
        out[: n - window][span > MAX_WINDOW_SPAN_DAYS] = np.nan
    return out


def forward_realised_calendar(
    spot: Floats, dates: Dates, calendar_days: int = IV_CALENDAR_DAYS, unbiased: bool = False
) -> Floats:
    """Annualised vol over the returns dated in ``(t, t + calendar_days]``, over 365.

    The window IV30 itself covers: its total variance ``IV30^2 * 30 / 365`` is what
    the market expects the squared returns of the next 30 calendar days to sum to,
    so this compares the two in the same units with no trading-day convention. NaN
    where the data end before the window does. ``unbiased`` divides by
    ``vol_bias`` of each window's own count of returns.
    """
    r = log_returns(spot)
    d = np.asarray(dates, dtype="datetime64[D]")
    if not d.size:
        return np.full(0, np.nan)
    bad = ~np.isfinite(r)
    cum = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, r * r))])  # sum of r[:j]^2
    n_bad = np.concatenate([[0], np.cumsum(bad)])
    end = d + np.timedelta64(calendar_days, "D")
    hi = np.searchsorted(d, end, side="right")  # first index dated after the window
    lo = np.arange(d.size) + 1
    out = np.sqrt((cum[hi] - cum[lo]) * 365.0 / calendar_days)
    count = hi - lo
    out[(end > d[-1]) | (n_bad[hi] > n_bad[lo]) | (count < 1)] = np.nan
    if unbiased:
        out /= np.array([vol_bias(int(k)) if k > 0 else 1.0 for k in count])
    return np.asarray(out, dtype=np.float64)


def vol_bias(k: int) -> float:
    """``E[sqrt(mean of k squared N(0, s^2) draws)] / s``: below 1, by Jensen.

    ``sqrt(2 / k) * Gamma((k + 1) / 2) / Gamma(k / 2)``, about ``1 - 1 / (4 k)``;
    0.9882 at k = 21. The simulator's premium is implied less the path's sigma, and a
    realised vol over k Gaussian returns reads sigma times this on average, so the raw
    difference ``IV - RV`` overstates the planted premium by ``(1 - vol_bias) * sigma``
    (0.18 vol points at 15% vol). Dividing ``RV`` by it removes that.
    """
    return math.sqrt(2.0 / k) * math.exp(math.lgamma((k + 1) / 2.0) - math.lgamma(k / 2.0))


def trailing_realised(
    spot: Floats, window: int = WINDOW, trading_days: int = TRADING_DAYS
) -> Floats:
    """Zero-mean vol over ``r[t-window+1] .. r[t]``: what had happened, for contrast."""
    r = log_returns(spot)
    out = np.full(r.size, np.nan)
    if r.size <= window:
        return out
    w = np.lib.stride_tricks.sliding_window_view(r[1:], window)
    out[window:] = np.sqrt((w * w).mean(axis=1) * trading_days)
    return out


def window_end(dates: Dates, window: int = WINDOW) -> Dates:
    """The date of the last return in each forward window; NaT past the data."""
    d = np.asarray(dates, dtype="datetime64[D]")
    out = np.full(d.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    if d.size > window:
        out[: d.size - window] = d[window:]
    return out


def premium_vol_pts(iv: Floats, rv: Floats) -> Floats:
    """Implied less realised, in vol points: the quantity ``iv_premium_vol_pts`` plants."""
    return np.asarray(100.0 * (np.asarray(iv) - np.asarray(rv)), dtype=np.float64)


def variance_premium(iv: Floats, rv: Floats) -> Floats:
    """``IV^2 - RV^2``, annualised variance: the Carr and Wu (2009) units."""
    iv_, rv_ = np.asarray(iv, dtype=np.float64), np.asarray(rv, dtype=np.float64)
    return np.asarray(iv_ * iv_ - rv_ * rv_, dtype=np.float64)


def variance_to_vol_pts(v: float, sigma: float = SIGMA) -> float:
    """The planted premium ``p`` whose variance premium at ``sigma`` is ``v``.

    The simulator prices at ``sigma + p`` and realises ``sigma``, so its variance
    premium is ``(sigma + p)^2 - sigma^2``; inverted, ``p = sqrt(sigma^2 + v) - sigma``.
    Monotone in ``v``, so it maps a mean, a median and an interval's ends alike.
    """
    return 100.0 * (math.sqrt(max(sigma * sigma + v, 0.0)) - sigma)


# -- the bootstrap ------------------------------------------------------------ #


def politis_white_block(x: Floats) -> float:
    """The Politis and White (2004) optimal mean block for the stationary bootstrap.

    With the Patton, Politis and White (2009) correction (``D_SB = 2 g(0)^2``). The
    bandwidth ``M`` is twice the first lag after which ``K_N = max(5, sqrt(log10 n))``
    consecutive autocorrelations are insignificant, with the flat-top lag window; capped
    at ``min(3 sqrt(n), n / 3)`` as in the paper. An autocorrelation is insignificant
    under ``1.96 sqrt(log10 n / n)``, the two-sided 95% bound, where the paper
    suggests ``c = 2``.
    """
    e = np.asarray(x, dtype=np.float64)
    e = e[np.isfinite(e)]
    n = e.size
    if n < 8:
        return 1.0
    e = e - e.mean()
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = min(math.ceil(math.sqrt(n)) + k_n, n - 1)
    acv = np.array([e[k:] @ e[: n - k] / n for k in range(m_max + 1)])
    if acv[0] <= 0.0:
        return 1.0
    rho = np.abs(acv[1:] / acv[0])
    small = rho < norm.ppf(0.975) * math.sqrt(math.log10(n) / n)
    m_hat: int | None = None
    run = 0
    for lag, ok in enumerate(small, start=1):
        run = run + 1 if ok else 0
        if run == k_n:
            m_hat = lag - k_n
            break
    if m_hat is None:  # never settles inside m_max: the last significant lag
        big = np.flatnonzero(~small)
        m_hat = int(big[-1]) + 1 if big.size else 1
    m = max(1, min(2 * m_hat, m_max))
    g, lr = 0.0, float(acv[0])
    for k in range(1, m + 1):
        lam = 1.0 if k / m <= 0.5 else 2.0 * (1.0 - k / m)
        g += 2.0 * lam * k * float(acv[k])
        lr += 2.0 * lam * float(acv[k])
    if lr <= 0.0:
        return float(b_max)
    b = (2.0 * g * g / (2.0 * lr * lr)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), b_max))


def block_length(x: Floats, window: int = WINDOW) -> float:
    """The mean block used here: Politis-White, never shorter than the overlap."""
    return max(float(window), politis_white_block(x))


def stationary_indices(
    n: int, block: float, n_boot: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """``(n_boot, n)`` resample indices: blocks of geometric length, mean ``block``.

    Each position starts a new block with probability ``1 / block`` (always at 0),
    at a uniform start, and otherwise continues the previous one, wrapping at ``n``.
    """
    p = 1.0 / max(block, 1.0)
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.arange(n)
    head = np.maximum.accumulate(np.where(new, pos, 0), axis=1)  # where each block began
    begin = np.take_along_axis(starts, head, axis=1)
    return np.asarray((begin + (pos - head)) % n, dtype=np.int64)


def bootstrap_mean_median(
    x: Floats, block: float, n_boot: int = N_BOOT, seed: int = SEED, chunk: int = 500
) -> tuple[Floats, Floats]:
    """Stationary-bootstrap replicates of the mean and the median of ``x``."""
    v = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means, medians = [], []
    for start in range(0, n_boot, chunk):
        idx = stationary_indices(v.size, block, min(chunk, n_boot - start), rng)
        sample = v[idx]
        means.append(sample.mean(axis=1))
        medians.append(np.median(sample, axis=1))
    return np.concatenate(means), np.concatenate(medians)


@dataclass(frozen=True, slots=True)
class Estimate:
    """One sample's premium in vol points: mean and median, each with a 95% interval."""

    label: str
    n_days: int
    first: date | None
    last: date | None
    mean: float
    mean_low: float
    mean_high: float
    median: float
    median_low: float
    median_high: float
    share_positive: float
    block: float


def estimate(
    x: Floats,
    dates: Dates,
    label: str,
    block: float | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    to_vol_pts: Callable[[float], float] | None = None,
) -> Estimate:
    """Mean, median and their stationary-bootstrap percentile intervals.

    ``x`` is a daily series; NaN days are dropped first. ``to_vol_pts`` maps a
    statistic of ``x`` onto vol points after the fact (the variance-unit
    sensitivity); it must be increasing, so it carries the interval's ends with it.
    ``block`` defaults to ``block_length`` of the sample itself.
    """
    v = np.asarray(x, dtype=np.float64)
    d = np.asarray(dates, dtype="datetime64[D]")
    ok = np.isfinite(v)
    v, d = v[ok], d[ok]
    f = to_vol_pts or (lambda s: s)
    if v.size < 2:
        nan = float("nan")
        return Estimate(label, int(v.size), None, None, nan, nan, nan, nan, nan, nan, nan, nan)
    b = block_length(v) if block is None else block
    means, medians = bootstrap_mean_median(v, b, n_boot, seed)
    q = (100.0 * ALPHA / 2.0, 100.0 * (1.0 - ALPHA / 2.0))
    m_lo, m_hi = np.percentile(means, q)
    d_lo, d_hi = np.percentile(medians, q)
    return Estimate(
        label=label,
        n_days=int(v.size),
        first=as_date(d.min()),
        last=as_date(d.max()),
        mean=f(float(v.mean())),
        mean_low=f(float(m_lo)),
        mean_high=f(float(m_hi)),
        median=f(float(np.median(v))),
        median_low=f(float(d_lo)),
        median_high=f(float(d_hi)),
        share_positive=float((v > 0).mean()),
        block=float(b),
    )


# -- sub-periods -------------------------------------------------------------- #


def before(ends: Dates, boundary: date) -> NDArray[np.bool_]:
    """Days whose whole forward window closes on or before ``boundary``."""
    e = np.asarray(ends, dtype="datetime64[D]")
    return np.asarray(~np.isnat(e) & (e <= np.datetime64(boundary, "D")), dtype=np.bool_)


def after(dates: Dates, boundary: date) -> NDArray[np.bool_]:
    """Days quoted after ``boundary``, so their windows lie wholly after it too."""
    d = np.asarray(dates, dtype="datetime64[D]")
    return np.asarray(d > np.datetime64(boundary, "D"), dtype=np.bool_)


def years(dates: Dates) -> NDArray[np.int64]:
    """Calendar year of each quote date (a window may run into the next year)."""
    d = np.asarray(dates, dtype="datetime64[D]")
    return np.asarray(d.astype("datetime64[Y]").astype(np.int64) + 1970, dtype=np.int64)


# -- the power curve at the mapped premium ------------------------------------ #


@dataclass(frozen=True, slots=True)
class MappedPower:
    """Detection probability at the mapped premium, at its interval's ends, and over it."""

    premium: float
    low: float
    high: float
    power: float
    power_low: float  # at the interval's lower end
    power_high: float  # at its upper end
    power_min: float  # the lowest power anywhere in the interval
    power_max: float  # the highest
    clipped: bool  # some premium lay outside the curve's grid and took the end value


def _axis(
    premiums: Sequence[float] | Floats, values: Sequence[float] | Floats, what: str
) -> tuple[Floats, Floats]:
    x = np.asarray(premiums, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError(f"premiums and {what} must be 1-D arrays of the same length, at least 2")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"premiums and {what} must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("premiums must be strictly increasing")
    return x, y


def _read(
    x: Floats, estimate: float, interval: tuple[float, float], at: Callable[[float], float]
) -> MappedPower:
    """``at`` at the point, at the interval's ends and at every knot inside the interval.

    Off the grid a premium is held at the end knot. ``at`` is linear between knots in
    whatever it interpolates, and power is monotone in that, so the extremes over the
    interval lie at its ends or at a knot inside it.
    """
    lo, hi = sorted(interval)

    def read(p: float) -> float:
        return at(float(np.clip(p, x[0], x[-1])))

    over = [read(lo), read(hi), *(read(float(k)) for k in x if lo < k < hi)]
    return MappedPower(
        premium=float(estimate),
        low=float(lo),
        high=float(hi),
        power=read(estimate),
        power_low=over[0],
        power_high=over[1],
        power_min=min(over),
        power_max=max(over),
        clipped=any(p < x[0] or p > x[-1] for p in (estimate, lo, hi)),
    )


def power_at(
    premiums: Sequence[float] | Floats,
    power: Sequence[float] | Floats,
    estimate: float,
    interval: tuple[float, float],
) -> MappedPower:
    """Read a sampled power curve at ``estimate``, at both ends of ``interval`` and over it.

    ``premiums`` must be strictly increasing and ``power`` in [0, 1]: detection
    probability against planted vol points, as the M4 sidecar's ``interpolated`` list
    holds it. The curve need not rise: a dip is read as it is, and ``power_min`` and
    ``power_max`` say how far the power ranges over the interval. Linear between
    samples; outside the grid it is held at the end value and ``clipped`` says so (below
    the grid that is the test's size, above it a lower bound).
    """
    x, y = _axis(premiums, power, "power")
    if y.min() < 0.0 or y.max() > 1.0:
        raise ValueError("power must lie in [0, 1]")
    return _read(x, estimate, interval, lambda p: float(np.interp(p, x, y)))


def noncentral_power(noncentrality: float, n_seeds: int, alpha: float = ALPHA) -> float:
    """``P(nct(S - 1, lambda) > t_{1-alpha, S-1})``, as ``m4analysis.power_at`` computes it."""
    if math.isinf(noncentrality):
        return 1.0 if noncentrality > 0.0 else 0.0
    if noncentrality == 0.0:
        return alpha
    return float(nct.sf(t_critical(n_seeds, alpha), n_seeds - 1, noncentrality))


def power_at_noncentrality(
    premiums: Sequence[float] | Floats,
    noncentrality: Sequence[float] | Floats,
    n_seeds: int,
    estimate: float,
    interval: tuple[float, float],
    alpha: float = ALPHA,
) -> MappedPower:
    """Read the M4 curve from its knots, through the analysis's own interpolation.

    ``premiums`` and ``noncentrality`` are the curve's knots, the anchor at (0, 0) first:
    the sidecar's ``curve.points``, with ``noncentrality`` (the curve),
    ``noncentrality_lo`` or ``noncentrality_hi`` (the band's lower and upper edges). The
    noncentrality is linear in premium between knots and the power is the exact
    noncentral t on ``n_seeds - 1`` df there, so this is the curve the analysis draws,
    not an approximation of it, and it is read as it is wherever it dips. Off the grid
    the end knot's value is held and ``clipped`` says so.
    """
    x, lam = _axis(premiums, noncentrality, "noncentrality")
    t_critical(n_seeds, alpha)  # range checks only
    return _read(
        x,
        estimate,
        interval,
        lambda p: noncentral_power(float(np.interp(p, x, lam)), n_seeds, alpha),
    )


def bracket(value: float, axis: Sequence[float]) -> tuple[float | None, float | None]:
    """The axis levels either side of ``value`` (None past an end)."""
    levels = sorted(axis)
    below = [a for a in levels if a <= value]
    above = [a for a in levels if a > value]
    return (below[-1] if below else None, above[0] if above else None)


def as_date(value: np.datetime64) -> date:
    """A numpy day as a ``datetime.date``."""
    out = value.astype("datetime64[D]").astype(date)
    assert isinstance(out, date)
    return out
