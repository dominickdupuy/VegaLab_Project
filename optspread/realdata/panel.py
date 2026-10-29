"""The stitched SPX panel, its daily series, its coverage and the seam between vendors.

Lane: Hemdutt Rao (data adapter).

One row per (date, root, expiry, right, strike) with a two-sided closing quote,
priced by ``optspread.realdata.surface``. OptionMetrics supplies every date up to
its ceiling and Databento every date after it; over the months both cover, both
are priced and compared (``seam_frame``) but only OptionMetrics enters the panel,
so no date has two answers.

Kept: ``|ln(K / F)| <= 0.7`` and at most 800 calendar days to expiry. Outside that
band the quotes are stale or one-sided often enough to add noise and size without
adding information the simulator or the VRP series uses. An expiry whose forward is
off the day's forward curve is stale and dropped whole (``stale_expiries``). Dropped
rows are counted in the coverage table, never silently.

The daily series: the 30-day at-the-money IV (OTM IVs interpolated to ``k = 0``
per expiry, then total variance interpolated to 30 days), close-to-close realised
volatility over the trailing and the following 21 trading days, and the variance
risk premium as implied minus the realised that followed, in volatility units.
That last series is the external anchor of the project document's section 4.5.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MAX_ABS_LOG_MONEYNESS = 0.7
MAX_TAU = 800.0 / 365.0
TARGET_TAU = 30.0 / 365.0
RV_WINDOW = 21
TRADING_DAYS = 252
K_GRID = np.round(np.arange(-0.5, 0.3 + 1e-9, 0.025), 4)  # log-moneyness ln(K / F)
TENOR_GRID = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)  # calendar days
# An expiry whose parity forward is further than this from the day's forward curve is
# stale (``stale_expiries``). One percent is half the 2018-02-05 gap; at 75 bp the rule
# would take 37 expiries on 34 days, nearly all the longest listed expiry in early 2010,
# where a straight line through six neighbours a few months apart misses the curve.
STALE_FORWARD_BP = 100.0

PANEL_COLUMNS = [
    "date",
    "root",
    "expiry",
    "dte",
    "tau",
    "cp",
    "strike",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_interest",
    "forward",
    "discount",
    "carry_fit",
    "iv",
    "iv_vendor",
    "delta",
    "gamma",
    "vega",
    "theta",
    "source",
]

# NYSE full-day closures, 2010 to 2027, in three blocks. 2010 to 2017: until February
# 2015 the vendor dates an AM-settled monthly on the Saturday after its last trading
# day, and the normaliser moves it to the trading day before
# (``optionmetrics._settlement_day``), which is a Thursday when the Friday is a closure
# (Good Friday, 2014-04-18). The block runs to 2017 so every Saturday expiry within
# ``MAX_TAU`` of a February 2015 date is covered. 2018 to the OptionMetrics ceiling: the
# vendor's own price file is the trading calendar of the panel, and this block is what
# a window's gap check and an expiry dated on a closure are read against. After the
# ceiling: the trading calendar (``trading_calendar``).
SATURDAY_ERA_HOLIDAYS = tuple(
    date.fromisoformat(d)
    for d in (
        "2010-01-01",
        "2010-01-18",
        "2010-02-15",
        "2010-04-02",
        "2010-05-31",
        "2010-07-05",
        "2010-09-06",
        "2010-11-25",
        "2010-12-24",
        "2011-01-17",
        "2011-02-21",
        "2011-04-22",
        "2011-05-30",
        "2011-07-04",
        "2011-09-05",
        "2011-11-24",
        "2011-12-26",
        "2012-01-02",
        "2012-01-16",
        "2012-02-20",
        "2012-04-06",
        "2012-05-28",
        "2012-07-04",
        "2012-09-03",
        "2012-10-29",  # Hurricane Sandy, two days
        "2012-10-30",
        "2012-11-22",
        "2012-12-25",
        "2013-01-01",
        "2013-01-21",
        "2013-02-18",
        "2013-03-29",
        "2013-05-27",
        "2013-07-04",
        "2013-09-02",
        "2013-11-28",
        "2013-12-25",
        "2014-01-01",
        "2014-01-20",
        "2014-02-17",
        "2014-04-18",
        "2014-05-26",
        "2014-07-04",
        "2014-09-01",
        "2014-11-27",
        "2014-12-25",
        "2015-01-01",
        "2015-01-19",
        "2015-02-16",
        "2015-04-03",
        "2015-05-25",
        "2015-07-03",
        "2015-09-07",
        "2015-11-26",
        "2015-12-25",
        "2016-01-01",
        "2016-01-18",
        "2016-02-15",
        "2016-03-25",
        "2016-05-30",
        "2016-07-04",
        "2016-09-05",
        "2016-11-24",
        "2016-12-26",
        "2017-01-02",
        "2017-01-16",
        "2017-02-20",
        "2017-04-14",
        "2017-05-29",
        "2017-07-04",
        "2017-09-04",
        "2017-11-23",
        "2017-12-25",
    )
)
# Closures no exchange calendar scheduled: Hurricane Sandy (also in the block above),
# the national days of mourning for George H. W. Bush and Jimmy Carter. The vendor
# keeps an expiry dated on one (SPXW 2018-12-05, SPXW 2025-01-09); it settled at the
# close of the trading day before, which is where its last quotes price it (an ATM IV
# of 3.1% and 1.0% with a day left on the clock), so the normaliser moves it there.
UNSCHEDULED_CLOSURES = tuple(
    date.fromisoformat(d) for d in ("2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09")
)
VENDOR_ERA_HOLIDAYS = tuple(
    date.fromisoformat(d)
    for d in (
        "2018-01-01",
        "2018-01-15",
        "2018-02-19",
        "2018-03-30",
        "2018-05-28",
        "2018-07-04",
        "2018-09-03",
        "2018-11-22",
        "2018-12-05",  # George H. W. Bush
        "2018-12-25",
        "2019-01-01",
        "2019-01-21",
        "2019-02-18",
        "2019-04-19",
        "2019-05-27",
        "2019-07-04",
        "2019-09-02",
        "2019-11-28",
        "2019-12-25",
        "2020-01-01",
        "2020-01-20",
        "2020-02-17",
        "2020-04-10",
        "2020-05-25",
        "2020-07-03",
        "2020-09-07",
        "2020-11-26",
        "2020-12-25",
        "2021-01-01",
        "2021-01-18",
        "2021-02-15",
        "2021-04-02",
        "2021-05-31",
        "2021-07-05",
        "2021-09-06",
        "2021-11-25",
        "2021-12-24",
        "2022-01-17",
        "2022-02-21",
        "2022-04-15",
        "2022-05-30",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2022-12-26",
        "2023-01-02",
        "2023-01-16",
        "2023-02-20",
        "2023-04-07",
        "2023-05-29",
        "2023-06-19",
        "2023-07-04",
        "2023-09-04",
        "2023-11-23",
        "2023-12-25",
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-03-29",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-11-28",
        "2024-12-25",
        "2025-01-01",
        "2025-01-09",  # Jimmy Carter
        "2025-01-20",
        "2025-02-17",
        "2025-04-18",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
    )
)
NYSE_HOLIDAYS = (
    SATURDAY_ERA_HOLIDAYS
    + VENDOR_ERA_HOLIDAYS
    + tuple(
        date.fromisoformat(d)
        for d in (
            "2025-09-01",
            "2025-11-27",
            "2025-12-25",
            "2026-01-01",
            "2026-01-19",
            "2026-02-16",
            "2026-04-03",
            "2026-05-25",
            "2026-06-19",
            "2026-07-03",
            "2026-09-07",
            "2026-11-26",
            "2026-12-25",
            "2027-01-01",
            "2027-01-18",
            "2027-02-15",
            "2027-03-26",
            "2027-05-31",
            "2027-06-18",
            "2027-07-05",
            "2027-09-06",
            "2027-11-25",
            "2027-12-24",
        )
    )
)
HOLIDAYS_FROM = date(2010, 1, 1)
HOLIDAYS_THROUGH = date(2027, 12, 31)


def finalise(priced: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.Series]:
    """Panel rows from priced quotes, and per date how many the band and the stale-expiry
    rule (``stale_expiries``) dropped."""
    q = priced
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log(q["strike"].to_numpy(dtype=np.float64) / q["forward"].to_numpy(dtype=np.float64))
    keys = ["date", "root", "expiry"]
    groups = q[[*keys, "tau", "forward"]].drop_duplicates(keys)
    stale = groups[stale_expiries(groups)][keys].assign(_stale=True)
    is_stale = q[keys].merge(stale, on=keys, how="left")["_stale"].notna()
    keep = (
        q["mid"].notna().to_numpy()
        & np.isfinite(k)
        & (np.abs(k) <= MAX_ABS_LOG_MONEYNESS)
        & (q["tau"].to_numpy() <= MAX_TAU)
        & ~is_stale.to_numpy(dtype=bool)
    )
    out = q[keep].copy()
    out["dte"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days
    out["source"] = source
    for col in ("volume", "open_interest", "iv_vendor"):
        if col not in out:
            out[col] = np.nan
    dropped = pd.Series(~keep, index=q.index).groupby(q["date"].to_numpy()).sum().astype(np.int64)
    return out[PANEL_COLUMNS].reset_index(drop=True), dropped.rename("n_dropped")


def stale_expiries(groups: pd.DataFrame) -> NDArray[np.bool_]:
    """Which expiries quote a forward off the day's forward curve: stale quotes.

    ``groups`` has one row per (date, root, expiry) with its ``tau`` and parity
    ``forward``. Each expiry is compared with the line through its six nearest expiries
    in tau on the day (itself left out): the line's slope is the median of their
    pairwise slopes of ``ln F`` against ``tau``, its level the median of their ``ln F``
    carried to this tau at that slope, so a minority of stale neighbours moves neither.
    The line is local because the carry curves over two years (a slope fitted to the
    whole day misses a long expiry by more than the threshold on hundreds of days). An
    expiry more than ``STALE_FORWARD_BP`` from it is stale, and every quote on it is
    dropped. On the v3 panel's ATM term structure (``atm_term.parquet``) the rule takes
    six expiries on three days: the four SPXW expiries of 2018-02-05 whose forwards sit
    about 208 bp below every other (quotes left from earlier in a day the index fell 4%,
    their ATM IVs 5 to 7 vol points rich), and the longest SPX expiry on 2010-02-10 and
    on 2012-05-08 (1.8 and 1.6 years, past the grid's last tenor, so neither the grid nor
    IV30 reads them).
    """
    out = np.zeros(len(groups), dtype=bool)
    g = groups.reset_index(drop=True)
    fwd = g["forward"].to_numpy(dtype=np.float64)
    ok = np.isfinite(fwd) & (fwd > 0)
    for _, day in g[ok].groupby("date", sort=False):
        y = np.log(day["forward"].to_numpy(dtype=np.float64))
        tau = day["tau"].to_numpy(dtype=np.float64)
        rows = day.index.to_numpy()
        for i in range(len(y)):
            others = np.delete(np.arange(len(y)), i)
            near = others[np.argsort(np.abs(tau[others] - tau[i]), kind="stable")[:6]]
            a, b = np.triu_indices(near.size, 1)
            dt = tau[near[b]] - tau[near[a]]
            apart = np.abs(dt) > 1e-9
            if near.size < 2 or not apart.any():
                continue
            slope = float(np.median((y[near[b]] - y[near[a]])[apart] / dt[apart]))
            level = float(np.median(y[near] + slope * (tau[i] - tau[near])))
            out[rows[i]] = abs(y[i] - level) * 1e4 > STALE_FORWARD_BP
    return out


def atm_term(panel: pd.DataFrame) -> pd.DataFrame:
    """At-the-money IV per (date, root, expiry), interpolated in log-moneyness.

    Uses the out-of-the-money side only (puts below the forward, calls at or above),
    the convention for a smile, and the nearest strike on each side of ``k = 0``.
    """
    p = panel[panel["iv"].notna()].copy()
    p["k"] = np.log(p["strike"] / p["forward"])
    otm = p[((p["cp"] == "C") & (p["k"] >= 0.0)) | ((p["cp"] == "P") & (p["k"] < 0.0))]
    keys = ["date", "root", "expiry"]
    below = otm[otm["k"] < 0.0]
    above = otm[otm["k"] >= 0.0]
    lo = below.loc[
        below.groupby(keys)["k"].idxmax(), [*keys, "tau", "forward", "discount", "k", "iv"]
    ]
    hi = above.loc[above.groupby(keys)["k"].idxmin(), [*keys, "k", "iv"]]
    both = lo.merge(hi, on=keys, suffixes=("_lo", "_hi"))
    w = (0.0 - both["k_lo"]) / (both["k_hi"] - both["k_lo"])
    both["atm_iv"] = both["iv_lo"] + w * (both["iv_hi"] - both["iv_lo"])
    return both[[*keys, "tau", "forward", "discount", "atm_iv"]].reset_index(drop=True)


def surface_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """IV per (date, tenor, log-moneyness) on the fixed grids: the compact surface.

    Each expiry's out-of-the-money smile is interpolated onto ``K_GRID`` (no
    extrapolation past its quoted strikes), then total variance is interpolated
    linearly in time onto ``TENOR_GRID`` between the bracketing expiries. A grid
    point with no quote on either side stays NaN. This is what the real chain
    provider and the surface view read, so neither has to load the full panel.
    """
    p = panel[panel["iv"].notna()].copy()
    p["k"] = np.log(p["strike"] / p["forward"])
    p = p[((p["cp"] == "C") & (p["k"] >= 0.0)) | ((p["cp"] == "P") & (p["k"] < 0.0))]
    p = p.sort_values(["date", "expiry", "root", "k"], ascending=[True, True, False, True])
    rows: list[pd.DataFrame] = []
    tenors = np.asarray(TENOR_GRID, dtype=np.float64) / 365.0
    for day, g in p.groupby("date", sort=True):
        taus: list[float] = []
        wvar: list[np.ndarray] = []
        for _, e in g.groupby("expiry", sort=True):
            e = e[e["root"] == e["root"].iloc[0]]  # one root per expiry date, SPXW first
            if len(e) < 3:
                continue
            tau = float(e["tau"].iloc[0])
            iv = np.interp(K_GRID, e["k"].to_numpy(), e["iv"].to_numpy(), left=np.nan, right=np.nan)
            taus.append(tau)
            wvar.append(iv * iv * tau)
        if len(taus) < 2:
            continue
        t_arr, w_arr = np.asarray(taus), np.vstack(wvar)
        hi = np.searchsorted(t_arr, tenors, side="right")
        exact = (hi > 0) & np.isclose(t_arr[np.clip(hi - 1, 0, len(t_arr) - 1)], tenors)
        ok = exact | ((hi > 0) & (hi < len(t_arr)))
        lo_i, hi_i = np.clip(hi - 1, 0, len(t_arr) - 1), np.clip(hi, 0, len(t_arr) - 1)
        frac = (tenors - t_arr[lo_i]) / np.where(ok & ~exact, t_arr[hi_i] - t_arr[lo_i], 1.0)
        w = w_arr[lo_i] + frac[:, None] * (w_arr[hi_i] - w_arr[lo_i])
        w[exact] = w_arr[lo_i[exact]]  # a tenor on an expiry is that expiry, not a blend
        iv_grid = np.sqrt(np.maximum(w, 0.0) / tenors[:, None])
        iv_grid[~ok] = np.nan
        rows.append(
            pd.DataFrame(
                {
                    "date": day,
                    "tenor_days": np.repeat(np.asarray(TENOR_GRID), len(K_GRID)),
                    "k": np.tile(K_GRID, len(TENOR_GRID)),
                    "iv": iv_grid.ravel(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def iv30(term: pd.DataFrame) -> pd.Series:
    """30-day ATM IV per date: total variance linear in time between bracketing expiries.

    Where both roots list the same expiry the PM-settled SPXW row is used. A date with
    no expiry on one side of 30 days has no value rather than an extrapolated one.
    """
    # Choose the root per (date, expiry) first: sorting by tau first would keep the
    # AM-settled SPX row, whose tau is 6.5 hours shorter on a shared third Friday.
    t = term.sort_values(["date", "expiry", "root"], ascending=[True, True, False])
    t = t.drop_duplicates(["date", "expiry"], keep="first").sort_values(["date", "tau"])
    t = t.assign(w=t["atm_iv"] ** 2 * t["tau"])
    near = t[t["tau"] <= TARGET_TAU].groupby("date").last()
    far = t[t["tau"] > TARGET_TAU].groupby("date").first()
    j = near[["tau", "w"]].join(far[["tau", "w"]], lsuffix="_1", rsuffix="_2", how="inner")
    frac = (TARGET_TAU - j["tau_1"]) / (j["tau_2"] - j["tau_1"])
    w30 = j["w_1"] + frac * (j["w_2"] - j["w_1"])
    return pd.Series(np.sqrt(np.maximum(w30, 0.0) / TARGET_TAU), index=j.index, name="iv30_atm")


def parity_spot(panel: pd.DataFrame) -> pd.Series:
    """The prepaid forward of each date's shortest listed expiry, one day out or more."""
    g = panel[panel["tau"] >= 1.0 / 365.0].sort_values(["date", "tau"])
    first = g.drop_duplicates("date", keep="first")
    return pd.Series(
        (first["forward"] * first["discount"]).to_numpy(), index=first["date"], name="spot_parity"
    )


def daily_table(
    term: pd.DataFrame, spot: pd.Series, spot_parity: pd.Series, sources: pd.Series, n: pd.Series
) -> pd.DataFrame:
    """The daily series the VRP anchor and the dashboard read."""
    d = pd.DataFrame({"spot": spot}).join(spot_parity, how="outer")
    d["spot"] = d["spot"].fillna(d["spot_parity"])
    d = d.join(iv30(term)).join(sources.rename("source")).join(n.rename("n_quotes"))
    d = d.sort_index()
    ret = np.log(d["spot"]).diff()
    d["rv21"] = ret.rolling(RV_WINDOW).std() * np.sqrt(TRADING_DAYS)
    d["rv21_fwd"] = d["rv21"].shift(-RV_WINDOW)
    d["vrp30"] = d["iv30_atm"] - d["rv21_fwd"]
    d = d.join(carry30(term, d["spot_parity"].fillna(d["spot"])))
    d.index.name = "date"
    return d.reset_index()


def carry30(term: pd.DataFrame, spot_parity: pd.Series) -> pd.DataFrame:
    """Per date, from the expiry nearest 30 days: the carry ``ln(F / S) / T`` and the rate.

    The simulator's pricer has no dividend yield, so a real chain hands it the carry
    (rate less dividends) as ``r``: that puts the engine's forward where the market's
    is. ``S`` is the parity spot (``parity_spot``), the spot the option snapshot itself
    implies, and the provider prices on that spot too (panel version 4). Against the
    index close, as before, the carry also carried the gap between the close and the
    option snapshot, divided by 30 days: +13.3% on 2011-11-01 (a 74 bp gap) and -18.3%
    on 2020-03-13 (142 bp), where rate less dividends was a few percent; against the
    parity spot the series runs from about -4% to +6% (data audit MINOR-3). The column
    is ``carry30_parity`` so a table built the old way (``carry30``) is never read as
    this one. The discount rate itself is kept beside it for the record.
    """
    t = term.assign(gap=(term["tau"] - TARGET_TAU).abs()).sort_values(["date", "gap"])
    near = t.drop_duplicates("date").set_index("date")
    s = spot_parity.reindex(near.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        carry = np.log(near["forward"] / s) / near["tau"]
        rate = -np.log(near["discount"]) / near["tau"]
    return pd.DataFrame({"carry30_parity": carry, "rate30": rate})


def nyse_trading_days(first: date, last: date) -> list[date]:
    """NYSE trading days from ``first`` to ``last`` that the holiday list reaches
    (``HOLIDAYS_FROM`` to ``HOLIDAYS_THROUGH``); outside that, nothing rather than a
    guess, since an unlisted closure would read as a missing day."""
    lo, hi = max(first, HOLIDAYS_FROM), min(last, HOLIDAYS_THROUGH)
    if lo > hi:
        return []
    closed = set(NYSE_HOLIDAYS)
    return [d.date() for d in pd.bdate_range(lo, hi) if d.date() not in closed]


def trading_calendar(vendor_days: pd.Index, through: date) -> list[date]:
    """Vendor trading days up to its ceiling, then NYSE business days to ``through``.

    Refuses a date past the end of the holiday list rather than counting a closure
    as a missing day.
    """
    if through > HOLIDAYS_THROUGH:
        raise ValueError(f"NYSE_HOLIDAYS ends {HOLIDAYS_THROUGH}; extend it to reach {through}")
    last = max(vendor_days)
    after = pd.bdate_range(pd.Timestamp(last) + pd.Timedelta(days=1), pd.Timestamp(through))
    extra = [d.date() for d in after if d.date() not in NYSE_HOLIDAYS]
    return sorted([*vendor_days, *extra])


def coverage(calendar: list[date], counts: pd.DataFrame) -> pd.DataFrame:
    """One row per expected trading day: which source served it, how much, how much dropped."""
    cal = pd.DataFrame({"date": calendar})
    out = cal.merge(counts, on="date", how="left")
    out["n_contracts"] = out["n_contracts"].fillna(0).astype(np.int64)
    if "n_dropped" in out:
        out["n_dropped"] = out["n_dropped"].fillna(0).astype(np.int64)
    out["missing"] = out["n_contracts"] == 0
    out["source"] = out["source"].fillna("none")
    return out


def seam_frame(om: pd.DataFrame, dbn: pd.DataFrame) -> pd.DataFrame:
    """Per overlap day: the same contracts from both vendors, priced the same way."""
    keys = ["date", "root", "expiry", "cp", "strike"]
    cols = ["mid", "bid", "ask", "iv", "forward", "tau"]
    j = om[[*keys, *cols, "iv_vendor"]].merge(dbn[[*keys, *cols]], on=keys, suffixes=("_om", "_db"))
    j["k"] = np.log(j["strike"] / j["forward_om"])
    j["otm"] = ((j["cp"] == "C") & (j["k"] >= 0)) | ((j["cp"] == "P") & (j["k"] < 0))
    band = j[j["otm"] & (j["k"].abs() <= 0.2) & (j["tau_om"] > 7 / 365)]
    rows = []
    for day, g in band.groupby("date"):
        rows.append(
            {
                "date": day,
                "n_matched": len(g),
                "iv_diff_median_volpts": float((g["iv_db"] - g["iv_om"]).median() * 100),
                "iv_diff_p95_abs_volpts": float(
                    (g["iv_db"] - g["iv_om"]).abs().quantile(0.95) * 100
                ),
                "mid_diff_median_rel": float((g["mid_db"] / g["mid_om"] - 1).median()),
                "forward_diff_median_bp": float(
                    (g["forward_db"] / g["forward_om"] - 1).median() * 1e4
                ),
                "ours_vs_vendor_median_volpts": float((g["iv_om"] - g["iv_vendor"]).median() * 100),
            }
        )
    return pd.DataFrame(rows)
