"""The stitched SPX panel, its daily series, its coverage and the seam between vendors.

Lane: Hemdutt Rao (data adapter).

One row per (date, root, expiry, right, strike) with a two-sided closing quote,
priced by ``optspread.realdata.surface``. OptionMetrics supplies every date up to
its ceiling and Databento every date after it; over the months both cover, both
are priced and compared (``seam_frame``) but only OptionMetrics enters the panel,
so no date has two answers.

Kept: ``|ln(K / F)| <= 0.7`` and at most 800 calendar days to expiry. Outside that
band the quotes are stale or one-sided often enough to add noise and size without
adding information the simulator or the VRP series uses. Dropped rows are counted
in the coverage table, never silently.

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

MAX_ABS_LOG_MONEYNESS = 0.7
MAX_TAU = 800.0 / 365.0
TARGET_TAU = 30.0 / 365.0
RV_WINDOW = 21
TRADING_DAYS = 252
K_GRID = np.round(np.arange(-0.5, 0.3 + 1e-9, 0.025), 4)  # log-moneyness ln(K / F)
TENOR_GRID = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)  # calendar days

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

# NYSE full-day closures after the OptionMetrics ceiling. Before it, the vendor's
# own price file is the calendar.
NYSE_HOLIDAYS = tuple(
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
HOLIDAYS_THROUGH = date(2027, 12, 31)


def finalise(priced: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.Series]:
    """Panel rows from priced quotes, and per date how many the band dropped."""
    q = priced
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log(q["strike"].to_numpy(dtype=np.float64) / q["forward"].to_numpy(dtype=np.float64))
    keep = (
        q["mid"].notna().to_numpy()
        & np.isfinite(k)
        & (np.abs(k) <= MAX_ABS_LOG_MONEYNESS)
        & (q["tau"].to_numpy() <= MAX_TAU)
    )
    out = q[keep].copy()
    out["dte"] = (pd.to_datetime(out["expiry"]) - pd.to_datetime(out["date"])).dt.days
    out["source"] = source
    for col in ("volume", "open_interest", "iv_vendor"):
        if col not in out:
            out[col] = np.nan
    dropped = pd.Series(~keep, index=q.index).groupby(q["date"].to_numpy()).sum().astype(np.int64)
    return out[PANEL_COLUMNS].reset_index(drop=True), dropped.rename("n_dropped")


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
    t = term.sort_values(["date", "tau", "root"], ascending=[True, True, False])
    t = t.drop_duplicates(["date", "expiry"], keep="first")
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
    d = d.join(carry30(term, d["spot"]))
    d.index.name = "date"
    return d.reset_index()


def carry30(term: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    """Per date, from the expiry nearest 30 days: the carry ``ln(F / S) / T`` and the rate.

    The simulator's pricer has no dividend yield, so a real chain hands it the carry
    (rate less dividends) as ``r``: that puts the engine's forward where the market's
    is. The discount rate itself is kept beside it for the record.
    """
    t = term.assign(gap=(term["tau"] - TARGET_TAU).abs()).sort_values(["date", "gap"])
    near = t.drop_duplicates("date").set_index("date")
    s = spot.reindex(near.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        carry = np.log(near["forward"] / s) / near["tau"]
        rate = -np.log(near["discount"]) / near["tau"]
    return pd.DataFrame({"carry30": carry, "rate30": rate})


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
