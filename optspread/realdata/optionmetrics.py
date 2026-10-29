"""WRDS OptionMetrics, as Dominick's extraction left it: ``opprcd`` and ``secprd`` by year.

Lane: Hemdutt Rao (data adapter).

The extraction (``scripts/wrds_extract.py``) writes one zstd Parquet per (table,
ticker, year) under the shared folder. This module reads the SPX files and puts
them into the panel's quote schema: strikes in index points rather than
thousandths, one of two roots (``SPX`` or ``SPXW``, see ``normalise_opprcd``), and
the vendor's IV, greeks and forward kept under ``*_vendor`` names so they are never
mistaken for ours.

The vendor stores ``strike_price`` as strike x 1000, but a copy may already have
divided it out (the team's Hugging Face mirror, ``Algogators/options_data``, did).
The unit is read from the data rather than assumed: ``strike_divisor`` compares
the median strike with the year's median close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from optspread.realdata import ROOTS
from optspread.realdata.panel import NYSE_HOLIDAYS

OPPRCD_COLUMNS = [
    "date",
    "symbol",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "am_settlement",
    "forward_price",
]


def opprcd_path(root: Path, year: int, ticker: str = "SPX") -> Path:
    return root / "opprcd" / f"year={year}" / f"{ticker}.parquet"


def secprd_path(root: Path, year: int, ticker: str = "SPX") -> Path:
    return root / "secprd" / f"year={year}" / f"{ticker}.parquet"


def strike_divisor(strike: pd.Series, close: pd.Series) -> float:
    """1000 for the vendor's strike x 1000, 1 for strikes already in index points.

    Listed strikes sit within a few multiples of the underlying, so the ratio of the
    medians is near 1 in index points and near 1000 in the vendor's unit. Anything
    else is neither, and is refused rather than guessed.
    """
    ratio = float(np.nanmedian(strike.to_numpy(dtype=np.float64))) / float(
        np.nanmedian(close.to_numpy(dtype=np.float64))
    )
    if 1 / 5 < ratio < 5:
        return 1.0
    if 200 < ratio < 5000:
        return 1000.0
    raise ValueError(f"median strike is {ratio:.4g}x the median close: neither unit")


def normalise_opprcd(raw: pd.DataFrame, divisor: float = 1000.0) -> pd.DataFrame:
    """Vendor rows in the panel's quote schema (see module docstring).

    The panel has two roots, and they mean settlement: ``SPX`` is the AM-settled
    monthly, ``SPXW`` the PM-settled family (weeklies, quarterlies, end-of-month).
    Every row takes its root from the vendor's settlement flag, AM to SPX and PM to
    SPXW, whatever its symbol says. An OSI symbol whose root agrees with its flag is
    *listed*; every other row is *remapped*:

    - in 2010 the PM-settled Friday weeklies are OSI symbols on the SPX root
      (``SPX 100528C850000``, 3,990 rows, 2010-05-21 to 2010-12-02), so the symbol's
      root alone would put PM contracts under the AM root (data audit MINOR-6);
    - before the OSI migration (completed 2010-02-12) the vendor's ``symbol`` is the
      old exchange code (``SPX.AT``, ``SPZ.MJ``), one code per contract, which says
      nothing about the root, and through 2010 the vendor still carries other
      root codes beside SPX and SPXW;
    - to 2014 the PM-settled series had roots of their own, ``SPXPM`` (third-Friday
      monthlies settled at the close) and ``SPXQ`` (end-of-quarter), both later
      folded into SPXW. They settle and price as SPXW does, so ``iv30``'s "prefer
      SPXW" rule and the dashboard's root filter should read them as SPXW.

    A remapped row that lands on a contract already listed under the same root
    (same date, expiry, right and strike) is the same contract quoted twice and is
    dropped; the parity fit would otherwise count its pair twice.

    Until February 2015 the vendor dates an AM-settled monthly on the Saturday after
    its last trading day. It settles at the open of that trading day, so those
    expiries move to it: the Friday, or the Thursday when the Friday is a closure
    (Good Friday, 2014-04-18). Left on the Saturday, every tau is a day long (about
    2% low on a 30-day IV, 7% on a 7-day one). An expiry the vendor dates on a weekday
    the exchange was closed (SPXW 2018-12-05 and 2025-01-09, national days of mourning,
    ``panel.UNSCHEDULED_CLOSURES``) settled on the trading day before, and moves there
    too (``_settlement_day``).
    """
    sym = raw["symbol"].astype("string")
    osi = sym.str.contains(" ", regex=False).fillna(False).to_numpy(dtype=bool)
    am = raw["am_settlement"].fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    vendor_root = sym.str.split(" ", n=1).str[0].to_numpy()
    root = np.where(am, ROOTS[0], ROOTS[1])  # SPX is AM, SPXW is PM
    agrees = pd.Series(vendor_root, dtype="string").eq(pd.Series(root, dtype="string"))
    listed = osi & agrees.fillna(False).to_numpy(dtype=bool)
    bid = raw["best_bid"].to_numpy(dtype=np.float64)
    ask = raw["best_offer"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]).dt.date,
            "root": root,
            "expiry": _settlement_day(pd.to_datetime(raw["exdate"]), am).dt.date,
            "cp": raw["cp_flag"].to_numpy(),
            "strike": raw["strike_price"].to_numpy(dtype=np.float64) / divisor,
            "bid": np.where(bid > 0.0, bid, np.nan),
            "ask": np.where(ask > 0.0, ask, np.nan),
            "volume": raw["volume"].to_numpy(dtype=np.float64),
            "open_interest": raw["open_interest"].to_numpy(dtype=np.float64),
            "iv_vendor": raw["impl_volatility"].to_numpy(dtype=np.float64),
            "delta_vendor": raw["delta"].to_numpy(dtype=np.float64),
            "forward_vendor": raw["forward_price"].to_numpy(dtype=np.float64),
            "am_settled": am,
        }
    )
    if listed.all():
        return out
    # Listed rows first, so a remapped duplicate is the one dropped; listed rows are
    # never dropped here, so a year with only SPX and SPXW symbols is unchanged.
    order = np.argsort(~listed, kind="stable")
    ranked = out.iloc[order]
    dup = ranked.duplicated(["date", "root", "expiry", "cp", "strike"]).to_numpy()
    return ranked[~(dup & ~listed[order])].sort_index().reset_index(drop=True)


def _settlement_day(expiry: pd.Series, am: np.ndarray) -> pd.Series:
    """The trading day an expiry settles on.

    A Saturday AM expiry settles at the open of the trading day before it. Any expiry
    dated on a weekday NYSE closure (``panel.NYSE_HOLIDAYS``, which lists the
    unscheduled ones) settles on the trading day before it as well: the exchange moved
    the SPXW expiries of 2018-12-05 and 2025-01-09 to the previous close, and the last
    quotes of each price it there.
    """
    days = expiry.to_numpy(dtype="datetime64[D]")
    holidays = np.asarray(NYSE_HOLIDAYS, dtype="datetime64[D]")
    saturday = (expiry.dt.weekday == 5).to_numpy() & am
    closed = np.isin(days, holidays) & (expiry.dt.weekday < 5).to_numpy()
    move = saturday | closed
    out = days.copy()
    out[move] = np.busday_offset(days[move], 0, roll="backward", holidays=holidays)
    return pd.Series(pd.to_datetime(out), index=expiry.index)


def read_opprcd(root: Path, year: int, ticker: str = "SPX") -> pd.DataFrame:  # pragma: no cover
    raw = pq.read_table(opprcd_path(root, year, ticker), columns=OPPRCD_COLUMNS).to_pandas()
    close = read_secprd(root, year, ticker)["close"]
    return normalise_opprcd(raw, strike_divisor(raw["strike_price"], close))


def read_secprd(root: Path, year: int, ticker: str = "SPX") -> pd.DataFrame:  # pragma: no cover
    table = pq.read_table(secprd_path(root, year, ticker), columns=["date", "close"])
    frame = pd.DataFrame(table.to_pandas())
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame
