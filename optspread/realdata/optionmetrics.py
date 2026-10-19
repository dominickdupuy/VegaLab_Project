"""WRDS OptionMetrics, as Dominick's extraction left it: ``opprcd`` and ``secprd`` by year.

Lane: Hemdutt Rao (data adapter).

The extraction (``scripts/wrds_extract.py``) writes one zstd Parquet per (table,
ticker, year) under the shared folder. This module reads the SPX files and puts
them into the panel's quote schema: strikes in index points rather than
thousandths, the root read from the OSI symbol (``SPX`` or ``SPXW``), and the
vendor's IV, greeks and forward kept under ``*_vendor`` names so they are never
mistaken for ours.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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


def normalise_opprcd(raw: pd.DataFrame) -> pd.DataFrame:
    """Vendor rows in the panel's quote schema (see module docstring).

    Before the OSI migration (completed 2010-02-12) the vendor's ``symbol`` is the
    old exchange code (``SPX.AT``, ``SPZ.MJ``), one code per contract, which says
    nothing about the root. Those rows take their root from settlement instead:
    AM-settled is the SPX root, PM-settled the SPXW family.

    Until February 2015 the vendor dates an AM-settled monthly on the Saturday after
    its last trading day. It settles at Friday's open, so those expiries move to the
    Friday: left on the Saturday, every tau is a day long (about 2% low on a 30-day
    IV, 7% on a 7-day one).
    """
    sym = raw["symbol"].astype("string")
    osi = sym.str.contains(" ", regex=False).fillna(False).to_numpy(dtype=bool)
    am = raw["am_settlement"].fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    root = np.where(osi, sym.str.split(" ", n=1).str[0].to_numpy(), np.where(am, "SPX", "SPXW"))
    bid = raw["best_bid"].to_numpy(dtype=np.float64)
    ask = raw["best_offer"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]).dt.date,
            "root": root,
            "expiry": _settlement_day(pd.to_datetime(raw["exdate"]), am).dt.date,
            "cp": raw["cp_flag"].to_numpy(),
            "strike": raw["strike_price"].to_numpy(dtype=np.float64) / 1000.0,
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


def _settlement_day(expiry: pd.Series, am: np.ndarray) -> pd.Series:
    """A Saturday AM expiry settles at the open of the Friday before it."""
    saturday = (expiry.dt.weekday == 5).to_numpy() & am
    return expiry - pd.to_timedelta(np.where(saturday, 1, 0), unit="D")


def read_opprcd(root: Path, year: int, ticker: str = "SPX") -> pd.DataFrame:  # pragma: no cover
    table = pq.read_table(opprcd_path(root, year, ticker), columns=OPPRCD_COLUMNS)
    return normalise_opprcd(table.to_pandas())


def read_secprd(root: Path, year: int, ticker: str = "SPX") -> pd.DataFrame:  # pragma: no cover
    table = pq.read_table(secprd_path(root, year, ticker), columns=["date", "close"])
    frame = pd.DataFrame(table.to_pandas())
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame
