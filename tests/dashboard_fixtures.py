"""Builders for the dashboard test fixtures: an SPX panel in miniature, in the W6-3 schema.

The chain is not written column by column: priced quotes go through the real
builder (``optspread.realdata``), so the views are tested on the schema the build
writes, string ``carry_fit`` label and all, and the fixture cannot drift from it.

The panel has a smile skewed the way SPX is (puts richer than calls), two sources
overlapping on a few days, one day flagged missing, a daily table whose VRP is
positive on average, and the seam statistics over the overlap. Every number is
synthetic; the shapes are what the views are tested against. Plain functions, so
the test modules import them directly; ``conftest.py`` wraps them as fixtures.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from optspread.pricing import bs_price
from optspread.realdata.panel import PANEL_COLUMNS, atm_term, finalise, surface_grid
from optspread.realdata.surface import price_surface

# Panel fixture: two OptionMetrics days before the overlap, three shared days, two
# Databento days after; Databento misses the middle shared day.
OM_DAYS = (date(2025, 6, 27), date(2025, 6, 30), date(2025, 7, 1), date(2025, 7, 2))
OM_DAYS += (date(2025, 7, 3),)
DB_DAYS = (date(2025, 7, 1), date(2025, 7, 3), date(2025, 7, 7), date(2025, 7, 8))
DB_MISSING = (date(2025, 7, 2),)
CHAIN_DTES = (7, 30, 60, 120)
RATE = 0.04


# -- the SPX panel ------------------------------------------------------------ #


def chain_frame(days: tuple[date, ...], source: str, spot: float = 6200.0) -> pd.DataFrame:
    """Panel rows built the way the real build builds them: quotes, priced, finalised.

    The quotes are priced by ``optspread.pricing`` on a skewed smile and go through
    ``price_surface`` and ``finalise``, so the columns and their types (``tau``, the
    ``carry_fit`` label, ``iv`` solved from the mid) are the builder's and cannot drift
    from it. Databento quotes 0.2 vol points over OptionMetrics, which alone carries a
    vendor IV, one tenth of a point over ours.
    """
    rows: list[dict[str, Any]] = []
    for i, day in enumerate(days):
        s = spot * (1.0 + 0.002 * i)
        for dte in CHAIN_DTES:
            t = dte / 365.0
            fwd = s * float(np.exp(RATE * t))
            root = "SPXW" if dte == 7 else "SPX"
            for k in np.arange(0.8, 1.2001, 0.025) * fwd:
                strike = round(float(k), 0)
                x = float(np.log(strike / fwd))
                iv = 0.16 + 0.02 * np.sqrt(t) - 0.15 * x + 0.5 * x * x
                iv += 0.002 if source == "databento" else 0.0
                for cp in ("C", "P"):
                    mid = float(bs_price(cp, fwd * np.exp(-RATE * t), strike, RATE, iv, t))
                    half = max(0.05, 0.01 * mid)
                    rows.append(
                        {
                            "date": day,
                            "root": root,
                            "expiry": day + timedelta(days=dte),
                            "cp": cp,
                            "strike": strike,
                            "bid": mid - half if mid - half > 0.05 else np.nan,
                            "ask": mid + half,
                            "am_settled": False,
                            "volume": 10.0,
                            "open_interest": np.nan if source == "databento" else 100.0,
                        }
                    )
    priced = price_surface(pd.DataFrame(rows))
    if source == "optionmetrics":
        priced["iv_vendor"] = priced["iv"] + 0.001
    panel, _ = finalise(priced, source)
    return panel


def chain_columns(days: tuple[date, ...], source: str, spot: float = 6200.0) -> dict[str, Any]:
    frame = chain_frame(days, source, spot)
    return {k: frame[k].tolist() for k in PANEL_COLUMNS}


def daily_columns(start: date, n_days: int, source: str, seed: int = 0) -> dict[str, list[Any]]:
    rng = np.random.default_rng(seed)
    days = [
        start + timedelta(days=i) for i in range(n_days * 2) if (start + timedelta(i)).weekday() < 5
    ]
    days = days[:n_days]
    iv30 = 0.18 + 0.03 * np.sin(np.arange(n_days) / 20.0)
    rv_fwd = iv30 - 0.03 + 0.02 * rng.standard_normal(n_days)
    rv_fwd[-3:] = np.nan  # the forward window runs past the end
    return {
        "date": days,
        "spot": list(5000.0 + np.arange(n_days, dtype=float)),
        "spot_parity": list(5000.5 + np.arange(n_days, dtype=float)),
        "carry30": [0.03] * n_days,
        "rate30": [RATE] * n_days,
        "iv30_atm": list(iv30),
        "rv21": list(iv30 - 0.02),
        "rv21_fwd": list(rv_fwd),
        "vrp30": list(iv30 - rv_fwd),
        "source": [source] * n_days,
        "n_quotes": [1000] * n_days,
    }


def coverage_columns() -> dict[str, list[Any]]:
    rows: list[tuple[date, str, int, bool]] = []
    rows += [(d, "optionmetrics", 2000, False) for d in OM_DAYS]
    rows += [(d, "databento", 2400, False) for d in DB_DAYS]
    rows += [(d, "databento", 0, True) for d in DB_MISSING]
    return {
        "date": [r[0] for r in rows],
        "source": [r[1] for r in rows],
        "n_contracts": [r[2] for r in rows],
        "missing": [r[3] for r in rows],
    }


def seam_columns() -> dict[str, list[Any]]:
    """The seam report over the shared days: both sources matched contract for contract."""
    shared = sorted(set(OM_DAYS) & set(DB_DAYS))
    n = len(shared)
    return {
        "date": shared,
        "n_matched": [1800 + i for i in range(n)],
        "iv_diff_median_volpts": [0.2] * n,
        "iv_diff_p95_abs_volpts": [0.9 + 0.1 * i for i in range(n)],
        "mid_diff_median_rel": [0.004] * n,
        "forward_diff_median_bp": [1.5] * n,
        "ours_vs_vendor_median_volpts": [-0.1] * n,
    }


def merged(*parts: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {k: [v for p in parts for v in p[k]] for k in parts[0]}


def write_panel(root: Path) -> Path:
    """Write the fixture panel under ``root/panel``, every file the real build writes."""
    pytest.importorskip("pyarrow")
    panel = root / "panel"
    chain = pd.concat(
        [chain_frame(OM_DAYS, "optionmetrics"), chain_frame(DB_DAYS, "databento")],
        ignore_index=True,
    )
    years = np.asarray([d.year for d in chain["date"]])
    for year in sorted(set(years.tolist())):
        target = panel / "spx" / f"year={year}" / "part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        chain[years == year].to_parquet(target, index=False, compression="zstd")
    daily = merged(
        daily_columns(date(2025, 1, 2), 130, "optionmetrics", seed=1),
        daily_columns(date(2025, 7, 1), 30, "databento", seed=2),
    )
    pd.DataFrame(daily).to_parquet(panel / "spx_daily.parquet", index=False)
    pd.DataFrame(coverage_columns()).to_parquet(panel / "coverage.parquet", index=False)
    pd.DataFrame(seam_columns()).to_parquet(panel / "seam_overlap.parquet", index=False)
    atm_term(chain).to_parquet(panel / "atm_term.parquet", index=False)
    surface_grid(chain).to_parquet(panel / "surface_grid.parquet", index=False)
    return root
