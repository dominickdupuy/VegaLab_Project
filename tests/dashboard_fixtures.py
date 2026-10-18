"""Builders for the dashboard test fixtures: an SPX panel in miniature, in the W6-3 schema.

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
import pytest

# Panel fixture: two OptionMetrics days before the overlap, three shared days, two
# Databento days after; Databento misses the middle shared day.
OM_DAYS = (date(2025, 6, 27), date(2025, 6, 30), date(2025, 7, 1), date(2025, 7, 2))
OM_DAYS += (date(2025, 7, 3),)
DB_DAYS = (date(2025, 7, 1), date(2025, 7, 3), date(2025, 7, 7), date(2025, 7, 8))
DB_MISSING = (date(2025, 7, 2),)
CHAIN_DTES = (7, 30, 60, 120)


# -- the SPX panel ------------------------------------------------------------ #


def chain_columns(days: tuple[date, ...], source: str, spot: float = 6200.0) -> dict[str, Any]:
    cols: dict[str, list[Any]] = {k: [] for k in _CHAIN_KEYS}
    for i, day in enumerate(days):
        s = spot * (1.0 + 0.002 * i)
        for dte in CHAIN_DTES:
            t = dte / 365.0
            fwd = s * float(np.exp(0.04 * t))
            root = "SPXW" if dte == 7 else "SPX"
            for k in np.arange(0.8, 1.2001, 0.025) * fwd:
                x = float(np.log(k / fwd))
                iv = 0.16 + 0.02 * np.sqrt(t) - 0.15 * x + 0.5 * x * x
                for cp in ("C", "P"):
                    mid = max(0.05, abs(fwd - k) * 0.1 + 20.0 * iv)
                    row = {
                        "date": day,
                        "root": root,
                        "expiry": day + timedelta(days=dte),
                        "dte": dte,
                        "cp": cp,
                        "strike": round(float(k), 0),
                        "bid": mid * 0.98,
                        "ask": mid * 1.02,
                        "mid": mid,
                        "volume": 10.0,
                        "open_interest": None if source == "databento" else 100.0,
                        "forward": fwd,
                        "discount": float(np.exp(-0.04 * t)),
                        "iv": iv + (0.002 if source == "databento" else 0.0),
                        "iv_vendor": iv + 0.001 if source == "optionmetrics" else None,
                        "delta": 0.5,
                        "gamma": 0.001,
                        "vega": 1.0,
                        "theta": -1.0,
                        "tau": t,
                        "carry_fit": 0.9999,
                        "source": source,
                    }
                    for key in _CHAIN_KEYS:
                        cols[key].append(row[key])
    return cols


_CHAIN_KEYS = (
    "date",
    "root",
    "expiry",
    "dte",
    "cp",
    "strike",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_interest",
    "forward",
    "discount",
    "iv",
    "iv_vendor",
    "delta",
    "gamma",
    "vega",
    "theta",
    "tau",
    "carry_fit",
    "source",
)


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
    """Write the fixture panel under ``root/panel``; needs pyarrow."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    panel = root / "panel"
    chain = merged(chain_columns(OM_DAYS, "optionmetrics"), chain_columns(DB_DAYS, "databento"))
    by_year: dict[int, list[int]] = {}
    for i, d in enumerate(chain["date"]):
        by_year.setdefault(d.year, []).append(i)
    for year, idx in by_year.items():
        target = panel / "spx" / f"year={year}" / "part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({k: [v[i] for i in idx] for k, v in chain.items()})
        pq.write_table(table, target)
    daily = merged(
        daily_columns(date(2025, 1, 2), 130, "optionmetrics", seed=1),
        daily_columns(date(2025, 7, 1), 30, "databento", seed=2),
    )
    pq.write_table(pa.table(daily), panel / "spx_daily.parquet")
    pq.write_table(pa.table(coverage_columns()), panel / "coverage.parquet")
    pq.write_table(pa.table(seam_columns()), panel / "seam_overlap.parquet")
    return root
