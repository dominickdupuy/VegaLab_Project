"""Fit the cost model's two parameters to real SPX closing spreads.

Lane: Hemdutt Rao (market layer).

    python scripts/calibrate_costs.py [--years 2023 2025] [--out FILE]

The model stays what it is (``optspread.costs.QuotedSpreadCost``): a half-spread of
``max(min_half_spread, half_spread_frac * mid)`` per share, crossed on every leg,
plus commission. Only its two numbers are fitted, on the contracts the environment
actually trades: out-of-the-money legs between 5 and 16 delta (the wing and the
short strike) at 20 to 70 calendar days, which brackets the 21 and 42 trading-day
tenors. The quote is the closing consolidated best bid and offer, so the fitted
half-spread is what crossing the displayed market costs; fills inside it would
cost less, which makes the fit conservative rather than flattering.

The fit minimises absolute error in the half-spread over a grid of both
parameters, so a few wide quotes cannot drag it. It prints the fit, the error by
delta bucket and tenor, and the current parameters' error for comparison, and
writes the same tables to ``--out``. Nothing in the package changes until the
fitted numbers are written into ``CostConfig`` by hand, in their own commit; that
was done on 2026-10-25 (#8), so "current" now prints the fitted pair, and
docs/COSTS.md keeps the parameters from before the freeze in that column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from optspread.config import CostConfig
from optspread.realdata import data_root

COLUMNS = ["date", "cp", "dte", "bid", "ask", "mid", "delta", "source"]
DELTA_BAND = (0.05, 0.16)
DTE_BAND = (20, 70)
FLOORS = np.round(np.arange(0.0, 1.0001, 0.01), 2)
FRACS = np.round(np.arange(0.0, 0.10001, 0.0025), 4)


def load(years: range) -> pd.DataFrame:  # pragma: no cover - reads the shared folder
    root = data_root() / "panel" / "spx"
    frames = [
        pd.read_parquet(root / f"year={y}" / "part.parquet", columns=COLUMNS)
        for y in years
        if (root / f"year={y}" / "part.parquet").exists()
    ]
    return pd.concat(frames, ignore_index=True)


def trade_region(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel["delta"].abs()
    keep = (
        d.between(*DELTA_BAND)
        & panel["dte"].between(*DTE_BAND)
        & (panel["bid"] > 0)
        & (panel["ask"] > panel["bid"])
    )
    out = panel[keep].copy()
    out["half"] = 0.5 * (out["ask"] - out["bid"])
    out["delta_bucket"] = pd.cut(d[keep], [0.05, 0.08, 0.12, 0.16], include_lowest=True)
    out["tenor"] = np.where(out["dte"] <= 45, "~21 trading days", "~42 trading days")
    return out


def model_half(mid: np.ndarray, floor: float, frac: float) -> np.ndarray:
    return np.asarray(np.maximum(floor, frac * mid), dtype=np.float64)


def fit(region: pd.DataFrame) -> tuple[float, float, float]:
    """Grid-search the (floor, frac) pair with the least mean absolute error."""
    mid = region["mid"].to_numpy(dtype=np.float64)
    half = region["half"].to_numpy(dtype=np.float64)
    best = (np.inf, 0.0, 0.0)
    for frac in FRACS:
        scaled = frac * mid
        for floor in FLOORS:
            err = float(np.mean(np.abs(np.maximum(floor, scaled) - half)))
            if err < best[0]:
                best = (err, float(floor), float(frac))
    return best[1], best[2], best[0]


def table(region: pd.DataFrame, floor: float, frac: float, current: CostConfig) -> pd.DataFrame:
    r = region.assign(
        fitted=model_half(region["mid"].to_numpy(), floor, frac),
        current=model_half(
            region["mid"].to_numpy(), current.min_half_spread, current.half_spread_frac
        ),
    )
    g = r.groupby(["tenor", "delta_bucket"], observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "median mid": g["mid"].median(),
            "median half-spread": g["half"].median(),
            "fitted": g["fitted"].median(),
            "current": g["current"].median(),
        }
    ).round(3)


def markdown_table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--years", type=int, nargs=2, default=(2023, 2025))
    # No default file: docs/COSTS.md also holds a hand-written section on the gates,
    # so writing it is a choice made on the command line, not a side effect.
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    region = trade_region(load(range(args.years[0], args.years[1] + 1)))
    floor, frac, mae = fit(region)
    current = CostConfig()
    cur_mae = float(
        np.mean(
            np.abs(
                model_half(
                    region["mid"].to_numpy(), current.min_half_spread, current.half_spread_frac
                )
                - region["half"].to_numpy()
            )
        )
    )
    tab = table(region, floor, frac, current)
    lines = [
        "# Cost model calibration",
        "",
        "Owner: Hemdutt Rao. Built by `scripts/calibrate_costs.py`; its docstring has the method.",
        "",
        f"Data: real SPX closing quotes, {args.years[0]} to {args.years[1]}, "
        f"{len(region):,} contract-days between {DELTA_BAND[0]:.2f} and {DELTA_BAND[1]:.2f} "
        f"delta at {DTE_BAND[0]} to {DTE_BAND[1]} calendar days "
        f"({', '.join(sorted(region['source'].unique()))}).",
        "",
        "| | min_half_spread | half_spread_frac | mean abs error per share |",
        "|---|---|---|---|",
        f"| current | {current.min_half_spread:.2f} | {current.half_spread_frac:.4f} "
        f"| {cur_mae:.3f} |",
        f"| fitted | {floor:.2f} | {frac:.4f} | {mae:.3f} |",
        "",
        "Median half-spread per share by tenor and delta, observed against each model:",
        "",
        markdown_table(tab.reset_index()),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
