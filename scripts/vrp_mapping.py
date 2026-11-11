"""The VRP mapping (#15): the real SPX premium in planted vol points, with its interval.

Lane: Hemdutt Rao (market layer: the VRP mapping).

    python -m scripts.vrp_mapping                                  # $VEGALAB_DATA's panel
    python -m scripts.vrp_mapping --daily /scratch/spx_daily.parquet --out docs/VRP_MAPPING

Reads ``panel/spx_daily.parquet`` (date, spot, iv30_atm, rv21_fwd, vrp30; see
docs/DATA.md) and writes ``<out>.md`` and ``<out>.json``. The definitions, the
bootstrap and the reasons for each choice are in ``optspread.vrp``; this script only
reads the panel, runs every estimate the report lists and renders it. The panel is
read once, with retries, since the shared folder is files-on-demand and a first read
can time out. Every number is from the real panel; nothing is synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from optspread import vrp
from optspread.dashboard.panel_views import Columns, PanelLayout, data_root, read_table
from optspread.experiment.grid import SWEEP_PREMIUMS

OUT = Path("docs/VRP_MAPPING")
MANIFEST = Path("docs/DATA_MANIFEST.json")
PANEL_FILE = "panel/spx_daily.parquet"
COLUMNS = ("date", "spot", "iv30_atm", "rv21_fwd", "vrp30", "source")
BLOCKS = (21.0, 63.0, 126.0, 252.0)  # the block-length sensitivity, in trading days
SPLIT_2020 = date(2019, 12, 31)


def read_daily(path: Path, attempts: int = 5, wait: float = 5.0) -> Columns:
    """The daily table, retried with backoff: a OneDrive placeholder can time out once."""
    for attempt in range(1, attempts + 1):
        try:
            return read_table(path, COLUMNS)
        except (OSError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            print(f"read failed ({exc}); retry {attempt} in {wait * attempt:.0f}s", flush=True)
            time.sleep(wait * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digest(manifest: Path = MANIFEST) -> str | None:
    """The sha256 the panel build recorded for the daily table, if the manifest has it."""
    try:
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        return None
    entry = files.get(PANEL_FILE) if isinstance(files, dict) else None
    return str(entry["sha256"]) if isinstance(entry, dict) and "sha256" in entry else None


@dataclass(frozen=True, slots=True)
class Series:
    """The panel by date, with every realised-vol definition the report uses."""

    date: vrp.Dates
    iv: vrp.Floats
    rv: vrp.Floats  # primary: zero mean, 21 trading days, /252, Jensen-corrected
    rv_raw: vrp.Floats  # the same, uncorrected
    rv_sd: vrp.Floats  # sample sd, ddof 1: the panel's rv21_fwd
    rv_cal: vrp.Floats  # the 30 calendar days IV30 covers, /365, corrected
    rv_trailing: vrp.Floats  # the 21 days before, for contrast only
    end: vrp.Dates  # last date of each forward window
    vrp30: vrp.Floats  # the panel's own column, vol units

    @property
    def premium(self) -> vrp.Floats:
        return vrp.premium_vol_pts(self.iv, self.rv)


def series(daily: Columns) -> Series:
    order = np.argsort(daily["date"], kind="stable")
    d = np.asarray(daily["date"][order], dtype="datetime64[D]")
    if np.unique(d).size != d.size:
        raise ValueError("the daily table has a date twice; the panel should not")
    spot = np.asarray(daily["spot"][order], dtype=np.float64)
    return Series(
        date=d,
        iv=np.asarray(daily["iv30_atm"][order], dtype=np.float64),
        rv=vrp.forward_realised(spot, d, unbiased=True),
        rv_raw=vrp.forward_realised(spot, d),
        rv_sd=vrp.forward_realised(spot, d, demean=True),
        rv_cal=vrp.forward_realised_calendar(spot, d, unbiased=True),
        rv_trailing=vrp.trailing_realised(spot),
        end=vrp.window_end(d),
        vrp30=np.asarray(daily["vrp30"][order], dtype=np.float64)
        if "vrp30" in daily
        else np.full(d.size, np.nan),
    )


@dataclass(frozen=True, slots=True)
class Mapping:
    headline: vrp.Estimate
    sensitivity: list[vrp.Estimate]
    blocks: list[vrp.Estimate]
    periods: list[vrp.Estimate]
    years: list[vrp.Estimate]
    bias_vol_pts: float  # mean of the Jensen correction over the headline sample
    panel_check_vol_pts: float  # max |vrp30 - (IV - rv_sd)|, vol points
    panel_vrp30_median: float
    panel_vrp30_mean: float
    trailing_median: float
    n_rows: int
    first_row: date
    last_row: date
    no_window: int  # rows at the end with no complete forward window
    no_iv: list[date]


def analyse(s: Series, n_boot: int = vrp.N_BOOT, seed: int = vrp.SEED) -> Mapping:
    est = partial(vrp.estimate, n_boot=n_boot, seed=seed)
    x = s.premium
    head = est(x, s.date, "headline: 21 trading days, zero mean, /252, corrected")
    var_sd = vrp.variance_premium(s.iv, s.rv_raw)
    ok = np.isfinite(var_sd)
    rv_level = float(np.sqrt(np.mean(s.rv_raw[ok] ** 2)))

    def at(sigma: float) -> Callable[[float], float]:
        return partial(vrp.variance_to_vol_pts, sigma=sigma)

    sensitivity = [
        head,
        est(vrp.premium_vol_pts(s.iv, s.rv_raw), s.date, "zero mean, uncorrected"),
        est(vrp.premium_vol_pts(s.iv, s.rv_sd), s.date, "sample sd, ddof 1 (the panel's vrp30)"),
        est(vrp.premium_vol_pts(s.iv, s.rv_cal), s.date, "30 calendar days, /365, corrected"),
        est(
            var_sd,
            s.date,
            f"variance units, at the simulator's sigma {vrp.SIGMA:.2f}",
            to_vol_pts=at(vrp.SIGMA),
        ),
        est(
            var_sd,
            s.date,
            f"variance units, at the sample's RV {rv_level:.3f}",
            to_vol_pts=at(rv_level),
        ),
        est(
            vrp.premium_vol_pts(s.iv, s.rv_trailing),
            s.date,
            "trailing 21d (not the simulator's definition)",
        ),
    ]
    blocks = [est(x, s.date, f"block {b:.0f}", block=b) for b in BLOCKS]
    periods = [
        est(np.where(vrp.before(s.end, SPLIT_2020), x, np.nan), s.date, "before 2020"),
        est(np.where(vrp.after(s.date, SPLIT_2020), x, np.nan), s.date, "2020 on"),
        est(
            np.where(vrp.before(s.end, vrp.HOLDOUT_AFTER), x, np.nan),
            s.date,
            f"development (windows to {vrp.HOLDOUT_AFTER})",
        ),
        est(
            np.where(vrp.after(s.date, vrp.HOLDOUT_AFTER), x, np.nan),
            s.date,
            f"holdout (after {vrp.HOLDOUT_AFTER})",
        ),
    ]
    yr = vrp.years(s.date)
    by_year = [est(np.where(yr == y, x, np.nan), s.date, str(y)) for y in sorted(set(yr.tolist()))]
    both = np.isfinite(s.vrp30) & np.isfinite(s.rv_sd) & np.isfinite(s.iv)
    check = (
        float(np.max(np.abs(s.vrp30[both] - (s.iv[both] - s.rv_sd[both])))) * 100.0
        if both.any()
        else float("nan")
    )
    have = np.isfinite(x)
    v30 = s.vrp30[np.isfinite(s.vrp30)] * 100.0
    trail = vrp.premium_vol_pts(s.iv, s.rv_trailing)
    no_iv = [vrp.as_date(d) for d in s.date[~np.isfinite(s.iv)]]
    return Mapping(
        headline=head,
        sensitivity=sensitivity,
        blocks=[est(x, s.date, f"auto (Politis-White, floor {vrp.WINDOW})"), *blocks],
        periods=periods,
        years=by_year,
        bias_vol_pts=float(np.mean((s.rv[have] - s.rv_raw[have]) * 100.0)),
        panel_check_vol_pts=check,
        panel_vrp30_median=float(np.median(v30)) if v30.size else float("nan"),
        panel_vrp30_mean=float(np.mean(v30)) if v30.size else float("nan"),
        trailing_median=float(np.nanmedian(trail)) if np.isfinite(trail).any() else float("nan"),
        n_rows=int(s.date.size),
        first_row=vrp.as_date(s.date[0]),
        last_row=vrp.as_date(s.date[-1]),
        no_window=int(np.isnat(s.end).sum()),
        no_iv=no_iv,
    )


# -- rendering ---------------------------------------------------------------- #


def _f(v: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:+.{digits}f}"


def _row(e: vrp.Estimate) -> str:
    return (
        f"| {e.label} | {e.n_days:,} | {_f(e.mean)} | [{_f(e.mean_low)}, {_f(e.mean_high)}] "
        f"| {_f(e.median)} | [{_f(e.median_low)}, {_f(e.median_high)}] "
        f"| {_pct(e.share_positive)} | {_int(e.block)} |"
    )


def _pct(v: float) -> str:
    return f"{v:.0%}" if np.isfinite(v) else "n/a"


def _int(v: float) -> str:
    return f"{v:.0f}" if np.isfinite(v) else "n/a"


HEAD = (
    "| sample | days | mean | 95% interval | median | 95% interval | positive | block |\n"
    "|---|---:|---:|---|---:|---|---:|---:|"
)


def placement(e: vrp.Estimate, axis: Sequence[float]) -> str:
    lo, hi = vrp.bracket(e.mean, axis)
    inside = [a for a in axis if e.mean_low <= a <= e.mean_high]
    where = (
        f"between the sweep's {lo:g} and {hi:g} vol-point levels"
        if lo is not None and hi is not None
        else ("below the sweep's lowest level" if lo is None else "above its highest level")
    )
    cover = (
        "the interval contains " + ", ".join(f"{a:g}" for a in inside)
        if inside
        else "the interval contains no sweep level"
    )
    return f"{_f(e.mean)} vol points sits {where}; {cover}."


def render_md(m: Mapping, source: str, digest: str, n_boot: int, seed: int) -> str:
    h = m.headline
    low = min((e for e in m.years if np.isfinite(e.mean)), key=lambda e: e.mean, default=h)
    axis = ", ".join(f"{a:g}" for a in SWEEP_PREMIUMS)
    lines = [
        "# The VRP mapping",
        "",
        f"Generated by `make vrp` (`python -m scripts.vrp_mapping`) on {date.today()}. "
        "Owner: Hemdutt Rao. Issue #15.",
        "",
        f"Source: `{source}`, sha256 `{digest[:16]}`, {m.n_rows:,} trading days, "
        f"{m.first_row} to {m.last_row}. The last {m.no_window} have no complete forward "
        "window and are left out"
        + (
            f", as are {len(m.no_iv)} with no 30-day IV ("
            + ", ".join(str(d) for d in m.no_iv)
            + ")."
            if m.no_iv
            else "."
        ),
        "",
        "## Result",
        "",
        f"**The real SPX premium maps to {_f(h.mean)} planted vol points, 95% interval "
        f"[{_f(h.mean_low)}, {_f(h.mean_high)}]** (median {_f(h.median)}, "
        f"[{_f(h.median_low)}, {_f(h.median_high)}]), over {h.n_days:,} days "
        f"from {h.first} to {h.last}. Positive on {h.share_positive:.0%} of days.",
        "",
        f"On the sweep's axis ({axis}): {placement(h, SWEEP_PREMIUMS)}",
        "",
        f"Why it differs from the panel's `vrp30` median of {_f(m.panel_vrp30_median)}: that "
        "column is already IV30 less the realised vol of the following 21 days (`rv21_fwd`, "
        f"reproduced here to {m.panel_check_vol_pts:.0e} vol points), not trailing, so the gap is "
        "not the horizon. It is the statistic and the RV: the premium is left-skewed "
        f"(volatility spikes; the lowest year is {low.label} at {_f(low.mean)}), so its median "
        f"sits above its mean ({_f(m.panel_vrp30_mean)} for `vrp30`), and the headline removes "
        f"the Jensen gap ({_f(m.bias_vol_pts)}) that the uncorrected sample sd keeps. Against "
        f"trailing RV21, for contrast, the median would be {_f(m.trailing_median)}.",
        "",
        "## Definition",
        "",
        "The simulator prices every contract at `sigma + iv_premium_vol_pts / 100` and drives "
        "the path with `sigma` (`optspread/market.py`), so the planted premium is implied vol "
        "less the vol that then drives the path. The real analogue, per day t:",
        "",
        "    premium_t = 100 * (IV30_t - RV_t / c_21)",
        "    RV_t      = sqrt(252 / 21 * sum_{i=1..21} r_{t+i}^2),  r = close-to-close log return",
        "",
        "- **Horizon.** 21 trading days, the simulator's front tenor (`MarketConfig.tenor_days`"
        " = (21, 42)), against the panel's 30-calendar-day ATM IV; 252 days a year, "
        "`trading_days_per_year`.",
        "- **Zero mean.** The simulated path has no drift and a hedged seller is paid on "
        "squared returns, so RV is a root mean square, not a standard deviation.",
        "- **Jensen.** A vol over 21 Gaussian returns reads "
        f"`c_21 = {vrp.vol_bias(vrp.WINDOW):.4f}` times the true vol on average, "
        "so `IV - RV` overstates the planted premium; "
        f"dividing by `c_21` removes it ({_f(m.bias_vol_pts)} vol points on average here).",
        "- **Spot.** The panel's `spot`: the vendor close to 2025-08-29, then the parity "
        "prepaid forward.",
        "",
        "## Uncertainty",
        "",
        "Stationary bootstrap (Politis and Romano 1994), "
        f"{n_boot:,} replicates, seed {seed}, percentile intervals. Consecutive days share 20 "
        "of their 21 forward returns and volatility clusters, so the mean block is the "
        "Politis-White automatic length, never below the 21-day window. Each sample takes its "
        "own. The headline's sensitivity to the block:",
        "",
        HEAD,
        *[_row(e) for e in m.blocks],
        "",
        "## Sensitivity to the realised-vol definition",
        "",
        HEAD,
        *[_row(e) for e in m.sensitivity],
        "",
        "Variance units: the mean of `IV^2 - RV^2`, converted to the planted premium whose "
        "variance premium at sigma it is, `sqrt(sigma^2 + v) - sigma`; at the simulator's "
        "sigma and at the sample's own realised level.",
        "",
        "## Sub-periods",
        "",
        "A split keeps only days whose whole forward window falls on its side, so the "
        "halves share no return; days that straddle a boundary are in the full sample only. "
        f"After {vrp.HOLDOUT_AFTER} is the development extract's holdout.",
        "",
        HEAD,
        *[_row(e) for e in m.periods],
        "",
        "By year (of the quote date):",
        "",
        HEAD,
        *[_row(e) for e in m.years],
        "",
        "## Reading a power curve at it",
        "",
        "`optspread.vrp.power_at_noncentrality(premiums, noncentrality, n_seeds, estimate, "
        "interval)` reads the M4 curve from its knots through the analysis's own interpolation "
        "(noncentrality linear in premium, exact noncentral t) and returns detection "
        "probability at the headline, at both interval ends and its range over the interval; "
        "a curve that dips is read as it is. `optspread.vrp.power_at(premiums, power, ...)` "
        "reads sampled powers linearly. `VRP_MAPPING.json` carries the inputs under "
        "`power_inputs`.",
        "",
        "## Not settled here",
        "",
        "- The simulator plants a constant premium at every strike and tenor; the real one "
        "moves with the regime (by year above) and is larger in the put wing than at the money. "
        "This maps the at-the-money, 30-day premium only.",
        "- Section 5.5's position size, multiplier and holding period do not enter a premium in "
        "vol points; what a defined-risk structure harvests of it is what the sweep measures, "
        "so it is in the power curve, not here.",
        "",
    ]
    return "\n".join(lines)


def to_json(m: Mapping, source: str, digest: str, n_boot: int, seed: int) -> dict[str, Any]:
    def enc(e: vrp.Estimate) -> dict[str, Any]:
        out = asdict(e)
        out["first"] = e.first.isoformat() if e.first else None
        out["last"] = e.last.isoformat() if e.last else None
        return out

    h = m.headline
    lo, hi = vrp.bracket(h.mean, SWEEP_PREMIUMS)
    return {
        "issue": 15,
        "owner": "Hemdutt Rao",
        "generated": date.today().isoformat(),
        "source": source,
        "source_sha256": digest,
        "definition": {
            "premium": "100 * (iv30_atm - rv_fwd / c_window)",
            "rv_fwd": "sqrt(trading_days / window * sum of the next window squared log returns)",
            "window_trading_days": vrp.WINDOW,
            "trading_days_per_year": vrp.TRADING_DAYS,
            "c_window": vrp.vol_bias(vrp.WINDOW),
            "statistic": "mean",
        },
        "bootstrap": {
            "method": "stationary (Politis-Romano), percentile",
            "block": "Politis-White automatic, floored at the window",
            "n_boot": n_boot,
            "seed": seed,
            "alpha": vrp.ALPHA,
        },
        "headline": enc(h),
        "power_inputs": {"estimate": h.mean, "interval": [h.mean_low, h.mean_high]},
        "sweep_axis": list(SWEEP_PREMIUMS),
        "bracket": [lo, hi],
        "sensitivity": [enc(e) for e in m.sensitivity],
        "blocks": [enc(e) for e in m.blocks],
        "periods": [enc(e) for e in m.periods],
        "years": [enc(e) for e in m.years],
        "checks": {
            "jensen_correction_vol_pts": m.bias_vol_pts,
            "panel_vrp30_reproduced_max_abs_vol_pts": m.panel_check_vol_pts,
            "panel_vrp30_median_vol_pts": m.panel_vrp30_median,
            "panel_vrp30_mean_vol_pts": m.panel_vrp30_mean,
            "trailing_rv21_median_vol_pts": m.trailing_median,
        },
        "rows": {
            "n": m.n_rows,
            "first": m.first_row.isoformat(),
            "last": m.last_row.isoformat(),
            "no_forward_window": m.no_window,
            "no_iv30": [d.isoformat() for d in m.no_iv],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--daily", type=Path, default=None, help="spx_daily.parquet (default: panel)")
    p.add_argument("--out", type=Path, default=OUT, help="writes <out>.md and <out>.json")
    p.add_argument("--n-boot", type=int, default=vrp.N_BOOT)
    p.add_argument("--seed", type=int, default=vrp.SEED)
    args = p.parse_args(argv)
    path = args.daily or PanelLayout(data_root()).daily
    daily = read_daily(path)
    digest = sha256(path)
    recorded = manifest_digest()
    source = PANEL_FILE + (
        ", as recorded in docs/DATA_MANIFEST.json"
        if digest == recorded
        else " (not the build docs/DATA_MANIFEST.json records)"
    )
    m = analyse(series(daily), args.n_boot, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = render_md(m, source, digest, args.n_boot, args.seed)
    blob = to_json(m, source, digest, args.n_boot, args.seed)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8", newline="\n")
    args.out.with_suffix(".json").write_text(
        json.dumps(blob, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    h = m.headline
    print(
        f"premium {h.mean:+.2f} vol pts, 95% [{h.mean_low:+.2f}, {h.mean_high:+.2f}], "
        f"median {h.median:+.2f}; wrote {args.out}.md and .json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
