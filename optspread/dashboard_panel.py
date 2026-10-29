"""Read-side views of the stitched SPX panel: coverage, smile, surface, term structure, VRP.

Lane: Ilan Danial (machinery). Reads what scripts/build_spx_panel.py writes; never writes.

The panel lives in the team's shared data folder, not in the repository, and is
gigabytes, so this module reads only what a view needs: the two small tables
whole, and one year's chain partition, filtered to one date, for the chain views.

    panel/spx/year=YYYY/part.parquet   one row per contract per date, both sources
    panel/spx_daily.parquet            spot, IV30, trailing and forward RV21, VRP30
    panel/coverage.parquet             contracts per date per source; missing days flagged
    panel/seam_overlap.parquet         the seam report: both sources, same contracts, same days
    panel/surface_grid.parquet         IV per date on the build's fixed tenor x log-moneyness grid
    panel/atm_term.parquet             ATM IV per (date, root, expiry)

Every aggregation works on plain numpy columns (``Columns``), so it is tested on
the ``dev`` install without pyarrow. Only ``read_table`` needs pyarrow, imported
when it is called. Volatilities are decimals on disk and stay decimals here; the
pages convert to vol points for display.

Nothing here interpolates a missing trading day. Coverage reports what is there,
and a day the build could not fill is listed, not smoothed over.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DATA_ENV = "VEGALAB_DATA"
DEFAULT_DATA_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-UniversityofFlorida"
    / "Dupuy, Dominick's files - wrds_optionmetrics"
)

Columns = dict[str, NDArray[Any]]

DATE_COLUMNS = ("date", "expiry")
CHAIN_COLUMNS = (
    "date",
    "root",
    "expiry",
    "dte",
    "cp",
    "strike",
    "bid",
    "forward",
    "tau",
    "carry_fit",
    "iv",
    "iv_vendor",
    "source",
)


def data_root(env: Mapping[str, str] | None = None) -> Path:
    """``$VEGALAB_DATA`` if set, else the team's shared OneDrive folder."""
    value = (os.environ if env is None else env).get(DATA_ENV, "")
    return Path(value).expanduser() if value else DEFAULT_DATA_ROOT


@dataclass(frozen=True, slots=True)
class PanelLayout:
    """Where each panel file lives under a data root."""

    root: Path

    @property
    def panel(self) -> Path:
        return self.root / "panel"

    @property
    def chain_dir(self) -> Path:
        return self.panel / "spx"

    @property
    def daily(self) -> Path:
        return self.panel / "spx_daily.parquet"

    @property
    def coverage(self) -> Path:
        return self.panel / "coverage.parquet"

    @property
    def seam_overlap(self) -> Path:
        return self.panel / "seam_overlap.parquet"

    @property
    def surface_grid(self) -> Path:
        return self.panel / "surface_grid.parquet"

    @property
    def atm_term(self) -> Path:
        return self.panel / "atm_term.parquet"

    def chain_file(self, year: int) -> Path:
        return self.chain_dir / f"year={year}" / "part.parquet"

    def years(self) -> tuple[int, ...]:
        """Years with a chain partition on disk."""
        if not self.chain_dir.is_dir():
            return ()
        found = []
        for d in self.chain_dir.glob("year=*"):
            stem = d.name.removeprefix("year=")
            if stem.isdigit() and (d / "part.parquet").is_file():
                found.append(int(stem))
        return tuple(sorted(found))

    def missing(self) -> tuple[str, ...]:
        """Panel files a view needs that are not there yet, as paths relative to the root."""
        out = [p for p in (self.daily, self.coverage) if not p.is_file()]
        names = [p.relative_to(self.root).as_posix() for p in out]
        if not self.years():
            names.append(
                (self.chain_dir / "year=YYYY" / "part.parquet").relative_to(self.root).as_posix()
            )
        return tuple(names)

    @property
    def built(self) -> bool:
        return not self.missing()


# -- reading ------------------------------------------------------------------ #


def _normalise(name: str, values: NDArray[Any]) -> NDArray[Any]:
    if name in DATE_COLUMNS:
        if values.dtype.kind == "M":
            return values.astype("datetime64[D]")
        return np.asarray(values, dtype="datetime64[D]")
    if values.dtype.kind == "O":  # nullable numbers or strings
        present = [v for v in values if v is not None]
        if all(isinstance(v, int | float) and not isinstance(v, bool) for v in present):
            return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
        return np.asarray(["" if v is None else str(v) for v in values], dtype=str)
    return values


def normalise(columns: Mapping[str, Sequence[Any] | NDArray[Any]]) -> Columns:
    """Dates to ``datetime64[D]``, nullable numbers to float with NaN, strings to ``str``."""
    return {k: _normalise(k, np.asarray(v)) for k, v in columns.items()}


def read_table(
    path: Path, columns: Sequence[str] | None = None, day: date | None = None
) -> Columns:
    """Read a parquet file into numpy columns, optionally only the rows on ``day``.

    The date filter is pushed down to pyarrow so a partition sorted by date reads
    only the row groups that hold that day, then applied again here in case the
    file's ``date`` type did not admit the pushdown.
    """
    import pyarrow.parquet as pq

    wanted = None
    if columns is not None:
        available = set(pq.read_schema(path).names)
        wanted = [c for c in columns if c in available]
    try:
        table = pq.read_table(path, columns=wanted, filters=[("date", "=", day)] if day else None)
    except (TypeError, ValueError, NotImplementedError):  # a date type the pushdown rejects
        table = pq.read_table(path, columns=wanted)
    out = normalise({name: table.column(name).to_numpy() for name in table.column_names})
    if day is not None and "date" in out:
        keep = out["date"] == np.datetime64(day, "D")
        out = {k: v[keep] for k, v in out.items()}
    return out


def read_daily(layout: PanelLayout) -> Columns:
    return read_table(layout.daily)


def read_coverage(layout: PanelLayout) -> Columns:
    return read_table(layout.coverage)


def read_seam_overlap(layout: PanelLayout) -> Columns:
    """The seam report, or no columns when the build has not written it."""
    return read_table(layout.seam_overlap) if layout.seam_overlap.is_file() else {}


def read_chain(layout: PanelLayout, day: date, columns: Sequence[str] = CHAIN_COLUMNS) -> Columns:
    """The chain on ``day``: one partition read, one date kept."""
    path = layout.chain_file(day.year)
    if not path.is_file():
        return {}
    return read_table(path, columns, day)


def take(columns: Columns, mask: NDArray[np.bool_]) -> Columns:
    return {k: v[mask] for k, v in columns.items()}


def n_rows(columns: Columns) -> int:
    return len(next(iter(columns.values()))) if columns else 0


# -- coverage ---------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SourceSpan:
    source: str
    first: date
    last: date
    n_days: int  # dates with data
    n_missing: int  # dates the build flagged as missing
    median_contracts: float


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    spans: tuple[SourceSpan, ...]
    missing: tuple[tuple[date, str], ...]  # (date, source), in date order
    overlap: tuple[date, date, int] | None  # first, last, days both sources have data

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(s.source for s in self.spans)


def _day(value: np.datetime64) -> date:
    out = value.astype("datetime64[D]").astype(date)
    assert isinstance(out, date)
    return out


def coverage_summary(cov: Columns) -> CoverageSummary:
    """Per-source span, the missing days listed, and the window both sources cover."""
    if not n_rows(cov):
        return CoverageSummary((), (), None)
    dates, sources = cov["date"], cov["source"]
    missing = cov["missing"].astype(bool) if "missing" in cov else np.zeros(len(dates), bool)
    counts = cov["n_contracts"].astype(np.float64)
    have = ~missing & (counts > 0)
    spans: list[SourceSpan] = []
    present: dict[str, set[np.datetime64]] = {}
    for src in sorted(set(sources.tolist())):
        mine = sources == src
        ok = mine & have
        if not ok.any():
            continue
        present[src] = set(dates[ok].tolist())
        spans.append(
            SourceSpan(
                source=src,
                first=_day(dates[ok].min()),
                last=_day(dates[ok].max()),
                n_days=len(np.unique(dates[ok])),
                n_missing=len(np.unique(dates[mine & missing])),
                median_contracts=float(np.median(counts[ok])),
            )
        )
    order = np.argsort(dates[missing], kind="stable")
    listed = tuple(
        (_day(d), str(s))
        for d, s in zip(dates[missing][order], sources[missing][order], strict=True)
    )
    overlap = None
    if len(present) >= 2:
        both = set.intersection(*present.values())
        if both:
            days = sorted(both)
            overlap = (_day(np.datetime64(days[0])), _day(np.datetime64(days[-1])), len(days))
    return CoverageSummary(tuple(spans), listed, overlap)


def available_dates(cov: Columns, source: str | None = None) -> NDArray[np.datetime64]:
    """Distinct dates with data (any source unless one is named), ascending."""
    if not n_rows(cov):
        return np.asarray([], dtype="datetime64[D]")
    ok = cov["n_contracts"].astype(np.float64) > 0
    if "missing" in cov:
        ok &= ~cov["missing"].astype(bool)
    if source is not None:
        ok &= cov["source"] == source
    return np.unique(cov["date"][ok]).astype("datetime64[D]")


# -- the chain on one date ---------------------------------------------------- #


def chain_sources(chain: Columns) -> tuple[str, ...]:
    return tuple(sorted(set(chain["source"].tolist()))) if n_rows(chain) else ()


def select(chain: Columns, source: str | None = None, roots: Sequence[str] = ()) -> Columns:
    """Rows of one source and (when given) the named roots, SPX and/or SPXW."""
    if not n_rows(chain):
        return chain
    keep = np.ones(n_rows(chain), dtype=bool)
    if source is not None:
        keep &= chain["source"] == source
    if roots:
        keep &= np.isin(chain["root"], list(roots))
    return take(chain, keep)


def otm_quotes(chain: Columns, max_iv: float = 5.0) -> Columns:
    """Out-of-the-money quotes with a usable IV, plus ``moneyness = K / F``.

    Puts below the forward and calls at or above it: the side that is liquid and
    whose IV is not dominated by intrinsic value. A zero bid or a non-finite,
    non-positive or absurd IV is dropped rather than plotted.
    """
    if not n_rows(chain):
        return {}
    m = chain["strike"].astype(np.float64) / chain["forward"].astype(np.float64)
    iv = chain["iv"].astype(np.float64)
    cp = chain["cp"]
    keep = np.isfinite(m) & np.isfinite(iv) & (iv > 0.0) & (iv < max_iv)
    keep &= ((cp == "P") & (m < 1.0)) | ((cp == "C") & (m >= 1.0))
    if "bid" in chain:
        keep &= chain["bid"].astype(np.float64) > 0.0
    out = take(chain, keep)
    out["moneyness"] = m[keep]
    return out


@dataclass(frozen=True, slots=True)
class Expiry:
    expiry: date
    dte: int
    n_quotes: int
    forward: float  # the parity-implied forward the moneyness is measured against
    # How the build fitted the carry: "near", "wide", either with "+r_ref" when the
    # day's reference rate was borrowed, or "none". Labels joined when roots differ;
    # empty when the panel does not carry the column.
    carry_fit: str


def expiries(chain: Columns) -> tuple[Expiry, ...]:
    """Listed expiries on the date, nearest first, with their usable quote counts."""
    otm = otm_quotes(chain)
    if not n_rows(otm):
        return ()
    out = []
    for exp in np.unique(otm["expiry"]):
        mine = otm["expiry"] == exp
        fit = otm["carry_fit"][mine].astype(str) if "carry_fit" in otm else ()
        out.append(
            Expiry(
                expiry=_day(exp),
                dte=int(otm["dte"][mine][0]),
                n_quotes=int(mine.sum()),
                forward=float(np.median(otm["forward"][mine].astype(np.float64))),
                carry_fit=", ".join(sorted({str(f) for f in fit if f})),
            )
        )
    return tuple(sorted(out, key=lambda e: e.dte))


def nearest_expiry(chain: Columns, target_dte: int = 30, min_quotes: int = 5) -> Expiry | None:
    """The expiry closest to ``target_dte`` with at least ``min_quotes`` usable quotes."""
    usable = [e for e in expiries(chain) if e.n_quotes >= min_quotes]
    return min(usable, key=lambda e: (abs(e.dte - target_dte), e.dte)) if usable else None


@dataclass(frozen=True, slots=True)
class Smile:
    expiry: date
    dte: int
    moneyness: NDArray[np.float64]
    strike: NDArray[np.float64]
    iv: NDArray[np.float64]
    iv_vendor: NDArray[np.float64]  # all NaN when the source has no vendor IV
    cp: NDArray[np.str_]

    @property
    def has_vendor(self) -> bool:
        return bool(np.isfinite(self.iv_vendor).any())


def _curve(otm: Columns) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One IV per distinct moneyness (duplicates across roots averaged), ascending."""
    m = otm["moneyness"].astype(np.float64)
    iv = otm["iv"].astype(np.float64)
    grid, inverse = np.unique(m, return_inverse=True)
    sums = np.bincount(inverse, weights=iv, minlength=len(grid))
    counts = np.bincount(inverse, minlength=len(grid))
    return grid, sums / counts


def smile(chain: Columns, expiry: date) -> Smile:
    otm = otm_quotes(chain)
    if n_rows(otm):
        otm = take(otm, otm["expiry"] == np.datetime64(expiry, "D"))
    order = np.argsort(otm["strike"], kind="stable") if n_rows(otm) else np.asarray([], int)
    n = len(order)

    def col(name: str) -> NDArray[np.float64]:
        if name not in otm or not n:
            return np.full(n, np.nan)
        return np.asarray(otm[name][order], dtype=np.float64)

    return Smile(
        expiry=expiry,
        dte=int(otm["dte"][order][0]) if n else 0,
        moneyness=col("moneyness"),
        strike=col("strike"),
        iv=col("iv"),
        iv_vendor=col("iv_vendor"),
        cp=np.asarray(otm["cp"][order], dtype=str) if n else np.asarray([], dtype=str),
    )


@dataclass(frozen=True, slots=True)
class TermStructure:
    expiry: tuple[date, ...]
    dte: NDArray[np.int64]
    atm_iv: NDArray[np.float64]


def term_structure(chain: Columns, min_quotes: int = 3) -> TermStructure:
    """ATM IV per expiry: the OTM curve interpolated at ``K / F = 1``.

    An expiry whose quotes do not straddle the forward has no ATM point and is left
    out rather than extrapolated.
    """
    otm = otm_quotes(chain)
    exps: list[date] = []
    dtes: list[int] = []
    ivs: list[float] = []
    for e in expiries(chain):
        mine = take(otm, otm["expiry"] == np.datetime64(e.expiry, "D"))
        if n_rows(mine) < min_quotes:
            continue
        m, iv = _curve(mine)
        if m[0] > 1.0 or m[-1] < 1.0:
            continue
        exps.append(e.expiry)
        dtes.append(e.dte)
        ivs.append(float(np.interp(1.0, m, iv)))
    return TermStructure(tuple(exps), np.asarray(dtes, np.int64), np.asarray(ivs, np.float64))


@dataclass(frozen=True, slots=True)
class Surface:
    moneyness: NDArray[np.float64]  # (M,)
    dte: NDArray[np.int64]  # (E,)
    iv: NDArray[np.float64]  # (E, M); NaN outside each expiry's quoted range


def iv_surface(
    chain: Columns,
    moneyness_grid: NDArray[np.float64] | None = None,
    min_dte: int = 1,
    max_dte: int = 365,
    min_quotes: int = 5,
) -> Surface:
    """IV on a moneyness x DTE grid, interpolated along strike within each expiry only.

    No interpolation across expiries and no extrapolation past an expiry's quoted
    strikes: the surface is what was quoted, gridded, with holes where nothing was.
    """
    grid = moneyness_grid if moneyness_grid is not None else np.linspace(0.8, 1.2, 41)
    otm = otm_quotes(chain)
    dtes: list[int] = []
    rows: list[NDArray[np.float64]] = []
    for e in expiries(chain):
        if not (min_dte <= e.dte <= max_dte):
            continue
        mine = take(otm, otm["expiry"] == np.datetime64(e.expiry, "D"))
        if n_rows(mine) < min_quotes:
            continue
        m, iv = _curve(mine)
        row = np.interp(grid, m, iv)
        row[(grid < m[0]) | (grid > m[-1])] = np.nan
        dtes.append(e.dte)
        rows.append(row)
    iv_grid = np.vstack(rows) if rows else np.empty((0, len(grid)))
    return Surface(grid, np.asarray(dtes, np.int64), iv_grid)


# -- IV30, realised vol and the variance risk premium ------------------------- #


@dataclass(frozen=True, slots=True)
class VrpSeries:
    source: str
    date: NDArray[np.datetime64]
    spot: NDArray[np.float64]
    spot_parity: NDArray[np.float64]  # spot implied by put-call parity; NaN when absent
    iv30: NDArray[np.float64]
    rv21: NDArray[np.float64]
    rv21_fwd: NDArray[np.float64]
    vrp30: NDArray[np.float64]


def daily_sources(daily: Columns) -> tuple[str, ...]:
    if not n_rows(daily):
        return ()
    return tuple(sorted(set(daily["source"].tolist()))) if "source" in daily else ("",)


def vrp_series(daily: Columns, source: str | None = None) -> VrpSeries:
    """The daily table for one source (all rows when ``source`` is None), by date."""
    rows = daily
    if source is not None and "source" in daily and n_rows(daily):
        rows = take(daily, daily["source"] == source)
    order = np.argsort(rows["date"], kind="stable") if n_rows(rows) else np.asarray([], int)

    def col(name: str) -> NDArray[np.float64]:
        if name not in rows or not len(order):
            return np.full(len(order), np.nan)
        return np.asarray(rows[name][order], dtype=np.float64)

    dates = rows["date"][order] if "date" in rows else np.asarray([], "datetime64[D]")
    return VrpSeries(
        source=source or "",
        date=dates.astype("datetime64[D]"),
        spot=col("spot"),
        spot_parity=col("spot_parity"),
        iv30=col("iv30_atm"),
        rv21=col("rv21"),
        rv21_fwd=col("rv21_fwd"),
        vrp30=col("vrp30"),
    )


@dataclass(frozen=True, slots=True)
class VrpStats:
    n_days: int
    mean_vol_pts: float
    median_vol_pts: float
    share_positive: float
    first: date | None
    last: date | None


def vrp_stats(series: VrpSeries) -> VrpStats:
    """Summary of VRP30 in vol points, over the days where it is defined.

    The last 21 trading days have no forward realised vol yet, so they are excluded
    by construction rather than filled.
    """
    ok = np.isfinite(series.vrp30)
    if not ok.any():
        return VrpStats(0, float("nan"), float("nan"), float("nan"), None, None)
    vrp = series.vrp30[ok] * 100.0
    dates = series.date[ok]
    return VrpStats(
        n_days=int(ok.sum()),
        mean_vol_pts=float(vrp.mean()),
        median_vol_pts=float(np.median(vrp)),
        share_positive=float((vrp > 0).mean()),
        first=_day(dates.min()),
        last=_day(dates.max()),
    )


@dataclass(frozen=True, slots=True)
class Seam:
    """IV30 from both sources on the days both cover: the cut-over check."""

    sources: tuple[str, str]
    date: NDArray[np.datetime64]
    iv30_a: NDArray[np.float64]
    iv30_b: NDArray[np.float64]

    @property
    def diff_vol_pts(self) -> NDArray[np.float64]:
        return (self.iv30_b - self.iv30_a) * 100.0


def seam(daily: Columns) -> Seam | None:
    """None unless two sources share at least one date."""
    sources = daily_sources(daily)
    if len(sources) < 2:
        return None
    a, b = vrp_series(daily, sources[0]), vrp_series(daily, sources[1])
    common, ia, ib = np.intersect1d(a.date, b.date, return_indices=True)
    if not len(common):
        return None
    return Seam((sources[0], sources[1]), common, a.iv30[ia], b.iv30[ib])


SEAM_STATS = (
    "n_matched",
    "iv_diff_median_volpts",
    "iv_diff_p95_abs_volpts",
    "mid_diff_median_rel",
    "forward_diff_median_bp",
    "ours_vs_vendor_median_volpts",
)


@dataclass(frozen=True, slots=True)
class SeamReport:
    """The build's per-day seam statistics over the overlap, and their medians.

    The columns are the build's, passed through: this module adds the summary, not
    the comparison. A statistic the file does not carry reads as NaN.
    """

    date: NDArray[np.datetime64]
    stats: dict[str, NDArray[np.float64]]

    @property
    def n_days(self) -> int:
        return len(self.date)

    def median(self, name: str) -> float:
        values = self.stats.get(name)
        if values is None or not np.isfinite(values).any():
            return float("nan")
        return float(np.nanmedian(values))

    def mean(self, name: str) -> float:
        values = self.stats.get(name)
        if values is None or not np.isfinite(values).any():
            return float("nan")
        return float(np.nanmean(values))

    def worst(self, name: str) -> float:
        values = self.stats.get(name)
        if values is None or not np.isfinite(values).any():
            return float("nan")
        return float(np.nanmax(values))


# The seam's headline, one definition for the page and docs/DATA.md: each overlap day's
# 95th percentile of |IV_databento - IV_optionmetrics| over the contracts both quote,
# averaged over the overlap days ("on the average day"), with the worst day beside it.
SEAM_HEADLINE = "iv_diff_p95_abs_volpts"


def seam_report(cols: Columns) -> SeamReport | None:
    """None when the build has not written the seam report yet."""
    if not n_rows(cols) or "date" not in cols:
        return None
    order = np.argsort(cols["date"], kind="stable")
    stats = {
        name: (
            np.asarray(cols[name][order], dtype=np.float64)
            if name in cols
            else np.full(len(order), np.nan)
        )
        for name in SEAM_STATS
    }
    return SeamReport(cols["date"][order].astype("datetime64[D]"), stats)


# -- the gridded surface through time ------------------------------------------ #

# The build's tenor grid (``optspread.realdata.panel.TENOR_GRID``), in calendar days:
# where the ATM term structure is read off through time.
TENOR_GRID = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)


@dataclass(frozen=True, slots=True)
class GridSurface:
    """One day of ``surface_grid.parquet``: IV on the build's fixed grid, holes kept."""

    k: NDArray[np.float64]  # (K,) log-moneyness ln(K / F), ascending
    tenor_days: NDArray[np.int64]  # (T,) calendar days, ascending
    iv: NDArray[np.float64]  # (T, K); NaN where nothing was quoted

    def at(self, k: NDArray[np.float64], tenor_days: float) -> NDArray[np.float64]:
        """IV at the points ``k`` of one tenor, read off the grid the way the build made it.

        Along log-moneyness within each tenor row, then total variance linear in time
        between the two rows that bracket the tenor. A point outside a row's quoted
        range, or a tenor outside the grid, is NaN: nothing is extrapolated.
        """
        tenors = self.tenor_days.astype(np.float64)
        hi = int(np.searchsorted(tenors, tenor_days, side="left"))
        if not len(tenors) or hi == len(tenors) or (hi == 0 and tenors[0] != tenor_days):
            return np.full(len(k), np.nan)
        lo = hi if tenors[hi] == tenor_days else hi - 1

        def row(i: int) -> NDArray[np.float64]:
            ok = np.isfinite(self.iv[i])
            if not ok.any():
                return np.full(len(k), np.nan)
            vals: NDArray[np.float64] = np.interp(
                k, self.k[ok], self.iv[i][ok], left=np.nan, right=np.nan
            )
            return vals

        if lo == hi:
            return row(lo)
        t0, t1 = tenors[lo] / 365.0, tenors[hi] / 365.0
        w0, w1 = row(lo) ** 2 * t0, row(hi) ** 2 * t1
        t = tenor_days / 365.0
        w = w0 + (t - t0) / (t1 - t0) * (w1 - w0)
        iv: NDArray[np.float64] = np.sqrt(np.maximum(w, 0.0) / t)
        return iv


def grid_axes(grid: Columns) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """The grid's log-moneyness and tenor axes, as the file lays them out."""
    if not n_rows(grid):
        return np.asarray([], np.float64), np.asarray([], np.int64)
    k = np.unique(np.round(grid["k"].astype(np.float64), 6))
    return k, np.unique(grid["tenor_days"].astype(np.int64))


def grid_dates(grid: Columns) -> NDArray[np.datetime64]:
    """Dates with a gridded surface, ascending. A day the build could not grid is absent."""
    if not n_rows(grid):
        return np.asarray([], dtype="datetime64[D]")
    return np.unique(grid["date"]).astype("datetime64[D]")


def grid_surface(
    grid: Columns, day: date, axes: tuple[NDArray[np.float64], NDArray[np.int64]] | None = None
) -> GridSurface:
    """The grid on ``day``: every cell NaN when the day is not in the file."""
    k, tenors = axes if axes is not None else grid_axes(grid)
    iv = np.full((len(tenors), len(k)), np.nan)
    if n_rows(grid):
        rows = take(grid, grid["date"] == np.datetime64(day, "D"))
        ki = np.searchsorted(k, np.round(rows["k"].astype(np.float64), 6))
        ti = np.searchsorted(tenors, rows["tenor_days"].astype(np.int64))
        ok = (ki < len(k)) & (ti < len(tenors))
        iv[ti[ok], ki[ok]] = rows["iv"].astype(np.float64)[ok]
    return GridSurface(k, tenors, iv)


# -- the ATM term structure through time ---------------------------------------- #


@dataclass(frozen=True, slots=True)
class AtmLandscape:
    """ATM IV per date on a fixed tenor grid: the term structure as a landscape."""

    date: NDArray[np.datetime64]  # (D,) the last trading day of each row's bucket
    tenor_days: NDArray[np.int64]  # (T,)
    iv: NDArray[np.float64]  # (D, T); NaN where no expiry brackets the tenor
    stride: int  # trading days averaged into each row; 1 is every day as quoted


def atm_curves(
    term: Columns, tenor_days: Sequence[int] = TENOR_GRID
) -> tuple[NDArray[np.datetime64], NDArray[np.float64]]:
    """Per date, ATM IV at each tenor: total variance linear in time between expiries.

    The same rule as the build's IV30: where both roots list an expiry the PM-settled
    SPXW row is used, and a tenor no two expiries bracket is NaN, not extrapolated.
    """
    tenors = np.asarray(tenor_days, dtype=np.float64) / 365.0
    if not n_rows(term):
        return np.asarray([], dtype="datetime64[D]"), np.empty((0, len(tenors)))
    iv = term["atm_iv"].astype(np.float64)
    tau = term["tau"].astype(np.float64)
    ok = np.isfinite(iv) & np.isfinite(tau) & (iv > 0.0) & (tau > 0.0)
    rows = take(term, ok)
    iv, tau = iv[ok], tau[ok]
    spxw_first = (rows["root"] != "SPXW").astype(np.int64)
    order = np.lexsort((spxw_first, tau, rows["date"]))
    days, expiry = rows["date"][order], rows["expiry"][order]
    iv, tau = iv[order], tau[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (days[1:] != days[:-1]) | (expiry[1:] != expiry[:-1])
    days, iv, tau = days[first], iv[first], tau[first]
    uniq, start = np.unique(days, return_index=True)
    stops = [*start[1:], len(days)]
    out = np.full((len(uniq), len(tenors)), np.nan)
    for i, (a, b) in enumerate(zip(start, stops, strict=True)):
        t, w = tau[a:b], iv[a:b] ** 2 * tau[a:b]
        keep = np.ones(len(t), dtype=bool)
        keep[1:] = t[1:] > t[:-1]  # an AM and a PM expiry a day apart can share a tau
        wt = np.interp(tenors, t[keep], w[keep], left=np.nan, right=np.nan)
        out[i] = np.sqrt(np.maximum(wt, 0.0) / tenors)
    return uniq.astype("datetime64[D]"), out


def atm_landscape(
    dates: NDArray[np.datetime64],
    curves: NDArray[np.float64],
    tenor_days: Sequence[int] = TENOR_GRID,
    start: date | None = None,
    end: date | None = None,
    max_dates: int = 420,
) -> AtmLandscape:
    """The curves between ``start`` and ``end``, averaged into at most ``max_dates`` rows.

    Sixteen years of trading days are four thousand rows; a landscape that size is
    unreadable and heavy to draw. Consecutive days are averaged in equal buckets
    (the stride is reported, so the reader knows a spike is smoothed), and a window
    short enough is sent day by day.
    """
    keep = np.ones(len(dates), dtype=bool)
    if start is not None:
        keep &= dates >= np.datetime64(start, "D")
    if end is not None:
        keep &= dates <= np.datetime64(end, "D")
    d, c = dates[keep], curves[keep]
    tenors = np.asarray(tenor_days, dtype=np.int64)
    stride = max(1, -(-len(d) // max(1, max_dates)))  # ceiling division
    if stride == 1 or not len(d):
        return AtmLandscape(d.astype("datetime64[D]"), tenors, c, 1)
    bucket = np.arange(len(d)) // stride
    n = int(bucket[-1]) + 1
    finite = np.isfinite(c)
    sums = np.zeros((n, c.shape[1]))
    counts = np.zeros((n, c.shape[1]))
    np.add.at(sums, bucket, np.where(finite, c, 0.0))
    np.add.at(counts, bucket, finite.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    last = np.r_[np.flatnonzero(np.diff(bucket)), len(d) - 1]
    return AtmLandscape(d[last].astype("datetime64[D]"), tenors, mean, stride)


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
