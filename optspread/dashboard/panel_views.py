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

The data page's one-sentence readings are built on these views, in
``optspread.dashboard.readings``. Moved from ``optspread.dashboard_panel``, which
re-exports both.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optspread.fileio import env_path
from optspread.realdata import DEFAULT_ROOT

DATA_ENV = "VEGALAB_DATA"
# The team's shared folder, as ``optspread.realdata.data_root`` falls back to it.
DEFAULT_DATA_ROOT = DEFAULT_ROOT

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
    """``$VEGALAB_DATA`` if set, else the team's shared OneDrive folder.

    The same path ``optspread.realdata.data_root()`` gives for the same environment;
    ``env`` replaces ``os.environ``, which that function has no way to take.
    """
    return env_path(DATA_ENV, DEFAULT_DATA_ROOT, env)


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


def _floats(columns: Columns, name: str, order: NDArray[Any]) -> NDArray[np.float64]:
    """Column ``name`` taken in ``order`` as floats; all NaN when the column is absent."""
    if name not in columns or not len(order):
        return np.full(len(order), np.nan)
    return np.asarray(columns[name][order], dtype=np.float64)


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


def _expiry_curves(
    chain: Columns, min_quotes: int, dte_range: tuple[int, int] | None = None
) -> Iterator[tuple[Expiry, NDArray[np.float64], NDArray[np.float64]]]:
    """Each expiry, nearest first, with at least ``min_quotes`` usable quotes (and its DTE
    within ``dte_range`` when given), and its OTM curve as ``_curve`` gives it."""
    otm = otm_quotes(chain)
    for e in expiries(chain):
        if dte_range is not None and not (dte_range[0] <= e.dte <= dte_range[1]):
            continue
        mine = take(otm, otm["expiry"] == np.datetime64(e.expiry, "D"))
        if n_rows(mine) < min_quotes:
            continue
        m, iv = _curve(mine)
        yield e, m, iv


def smile(chain: Columns, expiry: date) -> Smile:
    otm = otm_quotes(chain)
    if n_rows(otm):
        otm = take(otm, otm["expiry"] == np.datetime64(expiry, "D"))
    order = np.argsort(otm["strike"], kind="stable") if n_rows(otm) else np.asarray([], int)
    n = len(order)
    return Smile(
        expiry=expiry,
        dte=int(otm["dte"][order][0]) if n else 0,
        moneyness=_floats(otm, "moneyness", order),
        strike=_floats(otm, "strike", order),
        iv=_floats(otm, "iv", order),
        iv_vendor=_floats(otm, "iv_vendor", order),
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
    exps: list[date] = []
    dtes: list[int] = []
    ivs: list[float] = []
    for e, m, iv in _expiry_curves(chain, min_quotes):
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
    dtes: list[int] = []
    rows: list[NDArray[np.float64]] = []
    for e, m, iv in _expiry_curves(chain, min_quotes, (min_dte, max_dte)):
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
    dates = rows["date"][order] if "date" in rows else np.asarray([], "datetime64[D]")
    return VrpSeries(
        source=source or "",
        date=dates.astype("datetime64[D]"),
        spot=_floats(rows, "spot", order),
        spot_parity=_floats(rows, "spot_parity", order),
        iv30=_floats(rows, "iv30_atm", order),
        rv21=_floats(rows, "rv21", order),
        rv21_fwd=_floats(rows, "rv21_fwd", order),
        vrp30=_floats(rows, "vrp30", order),
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
