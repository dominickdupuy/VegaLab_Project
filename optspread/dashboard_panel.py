"""Read-side views of the stitched SPX panel: coverage, smile, surface, term structure, VRP.

Lane: Ilan Danial (machinery). Reads what the panel build (W6-3) writes; never writes.

The panel lives in the team's shared data folder, not in the repository, and is
gigabytes, so this module reads only what a view needs: the two small tables
whole, and one year's chain partition, filtered to one date, for the chain views.

    panel/spx/year=YYYY/part.parquet   one row per contract per date, both sources
    panel/spx_daily.parquet            spot, IV30, trailing and forward RV21, VRP30
    panel/coverage.parquet             contracts per date per source; missing days flagged
    panel/seam_overlap.parquet         the seam report: both sources, same contracts, same days

Every aggregation works on plain numpy columns (``Columns``), so it is tested on
the ``dev`` install without pyarrow. Only ``read_table`` needs pyarrow, imported
when it is called. Volatilities are decimals on disk and stay decimals here; the
pages convert to vol points for display.

Nothing here interpolates a missing trading day. Coverage reports what is there,
and a day the build could not fill is listed, not smoothed over.
"""

from __future__ import annotations

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
        names = [str(p.relative_to(self.root)) for p in out]
        if not self.years():
            names.append(
                str((self.chain_dir / "year=YYYY" / "part.parquet").relative_to(self.root))
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
