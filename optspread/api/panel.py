"""API payloads for the SPX panel, and ``PanelReader``, which reads them off the panel's tables.

Lane: Ilan Danial (machinery).

Every payload read from the panel names the file its numbers came from. The data page's
question models ``PanelReader`` builds are in ``optspread.api.readings``. Moved from
``optspread.webapi``.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optspread.api._base import DateText, Out, _from_attrs, _iso
from optspread.api.readings import (
    SIGMA_FROM,
    FredRowOut,
    PanelFileCheckOut,
    PremiumNowOut,
    PremiumOut,
    QualityOut,
    ShapeOut,
    SimulatedOut,
    simulated_surface,
)
from optspread.config import MarketConfig
from optspread.dashboard import panel_views as panel
from optspread.dashboard.readings import (
    SMOOTH_DAYS,
    PremiumSeries,
    ShapeHistory,
    premium_now,
    premium_reading,
    premium_series,
    quality_reading,
    shape_history,
    shape_on,
    shape_reading,
)

# -- the SPX panel ---------------------------------------------------------------- #


class PanelFileOut(Out):
    path: str
    present: bool
    modified: str | None


class DataStatusOut(Out):
    root: str
    built: bool
    missing: list[str]
    years: list[int]
    files: list[PanelFileOut]


class SpanOut(Out):
    source: str
    first: DateText
    last: DateText
    n_days: int
    n_missing: int
    median_contracts: float


class StripOut(Out):
    source: str
    date: list[str]
    n_contracts: list[float]
    missing: list[bool]


class CoverageOut(Out):
    spans: list[SpanOut]
    missing: list[tuple[str, str]]
    overlap: tuple[str, str, int] | None
    strips: list[StripOut]
    source: str


class ExpiryOut(Out):
    expiry: DateText
    dte: int
    n_quotes: int
    forward: float | None
    carry_fit: str  # the build's carry-fit label, e.g. "near+r_ref"; empty when absent


class ChainIndexOut(Out):
    date: str
    sources: list[str]
    roots: list[str]
    expiries: list[ExpiryOut]
    default_expiry: str | None
    source: str


class SmileOut(Out):
    date: str
    expiry: str
    dte: int
    data_source: str
    moneyness: list[float | None]
    strike: list[float | None]
    iv: list[float | None]
    iv_vendor: list[float | None]
    cp: list[str]
    has_vendor: bool
    forward: float | None
    source: str


class SurfaceOut(Out):
    date: str
    data_source: str
    moneyness: list[float]
    dte: list[int]
    iv: list[list[float | None]]
    source: str


class TermOut(Out):
    date: str
    data_source: str
    expiry: list[str]
    dte: list[int]
    atm_iv: list[float | None]
    source: str


class SeriesOut(Out):
    data_source: str
    date: list[str]
    spot: list[float | None]
    spot_parity: list[float | None]
    iv30: list[float | None]
    rv21: list[float | None]
    rv21_fwd: list[float | None]
    vrp30: list[float | None]


class VrpStatsOut(Out):
    n_days: int
    mean_vol_pts: float | None
    median_vol_pts: float | None
    share_positive: float | None
    first: DateText | None
    last: DateText | None


class DailyOut(Out):
    series: list[SeriesOut]
    stats: dict[str, VrpStatsOut]
    source: str


class SeamOut(Out):
    available: bool
    date: list[str]
    stats: dict[str, list[float | None]]
    medians: dict[str, float | None]
    # the headline (panel_views.SEAM_HEADLINE): each day's 95th-percentile |IV gap|,
    # averaged over the overlap days, and the worst day; what docs/DATA.md quotes
    p95_mean: float | None = None
    p95_worst: float | None = None
    iv30_sources: list[str]
    iv30_date: list[str]
    iv30_diff_vol_pts: list[float | None]
    source: str


class GridOut(Out):
    """One day of the gridded surface: what the 3D view draws."""

    date: str
    k: list[float]  # (K,) log-moneyness ln(K / F)
    tenor_days: list[int]  # (T,) calendar days
    iv: list[list[float | None]]  # (T, K); null where nothing was quoted
    source: str


class GridFrameOut(Out):
    date: str
    iv: list[list[float | None]]


class GridFramesOut(Out):
    """Consecutive days of the gridded surface on one set of axes, for playback."""

    k: list[float]
    tenor_days: list[int]
    frames: list[GridFrameOut]
    more: bool  # the window was cut at max_frames; ask again from the last date
    source: str


class LandscapeOut(Out):
    """ATM IV over date x tenor: the term structure through time."""

    date: list[str]  # (D,) the last trading day of each row's bucket
    tenor_days: list[int]  # (T,)
    iv: list[list[float | None]]  # (D, T)
    stride: int  # trading days averaged into each row
    n_days: int  # trading days in the window before averaging
    first: str | None
    last: str | None
    source: str


DATA_MANIFEST_FILE = "DATA_MANIFEST.json"


class PanelNotBuiltError(LookupError):
    """The view needs a panel file the build has not written yet."""


def _day_of(value: np.datetime64) -> date:
    out = value.astype("datetime64[D]").astype(date)
    assert isinstance(out, date)
    return out


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """docs/DATA_MANIFEST.json as a dict; empty, with the reason, when it does not read."""
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {}, str(exc)
    return (raw, "") if isinstance(raw, dict) else ({}, f"{path.name} is not a JSON object")


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


# The largest windows one request may ask for: a year of gridded frames, and every
# trading day of the panel (3,939 days to 2025-08-29) with room to spare.
MAX_FRAMES = 252
MAX_DATES = 4_200


class PanelReader:
    """The SPX panel, read lazily and cached by file modification time.

    The two small tables are read whole once per rebuild; chain partitions are read
    one date at a time and the last few dates kept, which is what flicking through
    a date picker needs.
    """

    def __init__(
        self,
        root: Path,
        chain_cache: int = 8,
        read: Callable[..., panel.Columns] = panel.read_table,
    ) -> None:
        self.layout = panel.PanelLayout(root)
        self._read = read  # injected so the payloads are tested without pyarrow
        self._tables: dict[str, tuple[float, panel.Columns]] = {}
        self._chains: OrderedDict[tuple[date, float], panel.Columns] = OrderedDict()
        self._chain_cache = chain_cache
        self._computed: dict[str, tuple[float, Any]] = {}

    def _table(self, name: str, path: Path) -> panel.Columns:
        if not path.is_file():
            raise PanelNotBuiltError(str(path))
        stamp = _mtime(path)
        cached = self._tables.get(name)
        if cached is None or cached[0] != stamp:
            cached = (stamp, self._read(path))
            self._tables[name] = cached
        return cached[1]

    def daily(self) -> panel.Columns:
        return self._table("daily", self.layout.daily)

    def coverage(self) -> panel.Columns:
        return self._table("coverage", self.layout.coverage)

    def seam(self) -> panel.Columns:
        if not self.layout.seam_overlap.is_file():
            return {}
        return self._table("seam", self.layout.seam_overlap)

    def chain(self, day: date) -> panel.Columns:
        path = self.layout.chain_file(day.year)
        if not path.is_file():
            raise PanelNotBuiltError(str(path))
        key = (day, _mtime(path))
        if key not in self._chains:
            self._chains[key] = self._read(path, panel.CHAIN_COLUMNS, day)
            while len(self._chains) > self._chain_cache:
                self._chains.popitem(last=False)
        self._chains.move_to_end(key)
        return self._chains[key]

    # -- payloads ------------------------------------------------------------- #

    def status(self) -> DataStatusOut:
        layout = self.layout
        files = [layout.daily, layout.coverage, layout.seam_overlap]
        files += [layout.chain_file(y) for y in layout.years()]
        return DataStatusOut(
            root=str(layout.root),
            built=layout.built,
            missing=list(layout.missing()),
            years=list(layout.years()),
            files=[
                PanelFileOut(
                    path=str(f.relative_to(layout.root)),
                    present=f.is_file(),
                    modified=str(np.datetime64(int(_mtime(f)), "s")) if f.is_file() else None,
                )
                for f in files
            ],
        )

    def coverage_out(self) -> CoverageOut:
        cov = self.coverage()
        summary = panel.coverage_summary(cov)
        strips = []
        for src in summary.sources:
            mine = panel.take(cov, cov["source"] == src)
            order = np.argsort(mine["date"], kind="stable")
            strips.append(
                StripOut(
                    source=src,
                    date=_iso(mine["date"][order]),
                    n_contracts=[float(v) for v in mine["n_contracts"][order]],
                    missing=[bool(v) for v in mine["missing"][order]]
                    if "missing" in mine
                    else [False] * len(order),
                )
            )
        return CoverageOut(
            spans=[SpanOut.model_validate(s, from_attributes=True) for s in summary.spans],
            missing=[(d.isoformat(), s) for d, s in summary.missing],
            overlap=(
                summary.overlap[0].isoformat(),
                summary.overlap[1].isoformat(),
                summary.overlap[2],
            )
            if summary.overlap
            else None,
            strips=strips,
            source=str(self.layout.coverage),
        )

    def dates(self, data_source: str | None = None) -> list[str]:
        return _iso(panel.available_dates(self.coverage(), data_source))

    def _selected(self, day: date, data_source: str | None, roots: Sequence[str]) -> panel.Columns:
        chain = self.chain(day)
        if data_source is None:
            sources = panel.chain_sources(chain)
            data_source = sources[-1] if sources else None  # optionmetrics sorts last
        return panel.select(chain, data_source, roots)

    def chain_index(
        self, day: date, data_source: str | None, roots: Sequence[str]
    ) -> ChainIndexOut:
        chain = self.chain(day)
        rows = self._selected(day, data_source, roots)
        near = panel.nearest_expiry(rows)
        return ChainIndexOut(
            date=day.isoformat(),
            sources=list(panel.chain_sources(chain)),
            roots=sorted(set(chain["root"].tolist())) if panel.n_rows(chain) else [],
            expiries=[
                ExpiryOut.model_validate(e, from_attributes=True) for e in panel.expiries(rows)
            ],
            default_expiry=near.expiry.isoformat() if near else None,
            source=str(self.layout.chain_file(day.year)),
        )

    def _source_name(self, rows: panel.Columns) -> str:
        names = panel.chain_sources(rows)
        return names[0] if len(names) == 1 else ", ".join(names)

    def smile_out(
        self, day: date, expiry: date | None, data_source: str | None, roots: Sequence[str]
    ) -> SmileOut:
        rows = self._selected(day, data_source, roots)
        if expiry is None:
            near = panel.nearest_expiry(rows)
            if near is None:
                raise PanelNotBuiltError(f"no usable expiry on {day}")
            expiry = near.expiry
        s = panel.smile(rows, expiry)
        fwd = next((e.forward for e in panel.expiries(rows) if e.expiry == expiry), None)
        return SmileOut(
            date=day.isoformat(),
            expiry=expiry.isoformat(),
            dte=s.dte,
            data_source=self._source_name(rows),
            moneyness=s.moneyness.tolist(),
            strike=s.strike.tolist(),
            iv=s.iv.tolist(),
            iv_vendor=s.iv_vendor.tolist(),
            cp=[str(c) for c in s.cp],
            has_vendor=s.has_vendor,
            forward=fwd,
            source=str(self.layout.chain_file(day.year)),
        )

    def surface_out(
        self,
        day: date,
        data_source: str | None,
        roots: Sequence[str],
        min_dte: int = 1,
        max_dte: int = 365,
    ) -> SurfaceOut:
        if min_dte > max_dte:
            raise ValueError(f"min_dte {min_dte} is above max_dte {max_dte}")
        rows = self._selected(day, data_source, roots)
        surf = panel.iv_surface(rows, min_dte=min_dte, max_dte=max_dte)
        return SurfaceOut(
            date=day.isoformat(),
            data_source=self._source_name(rows),
            moneyness=[float(m) for m in surf.moneyness],
            dte=[int(d) for d in surf.dte],
            iv=surf.iv.tolist(),
            source=str(self.layout.chain_file(day.year)),
        )

    def term_out(self, day: date, data_source: str | None, roots: Sequence[str]) -> TermOut:
        rows = self._selected(day, data_source, roots)
        ts = panel.term_structure(rows)
        return TermOut(
            date=day.isoformat(),
            data_source=self._source_name(rows),
            expiry=[e.isoformat() for e in ts.expiry],
            dte=[int(d) for d in ts.dte],
            atm_iv=ts.atm_iv.tolist(),
            source=str(self.layout.chain_file(day.year)),
        )

    def daily_out(self, start: date | None = None, end: date | None = None) -> DailyOut:
        daily = self.daily()
        if panel.n_rows(daily) and (start or end):
            keep = np.ones(panel.n_rows(daily), dtype=bool)
            if start:
                keep &= daily["date"] >= np.datetime64(start, "D")
            if end:
                keep &= daily["date"] <= np.datetime64(end, "D")
            daily = panel.take(daily, keep)
        series: list[SeriesOut] = []
        stats: dict[str, VrpStatsOut] = {}
        for src in panel.daily_sources(daily):
            s = panel.vrp_series(daily, src or None)
            # the series' arrays are read as they are: NaN where a day has no value
            series.append(_from_attrs(SeriesOut, s, data_source=src, date=_iso(s.date)))
            stats[src] = VrpStatsOut.model_validate(panel.vrp_stats(s), from_attributes=True)
        return DailyOut(series=series, stats=stats, source=str(self.layout.daily))

    def seam_out(self) -> SeamOut:
        report = panel.seam_report(self.seam())
        iv_seam = panel.seam(self.daily()) if self.layout.daily.is_file() else None
        return SeamOut(
            available=report is not None,
            date=_iso(report.date) if report else [],
            stats={k: v.tolist() for k, v in report.stats.items()} if report else {},
            medians={k: report.median(k) for k in panel.SEAM_STATS} if report else {},
            p95_mean=report.mean(panel.SEAM_HEADLINE) if report else None,
            p95_worst=report.worst(panel.SEAM_HEADLINE) if report else None,
            iv30_sources=list(iv_seam.sources) if iv_seam else [],
            iv30_date=_iso(iv_seam.date) if iv_seam else [],
            iv30_diff_vol_pts=iv_seam.diff_vol_pts.tolist() if iv_seam else [],
            source=str(self.layout.seam_overlap),
        )

    # -- the 3D views ---------------------------------------------------------- #

    def surface_grid(self) -> panel.Columns:
        return self._table("surface_grid", self.layout.surface_grid)

    def atm_term(self) -> panel.Columns:
        return self._table("atm_term", self.layout.atm_term)

    def _derived(self, name: str, path: Path, build: Callable[[], Any]) -> Any:
        """A value computed from a table, kept until the file changes."""
        stamp = _mtime(path)
        cached = self._computed.get(name)
        if cached is None or cached[0] != stamp:
            cached = (stamp, build())
            self._computed[name] = cached
        return cached[1]

    def _grid_axes(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        grid = self.surface_grid()
        axes: tuple[NDArray[np.float64], NDArray[np.int64]] = self._derived(
            "grid_axes", self.layout.surface_grid, lambda: panel.grid_axes(grid)
        )
        return axes

    def _grid_dates(self) -> NDArray[np.datetime64]:
        grid = self.surface_grid()
        dates: NDArray[np.datetime64] = self._derived(
            "grid_dates", self.layout.surface_grid, lambda: panel.grid_dates(grid)
        )
        return dates

    def grid_dates(self) -> list[str]:
        return _iso(self._grid_dates())

    def grid_day(self, day: date) -> panel.GridSurface:
        if np.datetime64(day, "D") not in self._grid_dates():
            raise PanelNotBuiltError(f"no gridded surface on {day} in {self.layout.surface_grid}")
        return panel.grid_surface(self.surface_grid(), day, self._grid_axes())

    def grid_out(self, day: date) -> GridOut:
        g = self.grid_day(day)
        return GridOut(
            date=day.isoformat(),
            k=[float(x) for x in g.k],
            tenor_days=[int(t) for t in g.tenor_days],
            iv=g.iv.tolist(),
            source=str(self.layout.surface_grid),
        )

    def grid_frames(self, start: date, end: date | None, max_frames: int = 126) -> GridFramesOut:
        """The gridded surface on every trading day from ``start``, up to ``max_frames``."""
        if not 1 <= max_frames <= MAX_FRAMES:
            raise ValueError(f"max_frames must be between 1 and {MAX_FRAMES}, got {max_frames}")
        dates = self._grid_dates()
        keep = dates >= np.datetime64(start, "D")
        if end is not None:
            keep &= dates <= np.datetime64(end, "D")
        days = dates[keep]
        grid = self.surface_grid()
        k, tenors = self._grid_axes()
        window = (
            panel.take(grid, (grid["date"] >= days[0]) & (grid["date"] <= days[-1]))
            if len(days)
            else {}
        )
        frames = [
            GridFrameOut(
                date=str(d),
                iv=panel.grid_surface(window, date.fromisoformat(str(d)), (k, tenors)).iv.tolist(),
            )
            for d in days[:max_frames]
        ]
        return GridFramesOut(
            k=[float(x) for x in k],
            tenor_days=[int(t) for t in tenors],
            frames=frames,
            more=len(days) > max_frames,
            source=str(self.layout.surface_grid),
        )

    def landscape_out(
        self, start: date | None = None, end: date | None = None, max_dates: int = 420
    ) -> LandscapeOut:
        if not 2 <= max_dates <= MAX_DATES:
            raise ValueError(f"max_dates must be between 2 and {MAX_DATES}, got {max_dates}")
        if start and end and start > end:
            raise ValueError(f"start {start} is after end {end}")
        term = self.atm_term()
        dates, curves = self._derived(
            "atm_curves", self.layout.atm_term, lambda: panel.atm_curves(term)
        )
        land = panel.atm_landscape(dates, curves, start=start, end=end, max_dates=max_dates)
        within = (dates >= np.datetime64(start, "D")) if start else np.ones(len(dates), bool)
        if end:
            within &= dates <= np.datetime64(end, "D")
        return LandscapeOut(
            date=_iso(land.date),
            tenor_days=[int(t) for t in land.tenor_days],
            iv=land.iv.tolist(),
            stride=land.stride,
            n_days=int(within.sum()),
            first=str(dates[within][0]) if within.any() else None,
            last=str(dates[within][-1]) if within.any() else None,
            source=str(self.layout.atm_term),
        )

    # -- the data page's questions ------------------------------------------------ #

    def premium_out(self, mapping: dict[str, Any] | None = None) -> PremiumOut:
        """The premium day by day; the headline and interval from ``mapping`` when given."""
        daily = self.daily()
        s: PremiumSeries = self._derived(
            "premium", self.layout.daily, lambda: premium_series(daily)
        )
        now = premium_now(s)
        ok = np.isfinite(s.premium)
        head = (mapping or {}).get("headline") or {}
        mean = head.get("mean", float(s.premium[ok].mean()) if ok.any() else None)
        low, high = head.get("mean_low"), head.get("mean_high")
        axis = [float(a) for a in (mapping or {}).get("sweep_axis", [])]
        from optspread import vrp

        bracket = list((mapping or {}).get("bracket") or vrp.bracket(mean or 0.0, axis))
        first = head.get("first") or (str(s.date[ok][0]) if ok.any() else "")
        last = head.get("last") or (str(s.date[ok][-1]) if ok.any() else "")
        reading = (
            premium_reading(mean, low, high, first, last, bracket, axis, now)
            if mean is not None
            else "The daily table has no day with a complete forward window yet."
        )

        def pts(x: NDArray[np.float64]) -> list[float]:
            # a thousandth of a vol point is below anything the chart can show
            return [round(v, 3) for v in x.tolist()]

        return PremiumOut(
            date=_iso(s.date),
            premium=pts(s.premium),
            smooth=pts(s.smooth),
            iv30=pts(s.iv30 * 100.0),
            rv_fwd=pts(s.rv_fwd * 100.0),
            rv_trailing=pts(s.rv_trailing * 100.0),
            smooth_days=SMOOTH_DAYS,
            now=PremiumNowOut(
                date=now.day.isoformat(),
                iv30=now.iv30 * 100.0,
                rv_trailing=now.rv_trailing * 100.0,
                priced=now.priced,
                measured_date=now.measured_date.isoformat() if now.measured_date else None,
                measured=now.measured,
                recent=now.recent,
                recent_share_below=now.recent_share_below,
            )
            if now
            else None,
            mean=mean,
            mean_low=low,
            mean_high=high,
            bracket=bracket,
            sweep_axis=axis,
            reading=reading,
            source=str(self.layout.daily),
        )

    def shape_out(self, day: date) -> ShapeOut:
        grid = self.surface_grid()
        h: ShapeHistory = self._derived(
            "shape", self.layout.surface_grid, lambda: shape_history(grid)
        )
        sd = shape_on(h, day)
        if sd is None:
            raise PanelNotBuiltError(f"no gridded surface on {day} in {self.layout.surface_grid}")
        return _from_attrs(
            ShapeOut,
            sd,
            date=day.isoformat(),
            reading=shape_reading(sd),
            source=str(self.layout.surface_grid),
        )

    def quality_out(self, docs: Path) -> QualityOut:
        """The build's version and its files against docs/DATA_MANIFEST.json, and FRED."""
        from optspread.realdata import panel_digests

        manifest, _ = _read_manifest(docs / DATA_MANIFEST_FILE)
        files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
        version = manifest.get("panel_version")
        actual = panel_digests(self.layout.panel)
        checked = []
        for name, digest in sorted(actual.items()):
            rec = files.get(name, {})
            recorded = rec.get("sha256") if isinstance(rec, dict) else None
            checked.append(
                PanelFileCheckOut(
                    path=name,
                    recorded=recorded,
                    actual=digest,
                    match=(recorded == digest) if recorded else None,
                )
            )
        verdicts = [c.match for c in checked]
        of_record = (
            None
            if not verdicts or any(v is None for v in verdicts)
            else all(bool(v) for v in verdicts)
        )
        fred_path = self.layout.panel / "fred_check.parquet"
        fred: list[FredRowOut] = []
        if fred_path.is_file():
            cols = self._table("fred", fred_path)
            order: Sequence[int] = (
                np.lexsort((cols["source"], cols["year"])).tolist() if panel.n_rows(cols) else []
            )
            for i in order:

                def col(name: str, i: int = int(i)) -> float | None:
                    return float(cols[name][i]) if name in cols else None

                fred.append(
                    FredRowOut(
                        year=int(cols["year"][i]),
                        source=str(cols["source"][i]),
                        rate_minus_bill_bp=col("rate_minus_bill_bp"),
                        rate_minus_sofr_bp=col("rate_minus_sofr_bp"),
                        spot_vs_sp500_bp=col("spot_vs_sp500_bp"),
                    )
                )
        reading = ""
        if self.layout.coverage.is_file():
            cov = panel.coverage_summary(self.coverage())
            days = panel.available_dates(self.coverage())
            report = panel.seam_report(self.seam())
            reading = quality_reading(
                n_days=len(days),
                first=str(days[0]) if len(days) else "",
                last=str(days[-1]) if len(days) else "",
                # a day one vendor misses and the other quotes is not missing from the panel
                n_missing=len({d for d, _ in cov.missing} - {_day_of(x) for x in days}),
                seam_days=report.n_days if report else 0,
                seam_p95=report.mean(panel.SEAM_HEADLINE) if report else None,
                seam_worst=report.worst(panel.SEAM_HEADLINE) if report else None,
                panel_version=int(version) if isinstance(version, int) else None,
                of_record=of_record,
            )
        return QualityOut(
            panel_version=int(version) if isinstance(version, int) else None,
            manifest_files=len(files),
            checked=checked,
            of_record=of_record,
            fred=fred,
            fred_available=fred_path.is_file(),
            reading=reading,
            source=str(docs / DATA_MANIFEST_FILE),
        )

    def _daily_row(self, day: date) -> dict[str, float]:
        """The day's spot, RV21 and IV30, OptionMetrics first; empty when not in the table."""
        if not self.layout.daily.is_file():
            return {}
        daily = self.daily()
        if not panel.n_rows(daily):
            return {}
        mine = daily["date"] == np.datetime64(day, "D")
        if not mine.any():
            return {}
        idx = np.flatnonzero(mine)
        if "source" in daily:
            om = idx[daily["source"][idx] == "optionmetrics"]
            idx = om if len(om) else idx
        i = int(idx[0])
        return {
            name: float(daily[col][i])
            for name, col in (("spot", "spot"), ("rv21", "rv21"), ("iv30", "iv30_atm"))
            if col in daily
        }

    def simulated_out(self, day: date, premium: float, sigma_from: str = "config") -> SimulatedOut:
        if sigma_from not in SIGMA_FROM:
            raise ValueError(
                f"sigma_from must be one of {', '.join(SIGMA_FROM)}, got {sigma_from!r}"
            )
        real = self.grid_day(day)
        row = self._daily_row(day)
        base = MarketConfig()
        spot = row.get("spot", math.nan)
        spot_from = "spx_daily.parquet" if math.isfinite(spot) and spot > 0 else "config"
        if spot_from == "config":
            spot = base.spot0
        if sigma_from == "config":
            sigma = base.sigma
        else:
            name = "rv21" if sigma_from == "rv21" else "iv30"
            value = row.get(name, math.nan)
            if not math.isfinite(value):
                raise ValueError(f"no {name} on {day} in {self.layout.daily}")
            sigma = value if sigma_from == "rv21" else value - premium / 100.0
        if not sigma > 0.0:
            raise ValueError(f"sigma {sigma:.4f} is not positive; lower the premium")
        cfg = base.model_copy(update={"spot0": spot, "sigma": sigma, "iv_premium_vol_pts": premium})
        return simulated_surface(day, cfg, real, spot_from, sigma_from)
