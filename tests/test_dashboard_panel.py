"""Ilan Danial: the SPX panel views read the W6-3 schema and invent nothing (week 6).

The aggregations run on numpy columns, so most of this file runs on the plain
``dev`` install. The parquet round trip at the end needs pyarrow and is skipped
without it.

The rules under test are the ones a reader of the charts relies on: a missing
day is listed and never filled, a surface has holes where nothing was quoted
rather than extrapolated values, an expiry whose quotes do not straddle the
forward has no ATM point, and VRP is summarised only where the forward realised
window exists.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from dashboard_fixtures import (
    CHAIN_DTES,
    DB_DAYS,
    DB_MISSING,
    OM_DAYS,
    chain_columns,
    coverage_columns,
    daily_columns,
    merged,
    seam_columns,
)

from optspread.dashboard_panel import (
    DATA_ENV,
    DEFAULT_DATA_ROOT,
    PanelLayout,
    available_dates,
    chain_sources,
    coverage_summary,
    daily_sources,
    data_root,
    expiries,
    iv_surface,
    n_rows,
    nearest_expiry,
    normalise,
    otm_quotes,
    read_chain,
    read_coverage,
    read_daily,
    read_seam_overlap,
    seam,
    seam_report,
    select,
    smile,
    term_structure,
    vrp_series,
    vrp_stats,
)

DAY = date(2025, 7, 1)  # both sources quote this day


@pytest.fixture(scope="module")
def chain() -> dict[str, np.ndarray]:
    return normalise(
        merged(chain_columns((DAY,), "optionmetrics"), chain_columns((DAY,), "databento"))
    )


@pytest.fixture(scope="module")
def daily() -> dict[str, np.ndarray]:
    return normalise(
        merged(
            daily_columns(date(2025, 1, 2), 130, "optionmetrics", seed=1),
            daily_columns(date(2025, 7, 1), 30, "databento", seed=2),
        )
    )


# -- where the panel is ----------------------------------------------------------- #


def test_the_data_root_comes_from_the_environment(tmp_path: Path) -> None:
    assert data_root({DATA_ENV: str(tmp_path)}) == tmp_path
    assert data_root({}) == DEFAULT_DATA_ROOT
    assert "wrds_optionmetrics" in DEFAULT_DATA_ROOT.name
    assert data_root() in (DEFAULT_DATA_ROOT, Path(os.environ.get(DATA_ENV, "x")))


def test_an_unbuilt_panel_says_what_is_missing(tmp_path: Path) -> None:
    layout = PanelLayout(tmp_path)
    assert not layout.built and layout.years() == ()
    assert layout.missing() == (
        "panel/spx_daily.parquet",
        "panel/coverage.parquet",
        "panel/spx/year=YYYY/part.parquet",
    )
    assert read_chain(layout, DAY) == {}
    (layout.chain_dir / "year=2024").mkdir(parents=True)  # a partition with no file
    (layout.chain_dir / "notes").mkdir()
    assert layout.years() == ()


# -- columns -------------------------------------------------------------------- #


def test_normalise_types_dates_nullable_numbers_and_strings() -> None:
    cols = normalise(
        {
            "date": [date(2025, 7, 1)],
            "expiry": np.asarray(["2025-07-31T00:00:00"], dtype="datetime64[ns]"),
            "iv_vendor": [None],
            "open_interest": np.asarray([None, 3], dtype=object),
            "root": np.asarray(["SPX", None], dtype=object),
            "dte": [30],
        }
    )
    assert cols["date"].dtype == np.dtype("datetime64[D]")
    assert cols["expiry"][0] == np.datetime64("2025-07-31")
    assert np.isnan(cols["iv_vendor"][0])
    assert cols["open_interest"].dtype == np.float64 and cols["open_interest"][1] == 3.0
    assert cols["root"].tolist() == ["SPX", ""]
    assert cols["dte"].dtype.kind == "i"
    assert n_rows({}) == 0


# -- coverage ---------------------------------------------------------------------- #


def test_coverage_lists_missing_days_and_finds_the_overlap() -> None:
    summary = coverage_summary(normalise(coverage_columns()))
    assert summary.sources == ("databento", "optionmetrics")
    om = summary.spans[1]
    assert (om.first, om.last, om.n_days, om.n_missing) == (OM_DAYS[0], OM_DAYS[-1], 5, 0)
    db = summary.spans[0]
    assert (db.first, db.last, db.n_days, db.n_missing) == (DB_DAYS[0], DB_DAYS[-1], 4, 1)
    assert db.median_contracts == 2400.0
    assert summary.missing == tuple((d, "databento") for d in DB_MISSING)
    # the missing middle day is not counted as overlap: only days both have data
    assert summary.overlap == (date(2025, 7, 1), date(2025, 7, 3), 2)


def test_coverage_of_one_source_has_no_overlap() -> None:
    cols = coverage_columns()
    keep = [i for i, s in enumerate(cols["source"]) if s == "optionmetrics"]
    summary = coverage_summary(normalise({k: [v[i] for i in keep] for k, v in cols.items()}))
    assert summary.overlap is None and summary.missing == ()
    assert coverage_summary({}).spans == () and coverage_summary({}).overlap is None


def test_disjoint_sources_have_no_overlap_and_empty_sources_are_dropped() -> None:
    cov = normalise(
        {
            "date": [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 3)],
            "source": ["a", "b", "c"],
            "n_contracts": [10, 10, 0],
        }
    )
    summary = coverage_summary(cov)
    assert summary.sources == ("a", "b") and summary.overlap is None


def test_available_dates_skip_missing_days() -> None:
    cov = normalise(coverage_columns())
    db = available_dates(cov, "databento")
    assert np.datetime64(DB_MISSING[0]) not in db and len(db) == len(DB_DAYS)
    assert len(available_dates(cov)) == len(set(OM_DAYS) | set(DB_DAYS))
    assert len(available_dates({})) == 0


# -- one date's chain --------------------------------------------------------------- #


def test_select_by_source_and_root(chain: dict[str, np.ndarray]) -> None:
    assert chain_sources(chain) == ("databento", "optionmetrics")
    om = select(chain, "optionmetrics")
    assert set(om["source"].tolist()) == {"optionmetrics"}
    weekly = select(chain, roots=["SPXW"])
    assert set(weekly["dte"].tolist()) == {7}
    assert select({}, "x") == {} and chain_sources({}) == ()


def test_otm_quotes_keep_puts_below_and_calls_above_the_forward(
    chain: dict[str, np.ndarray],
) -> None:
    otm = otm_quotes(select(chain, "optionmetrics"))
    puts, calls = otm["cp"] == "P", otm["cp"] == "C"
    assert (otm["moneyness"][puts] < 1.0).all() and (otm["moneyness"][calls] >= 1.0).all()
    bad = dict(select(chain, "optionmetrics"))
    bad["iv"] = bad["iv"].copy()
    bad["iv"][:] = np.nan
    assert n_rows(otm_quotes(bad)) == 0
    assert otm_quotes({}) == {}


def test_expiries_and_the_nearest_to_thirty_days(chain: dict[str, np.ndarray]) -> None:
    om = select(chain, "optionmetrics")
    exps = expiries(om)
    assert tuple(e.dte for e in exps) == CHAIN_DTES
    near = nearest_expiry(om)
    assert near is not None and near.dte == 30
    assert near.carry_fit == pytest.approx(0.9999) and near.forward > 6200.0
    assert nearest_expiry(om, target_dte=100).dte == 120  # type: ignore[union-attr]
    assert nearest_expiry(om, min_quotes=10_000) is None
    assert expiries({}) == ()


def test_the_smile_is_skewed_and_carries_the_vendor_iv(chain: dict[str, np.ndarray]) -> None:
    om = select(chain, "optionmetrics")
    near = nearest_expiry(om)
    assert near is not None
    s = smile(om, near.expiry)
    assert s.dte == 30 and len(s.iv) == len(s.strike) == len(s.cp) > 10
    assert (np.diff(s.strike) >= 0).all()
    assert s.iv[0] > s.iv[len(s.iv) // 2]  # downside puts richer than at the money
    assert s.has_vendor and np.allclose(s.iv_vendor - s.iv, 0.001)
    db = select(chain, "databento")
    assert not smile(db, near.expiry).has_vendor  # Databento carries no vendor IV
    empty = smile(om, date(2030, 1, 1))
    assert empty.dte == 0 and empty.iv.size == 0 and not empty.has_vendor


def test_term_structure_interpolates_at_the_forward_only(chain: dict[str, np.ndarray]) -> None:
    om = select(chain, "optionmetrics")
    ts = term_structure(om)
    assert tuple(ts.dte.tolist()) == CHAIN_DTES
    expected = 0.16 + 0.02 * np.sqrt(np.asarray(CHAIN_DTES) / 365.0)
    np.testing.assert_allclose(ts.atm_iv, expected, atol=5e-3)
    # quotes that do not straddle the forward have no ATM point
    calls_only = {k: v[om["cp"] == "C"] for k, v in om.items()}
    assert term_structure(calls_only).dte.size == 0
    assert term_structure(om, min_quotes=10_000).dte.size == 0


def test_the_surface_has_holes_rather_than_extrapolation(chain: dict[str, np.ndarray]) -> None:
    om = select(chain, "optionmetrics")
    grid = np.linspace(0.5, 1.5, 21)
    surf = iv_surface(om, grid, min_dte=20)
    assert tuple(surf.dte.tolist()) == (30, 60, 120)
    assert surf.iv.shape == (3, 21)
    assert np.isnan(surf.iv[:, 0]).all() and np.isnan(surf.iv[:, -1]).all()  # beyond 0.8, 1.2
    assert np.isfinite(surf.iv[:, 10]).all()
    default = iv_surface(om)
    assert default.moneyness.shape == (41,) and default.iv.shape[0] == len(CHAIN_DTES)
    assert iv_surface(om, min_quotes=10_000).iv.shape == (0, 41)


# -- IV30, realised vol and VRP ------------------------------------------------------- #


def test_vrp_series_per_source_in_date_order(daily: dict[str, np.ndarray]) -> None:
    assert daily_sources(daily) == ("databento", "optionmetrics")
    om = vrp_series(daily, "optionmetrics")
    assert om.source == "optionmetrics" and len(om.date) == 130
    np.testing.assert_allclose(om.spot_parity - om.spot, 0.5)
    assert (np.diff(om.date.astype("int64")) > 0).all()
    np.testing.assert_allclose(om.vrp30, om.iv30 - om.rv21_fwd, equal_nan=True)
    both = vrp_series(daily)
    assert len(both.date) == 160 and both.source == ""


def test_vrp_stats_skip_days_without_a_forward_window(daily: dict[str, np.ndarray]) -> None:
    stats = vrp_stats(vrp_series(daily, "optionmetrics"))
    assert stats.n_days == 127  # the last three have no forward realised vol
    assert 1.0 < stats.mean_vol_pts < 5.0  # the fixture plants about 3 vol points
    assert 0.5 < stats.share_positive <= 1.0
    assert stats.first == date(2025, 1, 2) and stats.last is not None
    empty = vrp_stats(vrp_series({}))
    assert empty.n_days == 0 and np.isnan(empty.mean_vol_pts) and empty.first is None


def test_a_series_without_a_column_reads_as_nan() -> None:
    cols = normalise({"date": [date(2025, 1, 2)], "iv30_atm": [0.2]})
    series = vrp_series(cols, "optionmetrics")  # no source column: nothing to filter on
    assert np.isnan(series.vrp30).all() and series.iv30[0] == 0.2
    assert daily_sources(cols) == ("",) and daily_sources({}) == ()


def test_the_seam_compares_both_sources_on_shared_days(daily: dict[str, np.ndarray]) -> None:
    s = seam(daily)
    assert s is not None and s.sources == ("databento", "optionmetrics")
    assert len(s.date) > 0 and s.date.min() >= np.datetime64("2025-07-01")
    assert s.diff_vol_pts.shape == s.date.shape
    one = {k: v[daily["source"] == "optionmetrics"] for k, v in daily.items()}
    assert seam(one) is None
    apart = normalise(
        merged(
            daily_columns(date(2020, 1, 2), 5, "optionmetrics"),
            daily_columns(date(2025, 1, 2), 5, "databento"),
        )
    )
    assert seam(apart) is None


def test_the_seam_report_passes_the_build_statistics_through() -> None:
    report = seam_report(normalise(seam_columns()))
    assert report is not None and report.n_days == 2
    assert report.median("iv_diff_median_volpts") == pytest.approx(0.2)
    assert report.median("iv_diff_p95_abs_volpts") == pytest.approx(0.95)
    assert report.median("n_matched") == pytest.approx(1800.5)
    partial = seam_report(normalise({"date": [date(2025, 7, 1)], "n_matched": [5]}))
    assert partial is not None and np.isnan(partial.median("forward_diff_median_bp"))
    assert np.isnan(partial.median("not_a_column"))
    assert seam_report({}) is None


# -- the parquet round trip (needs pyarrow) -------------------------------------------- #


def test_the_panel_reads_back_one_partition_and_one_day(panel_root: Path) -> None:
    pytest.importorskip("pyarrow")
    layout = PanelLayout(panel_root)
    assert layout.built and layout.years() == (2025,) and layout.missing() == ()
    chain = read_chain(layout, DAY)
    assert set(chain["date"].tolist()) == {DAY}
    assert chain_sources(chain) == ("databento", "optionmetrics")
    assert chain["expiry"].dtype == np.dtype("datetime64[D]")
    assert "theta" not in chain  # only the columns the views use are read
    assert np.isnan(select(chain, "databento")["iv_vendor"]).all()
    assert read_chain(layout, date(2019, 1, 2)) == {}
    summary = coverage_summary(read_coverage(layout))
    assert summary.overlap == (date(2025, 7, 1), date(2025, 7, 3), 2)
    assert daily_sources(read_daily(layout)) == ("databento", "optionmetrics")
    assert n_rows(read_chain(layout, date(2025, 7, 2))) > 0  # OptionMetrics only that day
    report = seam_report(read_seam_overlap(layout))
    assert report is not None and report.n_days == 2
    assert read_seam_overlap(PanelLayout(panel_root / "absent")) == {}
