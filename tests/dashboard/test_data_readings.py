"""Hemdutt Rao: the data page's four questions, each answered in one sentence (week 8).

The page leads with the questions a reader brings to the real data: the premium the
market pays for volatility, the surface on a day, how the simulator differs from the
market, and what the seller would open on a close. Each view carries one plain
sentence computed from the panel. These tests pin the numbers behind each sentence
and the sentence itself, on synthetic columns and on the fixture panel, so what the
page says is what the data say.

Where a payload's figure can be undefined, the check that it is sent as null is made on
the bytes the server sends (``_sent``, which needs the ``dashboard`` extra), whatever the
model holds; ``_finite`` is true of a number and false of None or NaN alike.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dashboard_fixtures import daily_columns, merged
from pydantic import BaseModel

from optspread import vrp
from optspread.api.panel import PanelNotBuiltError, PanelReader
from optspread.api.seller import seller_out
from optspread.dashboard.panel_views import Columns, normalise
from optspread.dashboard.readings import (
    axis_place,
    premium_now,
    premium_reading,
    premium_series,
    quality_reading,
    seller_reading,
    shape_history,
    shape_on,
    shape_reading,
    share_below,
    signed,
    sim_reading,
    sim_regions,
    trailing_mean,
)

M = "\u2212"  # the typographic minus the sentences use
REPO_DOCS = Path(__file__).resolve().parent.parent.parent / "docs"


def _sent(out: BaseModel) -> Any:
    """``out`` as the server sends it, parsed as strictly as a browser parses JSON.

    The model is returned from a route of the server's own app, so the bytes are the ones
    its redacting response writes: an undefined figure is null there, whatever the model
    holds. A NaN or an Infinity in the bytes fails the parse."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from optspread.api.server import Settings, create_app

    def refuse(token: str) -> Any:
        raise ValueError(f"{token} is not JSON")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            root / "runs",
            root / "docs",
            root / "data",
            root / "web",
            root / "README.md",
            attribution=root / "attribution",
        )
        app = create_app(settings)
        app.add_api_route("/sent", lambda: out, response_model=type(out))
        response = TestClient(app, base_url="http://127.0.0.1").get("/sent")
    assert response.status_code == 200, response.text
    return json.loads(response.content, parse_constant=refuse)


def _finite(x: float | None) -> bool:
    """A figure is there: a finite number, not None (undefined today) or NaN (undefined
    once the models keep what the views computed)."""
    return x is not None and math.isfinite(x)


# -- the small helpers ------------------------------------------------------------- #


@pytest.mark.behaviour
def test_signed_numbers_carry_a_sign_and_the_typographic_minus() -> None:
    assert signed(0.6396) == "+0.64" and signed(-0.3437) == f"{M}0.34"
    assert signed(0.0) == "+0.00" and signed(-2.68, 1) == f"{M}2.7"


@pytest.mark.behaviour
def test_the_trailing_mean_skips_holes_and_ends_where_the_data_do() -> None:
    x = np.asarray([1.0, 2.0, 3.0, np.nan, 5.0, np.nan])
    out = trailing_mean(x, window=2, min_count=1)
    assert out[:3].tolist() == [1.0, 1.5, 2.5] and out[4] == 5.0
    assert np.isnan(out[3]) and np.isnan(out[5])  # no value invented on a day without one
    assert np.isnan(trailing_mean(np.asarray([1.0, 2.0]), window=3)[0])  # 1 of 3 < 2/3


@pytest.mark.behaviour
def test_share_below_is_strict_and_ignores_nan() -> None:
    v_ = np.asarray([1.0, 2.0, 3.0, np.nan])
    assert share_below(v_, 2.0) == pytest.approx(1 / 3) and share_below(v_, 9.0) == 1.0
    assert math.isnan(share_below(np.asarray([np.nan]), 1.0))
    assert math.isnan(share_below(v_, float("nan")))


# -- question 1: the premium --------------------------------------------------------- #


def _daily() -> Columns:
    return normalise(
        merged(
            daily_columns(date(2025, 1, 2), 130, "optionmetrics", seed=1),
            daily_columns(date(2025, 7, 1), 30, "databento", seed=2),
        )
    )


@pytest.mark.invariant
def test_the_premium_is_the_mappings_definition_one_row_per_date() -> None:
    daily = _daily()
    s = premium_series(daily)
    assert (
        len(s.date) == len(np.unique(daily["date"]))
        and (np.diff(s.date) > np.timedelta64(0, "D")).all()
    )
    # on a day both sources quote, OptionMetrics' row is the one kept
    om = (daily["source"] == "optionmetrics") & (daily["date"] == np.datetime64("2025-07-01"))
    i = int(np.flatnonzero(s.date == np.datetime64("2025-07-01"))[0])
    assert s.iv30[i] == daily["iv30_atm"][om][0]
    # the mapping's premium: IV30 less the zero-mean forward RV over c21, in vol points
    spot = np.asarray([daily["spot"][(daily["date"] == d)][0] for d in s.date], dtype=np.float64)
    want = 100.0 * (s.iv30 - vrp.forward_realised(spot, s.date, unbiased=True))
    np.testing.assert_allclose(s.premium, want, equal_nan=True)
    assert np.isnan(s.premium[-vrp.WINDOW :]).all()  # no complete forward window yet
    trail = vrp.trailing_realised(spot) / vrp.vol_bias(vrp.WINDOW)
    np.testing.assert_allclose(s.priced, 100.0 * (s.iv30 - trail), equal_nan=True)
    ok = np.isfinite(s.smooth)
    assert ok.any() and np.isfinite(s.premium[ok]).all()
    j = int(np.flatnonzero(ok)[-1])
    assert s.smooth[j] == pytest.approx(np.nanmean(s.premium[max(0, j - 62) : j + 1]))


@pytest.mark.behaviour
def test_where_the_premium_stands_at_the_end_of_the_panel() -> None:
    s = premium_series(_daily())
    now = premium_now(s)
    assert now is not None and now.day == date.fromisoformat(str(s.date[-1]))
    done = np.flatnonzero(np.isfinite(s.premium))
    assert now.measured_date == date.fromisoformat(str(s.date[done[-1]]))
    assert now.measured == s.premium[done[-1]] and now.recent == s.smooth[done[-1]]
    assert 0.0 <= now.recent_share_below <= 1.0
    empty = premium_series({})
    assert len(empty.date) == 0 and premium_now(empty) is None


@pytest.mark.behaviour
def test_the_place_on_the_sweeps_axis_in_words() -> None:
    axis = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    assert axis_place(0.64, vrp.bracket(0.64, axis), axis) == (
        "between the sweep's 0.5 and 1 vol-point levels"
    )
    assert axis_place(1.0, vrp.bracket(1.0, axis), axis) == "at the sweep's 1 vol-point level"
    assert axis_place(-0.5, vrp.bracket(-0.5, axis), axis) == "below the sweep's lowest level (0)"
    assert axis_place(5.0, vrp.bracket(5.0, axis), axis) == "above the sweep's highest level (4)"


@pytest.mark.behaviour
def test_the_premium_reading_is_the_mappings_headline_in_words() -> None:
    m = json.loads((REPO_DOCS / "VRP_MAPPING.json").read_text())
    h = m["headline"]
    text = premium_reading(
        h["mean"],
        h["mean_low"],
        h["mean_high"],
        h["first"],
        h["last"],
        m["bracket"],
        m["sweep_axis"],
    )
    assert text == (
        "From 2010 to 2026 the market priced 30-day volatility +0.64 vol points above the "
        f"volatility that followed, on average (95% interval {M}0.34 to +1.47, so it could be "
        "zero): between the sweep's 0.5 and 1 vol-point levels."
    )
    clear = premium_reading(1.0, 0.2, 1.8, "2010-01-04", "2010-12-31", [1.0, 2.0], [1.0, 2.0])
    assert "From 2010 the" in clear and "clear of zero" in clear and "at the sweep's 1" in clear
    bare = premium_reading(0.5, None, None, "2025-01-02", "2025-06-01", [0.5, 1.0], [0.5, 1.0])
    assert "interval" not in bare
    now = premium_now(premium_series(_daily()))
    assert now is not None
    tail = premium_reading(0.5, None, None, "2025-01-02", "2025-06-01", [0.5, 1.0], [0.5], now)
    assert f"The latest quarter, to {now.measured_date}, averaged {signed(now.recent)}" in tail


# -- question 2: the surface's shape on a day ---------------------------------------- #


def _grid(days: list[str], atm: list[float], put: list[float], year: list[float]) -> Columns:
    rows: list[tuple[str, int, float, float]] = []
    for d, a, q, y in zip(days, atm, put, year, strict=True):
        rows += [(d, 30, 0.0, a), (d, 30, -0.1, q), (d, 30, 0.05, a - 0.01), (d, 365, 0.0, y)]
        rows += [(d, 30, -0.5, float("nan"))]
    return normalise(
        {
            "date": [r[0] for r in rows],
            "tenor_days": [r[1] for r in rows],
            "k": [r[2] for r in rows],
            "iv": [r[3] for r in rows],
        }
    )


@pytest.mark.behaviour
def test_the_shape_is_level_skew_and_term_slope_against_every_day() -> None:
    days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    grid = _grid(days, [0.15, 0.16, 0.40, 0.12], [0.24, 0.25, 0.46, 0.22], [0.18, 0.19, 0.30, 0.16])
    h = shape_history(grid)
    np.testing.assert_allclose(h.atm, [15.0, 16.0, 40.0, 12.0])
    np.testing.assert_allclose(h.skew, [9.0, 9.0, 6.0, 10.0])
    np.testing.assert_allclose(h.term, [3.0, 3.0, -10.0, 4.0])
    np.testing.assert_allclose(h.call, [-1.0] * 4)
    calm = shape_on(h, date(2025, 1, 7))
    assert calm is not None and calm.atm_share_below == 0.0 and calm.share_inverted == 0.25
    assert calm.skew_median == 9.0 and calm.term_median == 3.0
    assert shape_reading(calm) == (
        "On 2025-01-07 30-day at-the-money vol was 12.0 (lower than on 100% of days); a put "
        "10% below the forward cost 10.0 vol points more than at the money (9.0 on the "
        "typical day); and the term structure sloped up, a year out 4.0 above the 30-day "
        "(+3.0 typical)."
    )
    stressed = shape_on(h, date(2025, 1, 6))
    assert stressed is not None
    text = shape_reading(stressed)
    assert "(higher than on 75% of days)" in text
    assert "the term structure was inverted, a year out 10.0 below the 30-day" in text
    assert "as on only 25% of days (a stressed market)" in text
    assert shape_on(h, date(2025, 1, 4)) is None  # a Saturday: not in the grid


@pytest.mark.behaviour
def test_the_shape_reading_says_only_what_the_grid_quotes() -> None:
    days = ["2025-01-02"]
    grid = _grid(days, [0.15], [float("nan")], [float("nan")])
    one = shape_on(shape_history(grid), date(2025, 1, 2))
    assert one is not None
    text = shape_reading(one)
    assert text.startswith("On 2025-01-02 30-day at-the-money vol was 15.0")
    assert "put" not in text and "term structure" not in text
    blank = shape_on(shape_history(_grid(days, [float("nan")], [0.2], [0.2])), date(2025, 1, 2))
    assert blank is not None and "no 30-day at-the-money quote" in shape_reading(blank)


# -- question 3: the simulator against the market ------------------------------------- #


@pytest.mark.behaviour
def test_where_the_simulator_and_the_market_differ_most() -> None:
    k = np.asarray([[-0.2, -0.05, 0.0, 0.02, 0.1], [-0.2, -0.05, 0.0, 0.02, 0.1]])
    real = np.asarray([[0.35, 0.25, 0.21, 0.20, np.nan], [0.30, 0.24, 0.22, 0.21, 0.18]])
    sim = np.full(k.shape, 0.21)
    regions, widest = sim_regions(k, real, sim)
    by = {r.name: r for r in regions}
    assert by["put"].n == 4 and by["put"].mean_diff == pytest.approx((14 + 4 + 9 + 3) / 4)
    assert by["atm"].n == 4 and by["atm"].mean_diff == pytest.approx((0 - 1 + 1 + 0) / 4)
    assert by["call"].n == 1 and by["call"].mean_diff == pytest.approx(-3.0)
    assert widest is not None and (widest.row, widest.col) == (0, 0)
    assert widest.diff == pytest.approx(14.0) and widest.real == pytest.approx(35.0)
    text = sim_reading(0.21, regions, widest, 4400.0, 21)
    assert text.startswith("The simulator prices every contract at one level, 21.0 vol points")
    assert "furthest from it in the put wing, 7.5 vol points above on average" in text
    assert f"(near the money +0.0, the call wing {M}3.0)" in text
    assert "the widest gap is the 4,400 strike at 21 trading days, real 35.0 against 21.0" in text
    assert text.endswith("that slope is the market's skew, which the simulator does not have.")
    flat = sim_regions(k, np.full(k.shape, 0.22), sim)
    assert "skew" not in sim_reading(0.21, *flat)
    none = sim_regions(k, np.full(k.shape, np.nan), sim)
    assert none[1] is None and "quotes none of its contracts" in sim_reading(0.21, *none)


# -- question 4 and the data quality line -------------------------------------------- #


@pytest.mark.behaviour
def test_the_seller_reading_names_the_strikes_the_credit_and_the_risk() -> None:
    text = seller_reading(
        "2022-06-13",
        3746.0,
        42,
        3250,
        4150,
        2950,
        4400,
        4925.0,
        25075.0,
        25075.0,
        [3200.75, 4199.25],
        0.16,
    )
    assert text == (
        "At the close of 2022-06-13, with SPX at 3,746, the seller would sell the 3,250 put and "
        "the 4,150 call (about 13% below and 11% above, the 16-delta strikes), buy the 2,950 put "
        "and 4,400 call as wings, and collect $4,925 for 42 trading days; it keeps money if SPX "
        "ends between 3,201 and 4,199, can lose at most $25,075, and posts $25,075 of margin."
    )
    naked = seller_reading(
        "2022-06-13", 3746.0, 42, 3250, 4150, 2950, 4400, 1.0, None, 9.0, [], 0.16
    )
    assert "lose at most" not in naked and "keeps money" not in naked


@pytest.mark.behaviour
def test_the_quality_line() -> None:
    text = quality_reading(4213, "2010-01-04", "2026-10-02", 0, 43, 0.0027, 4, True, 0.0132)
    assert text == (
        "4,213 trading days from 2010-01-04 to 2026-10-02, none missing; on the 43 days both "
        "vendors quote, their IVs differ by 0.003 vol points at the 95th percentile on the "
        "average day (0.013 on the worst); panel version 4, the panel of record."
    )
    other = quality_reading(10, "2025-01-02", "2025-01-15", 2, 0, None, 4, False)
    assert "2 missing" in other and "vendors" not in other and "not the ones" in other
    assert "version" not in quality_reading(10, "a", "b", 0, 0, None, None, None)


# -- the payloads, on the fixture panel ------------------------------------------------ #


@pytest.fixture()
def built(panel_root: Path) -> PanelReader:
    return PanelReader(panel_root)


@pytest.mark.behaviour
def test_the_premium_payload(built: PanelReader) -> None:
    out = built.premium_out()
    s = premium_series(built.daily())
    assert out.date == [str(d) for d in s.date] and len(out.premium) == len(out.date)
    ok = np.isfinite(s.premium)
    assert out.mean == pytest.approx(float(s.premium[ok].mean())) and out.mean_low is None
    assert out.iv30[0] == pytest.approx(s.iv30[0] * 100.0, abs=1e-3) and out.smooth_days == 63
    assert out.now is not None and out.now.date == out.date[-1]
    assert _sent(out)["premium"][-1] is None  # no complete forward window yet: sent as null
    assert out.reading.startswith("From 2025 the market priced 30-day volatility")
    mapping = json.loads((REPO_DOCS / "VRP_MAPPING.json").read_text())
    headed = built.premium_out(mapping)
    assert headed.mean == mapping["headline"]["mean"] and headed.bracket == [0.5, 1.0]
    assert headed.sweep_axis == mapping["sweep_axis"] and "+0.64" in headed.reading


@pytest.mark.behaviour
def test_the_shape_payload(built: PanelReader) -> None:
    out = built.shape_out(date(2025, 7, 1))
    assert out.atm is not None and 5.0 < out.atm < 50.0 and out.tenor == 30
    assert out.skew is not None and out.skew > 0  # the fixture's smile is skewed
    # nothing past 120 days: the year-out point is not invented, and is sent as null
    assert _sent(out)["term"] is None
    assert out.reading.startswith("On 2025-07-01 30-day at-the-money vol was")
    assert "term structure" not in out.reading
    with pytest.raises(PanelNotBuiltError):
        built.shape_out(date(2025, 7, 4))


@pytest.mark.behaviour
def test_the_simulated_payload_says_where_the_two_differ(built: PanelReader) -> None:
    sim = built.simulated_out(date(2025, 7, 1), 1.0)
    assert [r.name for r in sim.regions] == ["put", "atm", "call"]
    assert sum(r.n for r in sim.regions) == sum(_finite(x) for row in sim.real for x in row)
    assert sim.widest is not None and sim.widest.strike in sim.strike
    assert sim.widest.sim == pytest.approx(sim.level * 100.0)
    assert abs(sim.widest.diff) == pytest.approx((sim.max_abs_diff or 0.0) * 100.0)
    assert sim.reading.startswith(
        f"The simulator prices every contract at one level, {sim.level * 100:.1f}"
    )
    assert "furthest from it in" in sim.reading


@pytest.mark.behaviour
def test_the_seller_payload_reads_in_words(built: PanelReader) -> None:
    out = seller_out(built, date(2025, 7, 1))
    shorts = sorted(leg.strike for leg in out.legs if leg.qty < 0)
    assert out.reading.startswith(f"At the close of 2025-07-01, with SPX at {out.spot:,.0f}")
    assert f"the {shorts[0]:,.0f} put and the {shorts[1]:,.0f} call" in out.reading
    assert f"collect ${out.credit:,.0f}" in out.reading


@pytest.mark.invariant
def test_the_quality_payload_checks_the_build_against_the_manifest(
    built: PanelReader, panel_root: Path, tmp_path: Path
) -> None:
    empty = built.quality_out(tmp_path / "nodocs")
    assert empty.panel_version is None and empty.of_record is None and empty.manifest_files == 0
    assert {c.path for c in empty.checked} == {
        "panel/spx_daily.parquet",
        "panel/surface_grid.parquet",
    }
    assert empty.reading.startswith("7 trading days from 2025-06-27 to 2025-07-08, none missing")
    assert not empty.fred_available and empty.fred == []

    docs = tmp_path / "docs"
    docs.mkdir()
    files = {c.path: {"sha256": c.actual, "bytes": 1} for c in empty.checked}
    (docs / "DATA_MANIFEST.json").write_text(json.dumps({"panel_version": 4, "files": files}))
    good = built.quality_out(docs)
    assert good.panel_version == 4 and good.of_record is True and all(c.match for c in good.checked)
    assert good.reading.endswith("panel version 4, the panel of record.")
    files["panel/spx_daily.parquet"]["sha256"] = "0" * 64
    (docs / "DATA_MANIFEST.json").write_text(json.dumps({"panel_version": 4, "files": files}))
    bad = built.quality_out(docs)
    assert bad.of_record is False and "not the ones" in bad.reading

    pd = pytest.importorskip("pandas")
    shutil.copytree(panel_root, tmp_path / "copy")
    pd.DataFrame(
        {
            "year": [2026, 2025, 2025],
            "source": ["databento", "optionmetrics", "databento"],
            "rate_minus_bill_bp": [50.6, 5.2, 32.5],
            "rate_minus_sofr_bp": [61.3, 8.1, None],
            "spot_vs_sp500_bp": [0.56, 0.0, 0.12],
        }
    ).to_parquet(tmp_path / "copy" / "panel" / "fred_check.parquet", index=False)
    quality = PanelReader(tmp_path / "copy").quality_out(docs)
    fred = quality.fred
    assert [(r.year, r.source) for r in fred] == [
        (2025, "databento"),
        (2025, "optionmetrics"),
        (2026, "databento"),
    ]
    # a rate the check could not read is sent as null
    assert _sent(quality)["fred"][0]["rate_minus_sofr_bp"] is None
    assert fred[2].rate_minus_bill_bp == 50.6
