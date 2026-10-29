"""Hemdutt Rao: the VRP mapping recovers a premium it was built with, and reads a curve.

Every fixture is constructed: a constant-vol path priced at a known implied vol, so
the right answer is the planted premium, the way the simulator itself is tested. No
test reads the shared data folder.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from optspread import vrp

SIGMA = 0.20
P = 1.5  # vol points


def business_days(n: int, start: date = date(2015, 1, 5)) -> vrp.Dates:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return np.asarray(out, dtype="datetime64[D]")


def gbm(n: int, sigma: float = SIGMA, seed: int = 0) -> vrp.Floats:
    """A zero-drift path at constant vol, on the simulator's 252-day year."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / vrp.TRADING_DAYS
    r = -0.5 * sigma * sigma * dt + sigma * math.sqrt(dt) * rng.standard_normal(n - 1)
    return 5000.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def alternating(n: int, step: float) -> vrp.Floats:
    """Every log return is +step or -step, so every window's RMS is exactly ``step``."""
    signs = np.where(np.arange(n - 1) % 2 == 0, 1.0, -1.0)
    return 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(step * signs)]))


# -- the forward-realised series --------------------------------------------- #


def test_forward_rv_is_the_rms_of_the_next_window_annualised_at_252() -> None:
    s = 0.01
    rv = vrp.forward_realised(alternating(60, s))
    assert np.isnan(rv[-vrp.WINDOW :]).all()  # no complete window at the end
    np.testing.assert_allclose(rv[: -vrp.WINDOW], s * math.sqrt(252.0))
    unbiased = vrp.forward_realised(alternating(60, s), unbiased=True)
    np.testing.assert_allclose(unbiased[:5], s * math.sqrt(252.0) / vrp.vol_bias(21))


def test_forward_rv_at_t_uses_only_the_returns_after_t() -> None:
    spot = np.full(80, 100.0)
    spot[40:] = 110.0  # the one move: r[40]
    rv = vrp.forward_realised(spot)
    moved = np.flatnonzero(rv[: 80 - vrp.WINDOW] > 0)
    # the windows r[t+1 .. t+21] that hold r[40]: t = 19 .. 39
    assert moved.tolist() == list(range(40 - vrp.WINDOW, 40))
    trailing = vrp.trailing_realised(spot)
    assert np.flatnonzero(trailing > 0).tolist() == list(range(40, 40 + vrp.WINDOW))


def test_the_demeaned_rv_is_the_panels_rv21_fwd() -> None:
    pd = pytest.importorskip("pandas")
    spot = gbm(300, seed=3)
    ret = np.log(pd.Series(spot)).diff()
    panel = (ret.rolling(21).std() * math.sqrt(252)).shift(-21).to_numpy()
    np.testing.assert_allclose(vrp.forward_realised(spot, demean=True), panel, equal_nan=True)


def test_a_window_across_a_hole_in_the_dates_is_dropped() -> None:
    dates = business_days(80)
    dates[50:] += np.timedelta64(30, "D")  # a month missing after day 49
    rv = vrp.forward_realised(alternating(80, 0.01), dates)
    assert np.isfinite(rv[: 50 - vrp.WINDOW]).all()
    assert np.isnan(rv[50 - vrp.WINDOW : 50]).all()
    assert np.isfinite(rv[50 : 80 - vrp.WINDOW]).all()


def test_the_calendar_rv_sums_the_returns_of_the_next_30_days_over_365() -> None:
    dates = business_days(60)
    s = 0.01
    rv = vrp.forward_realised_calendar(alternating(60, s), dates)
    for t in (0, 7, 13):
        k = int(((dates > dates[t]) & (dates <= dates[t] + np.timedelta64(30, "D"))).sum())
        assert rv[t] == pytest.approx(s * math.sqrt(k * 365.0 / 30.0))
    assert np.isnan(rv[dates + np.timedelta64(30, "D") > dates[-1]]).all()
    unbiased = vrp.forward_realised_calendar(alternating(60, s), dates, unbiased=True)
    assert (unbiased[:5] > rv[:5]).all()
    spot = alternating(60, s)
    spot[10] = np.nan  # r[10] and r[11] are NaN; the window after t = 10 starts at r[11]
    holed = vrp.forward_realised_calendar(spot, dates)
    assert np.isnan(holed[:11]).all() and np.isfinite(holed[11:20]).all()
    assert vrp.forward_realised_calendar(np.array([]), np.array([], "datetime64[D]")).size == 0


def test_short_series_have_no_window() -> None:
    assert np.isnan(vrp.forward_realised(np.full(10, 100.0))).all()
    assert np.isnan(vrp.trailing_realised(np.full(10, 100.0))).all()
    assert np.isnat(vrp.window_end(business_days(10))).all()


def test_vol_bias_is_the_mean_of_a_gaussian_rms() -> None:
    assert vrp.vol_bias(21) == pytest.approx(0.98817, abs=1e-5)
    assert vrp.vol_bias(21) == pytest.approx(1.0 - 1.0 / 84.0, abs=5e-4)
    draws = np.random.default_rng(0).standard_normal((200_000, 21))
    assert np.sqrt((draws**2).mean(axis=1)).mean() == pytest.approx(vrp.vol_bias(21), abs=1e-3)


def test_the_variance_conversion_inverts_the_simulators_premium() -> None:
    v = (SIGMA + P / 100.0) ** 2 - SIGMA**2
    assert vrp.variance_to_vol_pts(v, SIGMA) == pytest.approx(P)
    assert vrp.variance_to_vol_pts(0.0) == 0.0
    assert vrp.variance_to_vol_pts(-1.0) == pytest.approx(-100.0 * SIGMA)  # floored at zero vol


# -- the estimate, on a premium planted by construction ----------------------- #


def planted(n: int = 3000, seed: int = 0) -> tuple[vrp.Floats, vrp.Dates, vrp.Floats]:
    spot = gbm(n, seed=seed)
    return spot, business_days(n), np.full(n, SIGMA + P / 100.0)


def test_a_planted_premium_is_recovered_and_covered() -> None:
    spot, dates, iv = planted()
    rv = vrp.forward_realised(spot, dates, unbiased=True)
    e = vrp.estimate(vrp.premium_vol_pts(iv, rv), dates, "planted", n_boot=400)
    assert e.n_days == 3000 - vrp.WINDOW
    assert e.mean == pytest.approx(P, abs=0.15)
    assert e.mean_low < P < e.mean_high
    assert e.median_low < e.median < e.median_high
    assert e.block >= vrp.WINDOW and e.first == vrp.as_date(dates[0])
    # Uncorrected, the same path reads high by about (1 - c_21) * sigma.
    raw = vrp.forward_realised(spot, dates)
    biased = vrp.estimate(vrp.premium_vol_pts(iv, raw), dates, "raw", n_boot=50)
    gap = 100.0 * (1.0 - vrp.vol_bias(21)) * SIGMA
    assert biased.mean - e.mean == pytest.approx(gap, rel=0.05)


def test_the_variance_route_recovers_the_premium_without_a_correction() -> None:
    spot, dates, iv = planted(seed=1)
    v = vrp.variance_premium(iv, vrp.forward_realised(spot, dates))
    e = vrp.estimate(v, dates, "var", n_boot=200, to_vol_pts=vrp.variance_to_vol_pts)
    assert e.mean == pytest.approx(P, abs=0.25)
    assert e.mean_low < P < e.mean_high and e.mean_low < e.mean < e.mean_high


def test_the_interval_covers_a_known_mean_under_heavy_dependence() -> None:
    """AR(1) at 0.97 around 0.8: an iid bootstrap would be far too narrow here."""
    rng = np.random.default_rng(7)
    n = 3000
    from scipy.signal import lfilter

    x = lfilter([1.0], [1.0, -0.97], rng.standard_normal(n)) + 0.8
    dates = business_days(n)
    e = vrp.estimate(x, dates, "ar", n_boot=400)
    assert e.mean_low < 0.8 < e.mean_high
    iid = vrp.estimate(x, dates, "iid", block=1.0, n_boot=400)
    assert (e.mean_high - e.mean_low) > 3.0 * (iid.mean_high - iid.mean_low)


def test_the_estimate_is_reproducible_and_skips_nan_days() -> None:
    x = np.random.default_rng(2).standard_normal(400)
    x[::50] = np.nan
    dates = business_days(400)
    a = vrp.estimate(x, dates, "a", n_boot=100)
    assert a == vrp.estimate(x, dates, "a", n_boot=100)
    assert a.n_days == 392
    empty = vrp.estimate(np.full(5, np.nan), business_days(5), "none")
    assert empty.n_days == 0 and math.isnan(empty.mean) and empty.first is None


def test_the_block_length_rises_with_dependence_and_never_drops_below_the_window() -> None:
    rng = np.random.default_rng(4)
    iid = rng.standard_normal(4000)
    assert vrp.politis_white_block(iid) < 5.0
    assert vrp.block_length(iid) == vrp.WINDOW
    overlap = vrp.premium_vol_pts(
        np.full(4000, SIGMA), vrp.forward_realised(gbm(4000, seed=5), unbiased=True)
    )
    assert vrp.politis_white_block(overlap) > vrp.WINDOW  # the MA(20) the windows induce
    assert vrp.politis_white_block(np.ones(50)) == 1.0  # no variance
    assert vrp.politis_white_block(np.arange(5.0)) == 1.0  # too short to say
    assert vrp.politis_white_block(np.arange(400.0)) <= math.ceil(3 * math.sqrt(400))  # capped


def test_the_block_length_matches_its_closed_form_on_an_ar1() -> None:
    """For AR(1) at phi the optimal mean block is ``(2 phi / (1 - phi^2))^(2/3) n^(1/3)``."""
    from scipy.signal import lfilter

    n, phi = 20_000, 0.5
    x = lfilter([1.0], [1.0, -phi], np.random.default_rng(0).standard_normal(n))
    want = (2.0 * phi / (1.0 - phi * phi)) ** (2.0 / 3.0) * n ** (1.0 / 3.0)
    assert vrp.politis_white_block(x) == pytest.approx(want, rel=0.15)


def test_stationary_indices_wrap_and_have_the_mean_block() -> None:
    rng = np.random.default_rng(0)
    idx = vrp.stationary_indices(1000, 25.0, 200, rng)
    assert idx.shape == (200, 1000) and idx.min() >= 0 and idx.max() < 1000
    continues = (np.diff(idx, axis=1) % 1000) == 1
    assert 1.0 / (1.0 - continues.mean()) == pytest.approx(25.0, rel=0.1)


# -- sub-periods -------------------------------------------------------------- #


def test_a_split_shares_no_return_between_its_halves() -> None:
    dates = business_days(200, start=date(2019, 11, 1))
    ends = vrp.window_end(dates)
    left = vrp.before(ends, date(2019, 12, 31))
    right = vrp.after(dates, date(2019, 12, 31))
    assert not (left & right).any()
    assert (ends[left] <= np.datetime64("2019-12-31")).all()
    assert (dates[right] > np.datetime64("2019-12-31")).all()
    straddle = ~left & ~right & ~np.isnat(ends)
    assert straddle.sum() == vrp.WINDOW  # the last 21 quotes of 2019
    assert set(vrp.years(dates).tolist()) == {2019, 2020}


# -- the power curve ----------------------------------------------------------- #

AXIS = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
CURVE = 0.05 + 0.95 / (1.0 + np.exp(-(AXIS - 1.5) * 2.5)) - 0.95 / (1.0 + np.exp(3.75))


def test_power_at_hits_the_grid_and_brackets_the_point() -> None:
    m = vrp.power_at(AXIS, CURVE, 1.0, (0.5, 2.0))
    assert m.power == pytest.approx(CURVE[3]) and m.power_low == pytest.approx(CURVE[2])
    assert m.power_high == pytest.approx(CURVE[4]) and not m.clipped
    mid = vrp.power_at(AXIS, CURVE, 1.4, (2.0, 0.7))  # interval in either order
    assert (mid.low, mid.high) == (0.7, 2.0)
    assert m.power_low < mid.power_low < mid.power < mid.power_high


def test_power_at_is_monotone_between_points() -> None:
    grid = np.linspace(0.0, 4.0, 401)
    reads = [vrp.power_at(AXIS, CURVE, p, (p, p)).power for p in grid]
    assert np.all(np.diff(reads) >= -1e-12)
    flat = np.array([0.05, 0.05, 0.05, 0.4, 0.9, 0.9])  # flats stay flat: no overshoot
    reads = [vrp.power_at(AXIS, flat, p, (p, p)).power for p in grid]
    assert np.all(np.diff(reads) >= -1e-12) and max(reads) <= 0.9 and min(reads) >= 0.05


def test_power_at_holds_the_ends_outside_the_grid() -> None:
    m = vrp.power_at(AXIS, CURVE, -0.3, (-1.0, 6.0))
    assert m.clipped
    assert m.power == pytest.approx(CURVE[0]) and m.power_high == pytest.approx(CURVE[-1])


@pytest.mark.parametrize(
    ("premiums", "power"),
    [
        ([0.0, 1.0], [0.1]),
        ([0.0], [0.1]),
        ([0.0, 0.0, 1.0], [0.1, 0.2, 0.3]),
        ([0.0, 1.0], [0.1, 1.2]),
        ([0.0, float("nan")], [0.1, 0.2]),
    ],
)
def test_power_at_refuses_a_curve_it_cannot_read(premiums: list[float], power: list[float]) -> None:
    with pytest.raises(ValueError):
        vrp.power_at(premiums, power, 1.0, (0.5, 1.5))


def test_power_at_reads_a_curve_that_dips_as_it_is() -> None:
    # The real PPO curve's knots (0.475 at 0.25 vol points, 0.346 at 0.5): a dip by
    # sampling noise at S = 5, which the reader used to refuse.
    axis = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    power = [0.05, 0.475, 0.346, 0.889, 1.0, 0.991]
    m = vrp.power_at(axis, power, 0.6395, (-0.344, 1.474))
    assert m.power == pytest.approx(0.346 + (0.6395 - 0.5) / 0.5 * (0.889 - 0.346))
    assert m.clipped and m.power_low == 0.05
    assert m.power_high == pytest.approx(0.889 + 0.474 * (1.0 - 0.889))
    assert (m.power_min, m.power_max) == (0.05, m.power_high)
    inside = vrp.power_at(axis, power, 0.3, (0.25, 0.5))  # the dip lies inside
    assert inside.power_min == pytest.approx(0.346) and inside.power_max == pytest.approx(0.475)
    over = vrp.power_at(axis, power, 3.0, (1.5, 4.0))  # a peak inside, not at an end
    assert over.power_max == 1.0 > over.power_high


def _knots() -> tuple[list[float], list[float]]:
    return [0.0, 0.25, 0.5, 1.0, 2.0, 4.0], [0.0, 1.911, 1.4996, 3.5307, 6.9226, 5.0256]


def test_power_at_noncentrality_is_the_analysis_interpolation_exactly() -> None:
    from optspread import m4analysis as m4

    xs, lam = _knots()
    for p in (0.1, 0.25, 0.6395, 1.474, 3.0):
        m = vrp.power_at_noncentrality(xs, lam, 5, p, (p, p))
        assert m.power == m4.power_at(float(np.interp(p, xs, lam)), 5)
    curve = [
        m4.CurvePoint(x, x > 0, v, v, v, *(3 * [m4.power_at(v, 5)]), None)
        for x, v in zip(xs, lam, strict=True)
    ]
    for x, w, _, _ in m4.interpolate(curve, 5):
        assert vrp.power_at_noncentrality(xs, lam, 5, x, (x, x)).power == pytest.approx(
            w, abs=1e-15
        )
    assert vrp.noncentral_power(0.0, 5) == 0.05
    assert (vrp.noncentral_power(np.inf, 5), vrp.noncentral_power(-np.inf, 5)) == (1.0, 0.0)


def test_power_at_noncentrality_reads_the_real_ppo_curve_over_its_interval() -> None:
    xs, lam = _knots()
    m = vrp.power_at_noncentrality(xs, lam, 5, 0.6395, (-0.344, 1.474))
    assert 0.5 < m.power < 0.55 and m.clipped and m.power_low == 0.05
    assert m.power_min == 0.05 and m.power_max == m.power_high > 0.98
    dip = vrp.power_at_noncentrality(xs, lam, 5, 3.0, (1.5, 4.0))
    assert dip.power_max > dip.power_high  # 2 vol points is the curve's top
    with pytest.raises(ValueError, match="strictly increasing"):
        vrp.power_at_noncentrality([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], 5, 0.5, (0.0, 1.0))
    with pytest.raises(ValueError, match="at least 2 seeds"):
        vrp.power_at_noncentrality(xs, lam, 1, 0.5, (0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        vrp.power_at_noncentrality([0.0, 1.0], [0.0, float("nan")], 5, 0.5, (0.0, 1.0))


def test_bracket() -> None:
    assert vrp.bracket(0.8, AXIS.tolist()) == (0.5, 1.0)
    assert vrp.bracket(1.0, AXIS.tolist()) == (1.0, 2.0)
    assert vrp.bracket(-0.1, AXIS.tolist()) == (None, 0.0)
    assert vrp.bracket(5.0, AXIS.tolist()) == (4.0, None)


# -- the report, end to end on a constructed panel ---------------------------- #


def write_daily(path: Path, n: int = 1500) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    spot = gbm(n, seed=11)
    dates = business_days(n, start=date(2021, 1, 4))
    iv = np.full(n, SIGMA + P / 100.0)
    iv[5] = np.nan  # a day with no 30-day IV
    ret = np.log(pd.Series(spot)).diff()
    rv_fwd = (ret.rolling(21).std() * math.sqrt(252)).shift(-21)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "spot": spot,
            "iv30_atm": iv,
            "rv21_fwd": rv_fwd,
            "vrp30": iv - rv_fwd,
            "source": "optionmetrics",
        }
    )
    frame.to_parquet(path, index=False)


def test_the_report_recovers_the_planted_premium(tmp_path: Path) -> None:
    from scripts import vrp_mapping

    daily = tmp_path / "spx_daily.parquet"
    write_daily(daily)
    out = tmp_path / "docs" / "VRP_MAPPING"
    assert vrp_mapping.main(["--daily", str(daily), "--out", str(out), "--n-boot", "60"]) == 0
    blob = json.loads(out.with_suffix(".json").read_text())
    head = blob["headline"]
    assert head["mean"] == pytest.approx(P, abs=0.3)
    assert blob["power_inputs"]["interval"] == [head["mean_low"], head["mean_high"]]
    assert blob["checks"]["panel_vrp30_reproduced_max_abs_vol_pts"] < 1e-9
    assert blob["rows"]["no_iv30"] == ["2021-01-11"] and blob["rows"]["no_forward_window"] == 21
    assert {p["label"].split(" (")[0] for p in blob["periods"]} >= {"before 2020", "2020 on"}
    assert [y["label"] for y in blob["years"]][:2] == ["2021", "2022"]
    md = out.with_suffix(".md").read_text()
    assert "between the sweep's 1 and 2 vol-point levels" in md
    assert "not the build docs/DATA_MANIFEST.json records" in md  # a constructed file


def test_the_report_refuses_a_date_twice() -> None:
    from scripts import vrp_mapping

    d = np.asarray(["2021-01-04", "2021-01-04"], dtype="datetime64[D]")
    cols = {"date": d, "spot": np.ones(2), "iv30_atm": np.ones(2)}
    with pytest.raises(ValueError, match="twice"):
        vrp_mapping.series(cols)


def test_reading_retries_a_timed_out_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import vrp_mapping

    calls: list[int] = []

    def flaky(path: Path, columns: object) -> dict[str, object]:
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("Operation timed out")
        return {"ok": True}

    monkeypatch.setattr(vrp_mapping, "read_table", flaky)
    assert vrp_mapping.read_daily(tmp_path / "x", wait=0.0) == {"ok": True}
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(vrp_mapping, "read_table", lambda p, c: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        vrp_mapping.read_daily(tmp_path / "x", attempts=2, wait=0.0)
    assert vrp_mapping.manifest_digest(tmp_path / "missing.json") is None
