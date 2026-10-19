"""Hemdutt Rao: the real-data path, proven on quotes whose answer is known.

Every fixture here is a chain priced by ``optspread.pricing`` at a known vol,
rate and forward, quoted with a spread around that price. The pipeline has to
hand back what went in: the parity fit recovers the forward and the discount
factor, the IV solver recovers the vol, and the provider built from the result
clears the same conformance suite as the synthetic source.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optspread.config import MarketConfig
from optspread.data import OptionMetricsSource, conformance_report
from optspread.pricing import bs_price
from optspread.realdata import data_root
from optspread.realdata.opra import closing_quotes, parse_metadata, parse_osi
from optspread.realdata.optionmetrics import normalise_opprcd
from optspread.realdata.panel import (
    K_GRID,
    TENOR_GRID,
    atm_term,
    carry30,
    coverage,
    daily_table,
    finalise,
    iv30,
    parity_spot,
    seam_frame,
    surface_grid,
    trading_calendar,
)
from optspread.realdata.provider import HistoricalChainProvider, SurfaceStore
from optspread.realdata.surface import fit_carry, implied_vol, price_surface, time_to_expiry
from optspread.rng import SeededStepRNG

SIGMA, RATE, DIV = 0.20, 0.04, 0.015
SPOT = 5000.0
DAY0 = date(2024, 3, 1)


def _quotes(day: date, spot: float, sigma: float = SIGMA, *, root: str = "SPXW") -> pd.DataFrame:
    """A two-sided chain priced at a flat vol, for expiries 9 to 400 days out."""
    rows = []
    for dte in (9, 16, 30, 44, 65, 93, 184, 400):
        T = dte / 365.0
        fwd = spot * np.exp((RATE - DIV) * T)
        S_pre = fwd * np.exp(-RATE * T)  # the prepaid forward the engine prices on
        for K in np.arange(round(spot * 0.6, -1), spot * 1.4, 25.0):
            for cp in ("C", "P"):
                mid = float(bs_price(cp, S_pre, K, RATE, sigma, T))
                half = max(0.05, 0.01 * mid)
                rows.append(
                    {
                        "date": day,
                        "root": root,
                        "expiry": day + timedelta(days=dte),
                        "cp": cp,
                        "strike": float(K),
                        "bid": mid - half if mid - half > 0.05 else np.nan,
                        "ask": mid + half,
                        "am_settled": False,
                    }
                )
    return pd.DataFrame(rows)


# -- carry and IV ------------------------------------------------------------ #


def test_parity_recovers_the_forward_and_the_discount_factor() -> None:
    q = _quotes(DAY0, SPOT)
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    q["tau"] = time_to_expiry(q["date"], q["expiry"], q["am_settled"])
    fits = fit_carry(q).merge(q[["expiry", "tau"]].drop_duplicates(), on="expiry")
    long = fits[fits["tau"] >= 60 / 365]
    want_f = SPOT * np.exp((RATE - DIV) * long["tau"])
    np.testing.assert_allclose(long["forward"], want_f, rtol=2e-4)
    np.testing.assert_allclose(long["discount"], np.exp(-RATE * long["tau"]), atol=2e-4)
    short = fits[fits["tau"] < 60 / 365]
    assert (short["carry_fit"].str.endswith("+r_ref")).all(), "short expiries borrow the day's rate"
    np.testing.assert_allclose(short["discount"], np.exp(-RATE * short["tau"]), atol=5e-4)


def test_the_solver_recovers_the_vol_on_both_rights() -> None:
    T = np.full(6, 0.25)
    K = np.array([4500.0, 5000.0, 5500.0] * 2)
    is_call = np.array([True] * 3 + [False] * 3)
    S, r = np.full(6, SPOT), np.full(6, RATE)
    price = np.where(is_call, bs_price("C", S, K, r, 0.23, T), bs_price("P", S, K, r, 0.23, T))
    np.testing.assert_allclose(implied_vol(is_call, S, K, r, T, price), 0.23, atol=1e-8)


def test_a_price_outside_the_bounds_has_no_vol() -> None:
    iv = implied_vol(
        np.array([True, False]),
        np.full(2, SPOT),
        np.full(2, 5000.0),
        np.full(2, RATE),
        np.full(2, 0.25),
        np.array([SPOT + 1.0, -1.0]),  # a call above spot; a negative put
    )
    assert np.isnan(iv).all()


def test_the_priced_surface_is_flat_where_the_chain_was_priced_flat() -> None:
    s = price_surface(_quotes(DAY0, SPOT))
    s["k"] = np.log(s["strike"] / s["forward"])
    band = s[(s["k"].abs() < 0.15) & s["iv"].notna()]
    assert len(band) > 100
    assert float((band["iv"] - SIGMA).abs().quantile(0.95)) < 0.005


def test_am_settled_contracts_stop_at_the_open() -> None:
    tau = time_to_expiry(
        pd.Series([DAY0, DAY0]), pd.Series([DAY0 + timedelta(days=7)] * 2), pd.Series([1, 0])
    )
    assert tau[1] - tau[0] == pytest.approx(6.5 / 24 / 365)


# -- vendor schemas ------------------------------------------------------------ #


def test_osi_symbols_split_into_their_fields() -> None:
    out = parse_osi(pd.Series(["SPXW  250709C05500000", "SPX   251219P00255000", "junk"]))
    assert out["root"].iloc[0] == "SPXW" and out["root"].iloc[1] == "SPX"
    assert out["expiry"].iloc[0] == date(2025, 7, 9)
    assert out["cp"].iloc[1] == "P"
    assert out["strike"].tolist()[:2] == [5500.0, 255.0]
    assert pd.isna(out["strike"].iloc[2])


def test_the_closing_bar_is_the_first_and_other_roots_are_dropped() -> None:
    idx = pd.DatetimeIndex(
        ["2025-07-09 19:59", "2025-07-09 19:59", "2025-07-09 19:59", "2025-07-09 20:00"], tz="UTC"
    )
    frame = pd.DataFrame(
        {
            "symbol": [
                "SPXW  250711C05500000",
                "SPY   250711C00550000",
                "SPX   250718P05000000",
                "SPXW  250711C05500000",
            ],
            "bid_px_00": [10.0, 1.0, 0.0, 99.0],
            "ask_px_00": [10.5, 1.1, 0.5, 99.5],
            "bid_sz_00": [5, 5, 5, 5],
            "ask_sz_00": [5, 5, 5, 5],
        },
        index=idx,
    )
    q = closing_quotes(frame)
    assert sorted(q["root"]) == ["SPX", "SPXW"]
    assert q.loc[q["root"] == "SPXW", "bid"].item() == 10.0, "the 16:00 bar is not the close"
    assert np.isnan(q.loc[q["root"] == "SPX", "bid"].item()), "a zero bid is no bid"
    assert set(q["date"]) == {date(2025, 7, 9)}
    assert closing_quotes(frame.iloc[0:0]).empty


def test_batch_metadata_reads_nanosecond_and_iso_starts() -> None:
    info = parse_metadata(
        {
            "job_id": "J",
            "query": {
                "dataset": "OPRA.PILLAR",
                "schema": "cbbo-1m",
                "start": 1752091140000000000,
                "end": "2025-07-09T20:01:00Z",
                "symbols": ["SPX.OPT", "SPY.OPT"],
            },
        },
        "J.tar",
        ("a.dbn.zst",),
        10,
    )
    assert info.start == "2025-07-09" and info.end == "2025-07-09"
    assert info.is_spx_close and info.as_row()["symbols"] == "SPX.OPT,SPY.OPT"


def test_pre_osi_vendor_symbols_take_their_root_from_settlement() -> None:
    raw = pd.DataFrame(
        {
            "date": [DAY0] * 3,
            "symbol": ["SPX.AT", "SPZ.MJ", "SPXW 240315C5000000"],
            "exdate": [DAY0 + timedelta(days=20)] * 3,
            "cp_flag": ["C", "P", "C"],
            "strike_price": [5_000_000.0, 4_000_000.0, 5_000_000.0],
            "best_bid": [10.0, 0.0, 12.0],
            "best_offer": [11.0, 0.5, 13.0],
            "volume": [1.0, 2.0, 3.0],
            "open_interest": [4.0, 5.0, 6.0],
            "impl_volatility": [0.2, 0.3, 0.2],
            "delta": [0.5, -0.1, 0.5],
            "am_settlement": [1.0, 0.0, 0.0],
            "forward_price": [np.nan] * 3,
        }
    )
    q = normalise_opprcd(raw)
    assert q["root"].tolist() == ["SPX", "SPXW", "SPXW"]
    assert q["strike"].tolist() == [5000.0, 4000.0, 5000.0]
    assert np.isnan(q["bid"].iloc[1])
    assert q["am_settled"].tolist() == [True, False, False]


def test_saturday_am_expiries_settle_on_the_friday() -> None:
    raw = pd.DataFrame(
        {
            "date": [date(2012, 3, 1)] * 3,
            "symbol": ["SPX 120317C1400000", "SPXW 120316C1400000", "SPX 120316C1400000"],
            "exdate": [date(2012, 3, 17), date(2012, 3, 17), date(2012, 3, 16)],
            "cp_flag": ["C"] * 3,
            "strike_price": [1_400_000.0] * 3,
            "best_bid": [10.0] * 3,
            "best_offer": [11.0] * 3,
            "volume": [0.0] * 3,
            "open_interest": [0.0] * 3,
            "impl_volatility": [0.2] * 3,
            "delta": [0.5] * 3,
            "am_settlement": [1.0, 0.0, 1.0],
            "forward_price": [np.nan] * 3,
        }
    )
    q = normalise_opprcd(raw)
    # AM on a Saturday moves to Friday; a PM row, or an AM row already on a Friday, stays.
    assert q["expiry"].tolist() == [date(2012, 3, 16), date(2012, 3, 17), date(2012, 3, 16)]


# -- the panel and its series ------------------------------------------------- #


def _panel(n_days: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    days, frames, spots = [], [], []
    spot = SPOT
    rng = np.random.default_rng(0)
    d = DAY0
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
            spots.append(spot)
            frames.append(_quotes(d, spot))
            spot *= float(np.exp(0.01 * rng.standard_normal()))
        d += timedelta(days=1)
    priced = price_surface(pd.concat(frames, ignore_index=True))
    panel, dropped = finalise(priced, "optionmetrics")
    assert (dropped > 0).all(), "the far wings fall outside the band, every day"
    assert set(dropped.index) == set(panel["date"]), "dropped is counted per date"
    return panel, pd.Series(spots, index=days)


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    panel, spots = _panel()
    return panel, spots, atm_term(panel)


def test_finalise_keeps_the_schema_and_the_band(built: tuple[pd.DataFrame, ...]) -> None:
    panel = built[0]
    assert {"source", "dte", "iv_vendor", "volume"} <= set(panel.columns)
    k = np.log(panel["strike"] / panel["forward"])
    assert k.abs().max() <= 0.7 + 1e-12
    assert (panel["source"] == "optionmetrics").all()


def test_atm_iv_and_the_30_day_interpolation_recover_the_flat_vol(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    term = built[2]
    assert float((term["atm_iv"] - SIGMA).abs().max()) < 0.003
    series = iv30(term)
    assert len(series) == built[0]["date"].nunique()
    assert float((series - SIGMA).abs().max()) < 0.003


def test_daily_table_realised_vol_and_vrp(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    panel, spots, term = built
    sp = parity_spot(panel)
    per_day = panel.groupby("date").size()
    daily = daily_table(term, spots, sp, pd.Series("optionmetrics", index=per_day.index), per_day)
    assert daily["rv21"].notna().sum() == len(daily) - 21
    expect = spots.pct_change().std() * np.sqrt(252)
    assert float(daily["rv21"].mean()) == pytest.approx(float(expect), rel=0.35)
    assert (daily["vrp30"].dropna() == (daily["iv30_atm"] - daily["rv21_fwd"]).dropna()).all()
    np.testing.assert_allclose(daily["carry30"], RATE - DIV, atol=2e-3)
    np.testing.assert_allclose(daily["rate30"], RATE, atol=2e-3)
    np.testing.assert_allclose(sp.to_numpy(), spots.to_numpy() * np.exp(-DIV * 9 / 365), rtol=5e-4)


def test_carry_is_read_at_the_expiry_nearest_thirty_days() -> None:
    term = pd.DataFrame(
        {
            "date": [DAY0, DAY0],
            "tau": [10 / 365, 31 / 365],
            "forward": [5001.0, 5010.0],
            "discount": [0.999, np.exp(-0.05 * 31 / 365)],
        }
    )
    out = carry30(term, pd.Series([5000.0], index=[DAY0]))
    assert out["rate30"].item() == pytest.approx(0.05)
    assert out["carry30"].item() == pytest.approx(np.log(5010 / 5000) / (31 / 365))


def test_the_surface_grid_is_flat_inside_the_quoted_range(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    grid = surface_grid(built[0])
    assert set(grid["tenor_days"]) == set(TENOR_GRID)
    long = (grid["tenor_days"].between(45, 365)) & (grid["k"].abs() <= 0.2)
    short = (grid["tenor_days"].between(14, 30)) & (grid["k"].abs() <= 0.05)
    inside = grid[long | short]
    assert inside["iv"].notna().all()
    assert float((inside["iv"] - SIGMA).abs().max()) < 0.005
    assert grid[grid["tenor_days"] == 7]["iv"].isna().all(), "nothing before the first expiry"
    at_30 = grid[(grid["tenor_days"] == 30) & (grid["k"].abs() <= 0.15)]
    assert at_30["iv"].notna().all(), "a tenor on a listed expiry reads that expiry alone"


def test_the_calendar_refuses_a_date_past_its_holiday_list() -> None:
    vendor = pd.Index([date(2025, 8, 29)])
    cal = trading_calendar(vendor, date(2027, 7, 9))
    assert date(2027, 7, 5) not in cal, "Independence Day observed"
    with pytest.raises(ValueError, match="NYSE_HOLIDAYS"):
        trading_calendar(vendor, date(2028, 1, 3))


def test_coverage_lists_every_missing_day() -> None:
    vendor = pd.Index([date(2025, 8, 27), date(2025, 8, 28), date(2025, 8, 29)])
    cal = trading_calendar(vendor, date(2025, 9, 5))
    assert date(2025, 9, 1) not in cal, "Labor Day"
    assert cal[-1] == date(2025, 9, 5) and len(cal) == 7
    counts = pd.DataFrame(
        {"date": [d for d in cal if d != date(2025, 9, 3)], "source": "x", "n_contracts": 5}
    )
    cov = coverage(cal, counts)
    assert cov.loc[cov["missing"], "date"].tolist() == [date(2025, 9, 3)]
    assert cov.loc[cov["missing"], "source"].item() == "none"


def test_the_seam_of_two_identical_vendors_is_zero(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    panel = built[0].assign(iv_vendor=built[0]["iv"])
    seam = seam_frame(panel, panel.drop(columns="iv_vendor"))
    assert len(seam) == panel["date"].nunique()
    assert (seam["iv_diff_median_volpts"].abs() < 1e-9).all()
    assert (seam["forward_diff_median_bp"].abs() < 1e-9).all()


# -- the real provider on the conformance suite ------------------------------- #


def _store(built: tuple[pd.DataFrame, pd.Series, pd.DataFrame]) -> SurfaceStore:
    panel, spots, term = built
    per_day = panel.groupby("date").size()
    daily = daily_table(
        term, spots, parity_spot(panel), pd.Series("optionmetrics", index=per_day.index), per_day
    )
    return SurfaceStore.from_frames(surface_grid(panel), daily)


def test_the_store_fills_every_grid_gap(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    store = _store(built)
    assert store.n == built[0]["date"].nunique()
    assert store.ivs.shape[1:] == (len(TENOR_GRID), len(K_GRID))
    assert np.isfinite(store.ivs).all()


def test_the_historical_provider_conforms(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame], tmp_path: Path
) -> None:
    cfg = MarketConfig(n_days=15, tenor_days=(8, 15))
    store = _store(built)

    class _Source(OptionMetricsSource):
        @property
        def available(self) -> bool:
            return True

        def provider(self) -> HistoricalChainProvider:  # type: ignore[override]
            return HistoricalChainProvider(cfg, store)

    report = conformance_report(_Source(cfg, root=tmp_path))
    assert report.passed, report.render()

    p = HistoricalChainProvider(cfg, store)
    c0 = p.reset(SeededStepRNG(3))
    assert c0.t == 0 and c0.ivs.shape == (2, 2 * cfg.n_strikes_each_side + 1)
    assert float(np.abs(c0.ivs[:, cfg.n_strikes_each_side] - SIGMA).max()) < 0.01
    first = p.start_date
    p.reset(SeededStepRNG(3))
    assert p.start_date == first, "the same seed picks the same window"
    for _ in range(cfg.n_days):
        p.step()
    assert p.done
    with pytest.raises(RuntimeError, match="horizon"):
        p.step()
    with pytest.raises(RuntimeError, match="before reset"):
        HistoricalChainProvider(cfg, store).step()
    with pytest.raises(ValueError, match="episode needs"):
        HistoricalChainProvider(MarketConfig(n_days=500), store)


def test_the_provider_interpolates_total_variance_across_tenors() -> None:
    n_days = 30
    grid_t = np.asarray(TENOR_GRID, dtype=np.float64)
    vols = np.interp(grid_t, [7, 365], [0.10, 0.30])  # a steep term structure, flat in k
    cube = np.repeat(vols[None, :, None], n_days, axis=0) * np.ones((1, 1, len(K_GRID)))
    store = SurfaceStore(
        dates=tuple(str(DAY0 + timedelta(days=i)) for i in range(n_days)),
        spot=np.full(n_days, SPOT),
        carry=np.zeros(n_days),
        ivs=cube,
    )
    cfg = MarketConfig(n_days=10, tenor_days=(31,))  # 31 trading days = 44.9 calendar days
    chain = HistoricalChainProvider(cfg, store).reset(SeededStepRNG(0))
    cal = 31 * 365 / 252
    j = int(np.searchsorted(grid_t, cal))
    t1, t2, v1, v2 = grid_t[j - 1], grid_t[j], vols[j - 1], vols[j]
    w = v1**2 * t1 + (cal - t1) / (t2 - t1) * (v2**2 * t2 - v1**2 * t1)
    np.testing.assert_allclose(chain.ivs[0], np.sqrt(w / cal), rtol=1e-9)


def test_the_source_is_available_once_the_panel_files_exist(tmp_path: Path) -> None:
    src = OptionMetricsSource(root=tmp_path)
    assert not src.available
    (tmp_path / "panel").mkdir()
    for f in ("surface_grid.parquet", "spx_daily.parquet"):
        (tmp_path / "panel" / f).touch()
    assert src.available


def test_the_data_root_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGALAB_DATA", "/tmp/somewhere")
    assert data_root() == Path("/tmp/somewhere")
    monkeypatch.delenv("VEGALAB_DATA")
    assert "wrds_optionmetrics" in str(data_root())
