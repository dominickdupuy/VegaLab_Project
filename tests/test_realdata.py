"""Hemdutt Rao: the real-data path, proven on quotes whose answer is known.

Every fixture here is a chain priced by ``optspread.pricing`` at a known vol,
rate and forward, quoted with a spread around that price. The pipeline has to
hand back what went in: the parity fit recovers the forward and the discount
factor, the IV solver recovers the vol, and the provider built from the result
clears the same conformance suite as the synthetic source.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optspread.actions import IRON_CONDOR
from optspread.config import CostConfig, EnvConfig, MarketConfig
from optspread.costs import make_cost_model
from optspread.data import OptionMetricsSource, conformance_report, make_source
from optspread.env import SpreadEnv
from optspread.market import SyntheticChainProvider
from optspread.pricing import bs_price
from optspread.realdata import HOLDOUT_START, SPLITS, SplitRanges, data_root
from optspread.realdata.opra import closing_quotes, parse_metadata, parse_osi
from optspread.realdata.optionmetrics import normalise_opprcd, strike_divisor
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
from optspread.realdata.provider import (
    HistoricalChainProvider,
    SurfaceStore,
    TermStructureChain,
    market_surface,
)
from optspread.realdata.surface import fit_carry, implied_vol, price_surface, time_to_expiry
from optspread.rng import SeededStepRNG
from optspread.types import Chain

SIGMA, RATE, DIV = 0.20, 0.04, 0.015
SPOT = 5000.0
DAY0 = date(2024, 3, 1)
# Dates of a surface of your own, cut on its own ranges rather than A8's (#28).
OWN = SplitRanges(
    train=(date(2022, 5, 2), date(2022, 6, 30)), eval=(date(2022, 7, 1), date(2022, 8, 24))
)
# The priced fixture's dates (from DAY0), a surface of your own read as its eval split.
FIXTURE = SplitRanges(
    train=(date(2024, 1, 1), date(2024, 2, 29)), eval=(date(2024, 3, 1), date(2024, 12, 31))
)


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


def _bars(stamps: list[str], symbols: list[str], bids: list[float]) -> pd.DataFrame:
    n = len(stamps)
    return pd.DataFrame(
        {
            "symbol": symbols,
            "bid_px_00": bids,
            "ask_px_00": [b + 0.5 for b in bids],
            "bid_sz_00": [5] * n,
            "ask_sz_00": [5] * n,
        },
        index=pd.DatetimeIndex(stamps, tz="UTC"),
    )


def test_a_contract_missing_from_the_closing_bar_keeps_its_next_bar_in_the_window() -> None:
    # cbbo-1m writes a bar only when the book changed: the SPX put had no update at
    # 15:59, so the 16:00 bar is its first quote in the window; 16:01 is past it.
    frame = _bars(
        ["2025-07-09 19:59", "2025-07-09 20:00", "2025-07-09 20:00", "2025-07-09 20:01"],
        [
            "SPXW  250711C05500000",
            "SPXW  250711C05500000",
            "SPX   250718P05000000",
            "SPX   250718C05100000",
        ],
        [10.0, 99.0, 7.0, 3.0],
    )
    q = closing_quotes(frame).set_index(["root", "cp"])
    assert len(q) == 2, "the 16:01 record lies past the closing window"
    assert q.loc[("SPXW", "C"), "bid"] == 10.0, "a contract in the closing bar keeps it"
    assert q.loc[("SPX", "P"), "bid"] == 7.0, "an illiquid contract is not lost for the day"
    assert set(q["date"]) == {date(2025, 7, 9)}


def test_on_an_early_close_the_window_opens_at_the_first_bar() -> None:
    # 12:59 New York is 16:59 UTC in July: the close is still the file's first bar.
    frame = _bars(
        ["2025-07-03 16:59", "2025-07-03 17:00", "2025-07-03 17:00"],
        ["SPXW  250711C05500000", "SPXW  250711C05500000", "SPXW  250711P05400000"],
        [10.0, 99.0, 4.0],
    )
    q = closing_quotes(frame).set_index("cp")
    assert q.loc["C", "bid"] == 10.0 and q.loc["P", "bid"] == 4.0
    assert set(q["date"]) == {date(2025, 7, 3)}


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


def _vendor_rows(**cols: list[object]) -> pd.DataFrame:
    """Raw opprcd rows: the columns given, the rest filled with neutral values."""
    n = len(next(iter(cols.values())))
    base: dict[str, list[object]] = {
        "cp_flag": ["C"] * n,
        "strike_price": [1_800_000.0] * n,
        "best_bid": [10.0] * n,
        "best_offer": [11.0] * n,
        "volume": [0.0] * n,
        "open_interest": [0.0] * n,
        "impl_volatility": [0.2] * n,
        "delta": [0.5] * n,
        "forward_price": [np.nan] * n,
    }
    return pd.DataFrame(base | cols)


def test_a_saturday_expiry_after_good_friday_settles_on_the_thursday() -> None:
    # The April 2014 monthly: the vendor dates it Saturday 2014-04-19, the market
    # was shut on Good Friday 2014-04-18, so it settled at Thursday's open.
    raw = _vendor_rows(
        date=[date(2014, 4, 1)] * 3,
        symbol=["SPX 140419C1800000", "SPX 140517C1800000", "SPXW 140416C1800000"],
        exdate=[date(2014, 4, 19), date(2014, 5, 17), date(2014, 4, 16)],
        am_settlement=[1.0, 1.0, 0.0],
    )
    q = normalise_opprcd(raw)
    assert q["expiry"].tolist() == [date(2014, 4, 17), date(2014, 5, 16), date(2014, 4, 16)]
    tau = time_to_expiry(q["date"], q["expiry"], q["am_settled"])
    assert tau[0] == pytest.approx((16 - 6.5 / 24) / 365), "tau runs to Thursday's open"


def test_the_other_vendor_roots_map_to_spx_or_spxw_by_settlement() -> None:
    raw = _vendor_rows(
        date=[date(2012, 3, 1)] * 6,
        symbol=[
            "SPXPM 120316C1400000",  # PM third-Friday monthly
            "SPXQ 120330C1400000",  # PM end-of-quarter
            "SPT 120317C1400000",  # an old AM root code
            "SPX 120317P1400000",
            "SPXW 120309C1400000",
            "SPXQ 120309C1400000",  # the same contract as the SPXW row above
        ],
        exdate=[
            date(2012, 3, 16),
            date(2012, 3, 30),
            date(2012, 3, 17),
            date(2012, 3, 17),
            date(2012, 3, 9),
            date(2012, 3, 9),
        ],
        cp_flag=["C", "C", "C", "P", "C", "C"],
        strike_price=[1_400_000.0] * 6,
        best_bid=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        am_settlement=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    )
    q = normalise_opprcd(raw)
    assert q["root"].tolist() == ["SPXW", "SPXW", "SPX", "SPX", "SPXW"]
    assert q["bid"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0], "the listed SPXW row is kept"
    # The AM rows' Saturday moves to the Friday, the day the SPXPM monthly settles too.
    assert (
        q["expiry"].tolist()[:4] == [date(2012, 3, 16), date(2012, 3, 30)] + [date(2012, 3, 16)] * 2
    )
    assert q["am_settled"].tolist() == [False, False, True, True, False]


def test_a_year_of_listed_roots_is_unchanged_by_the_mapping() -> None:
    raw = _vendor_rows(
        date=[date(2016, 3, 1)] * 2,
        symbol=["SPXW 160304C1900000", "SPXW 160304C1900000"],
        exdate=[date(2016, 3, 4)] * 2,
        best_bid=[1.0, 2.0],
        am_settlement=[0.0, 0.0],
    )
    assert normalise_opprcd(raw)["bid"].tolist() == [1.0, 2.0]


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


def test_iv30_takes_spxw_where_both_roots_list_the_same_expiry() -> None:
    # A third Friday listed by both roots: the AM row is 6.5 hours shorter. Only the
    # SPXW row's vol (0.30) brackets 30 days with the far expiry here.
    day, third_friday, far = pd.Timestamp("2016-01-04"), "2016-01-15", "2016-02-19"
    term = pd.DataFrame(
        {
            "date": [day] * 3,
            "expiry": [third_friday, third_friday, far],
            "root": ["SPX", "SPXW", "SPXW"],
            "tau": [10.73 / 365, 11.0 / 365, 46.0 / 365],
            "atm_iv": [0.10, 0.30, 0.30],
        }
    )
    assert float(iv30(term).iloc[0]) == pytest.approx(0.30)


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
    # The carry is against the parity spot, the prepaid forward to the 9-day expiry, so
    # it is rate less dividends plus the dividends to that expiry spread over 30 days.
    np.testing.assert_allclose(daily["carry30_parity"], RATE - DIV + DIV * 9 / 30, atol=2e-3)
    np.testing.assert_allclose(daily["rate30"], RATE, atol=2e-3)
    np.testing.assert_allclose(sp.to_numpy(), spots.to_numpy() * np.exp(-DIV * 9 / 365), rtol=5e-4)
    assert "carry30" not in daily, "the carry against the close is gone (panel version 4)"


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
    assert out["carry30_parity"].item() == pytest.approx(np.log(5010 / 5000) / (31 / 365))


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
    return SurfaceStore.from_frames(surface_grid(panel), daily, "eval", FIXTURE)


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


# -- a held leg between listed tenors (#27) ------------------------------------ #


def _sloped_store(n_days: int, dates: tuple[str, ...] | None = None) -> SurfaceStore:
    """The same steep term structure every day, flat in k, at a constant spot."""
    vols = np.interp(np.asarray(TENOR_GRID, dtype=np.float64), [7, 60], [0.10, 0.30])
    cube = np.repeat(vols[None, :, None], n_days, axis=0) * np.ones((1, 1, len(K_GRID)))
    return SurfaceStore(
        dates=dates or tuple(str(DAY0 + timedelta(days=i)) for i in range(n_days)),
        spot=np.full(n_days, SPOT),
        carry=np.zeros(n_days),
        ivs=cube,
    )


TERM = MarketConfig(n_days=24, tenor_days=(10, 30))


def test_a_held_leg_marks_continuously_between_listed_tenors() -> None:
    p = HistoricalChainProvider(TERM, _sloped_store(40))
    chain = p.reset(SeededStepRNG(0))
    expiry, K = int(chain.expiry_days[1]), SPOT  # opened on the 30-day tenor, at the money
    near, far = float(chain.ivs[0, 16]), float(chain.ivs[1, 16])
    assert far - near > 0.02, "the term structure must be steep enough to show a step"
    new, old = [], []
    while True:
        assert isinstance(chain, TermStructureChain)
        new.append(chain.iv_at(expiry, K))
        old.append(Chain.iv_at(chain, expiry, K))  # the nearest-tenor lookup it replaces
        left = expiry - chain.t
        if left == 30:
            assert new[-1] == far, "on a listed tenor the listed vol, exactly"
        elif left == 10:
            assert new[-1] == near
        elif 10 < left < 30:
            w = near**2 * 10 + (left - 10) / 20 * (far**2 * 30 - near**2 * 10)
            assert new[-1] == pytest.approx(np.sqrt(w / left), rel=1e-12)
        else:
            assert new[-1] == near, "flat in vol inside the shortest tenor"
        if p.done:
            break
        chain = p.step()
    old_step = float(np.abs(np.diff(old)).max())
    new_step = float(np.abs(np.diff(new)).max())
    assert old_step == pytest.approx(far - near), "the old lookup jumps a whole tenor gap"
    # Forward variance is constant between the tenors, so the vol falls fastest near the
    # short one; it falls every day, and never by more than a fraction of the gap.
    assert np.all(np.diff(new) <= 0.0)
    assert new_step < old_step / 4
    # Beyond the longest listed tenor the vol is flat too.
    assert chain.iv_at(int(chain.expiry_days[-1]) + 5, K) == float(chain.ivs[-1, 16])


def test_the_book_marks_a_held_structure_through_the_interpolated_vol() -> None:
    """Equity under a held condor moves smoothly; under the nearest-tenor lookup it steps."""

    class _Nearest(HistoricalChainProvider):
        def _chain(self) -> Chain:
            c = super()._chain()
            return Chain(c.t, c.spot, c.r, c.strikes, c.expiry_days, c.ivs)

    def daily_pnl(provider: HistoricalChainProvider) -> np.ndarray:
        env = SpreadEnv(provider, make_cost_model(CostConfig()), EnvConfig(episode_days=24))
        env.reset(seed=0)
        pnl = [env.step(IRON_CONDOR)[4]["pnl"] for _ in range(19)]  # held, never expires
        return np.asarray(pnl[1:])  # the day it opens pays the spread

    smooth = daily_pnl(HistoricalChainProvider(TERM, _sloped_store(40)))
    stepped = daily_pnl(_Nearest(TERM, _sloped_store(40)))
    jump = float(np.abs(np.diff(stepped)).max())
    assert float(np.abs(np.diff(smooth)).max()) < jump / 5


def test_the_synthetic_world_keeps_the_plain_chain() -> None:
    chain = SyntheticChainProvider(MarketConfig()).reset(SeededStepRNG(0))
    assert type(chain) is Chain


# -- training and evaluation dates (#28) --------------------------------------- #


def _window(p: HistoricalChainProvider, seed: int) -> set[str]:
    p.reset(SeededStepRNG(seed))
    i = p.store.dates.index(p.start_date)
    return set(p.store.dates[i : i + p.n_days + 1])


def test_a_surface_of_your_own_cuts_its_splits_on_its_own_ranges() -> None:
    grid, daily = _frames(_business_days(date(2022, 5, 2), date(2022, 9, 30)))
    cfg = MarketConfig(n_days=10, tenor_days=(5, 10))
    train = HistoricalChainProvider(cfg, SurfaceStore.from_frames(grid, daily, "train", OWN))
    test = HistoricalChainProvider(cfg, SurfaceStore.from_frames(grid, daily, "eval", OWN))
    seen_train = set().union(*(_window(train, s) for s in range(1_000, 1_200)))
    seen_eval = set().union(*(_window(test, s) for s in range(100_000, 100_200)))
    assert not seen_train & seen_eval
    assert max(seen_train) <= "2022-06-30" < "2022-07-01" <= min(seen_eval)
    assert max(seen_eval) <= "2022-08-24", "nothing after the evaluation range is read"
    assert train.n_starts == 44 - 10 and test.n_starts == 39 - 10


def test_other_ranges_are_for_a_root_of_your_own_never_the_team_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a8 = {
        "train": (date(2010, 1, 1), date(2018, 12, 31)),
        "eval": (date(2019, 1, 1), date(2022, 8, 24)),
    }
    assert a8 == SPLITS, "pre-registration A8"
    assert OptionMetricsSource().split == "train" and OptionMetricsSource().ranges is None
    src = make_source("optionmetrics", TERM, split="eval")
    assert isinstance(src, OptionMetricsSource) and src.split == "eval"
    assert str(OWN) == "train 2022-05-02..2022-06-30, eval 2022-07-01..2022-08-24"
    with pytest.raises(ValueError, match="wholly before"):
        SplitRanges(
            train=(date(2020, 1, 1), date(2020, 7, 1)), eval=(date(2020, 7, 1), date(2021, 1, 1))
        )

    team = tmp_path / "team"
    monkeypatch.setenv("VEGALAB_DATA", str(team))
    with pytest.raises(ValueError, match="A8"):
        OptionMetricsSource(ranges=OWN)  # the default root is the team's
    with pytest.raises(ValueError, match="A8"):
        OptionMetricsSource(root=team, ranges=OWN)

    mine = tmp_path / "mine"
    (mine / "panel").mkdir(parents=True)
    grid, daily = _frames(_business_days(date(2022, 5, 2), date(2022, 9, 30)))
    grid.to_parquet(mine / "panel" / "surface_grid.parquet", index=False)
    daily.to_parquet(mine / "panel" / "spx_daily.parquet", index=False)
    source = OptionMetricsSource(MarketConfig(n_days=10, tenor_days=(5, 10)), mine, "eval", OWN)
    provider = source.provider()
    assert isinstance(provider, HistoricalChainProvider)
    assert provider.store.dates[0] == "2022-07-01" and provider.store.dates[-1] == "2022-08-24"


def test_no_window_runs_across_days_dropped_from_the_store(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    panel, spots, term = built
    days = sorted(panel["date"].unique())
    dropped = set(days[30:33])  # three days with no surface: in the calendar, not the store
    per_day = panel.groupby("date").size()
    daily = daily_table(
        term, spots, parity_spot(panel), pd.Series("optionmetrics", index=per_day.index), per_day
    )
    grid = surface_grid(panel)
    store = SurfaceStore.from_frames(grid[~grid["date"].isin(dropped)], daily, "eval", FIXTURE)
    assert store.n == 77 and store.calendar is not None
    assert int(np.diff(store.calendar).max()) == 4, "the gap is seen against the calendar"
    cfg = MarketConfig(n_days=15, tenor_days=(8, 15))
    p = HistoricalChainProvider(cfg, store)
    assert p.n_starts == (30 - 15) + (47 - 15), "a start whose window spans the gap is refused"
    position = {str(d): i for i, d in enumerate(days)}
    for seed in range(300):
        window = sorted(position[d] for d in _window(p, seed))
        assert window[-1] - window[0] == cfg.n_days, "consecutive trading days"


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


# -- strike units: the vendor's x 1000, or already divided out ----------------- #


def test_the_strike_unit_is_read_from_the_data() -> None:
    close = pd.Series([4000.0, 4100.0, 4200.0])
    points = pd.Series([3000.0, 4000.0, 5000.0])
    assert strike_divisor(points, close) == 1.0
    assert strike_divisor(points * 1000.0, close) == 1000.0
    with pytest.raises(ValueError, match="neither unit"):
        strike_divisor(points * 30.0, close)


def test_a_copy_already_in_index_points_is_not_divided_again() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2019-03-01"],
            "symbol": ["SPXW 190308C2800000"],
            "exdate": ["2019-03-08"],
            "cp_flag": ["C"],
            "strike_price": [2800.0],
            "best_bid": [10.0],
            "best_offer": [11.0],
            "volume": [1.0],
            "open_interest": [1.0],
            "impl_volatility": [0.15],
            "delta": [0.5],
            "am_settlement": [0.0],
            "forward_price": [2805.0],
        }
    )
    assert normalise_opprcd(raw, divisor=1.0)["strike"].item() == 2800.0
    vendor = raw.assign(strike_price=2_800_000.0)
    assert normalise_opprcd(vendor)["strike"].item() == 2800.0


# -- the date split and gaps (#28) -------------------------------------------- #


def _frames(days: list[date]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A flat 20% surface on every day given, and a daily series on the same days."""
    grid = pd.DataFrame(
        [
            {"date": d, "tenor_days": t, "k": k, "iv": SIGMA}
            for d in days
            for t in TENOR_GRID
            for k in K_GRID
        ]
    )
    daily = pd.DataFrame({"date": days, "spot": SPOT, "carry30": 0.0})
    return grid, daily


def _business_days(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


def test_train_and_eval_are_disjoint_and_neither_reaches_the_holdout() -> None:
    days = _business_days(date(2018, 12, 1), date(2018, 12, 31))
    days += _business_days(date(2022, 8, 1), date(2022, 9, 30))
    grid, daily = _frames(days)
    train = SurfaceStore.from_frames(grid, daily, "train")
    evaluation = SurfaceStore.from_frames(grid, daily, "eval")
    assert max(train.dates) == "2018-12-31" and min(evaluation.dates) == "2022-08-01"
    assert max(evaluation.dates) == "2022-08-24"
    assert not set(train.dates) & set(evaluation.dates)
    for store in (train, evaluation):
        assert all(date.fromisoformat(d) < HOLDOUT_START for d in store.dates)
    everything = market_surface(grid, daily)
    assert max(everything.dates) == "2022-09-30", "market measurement still reads every date"
    assert min(everything.dates) == "2018-12-03"
    assert set(SPLITS) == {"train", "eval"}, "no split serves the holdout"


def test_a_window_never_spans_a_day_dropped_as_unusable() -> None:
    days = _business_days(date(2019, 3, 1), date(2019, 3, 29))
    grid, daily = _frames(days)
    hole = days[10]
    grid = grid[grid["date"] != hole]  # the day is on the calendar but has no surface
    store = SurfaceStore.from_frames(grid, daily, "eval")
    assert str(hole) not in store.dates
    n_days = 5
    for i in store.window_starts(n_days):
        window = store.dates[i : i + n_days + 1]
        assert not window[0] < str(hole) < window[-1], f"window {window} spans {hole}"
    assert store.window_starts(n_days).size == store.n - n_days - n_days
    cfg = MarketConfig(n_days=n_days, tenor_days=(8,))
    for seed in range(40):
        p = HistoricalChainProvider(cfg, store)
        p.reset(SeededStepRNG(seed))
        end = store.dates[store.dates.index(p.start_date) + n_days]
        assert not p.start_date < str(hole) < end


def test_the_provider_refuses_a_surface_with_no_whole_window() -> None:
    days = _business_days(date(2019, 3, 1), date(2019, 3, 29))
    grid, daily = _frames(days)
    grid = grid[~grid["date"].isin(days[3::4])]  # a hole every fourth day
    store = SurfaceStore.from_frames(grid, daily, "eval")
    with pytest.raises(ValueError, match="span a gap"):
        HistoricalChainProvider(MarketConfig(n_days=5, tenor_days=(8,)), store)


# -- the data audit: a split is checked, every date is measurement only (#28) ----- #


def _write_panel(root: Path, days: list[date]) -> Path:
    """The two panel files a provider reads, a flat surface on ``days``, under ``root``."""
    grid, daily = _frames(days)
    (root / "panel").mkdir(parents=True)
    grid.to_parquet(root / "panel" / "surface_grid.parquet", index=False)
    daily.to_parquet(root / "panel" / "spx_daily.parquet", index=False)
    return root / "panel"


@pytest.mark.parametrize("split", [None, "holdout", "EVAL", ""])
def test_a_split_outside_splits_is_refused_everywhere(split: object, tmp_path: Path) -> None:
    from optspread.config import RunConfig
    from optspread.harness import build_env
    from optspread.manifest import DataRecord
    from optspread.realdata.provider import load_store

    panel = _write_panel(tmp_path / "mine", _business_days(date(2019, 3, 1), date(2019, 4, 30)))
    grid, daily = _frames(_business_days(date(2019, 3, 1), date(2019, 4, 30)))
    with pytest.raises(ValueError, match="split must be one of"):
        OptionMetricsSource(root=tmp_path / "mine", split=split)  # type: ignore[arg-type]
    for name in ("optionmetrics", "synthetic", "optionmetrics_stub"):
        with pytest.raises(ValueError, match="split must be one of"):
            make_source(name, split=split)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split must be one of"):
        build_env(RunConfig(), "optionmetrics", split=split)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split must be one of"):
        SurfaceStore.from_frames(grid, daily, split)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split must be one of"):
        load_store(panel, split)  # type: ignore[arg-type]
    store = SurfaceStore.from_frames(grid, daily, "eval")
    if split is not None:  # None is a store from arrays of your own, cut by nobody
        with pytest.raises(ValueError, match="split must be one of"):
            HistoricalChainProvider(
                MarketConfig(n_days=5, tenor_days=(8,)), replace(store, split=split)
            )
    # No real-data record can say None, so none reads as a legacy eval run.
    good = OptionMetricsSource(root=tmp_path / "mine", split="eval")
    assert DataRecord.of(good).split == "eval"
    with pytest.raises(ValueError):
        DataRecord.of(OptionMetricsSource(root=tmp_path / "mine", split=split))  # type: ignore[arg-type]


def test_every_date_is_a_market_surface_no_provider_takes() -> None:
    days = _business_days(date(2018, 12, 3), date(2023, 1, 31))
    grid, daily = _frames(days)
    whole = market_surface(grid, daily)
    assert whole.dates[0] == "2018-12-03" and whole.dates[-1] == "2023-01-31"
    assert any(date.fromisoformat(d) >= HOLDOUT_START for d in whole.dates)
    with pytest.raises(TypeError, match="market measurement only"):
        HistoricalChainProvider(MarketConfig(n_days=5, tenor_days=(8,)), whole)  # type: ignore[arg-type]
    for split in SPLITS:
        cut = SurfaceStore.from_frames(grid, daily, split)
        assert cut.split == split and all(date.fromisoformat(d) < HOLDOUT_START for d in cut.dates)


# -- the data audit: the team's panel is known by its content (#28) -------------- #


def _team(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A team folder under ``VEGALAB_DATA`` whose MANIFEST.json records its panel files."""
    import json

    from optspread.realdata import panel_digests

    team = tmp_path / "team"
    panel = _write_panel(team, _business_days(date(2022, 5, 2), date(2022, 10, 31)))
    files = {k: {"sha256": v, "bytes": 1} for k, v in panel_digests(panel).items()}
    (team / "MANIFEST.json").write_text(json.dumps({"files": files}))
    monkeypatch.setenv("VEGALAB_DATA", str(team))
    return team


def test_ranges_are_refused_on_a_symlink_or_a_copy_of_the_team_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    team = _team(tmp_path, monkeypatch)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "panel").symlink_to(team / "panel")  # bypass (a) of the audit
    copied = tmp_path / "copied"
    shutil.copytree(team / "panel", copied / "panel")
    files = tmp_path / "files"
    (files / "panel").mkdir(parents=True)
    for name in ("surface_grid.parquet", "spx_daily.parquet"):
        (files / "panel" / name).symlink_to(team / "panel" / name)
    for root in (team, linked, copied, files):
        with pytest.raises(ValueError, match="A8"):
            OptionMetricsSource(root=root, split="eval", ranges=OWN)
    # A copy whose daily table is your own is still the team's surface grid.
    pd.DataFrame({"date": [date(2022, 5, 2)], "spot": [1.0], "carry30": [0.0]}).to_parquet(
        copied / "panel" / "spx_daily.parquet", index=False
    )
    with pytest.raises(ValueError, match="A8"):
        OptionMetricsSource(root=copied, split="eval", ranges=OWN)
    # Without ranges the team panel serves its A8 splits as before.
    assert OptionMetricsSource(root=linked, split="eval").ranges is None


def test_the_team_panel_is_known_by_the_repository_manifest_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import optspread.realdata as realdata

    assert realdata.REPO_MANIFEST.name == "DATA_MANIFEST.json"
    recorded = json.loads(realdata.REPO_MANIFEST.read_text())["files"]
    assert recorded["panel/surface_grid.parquet"]["sha256"] in realdata.team_digests()
    team = _team(tmp_path, monkeypatch)
    repo = tmp_path / "DATA_MANIFEST.json"
    (team / "MANIFEST.json").rename(repo)  # only the repository's copy knows it now
    monkeypatch.setattr(realdata, "REPO_MANIFEST", repo)
    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="sha256"):
        OptionMetricsSource(root=team, split="eval", ranges=OWN)


def test_the_default_root_is_refused_wherever_vegalab_data_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from optspread.realdata import DEFAULT_ROOT

    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="A8"):  # bypass (b) of the audit
        OptionMetricsSource(root=DEFAULT_ROOT, split="eval", ranges=OWN)


def test_load_store_and_the_provider_refuse_ranges_on_the_team_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from optspread.realdata.provider import load_store

    team = _team(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="A8"):  # bypass (c) of the audit
        load_store(team / "panel", "eval", OWN)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "panel").symlink_to(team / "panel")
    with pytest.raises(ValueError, match="A8"):
        load_store(linked / "panel", "eval", OWN)
    # The team panel on A8's own split loads, and its store records what it read.
    store = load_store(team / "panel", "eval")
    assert store.split == "eval" and store.ranges is None and len(store.panel) == 2
    # A store of your own relabelled with the team's digests is refused (bypass (e)).
    own_days = _business_days(date(2022, 5, 2), date(2022, 9, 30))
    mine = load_store(_write_panel(tmp_path / "mine", own_days), "eval", OWN)
    assert mine.dates[0] == "2022-07-01" and mine.dates[-1] == "2022-08-24"
    cfg = MarketConfig(n_days=10, tenor_days=(5, 10))
    assert HistoricalChainProvider(cfg, mine).n_starts > 0
    with pytest.raises(ValueError, match="A8"):
        HistoricalChainProvider(cfg, replace(mine, panel=store.panel))


# -- the data audit: carry against the parity spot (MINOR-3) ---------------------- #


def test_the_carry_ignores_a_close_that_disagrees_with_the_option_snapshot(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    panel, spots, term = built
    sp = parity_spot(panel)
    per_day = panel.groupby("date").size()
    source = pd.Series("optionmetrics", index=per_day.index)
    # A close 74 bp below the options on every other day, as on 2011-11-01.
    off = spots * np.where(np.arange(len(spots)) % 2 == 0, 1.0, np.exp(-0.0074))
    clean = daily_table(term, spots, sp, source, per_day)
    noisy = daily_table(term, off, sp, source, per_day)
    np.testing.assert_allclose(noisy["carry30_parity"], clean["carry30_parity"], atol=1e-12)
    assert float(noisy["carry30_parity"].std()) < 1e-3, "no 9% swing from a 74 bp gap"
    # The engine prices on the parity spot and its carry, so its forward is the market's.
    grid = surface_grid(panel)
    store = SurfaceStore.from_frames(grid, noisy, "eval", FIXTURE)
    by_day = noisy.set_index("date")
    first = date.fromisoformat(store.dates[0])
    assert store.spot[0] == pytest.approx(float(by_day.loc[first, "spot_parity"]))
    assert store.carry[0] == pytest.approx(float(by_day.loc[first, "carry30_parity"]))
    near30 = term[term["date"] == first].assign(gap=lambda t: (t["tau"] - 30 / 365).abs())
    row = near30.sort_values("gap").iloc[0]
    engine_forward = store.spot[0] * np.exp(store.carry[0] * row["tau"])
    assert engine_forward == pytest.approx(float(row["forward"]), rel=1e-12)


def test_a_daily_table_from_before_version_4_keeps_its_own_pair() -> None:
    days = _business_days(date(2019, 3, 1), date(2019, 3, 29))
    grid, daily = _frames(days)
    daily = daily.assign(spot_parity=SPOT * 1.01, carry30=0.03)  # no carry30_parity
    store = SurfaceStore.from_frames(grid, daily, "eval")
    assert (store.spot == SPOT).all() and (store.carry == 0.03).all()


# -- the data audit: calendar arbitrage on the served surface (MINOR-4, #29) ------ #


def _wing_day(day: date) -> pd.DataFrame:
    """A grid day with both of the audit's causes: a deep-put wing quoted rich on the
    short tenors (2010-05-06, K = 750) and short tenors that stop at a higher call wing
    than the long ones (2011-06-17, the 30-day row ending at 0.209, the 60-day at 0.145)."""
    rows = []
    for t in TENOR_GRID:
        for k in K_GRID:
            iv = 0.20 + 0.02 * np.sqrt(t / 30.0) - 0.3 * min(k, 0.0)
            if k <= -0.35 and t <= 30:
                iv = 0.78  # a 0.05/4.00 market
            if k >= 0.125 and t > 30:
                iv = np.nan  # the long rows stop short in the call wing
            if 0.1 <= k <= 0.125 and t > 30:
                iv = 0.145
            if 0.1 <= k <= 0.125 and t <= 30:
                iv = 0.209
            if k > 0.125 and t <= 30:
                iv = np.nan
            rows.append({"date": day, "tenor_days": t, "k": float(k), "iv": iv})
    return pd.DataFrame(rows)


def test_the_real_provider_serves_no_calendar_arbitrage() -> None:
    from optspread.leaks import calendar_violations

    days = _business_days(date(2019, 3, 1), date(2019, 4, 30))
    grid = pd.concat([_wing_day(d) for d in days], ignore_index=True)
    daily = pd.DataFrame({"date": days, "spot": SPOT, "carry30": 0.01})
    store = SurfaceStore.from_frames(grid, daily, "eval")
    w = store.ivs**2 * np.asarray(TENOR_GRID, dtype=np.float64)[None, :, None]
    assert (np.diff(w, axis=1) >= -1e-12).all(), "total variance never falls with tenor"
    for tenors in ((21, 42), (5, 10, 21, 42, 63, 126, 252)):
        cfg = MarketConfig(n_days=20, tenor_days=tenors)
        p = HistoricalChainProvider(cfg, store)
        chains = [p.reset(SeededStepRNG(0))] + [p.step() for _ in range(cfg.n_days)]
        assert [v for c in chains for v in calendar_violations(c)] == [], tenors
    # Cells already in order are untouched: the ATM column as quoted.
    atm = int(np.argmin(np.abs(K_GRID)))
    quoted = _wing_day(days[0]).pivot_table(index="tenor_days", columns="k", values="iv")
    np.testing.assert_allclose(store.ivs[0][:, atm], quoted.iloc[:, atm].to_numpy())


def test_the_fill_runs_along_tenor_in_total_variance() -> None:
    from optspread.realdata.provider import calendar_free

    t = np.asarray(TENOR_GRID, dtype=np.float64)
    block = np.full((len(TENOR_GRID), len(K_GRID)), np.nan)
    block[2, 5] = 0.50  # 21 days
    block[6, 5] = 0.20  # 90 days, less total variance: the column is pooled to its mean
    block[3, 6] = 0.25  # one quote in its column: flat vol in tenor
    out = calendar_free(block)
    assert np.isfinite(out).all()
    w = (0.50**2 * 21 + 0.20**2 * 90) / 2
    assert out[2, 5] ** 2 * 21 == pytest.approx(w) and out[6, 5] ** 2 * 90 == pytest.approx(w)
    np.testing.assert_allclose(out[:, 6], 0.25)
    np.testing.assert_allclose(out[:, 0], out[:, 5])  # flat in k beyond the last column
    assert (np.diff(out**2 * t[:, None], axis=0) >= -1e-12).all()


# -- the data audit: stale expiries, roots, closures and the calendar (MINOR-5..8) - #


def test_an_expiry_off_the_day_forward_curve_is_stale_and_dropped(
    built: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> None:
    from optspread.realdata.panel import STALE_FORWARD_BP, stale_expiries

    days = [2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 37, 44, 46, 51, 58, 65]
    days += [72, 79, 86, 100, 117, 135, 163, 191, 226, 317]
    tau = np.asarray(days, dtype=np.float64) / 365
    forward = 2660.0 * np.exp(0.02 * tau)
    off = np.array([5, 12, 17, 26])  # four expiries 208 bp low, as on 2018-02-05
    forward[off] *= np.exp(-0.0208)
    forward[22] *= np.exp(0.006)  # 60 bp: noise, kept
    groups = pd.DataFrame(
        {"date": date(2018, 2, 5), "root": "SPXW", "expiry": days, "tau": tau, "forward": forward}
    )
    assert STALE_FORWARD_BP == 100.0
    assert np.flatnonzero(stale_expiries(groups)).tolist() == off.tolist()
    # finalise drops every quote on a stale expiry and counts them.
    panel = built[0]
    day = panel["date"].iloc[0]
    one = panel[panel["date"] == day].copy()
    expiries = sorted(one["expiry"].unique())
    stale = expiries[2]
    one.loc[one["expiry"] == stale, "forward"] *= np.exp(-0.0208)
    kept, dropped = finalise(one.assign(am_settled=False), "optionmetrics")
    assert stale not in set(kept["expiry"])
    assert int(dropped.loc[day]) == int((one["expiry"] == stale).sum())


def test_the_2010_pm_weeklies_under_spx_are_spxw() -> None:
    raw = _vendor_rows(
        date=[date(2010, 5, 21)] * 4,
        symbol=[
            "SPX 100604C850000",
            "SPX 100618C850000",
            "SPX 100528C850000",
            "SPXW 100528C850000",
        ],
        exdate=[date(2010, 6, 4), date(2010, 6, 19), date(2010, 5, 28), date(2010, 5, 28)],
        best_bid=[1.0, 2.0, 3.0, 4.0],
        am_settlement=[0.0, 1.0, 0.0, 0.0],
    )
    q = normalise_opprcd(raw)
    assert q["root"].tolist() == ["SPXW", "SPX", "SPXW"], "the root means settlement"
    assert q["bid"].tolist() == [1.0, 2.0, 4.0], "the remapped duplicate goes, the listed stays"
    assert q["am_settled"].tolist() == [False, True, False]


def test_an_expiry_on_an_unscheduled_closure_settles_the_day_before() -> None:
    from optspread.realdata.panel import NYSE_HOLIDAYS, UNSCHEDULED_CLOSURES

    assert {date(2018, 12, 5), date(2025, 1, 9)} <= set(UNSCHEDULED_CLOSURES) <= set(NYSE_HOLIDAYS)
    raw = _vendor_rows(
        date=[date(2018, 12, 3), date(2025, 1, 6), date(2018, 12, 3)],
        symbol=["SPXW 181205C2800000", "SPXW 250109C5900000", "SPXW 181207C2800000"],
        exdate=[date(2018, 12, 5), date(2025, 1, 9), date(2018, 12, 7)],
        am_settlement=[0.0, 0.0, 0.0],
    )
    q = normalise_opprcd(raw)
    assert q["expiry"].tolist() == [date(2018, 12, 4), date(2025, 1, 8), date(2018, 12, 7)]


def test_the_nyse_calendar_counts_every_year() -> None:
    from optspread.realdata.panel import nyse_trading_days

    counts = {
        2010: 252, 2011: 252, 2012: 250, 2013: 252, 2014: 252, 2015: 252, 2016: 252,
        2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 250,
        2024: 252, 2025: 250, 2026: 251, 2027: 251,
    }  # fmt: skip
    for year, n in counts.items():
        assert len(nyse_trading_days(date(year, 1, 1), date(year, 12, 31))) == n, year
    assert nyse_trading_days(date(2009, 1, 1), date(2009, 12, 31)) == []


def test_a_day_missing_from_both_tables_is_a_gap() -> None:
    days = _business_days(date(2019, 3, 1), date(2019, 3, 29))
    hole = date(2019, 3, 15)
    grid, daily = _frames([d for d in days if d != hole])  # gone from both, as in the audit
    store = SurfaceStore.from_frames(grid, daily, "eval")
    assert store.calendar is not None and int(np.diff(store.calendar).max()) == 2
    for i in store.window_starts(5):
        window = store.dates[i : i + 6]
        assert not window[0] < str(hole) < window[-1], f"window {window} spans {hole}"


def test_the_build_check_runs_the_chain_checks_on_both_splits(tmp_path: Path) -> None:
    from scripts.build_spx_panel import calendar_check

    days = _business_days(date(2018, 10, 1), date(2019, 3, 29))
    panel = _write_panel(tmp_path / "copy", days)
    assert calendar_check(panel) == 0
    grid = pd.concat([_wing_day(d) for d in days], ignore_index=True)
    grid.to_parquet(panel / "surface_grid.parquet", index=False)
    assert calendar_check(panel) == 0, "the wing day is served arbitrage-free"
