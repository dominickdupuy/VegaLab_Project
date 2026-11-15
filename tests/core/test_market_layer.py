"""Hemdutt Rao: pricing, cost and margin gates (project document 10.1, 10.2)."""

import numpy as np
import pytest

from optspread.actions import IRON_CONDOR, SHORT_STRANGLE, build_position
from optspread.config import CostConfig, EnvConfig, MarketConfig
from optspread.costs import QuotedSpreadCost, ZeroCost, trade_cost
from optspread.leaks import chain_violations, perturbation_check
from optspread.margin import is_defined_risk, margin_requirement, max_loss
from optspread.market import SyntheticChainProvider
from optspread.portfolio import Book
from optspread.pricing import bs_greeks, bs_price
from optspread.rng import SeededStepRNG
from optspread.types import Chain, OptionLeg


def make_chain(seed: int = 0, premium: float = 0.0) -> Chain:
    provider = SyntheticChainProvider(MarketConfig(iv_premium_vol_pts=premium))
    return provider.reset(SeededStepRNG(seed))


# -- pricing --------------------------------------------------------------- #


@pytest.mark.invariant
def test_put_call_parity_and_no_arb_bounds() -> None:
    S, r, sigma, T = 5000.0, 0.04, 0.2, 42 / 252
    K = np.linspace(4200, 5800, 33)
    C, P = bs_price("C", S, K, r, sigma, T), bs_price("P", S, K, r, sigma, T)
    disc = np.exp(-r * T)
    np.testing.assert_allclose(C - P, S - K * disc, atol=1e-8)
    assert np.all(C <= S) and np.all(np.maximum(S - K * disc, 0.0) <= C)
    assert np.all(K * disc >= P) and np.all(np.maximum(K * disc - S, 0.0) <= P)


@pytest.mark.invariant
@pytest.mark.parametrize("right", ["C", "P"])
def test_greeks_match_finite_differences(right: str) -> None:
    S, K, r, sigma, T = 5000.0, 4900.0, 0.04, 0.2, 21 / 252
    delta, gamma, vega, theta = (float(x) for x in bs_greeks(right, S, K, r, sigma, T))
    h, hv, ht = 0.5, 1e-4, 1e-5

    def p(S_: float = S, sig: float = sigma, T_: float = T) -> float:
        return float(bs_price(right, S_, K, r, sig, T_))

    assert delta == pytest.approx((p(S + h) - p(S - h)) / (2 * h), rel=1e-5)
    assert gamma == pytest.approx((p(S + h) - 2 * p() + p(S - h)) / h**2, rel=1e-3)
    assert vega == pytest.approx((p(sig=sigma + hv) - p(sig=sigma - hv)) / (2 * hv), rel=1e-5)
    assert theta == pytest.approx(-(p(T_=T + ht) - p(T_=T - ht)) / (2 * ht), rel=1e-4)


@pytest.mark.invariant
def test_expired_option_is_intrinsic_without_nan() -> None:
    assert float(bs_price("C", 5100.0, 5000.0, 0.04, 0.2, 0.0)) == 100.0
    assert float(bs_price("P", 5100.0, 5000.0, 0.04, 0.0, 0.1)) == 0.0
    assert np.isfinite(bs_greeks("P", 5100.0, 5000.0, 0.04, 0.2, 0.0)).all()


# -- market ---------------------------------------------------------------- #


@pytest.mark.invariant
def test_same_seed_same_path_and_premium_lifts_iv() -> None:
    a = [make_chain(3).spot]
    prov = SyntheticChainProvider(MarketConfig())
    prov.reset(SeededStepRNG(3))
    a += [prov.step().spot for _ in range(10)]
    prov.reset(SeededStepRNG(3))
    b = [prov.reset(SeededStepRNG(3)).spot] + [prov.step().spot for _ in range(10)]
    assert a == b
    assert make_chain(premium=2.0).ivs.min() == pytest.approx(0.22)


# -- market: the research knobs (GH-46) ----------------------------------- #


def walk(config: MarketConfig, seed: int, days: int) -> list[Chain]:
    provider = SyntheticChainProvider(config)
    chains = [provider.reset(SeededStepRNG(seed))]
    chains += [provider.step() for _ in range(days)]
    return chains


def atm_iv(chain: Chain) -> float:
    return float(chain.ivs[0, chain.ivs.shape[1] // 2])


@pytest.mark.invariant
def test_the_knobs_at_zero_are_the_study_world_bit_for_bit() -> None:
    zero = MarketConfig(alpha_bps=0.0, vol_of_vol=0.0, skew_slope=0.0)
    for a, b in zip(walk(MarketConfig(), 7, 63), walk(zero, 7, 63), strict=True):
        assert a.spot == b.spot
        assert np.array_equal(a.ivs, b.ivs) and np.array_equal(a.strikes, b.strikes)


@pytest.mark.behaviour
def test_vol_of_vol_moves_the_atm_vol_and_starts_at_sigma() -> None:
    chains = walk(MarketConfig(vol_of_vol=0.5), 42, 20)
    assert atm_iv(chains[0]) == pytest.approx(0.20)
    assert len({round(atm_iv(c), 9) for c in chains}) > 10
    flat = walk(MarketConfig(), 42, 20)
    assert {atm_iv(c) for c in flat} == {0.20}


@pytest.mark.invariant
def test_log_return_is_a_pure_read_with_vol_of_vol() -> None:
    # The leak detector reads log_return(t + 1) to look ahead without stepping; a draw on
    # each call would move the path it measures.
    provider = SyntheticChainProvider(MarketConfig(vol_of_vol=0.5, n_days=20))
    provider.reset(SeededStepRNG(99))
    first = [provider.log_return(d) for d in range(1, 21)]
    assert first == [provider.log_return(d) for d in range(1, 21)]
    spots = [provider.step().spot for _ in range(20)]
    assert spots[-1] == pytest.approx(5000.0 * float(np.exp(sum(first))))
    assert provider.log_return(21) == provider.log_return(21)  # past the horizon: still a read


@pytest.mark.invariant
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_knobs_keep_the_no_look_ahead_contract(seed: int) -> None:
    config = MarketConfig(vol_of_vol=0.8, skew_slope=0.3, alpha_bps=40.0, n_days=21)
    for t in (0, 5, 20):
        past_same, future_differs = perturbation_check(
            lambda: SyntheticChainProvider(config), seed, seed + 100, t
        )
        assert past_same and future_differs


@pytest.mark.behaviour
def test_skew_slope_tilts_the_smile_toward_the_downside() -> None:
    chain = SyntheticChainProvider(MarketConfig(skew_slope=0.10)).reset(SeededStepRNG(0))
    row = chain.ivs[0]
    assert np.all(np.diff(row) < 0), "dearer at every lower strike"
    assert np.array_equal(chain.ivs[0], chain.ivs[1]), "the same smile at every tenor"
    expected = 0.20 + 0.10 * np.log(chain.spot / chain.strikes)
    np.testing.assert_allclose(row, expected, rtol=1e-12)
    assert not chain_violations(chain)
    flat = SyntheticChainProvider(MarketConfig(skew_slope=0.0)).reset(SeededStepRNG(0))
    assert float(np.ptp(flat.ivs)) == 0.0


@pytest.mark.behaviour
def test_a_premium_is_planted_on_the_tilted_surface() -> None:
    fair = SyntheticChainProvider(MarketConfig(skew_slope=-0.2)).reset(SeededStepRNG(4))
    rich = MarketConfig(skew_slope=-0.2, iv_premium_vol_pts=2.0)
    planted = SyntheticChainProvider(rich).reset(SeededStepRNG(4))
    np.testing.assert_allclose(planted.ivs - fair.ivs, 0.02, atol=1e-12)


@pytest.mark.behaviour
def test_alpha_bps_is_a_drift_per_day_not_per_year() -> None:
    # With almost no vol the return into day 1 is the drift alone: 100 bps is 1% a day.
    config = MarketConfig(alpha_bps=100.0, sigma=0.001, n_days=5)
    returns = []
    for seed in range(200):
        provider = SyntheticChainProvider(config)
        provider.reset(SeededStepRNG(seed))
        returns.append(provider.log_return(1))
    assert float(np.mean(returns)) == pytest.approx(0.01, abs=1e-4)
    up = [walk(MarketConfig(alpha_bps=25.0), s, 63)[-1].spot for s in range(40)]
    level = [walk(MarketConfig(), s, 63)[-1].spot for s in range(40)]
    assert all(u > v for u, v in zip(up, level, strict=True)), "the same shocks, shifted up"


# -- costs: the round-trip gate ------------------------------------------- #


@pytest.mark.invariant
def test_round_trip_loses_exactly_the_modelled_cost() -> None:
    chain = make_chain()
    pos = build_position(IRON_CONDOR, chain, EnvConfig())
    assert pos is not None
    costs = QuotedSpreadCost(CostConfig())
    expected = 2 * trade_cost(costs, chain, pos.legs)

    book = Book(1_000_000.0)
    book.rebalance(pos, chain, costs)
    book.rebalance(None, chain, costs)
    assert book.equity(chain) == pytest.approx(1_000_000.0 - expected)
    assert expected > 0

    free = Book(1_000_000.0)
    free.rebalance(pos, chain, ZeroCost())
    free.rebalance(None, chain, ZeroCost())
    assert free.equity(chain) == pytest.approx(1_000_000.0)


@pytest.mark.invariant
def test_settlement_at_expiry_is_cash_at_intrinsic() -> None:
    prov = SyntheticChainProvider(MarketConfig())
    chain = prov.reset(SeededStepRNG(5))
    book = Book(1_000_000.0)
    pos = build_position(IRON_CONDOR, chain, EnvConfig(expiry_idx=0))
    assert pos is not None
    book.rebalance(pos, chain, ZeroCost())
    while chain.t < pos.first_expiry:
        chain = prov.step()
    before = book.equity(chain)
    assert book.settle_expired(chain)
    assert book.position is None
    assert book.equity(chain) == pytest.approx(before)


# -- margin ----------------------------------------------------------------- #


@pytest.mark.invariant
def test_defined_risk_margin_is_max_loss() -> None:
    chain = make_chain()
    ic = build_position(IRON_CONDOR, chain, EnvConfig())
    assert ic is not None and is_defined_risk(ic.legs)
    puts = sorted(leg.strike for leg in ic.legs if leg.right == "P")
    calls = sorted(leg.strike for leg in ic.legs if leg.right == "C")
    credit = -ic.entry_value()
    assert credit > 0
    width = max(puts[1] - puts[0], calls[1] - calls[0]) * 100.0
    assert margin_requirement(ic.legs, chain) == pytest.approx(max_loss(ic.legs))
    assert max_loss(ic.legs) == pytest.approx(width - credit)


@pytest.mark.behaviour
def test_undefined_risk_margin_exceeds_premium() -> None:
    chain = make_chain()
    strangle = build_position(SHORT_STRANGLE, chain, EnvConfig())
    assert strangle is not None and not is_defined_risk(strangle.legs)
    assert margin_requirement(strangle.legs, chain) > -strangle.entry_value()
    single = (OptionLeg("P", 4800.0, chain.t + 21, -1, 30.0),)
    assert margin_requirement(single, chain) > 0


# -- the pre-trade margin check --------------------------------------------- #


@pytest.mark.behaviour
def test_an_open_is_refused_one_dollar_under_the_line_and_made_one_dollar_over() -> None:
    chain = make_chain()
    pos = build_position(SHORT_STRANGLE, chain, EnvConfig())
    assert pos is not None
    costs = QuotedSpreadCost(CostConfig())
    line = margin_requirement(pos.legs, chain) + trade_cost(costs, chain, pos.legs)

    short = Book(line - 1.0)
    assert short.rebalance(pos, chain, costs) == 0.0, "a refused open charges nothing"
    assert short.position is None and short.refused_opens == 1 and short.n_trades == 0
    assert short.cash == line - 1.0

    enough = Book(line + 1.0)
    enough.rebalance(pos, chain, costs)
    assert enough.position is pos and enough.refused_opens == 0
    # Opened on the line, the book sits where the env's breach check would not fire.
    assert enough.equity(chain) >= enough.margin_used(chain)
    assert enough.equity(chain) - enough.margin_used(chain) == pytest.approx(1.0)


@pytest.mark.behaviour
def test_switching_into_an_uncovered_structure_still_closes_what_was_held() -> None:
    chain = make_chain()
    condor = build_position(IRON_CONDOR, chain, EnvConfig())
    strangle = build_position(SHORT_STRANGLE, chain, EnvConfig())
    assert condor is not None and strangle is not None
    costs = QuotedSpreadCost(CostConfig())

    book = Book(40_000.0)  # carries the condor (25,059 max loss), not the strangle (56,466)
    book.rebalance(condor, chain, costs)
    assert book.position is condor
    cost = book.rebalance(strangle, chain, costs)
    assert cost == pytest.approx(trade_cost(costs, chain, condor.legs)), "only the exit is paid"
    assert book.position is None, "the policy asked to leave the condor; it is left"
    assert book.refused_opens == 1 and book.n_trades == 2
