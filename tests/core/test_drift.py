"""Hemdutt Rao: the carry bias of the study's world, and the fair world (GH-36, A11).

The path is a zero-drift GBM and the book earns nothing on cash. Priced at r = 0.04 (the
study's world) a contract is not worth its expected payoff under that path, and a
seller of puts loses the carry before costs; priced at r = 0 (the default from A11 on)
every contract is. Both are pinned here, on the provider's own path. Every number is
synthetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optspread.actions import IRON_CONDOR, PUT_CREDIT_SPREAD, SHORT_STRANGLE, build_position
from optspread.config import STUDY_R, CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.harness import build_env, run_episode
from optspread.market import SyntheticChainProvider
from optspread.policies import DeltaSellerPolicy
from optspread.pricing import bs_price
from optspread.rng import SeededStepRNG
from optspread.types import MULTIPLIER

FAIR_R = 0.0  # forward = spot under the zero-drift path; MarketConfig's default from A11
N_EPISODES = 60
TENOR = 21  # trading days to the front expiry


def _world(r: float, alpha_bps: float = 0.0) -> RunConfig:
    return RunConfig(
        market=MarketConfig(r=r, alpha_bps=alpha_bps),
        cost=CostConfig(enabled=False),
        env=EnvConfig(),
    )


def _pnls(config: RunConfig, action: int) -> np.ndarray:
    env = build_env(config)
    policy = DeltaSellerPolicy(action, roll_dte=config.env.roll_dte)
    return np.array([run_episode(env, policy, seed).pnl for seed in range(N_EPISODES)])


def _t(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


@pytest.mark.invariant
@pytest.mark.slow
def test_a_paired_run_shows_the_study_rate_biases_the_put_spread_seller() -> None:
    # Paired on the same draws, costs off: the zero-drift path against the path with the
    # drift the chain's pricing assumes (r a year, as alpha_bps a day). At the study's rate
    # the rolled put credit spread does worse on the martingale, by about the $406 a held
    # spread gives up at entry (below) per roll; at r = 0 the two are one path, so the
    # bias is exactly zero, and the fair world's own mean is indistinguishable from zero.
    action = PUT_CREDIT_SPREAD
    drift_bps = STUDY_R / 252 * 10_000
    biased = _pnls(_world(STUDY_R), action) - _pnls(_world(STUDY_R, drift_bps), action)
    assert biased.mean() < -200.0 and _t(biased) < -3.0
    fair = _pnls(_world(FAIR_R), action)
    assert np.array_equal(fair, _pnls(_world(FAIR_R, 0.0), action))
    assert abs(_t(fair)) < 2.0


def _entry_bias(r: float, action: int) -> float:
    """Dollars a seller gives up opening ``action`` on the day-0 chain priced at ``r``: the
    structure's value at r = 0 (its expected payoff under the path, test below) less the
    credit it is opened at. Negative is a loss to the seller."""
    chain = SyntheticChainProvider(MarketConfig(r=r)).reset(SeededStepRNG(0))
    position = build_position(action, chain, EnvConfig())
    assert position is not None
    tau = chain.tau(position.legs[0].expiry_day)
    sigma = MarketConfig().sigma
    fair = math.fsum(
        leg.qty * float(bs_price(leg.right, chain.spot, leg.strike, FAIR_R, sigma, tau))
        for leg in position.legs
    )
    return (fair * MULTIPLIER) - position.entry_value()


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("action", "bias"),
    [(PUT_CREDIT_SPREAD, -406.35), (IRON_CONDOR, -95.25), (SHORT_STRANGLE, -133.80)],
)
def test_each_structure_opened_at_the_study_rate_gives_up_the_carry(
    action: int, bias: float
) -> None:
    # Exact, not sampled: at r = 0.04 the put legs are cheap and the call legs dear against
    # their expected payoff, so a put spread loses the most and a condor, whose call side
    # gains back part of it, the least (the outside review's -$406 and -$95 per held
    # structure). In the fair world a structure opens at exactly its expected payoff.
    assert _entry_bias(STUDY_R, action) == pytest.approx(bias, abs=0.01)
    assert _entry_bias(FAIR_R, action) == 0.0


def _terminal_spots(n_paths: int, r: float) -> np.ndarray:
    """The provider's own spot ``TENOR`` days on, from 5,000, over ``n_paths`` seeds."""
    provider = SyntheticChainProvider(MarketConfig(r=r))
    out = np.empty(n_paths)
    for seed in range(n_paths):
        provider.reset(SeededStepRNG(seed))
        log_s = sum(provider.log_return(day) for day in range(1, TENOR + 1))
        out[seed] = provider.config.spot0 * math.exp(log_s)
    return out


@pytest.mark.invariant
@pytest.mark.slow
def test_the_fair_world_prices_every_contract_at_its_expected_payoff() -> None:
    cfg = MarketConfig(r=FAIR_R)
    spots = _terminal_spots(10_000, FAIR_R)
    assert np.array_equal(spots[:500], _terminal_spots(500, STUDY_R))  # r never moves the path
    s0, sigma, tau = cfg.spot0, cfg.sigma, TENOR / cfg.trading_days_per_year
    # The path is a martingale: the forward is the spot, as r = 0 prices it.
    se = spots.std(ddof=1) / math.sqrt(len(spots))
    assert abs(spots.mean() - s0) < 3.0 * se
    for strike in (4600.0, 4800.0, 5000.0, 5200.0, 5400.0):
        for right in ("P", "C"):
            payoff = (
                np.maximum(strike - spots, 0.0) if right == "P" else np.maximum(spots - strike, 0.0)
            )
            mc, mc_se = payoff.mean(), payoff.std(ddof=1) / math.sqrt(len(payoff))
            price = float(bs_price(right, s0, strike, FAIR_R, sigma, tau))
            assert abs(mc - price) < 3.0 * mc_se, (right, strike, mc, price, mc_se)
    # At the study's rate the same path does not pay what the chain charges: the forward
    # S - K e^{-rT} (a call less a put) is priced K (1 - e^{-rT}) above its payoff's mean,
    # about 16.6 points at K = 5,000, far beyond the Monte Carlo error.
    strike = 5000.0
    forward_price = float(
        bs_price("C", s0, strike, STUDY_R, sigma, tau)
        - bs_price("P", s0, strike, STUDY_R, sigma, tau)
    )
    shortfall = forward_price - (spots - strike).mean()
    assert shortfall > 4.0 * se
    assert shortfall == pytest.approx(strike * (1.0 - math.exp(-STUDY_R * tau)), abs=3.0 * se)


@pytest.mark.invariant
def test_the_default_world_is_the_fair_one() -> None:
    assert MarketConfig().r == FAIR_R and STUDY_R == 0.04
