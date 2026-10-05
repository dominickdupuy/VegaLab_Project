"""Hemdutt Rao: the undefined-risk margin rule, hardened (week 5).

The naked rule in ``optspread.margin`` is the standard broker approximation for a
short index option: 20% of the underlying less the out-of-the-money amount, never
below 10% of the strike, plus the current premium, per contract; a strangle carries
the greater side plus the other side's premium. Until now it had two loose
assertions. Each test here works the rule by hand with marks taken from
``bs_price`` directly, so the module is checked against arithmetic rather than
against itself, and the last two tests show the requirement binds inside the
environment: a breach forces a flat that pays its exit.

Every number is synthetic (the fair-IV generator at its defaults).
"""

from __future__ import annotations

import numpy as np
import pytest

from optspread.actions import SHORT_STRANGLE, build_position
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.costs import make_cost_model, trade_cost
from optspread.env import OBS_NAMES
from optspread.harness import build_env
from optspread.margin import margin_requirement
from optspread.market import SyntheticChainProvider
from optspread.pricing import bs_price
from optspread.rng import SeededStepRNG
from optspread.types import MULTIPLIER, Chain, OptionLeg, Right

CENT = 0.005  # "to the cent": within half a cent of dollars


def default_chain() -> Chain:
    """The t=0 chain of the default world: spot 5,000, flat 20% vol, tenors 21 and 42 days."""
    return SyntheticChainProvider(MarketConfig()).reset(SeededStepRNG(0))


def mid(chain: Chain, leg: OptionLeg) -> float:
    """Per-share mid from Black-Scholes directly, not through the margin module."""
    T = chain.tau(leg.expiry_day)
    sigma = chain.iv_at(leg.expiry_day, leg.strike)
    return float(bs_price(leg.right, chain.spot, leg.strike, chain.r, sigma, T))


def naked_requirement(chain: Chain, leg: OptionLeg) -> float:
    """The rule for one naked short leg, worked by hand, in dollars.

    20% of the underlying less the OTM amount, floored at 10% of the strike, plus the
    current mid, times the multiplier, per contract.
    """
    spot = chain.spot
    otm = max(leg.strike - spot, 0.0) if leg.right == "C" else max(spot - leg.strike, 0.0)
    return abs(leg.qty) * (max(0.20 * spot - otm, 0.10 * leg.strike) + mid(chain, leg)) * MULTIPLIER


def short_leg(chain: Chain, right: Right, strike: float, qty: int = -1) -> OptionLeg:
    expiry = int(chain.expiry_days[1])
    return OptionLeg(right, strike, expiry, qty, chain.price(right, strike, expiry))


# -- the rule, to the cent -------------------------------------------------- #


def test_reg_t_strangle_matches_the_hand_worked_requirement_to_the_cent() -> None:
    chain = default_chain()
    pos = build_position(SHORT_STRANGLE, chain, EnvConfig())
    assert pos is not None
    put = next(leg for leg in pos.legs if leg.right == "P")
    call = next(leg for leg in pos.legs if leg.right == "C")
    # At spot 5,000 the 16-delta legs sit at 4,650 and 5,500. By hand:
    #   put:  0.20 * 5000 - (5000 - 4650) = 650 > 0.10 * 4650 = 465  -> 650 + mid
    #   call: 0.20 * 5000 - (5500 - 5000) = 500 < 0.10 * 5500 = 550  -> 550 + mid
    # so the put side is the greater and the call's premium is added to it.
    assert (put.strike, call.strike) == (4650.0, 5500.0)
    put_side, call_side = naked_requirement(chain, put), naked_requirement(chain, call)
    assert put_side > call_side
    expected = put_side + abs(call.qty) * mid(chain, call) * MULTIPLIER
    got = margin_requirement(pos.legs, chain)
    assert got == pytest.approx(expected, abs=CENT)
    assert got == pytest.approx(71_466.22, abs=CENT), "the hand-worked figure for this chain"
    assert got > -pos.entry_value(), "a naked structure needs more than the credit it brings in"


def test_the_twenty_percent_branch_binds_near_the_money() -> None:
    chain = default_chain()
    leg = short_leg(chain, "P", 4900.0)
    # 0.20 * 5000 - (5000 - 4900) = 900 beats the floor 0.10 * 4900 = 490.
    expected = (900.0 + mid(chain, leg)) * MULTIPLIER
    assert margin_requirement((leg,), chain) == pytest.approx(expected, abs=CENT)
    assert expected > (490.0 + mid(chain, leg)) * MULTIPLIER


def test_the_ten_percent_floor_binds_deep_out_of_the_money() -> None:
    chain = default_chain()
    leg = short_leg(chain, "P", 3000.0)
    # 0.20 * 5000 - (5000 - 3000) = -1000, so the floor 0.10 * 3000 = 300 is the requirement.
    expected = (300.0 + mid(chain, leg)) * MULTIPLIER
    assert margin_requirement((leg,), chain) == pytest.approx(expected, abs=CENT)
    assert 0.20 * chain.spot - (chain.spot - leg.strike) < 0.0


def test_a_same_right_naked_pair_counts_each_premium_once() -> None:
    chain = default_chain()
    near, far = short_leg(chain, "P", 4900.0), short_leg(chain, "P", 4700.0)
    expected = naked_requirement(chain, near) + naked_requirement(chain, far)
    got = margin_requirement((near, far), chain)
    assert got == pytest.approx(expected, abs=CENT)
    # The strangle rule (greater side plus the other side's premium) is for opposite rights
    # only. Applying it here would drop the lesser side's notional requirement entirely.
    lesser = min((near, far), key=lambda leg: naked_requirement(chain, leg))
    strangle_style = max(naked_requirement(chain, leg) for leg in (near, far))
    strangle_style += abs(lesser.qty) * mid(chain, lesser) * MULTIPLIER
    assert abs(got - strangle_style) > 1.0


def test_contract_quantity_scales_the_requirement_exactly() -> None:
    chain = default_chain()
    one, two = short_leg(chain, "C", 5300.0, -1), short_leg(chain, "C", 5300.0, -2)
    # Exact arithmetic: the per-contract figure is the same float, multiplied by |qty|.
    assert margin_requirement((two,), chain) == pytest.approx(
        2.0 * margin_requirement((one,), chain), rel=1e-12
    )


def test_a_short_put_requirement_is_non_increasing_in_spot() -> None:
    base = default_chain()
    leg = OptionLeg("P", 4800.0, int(base.expiry_days[1]), -1, 0.0)  # entry price plays no part
    spots = np.arange(4000.0, 6001.0, 100.0)
    reqs = np.asarray(
        [
            margin_requirement(
                (leg,), Chain(base.t, float(s), base.r, base.strikes, base.expiry_days, base.ivs)
            )
            for s in spots
        ]
    )
    # Below the strike the requirement is 0.2 S + P(S) and the put's delta is below -0.5,
    # so the slope is negative; above it, K - 0.8 S + P(S) falls until the floor binds and
    # then only the premium moves, downward. A rally never raises a short put's margin.
    assert np.all(np.diff(reqs) <= 1e-9), reqs
    assert reqs[0] > reqs[-1]


def test_no_legs_consume_no_buying_power() -> None:
    assert margin_requirement((), default_chain()) == 0.0


# -- the requirement binds inside the environment ------------------------------ #


def test_a_strangle_opened_on_too_little_cash_is_liquidated_and_pays_its_exit() -> None:
    cfg = RunConfig(env=EnvConfig(initial_cash=20_000.0), cost=CostConfig(enabled=True))
    env = build_env(cfg)
    obs, info = env.reset(seed=0)
    obs, _, _, _, info = env.step(SHORT_STRANGLE)

    # Replay the same two chains to work out what the two crossings must have cost.
    provider = SyntheticChainProvider(cfg.market)
    today = provider.reset(SeededStepRNG(0))
    pos = build_position(SHORT_STRANGLE, today, cfg.env)
    assert pos is not None
    tomorrow = provider.step()
    costs = make_cost_model(cfg.cost)
    expected = trade_cost(costs, today, pos.legs) + trade_cost(costs, tomorrow, pos.legs)

    assert margin_requirement(pos.legs, tomorrow) > cfg.env.initial_cash, "the breach is certain"
    assert info["position"] is None, "the breach must force a flat"
    assert info["n_trades"] == 2, "the open and the forced close"
    assert info["cost"] == pytest.approx(expected, rel=1e-12), "the forced close pays its exit"
    assert obs[OBS_NAMES.index("has_position")] == 0.0
    assert obs[OBS_NAMES.index("margin_frac")] == 0.0


def test_the_same_strangle_on_ample_cash_is_held() -> None:
    cfg = RunConfig(cost=CostConfig(enabled=True))  # 1,000,000 of initial cash
    env = build_env(cfg)
    obs, info = env.reset(seed=0)
    obs, _, _, _, info = env.step(SHORT_STRANGLE)
    assert info["position"] is not None
    assert info["n_trades"] == 1
    assert 0.0 < obs[OBS_NAMES.index("margin_frac")] < 1.0
