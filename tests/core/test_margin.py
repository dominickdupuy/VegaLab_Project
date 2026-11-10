"""Hemdutt Rao: the undefined-risk margin rule, hardened (week 5, re-worked week 6).

The naked rule in ``optspread.margin`` is Cboe Rule 10.3(c)(5)(A) for a short
option on a broad-based index: 15% of the underlying less the out-of-the-money
amount, never below 10% of the underlying for a call or 10% of the strike for a
put, plus the current premium, per contract; a strangle carries the greater side
plus the other side's premium. Each test here works the rule by hand with marks
taken from ``bs_price`` directly, so the module is checked against arithmetic
rather than against itself, and the last tests show the requirement binds inside
the environment: an open the account cannot carry is refused before it is made,
and one that clears the check but breaches later is forced flat and pays its exit.

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
from optspread.types import MULTIPLIER, Chain, OptionLeg, Position, Right

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

    15% of the underlying less the OTM amount, floored at 10% of the underlying for a
    call and 10% of the strike for a put, plus the current mid, times the multiplier,
    per contract.
    """
    spot = chain.spot
    if leg.right == "C":
        otm, floor = max(leg.strike - spot, 0.0), 0.10 * spot
    else:
        otm, floor = max(spot - leg.strike, 0.0), 0.10 * leg.strike
    return abs(leg.qty) * (max(0.15 * spot - otm, floor) + mid(chain, leg)) * MULTIPLIER


def short_leg(chain: Chain, right: Right, strike: float, qty: int = -1) -> OptionLeg:
    expiry = int(chain.expiry_days[1])
    return OptionLeg(right, strike, expiry, qty, chain.price(right, strike, expiry))


# -- the rule, to the cent -------------------------------------------------- #


@pytest.mark.invariant
def test_reg_t_strangle_matches_the_hand_worked_requirement_to_the_cent() -> None:
    chain = default_chain()
    pos = build_position(SHORT_STRANGLE, chain, EnvConfig())
    assert pos is not None
    put = next(leg for leg in pos.legs if leg.right == "P")
    call = next(leg for leg in pos.legs if leg.right == "C")
    # At spot 5,000 the 16-delta legs sit at 4,650 and 5,500, with mids 34.555208 and
    # 30.106994. By hand:
    #   put:  0.15 * 5000 - (5000 - 4650) = 400 < 0.10 * 4650 = 465  -> 465 + 34.555208
    #         = 499.555208, or 49,955.52 dollars
    #   call: 0.15 * 5000 - (5500 - 5000) = 250 < 0.10 * 5000 = 500  -> 500 + 30.106994
    #         = 530.106994, or 53,010.70 dollars
    # so the call side is the greater and the put's premium, 3,455.52, is added to it:
    # 53,010.70 + 3,455.52 = 56,466.22. Both floors bind, and the call's is 10% of the
    # underlying (500), not of its strike (550, which would have made it 58,010.70).
    assert (put.strike, call.strike) == (4650.0, 5500.0)
    put_side, call_side = naked_requirement(chain, put), naked_requirement(chain, call)
    assert put_side == pytest.approx(49_955.52, abs=CENT)
    assert call_side == pytest.approx(53_010.70, abs=CENT)
    expected = call_side + abs(put.qty) * mid(chain, put) * MULTIPLIER
    got = margin_requirement(pos.legs, chain)
    assert got == pytest.approx(expected, abs=CENT)
    assert got == pytest.approx(56_466.22, abs=CENT), "the hand-worked figure for this chain"
    assert got > -pos.entry_value(), "a naked structure needs more than the credit it brings in"


@pytest.mark.invariant
def test_the_fifteen_percent_branch_binds_near_the_money() -> None:
    chain = default_chain()
    leg = short_leg(chain, "P", 4900.0)
    # 0.15 * 5000 - (5000 - 4900) = 650 beats the floor 0.10 * 4900 = 490. With the mid at
    # 103.001331 that is 753.001331, or 75,300.13 dollars.
    expected = (650.0 + mid(chain, leg)) * MULTIPLIER
    assert margin_requirement((leg,), chain) == pytest.approx(expected, abs=CENT)
    assert margin_requirement((leg,), chain) == pytest.approx(75_300.13, abs=CENT)
    assert expected > (490.0 + mid(chain, leg)) * MULTIPLIER


@pytest.mark.invariant
def test_the_put_floor_is_ten_percent_of_the_strike_deep_out_of_the_money() -> None:
    chain = default_chain()
    leg = short_leg(chain, "P", 3000.0)
    # 0.15 * 5000 - (5000 - 3000) = -1250, so the floor 0.10 * 3000 = 300 is the requirement;
    # the mid is below a millionth of a point, so 30,000.00 dollars.
    expected = (300.0 + mid(chain, leg)) * MULTIPLIER
    assert margin_requirement((leg,), chain) == pytest.approx(expected, abs=CENT)
    assert margin_requirement((leg,), chain) == pytest.approx(30_000.00, abs=CENT)
    assert 0.15 * chain.spot - (chain.spot - leg.strike) < 0.0
    # A put's floor is a share of its exercise price, not of the underlying (500 here).
    assert expected < (0.10 * chain.spot + mid(chain, leg)) * MULTIPLIER


@pytest.mark.invariant
def test_the_call_floor_is_ten_percent_of_the_underlying_not_the_strike() -> None:
    chain = default_chain()
    leg = short_leg(chain, "C", 5500.0)
    # 0.15 * 5000 - (5500 - 5000) = 250, so the floor binds, and for a call it is
    # 0.10 * 5000 = 500, not 0.10 * 5500 = 550. With the mid at 30.106994 that is
    # 530.106994, or 53,010.70 dollars; the strike floor would have read 58,010.70.
    expected = (500.0 + mid(chain, leg)) * MULTIPLIER
    assert margin_requirement((leg,), chain) == pytest.approx(expected, abs=CENT)
    assert margin_requirement((leg,), chain) == pytest.approx(53_010.70, abs=CENT)
    assert margin_requirement((leg,), chain) < (550.0 + mid(chain, leg)) * MULTIPLIER - 1.0


@pytest.mark.invariant
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


@pytest.mark.invariant
def test_contract_quantity_scales_the_requirement_exactly() -> None:
    chain = default_chain()
    one, two = short_leg(chain, "C", 5300.0, -1), short_leg(chain, "C", 5300.0, -2)
    # Exact arithmetic: the per-contract figure is the same float, multiplied by |qty|.
    assert margin_requirement((two,), chain) == pytest.approx(
        2.0 * margin_requirement((one,), chain), rel=1e-12
    )


@pytest.mark.invariant
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
    # Below the strike the requirement is 0.15 S + P(S) and the put's delta is below -0.5,
    # so the slope is negative; above it, K - 0.85 S + P(S) falls until the floor 0.1 K
    # binds and then only the premium moves, downward. A rally never raises a short put's
    # margin.
    assert np.all(np.diff(reqs) <= 1e-9), reqs
    assert reqs[0] > reqs[-1]


@pytest.mark.behaviour
def test_no_legs_consume_no_buying_power() -> None:
    assert margin_requirement((), default_chain()) == 0.0


# -- the requirement binds inside the environment ------------------------------ #


@pytest.mark.behaviour
def test_a_strangle_on_too_little_cash_is_refused_at_open_not_liquidated_a_day_late() -> None:
    # 20,000 of cash against a 56,466.22 requirement (the hand-worked figure above). Before
    # the pre-trade check the strangle was opened, held overnight, and forced flat on the
    # next step, paying two crossings for a structure the account could never carry.
    cfg = RunConfig(env=EnvConfig(initial_cash=20_000.0), cost=CostConfig(enabled=True))
    env = build_env(cfg)
    obs, info = env.reset(seed=0)
    assert margin_requirement(_strangle_at_open(cfg, 0)[0].legs, info["chain"]) > 20_000.0

    obs, reward, _, _, info = env.step(SHORT_STRANGLE)
    assert info["position"] is None, "refused at the open, so nothing was held overnight"
    assert info["refused_opens"] == 1
    assert info["n_trades"] == 0, "no open and no forced close"
    assert info["cost"] == 0.0 and reward == 0.0
    assert obs[OBS_NAMES.index("has_position")] == 0.0
    assert obs[OBS_NAMES.index("margin_frac")] == 0.0

    # Asking again is refused again: the book stays flat rather than churning.
    obs, _, _, _, info = env.step(SHORT_STRANGLE)
    assert info["position"] is None and info["refused_opens"] == 2 and info["n_trades"] == 0
    assert info["equity"] == 20_000.0


@pytest.mark.behaviour
def test_a_strangle_opened_on_just_enough_cash_breaches_on_a_rally_and_pays_its_exit() -> None:
    # The breach path in the env is still live: the pre-trade check holds only at the chain
    # it is made on. Seed 7 rallies 1.75% on day one (5,000 to 5,087.56), which marks the
    # short call up and raises the call side's requirement, so a book opened 3.16 dollars
    # above the line at t=0 is under it at t=1.
    seed = 7
    cfg = RunConfig(env=EnvConfig(initial_cash=56_600.0), cost=CostConfig(enabled=True))
    pos, today, tomorrow = _strangle_at_open(cfg, seed)
    costs = make_cost_model(cfg.cost)
    open_cost = trade_cost(costs, today, pos.legs)
    assert open_cost + margin_requirement(pos.legs, today) <= cfg.env.initial_cash, "it opens"

    env = build_env(cfg)
    obs, info = env.reset(seed=seed)
    obs, _, _, _, info = env.step(SHORT_STRANGLE)

    equity_tomorrow = cfg.env.initial_cash - open_cost + pos.value(tomorrow) - pos.value(today)
    assert margin_requirement(pos.legs, tomorrow) > equity_tomorrow, "the breach is certain"
    assert info["refused_opens"] == 0, "the open cleared the pre-trade check"
    assert info["position"] is None, "the breach must force a flat"
    assert info["n_trades"] == 2, "the open and the forced close"
    expected = open_cost + trade_cost(costs, tomorrow, pos.legs)
    assert info["cost"] == pytest.approx(expected, rel=1e-12), "the forced close pays its exit"
    assert obs[OBS_NAMES.index("has_position")] == 0.0
    assert obs[OBS_NAMES.index("margin_frac")] == 0.0


@pytest.mark.behaviour
def test_the_same_strangle_on_ample_cash_is_held() -> None:
    cfg = RunConfig(cost=CostConfig(enabled=True))  # 1,000,000 of initial cash
    env = build_env(cfg)
    obs, info = env.reset(seed=0)
    obs, _, _, _, info = env.step(SHORT_STRANGLE)
    assert info["position"] is not None
    assert info["n_trades"] == 1
    assert info["refused_opens"] == 0
    assert 0.0 < obs[OBS_NAMES.index("margin_frac")] < 1.0


def _strangle_at_open(cfg: RunConfig, seed: int) -> tuple[Position, Chain, Chain]:
    """Replay the episode's first two chains and the strangle the env would build at t=0."""
    provider = SyntheticChainProvider(cfg.market)
    today = provider.reset(SeededStepRNG(seed))
    pos = build_position(SHORT_STRANGLE, today, cfg.env)
    assert pos is not None
    return pos, today, provider.step()
