"""Hemdutt Rao: the scripted sellers check the cost model against arithmetic.

In a fair-priced world a scripted seller expects to lose its costs and nothing
else, so these are tests with a known answer rather than performance checks.
"""

from __future__ import annotations

import pytest

from optspread.actions import IRON_CONDOR, PUT_CREDIT_SPREAD, build_position
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.margin import is_defined_risk, max_loss
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG
from optspread.scripted import EpisodeAudit, audit_episode, cost_drag_check, make_seller

SHORT = RunConfig(
    seed=2000,
    n_episodes=6,
    market=MarketConfig(n_days=42, tenor_days=(14, 28)),
    env=EnvConfig(episode_days=42),
    cost=CostConfig(enabled=True),
)


# -- the put credit spread joins the library -------------------------------- #


def test_put_credit_spread_is_two_puts_and_defined_risk() -> None:
    chain = SyntheticChainProvider(SHORT.market).reset(SeededStepRNG(0))
    pos = build_position(PUT_CREDIT_SPREAD, chain, SHORT.env)
    assert pos is not None
    assert [leg.right for leg in pos.legs] == ["P", "P"]
    assert sum(leg.qty for leg in pos.legs) == 0, "one short, one long"
    short = next(leg for leg in pos.legs if leg.qty < 0)
    long = next(leg for leg in pos.legs if leg.qty > 0)
    assert long.strike < short.strike, "the long wing sits below the short strike"
    assert is_defined_risk(pos.legs)
    assert max_loss(pos.legs) > 0.0


# -- check 1: costs change the accounting and nothing else ------------------ #


def test_cost_identity_holds_for_every_seller() -> None:
    report = cost_drag_check(n_episodes=6, seed=2000)
    for row in report.rows:
        assert row.identity_holds, (row.seller, row.identity_max_dev)


# -- check 2: cash settlement carries no exit cost -------------------------- #


def test_a_seller_held_to_expiry_pays_only_its_opens() -> None:
    audit = audit_episode(SHORT, make_seller(IRON_CONDOR, None), seed=2000)
    assert audit.n_settlements >= 1, "this config must reach at least one expiry"
    assert audit.n_early_closes == 0
    assert audit.n_crossings == audit.n_opens
    assert audit.settlement_crossings == 0


def test_settlement_crossings_counts_an_overcharge() -> None:
    # Positive control for check 2: the arithmetic must notice a crossing that no
    # open or early close accounts for, which is what a charge at settlement is.
    overcharged = EpisodeAudit(
        seed=0,
        pnl=0.0,
        cost_charged=0.0,
        n_crossings=3,
        n_opens=2,
        n_early_closes=0,
        n_settlements=1,
        unpaid_exit=0.0,
    )
    assert overcharged.settlement_crossings == 1
    clean = EpisodeAudit(0, 0.0, 0.0, 2, 2, 0, 1, 0.0)
    assert clean.settlement_crossings == 0


def test_a_rolled_seller_pays_for_the_close_it_actually_makes() -> None:
    audit = audit_episode(SHORT, make_seller(IRON_CONDOR, 5), seed=2000)
    assert audit.n_early_closes >= 1
    assert audit.n_crossings == audit.n_opens + audit.n_early_closes
    assert audit.cost_charged > 0.0


# -- check 3: the defect found on the first run ----------------------------- #


@pytest.mark.parametrize("action_id", [IRON_CONDOR, PUT_CREDIT_SPREAD])
def test_a_position_open_at_the_horizon_has_an_unpaid_exit(action_id: int) -> None:
    """Records the defect the drag check found, so the fix has a test to flip.

    A hold-to-expiry seller ends the episode holding an unexpired structure. It is
    marked at mid and no exit is ever charged, so the reported P&L is one crossing
    better than anything the episode could have realised.
    """
    audit = audit_episode(SHORT, make_seller(action_id, None), seed=2000)
    assert audit.open_at_horizon, "this config must end holding an open structure"
    assert audit.unpaid_exit > 0.0


def test_the_drag_check_reports_the_gap_rather_than_passing() -> None:
    report = cost_drag_check(n_episodes=6, seed=2000)
    assert not report.passed, "the unpaid exit at the horizon is still open"
    assert all(row.identity_holds and row.settlement_is_free for row in report.rows)
    assert any(row.drag_gap > 0.0 for row in report.rows)
    rendered = report.render()
    assert "[FAIL] no unpaid exit at the horizon" in rendered
