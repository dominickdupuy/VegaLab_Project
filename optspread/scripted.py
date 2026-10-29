"""Scripted sellers as a check on the cost model, not as strategies.

Lane: Hemdutt Rao (market layer).

In a world priced at fair implied volatility a seller has no premium to collect,
so the expected result of any scripted seller is minus its trading costs and
nothing else. That turns a scripted policy into a test with a known answer: if
the average loss does not match the drag the cost model predicts, either the cost
model or the prediction is wrong.

Three things get checked, each falsifiable on its own:

1. **Cost identity.** For every seed, ``pnl_costed == pnl_free - cost_charged``
   exactly. The arms share seeds, so anything else means costs perturbed the
   trajectory rather than only the accounting.
2. **Cash settlement is free.** SPX options are European and cash-settled, so a
   structure held to expiry closes at the settlement value with no second trade.
   Charged crossings must equal opens plus early closes, and never count a
   settlement.
3. **No unpaid exit.** Every structure the episode opened has either settled in
   cash or been charged an exit. A position still open at the horizon, marked at
   mid with no exit charged, is an unrealisable mark: the episode reports a P&L it
   could not have taken.

Check 3 is the one that failed on the first run. ``unpaid_exit`` is the size of
the gap.

The check takes any ``CostModel`` (``costs``), so a cost model of your own clears
the same three before it is trusted (docs/EXTENDING.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from optspread.actions import ACTION_NAMES, IRON_CONDOR, PUT_CREDIT_SPREAD
from optspread.config import CostConfig, RunConfig
from optspread.costs import CostModel, make_cost_model, trade_cost
from optspread.harness import build_env
from optspread.policies import AlwaysOnPolicy, DeltaSellerPolicy, Policy
from optspread.types import Position

# A hold-to-expiry seller of each defined-risk structure, and the same structures
# rolled on schedule. None means hold; an int is the roll trigger in days to expiry.
SELLERS: tuple[tuple[str, int, int | None], ...] = (
    ("hold_condor", IRON_CONDOR, None),
    ("hold_put_spread", PUT_CREDIT_SPREAD, None),
    ("roll_condor", IRON_CONDOR, 5),
    ("roll_put_spread", PUT_CREDIT_SPREAD, 5),
)

TOL = 1e-6


def make_seller(action_id: int, roll_dte: int | None) -> Policy:
    """A hold-to-expiry seller (``roll_dte is None``) or one rolled on schedule."""
    if roll_dte is None:
        return AlwaysOnPolicy(action_id)
    return DeltaSellerPolicy(action_id, roll_dte)


@dataclass(frozen=True, slots=True)
class EpisodeAudit:
    """One episode trade lifecycle, derived from the env public info stream."""

    seed: int
    pnl: float
    cost_charged: float
    n_crossings: int  # the book own trade count
    n_opens: int
    n_early_closes: int
    n_settlements: int
    unpaid_exit: float  # cost of closing whatever is still open at the horizon

    @property
    def open_at_horizon(self) -> bool:
        return self.unpaid_exit > 0.0

    @property
    def settlement_crossings(self) -> int:
        """Charged crossings not explained by an open or an early close."""
        return max(self.n_crossings - (self.n_opens + self.n_early_closes), 0)


def audit_episode(
    config: RunConfig, policy: Policy, seed: int, costs: CostModel | None = None
) -> EpisodeAudit:
    """Run one episode and classify every position change as open, close or settlement.

    ``costs`` defaults to the model ``config.cost`` builds."""
    if costs is None:
        costs = make_cost_model(config.cost)
    env = build_env(config, costs=costs)
    obs, info = env.reset(seed=seed)
    policy.reset(seed)

    equity0 = info["equity"]
    charged = 0.0
    opens = early = settled = 0
    prev: Position | None = info["position"]
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, info))
        charged += info["cost"]
        held: Position | None = info["position"]
        if prev is not None and held is not prev:
            # A position that reached its expiry settled in cash; anything else
            # that left the book was traded out of.
            if prev.first_expiry <= int(info["day"]):
                settled += 1
            else:
                early += 1
        if held is not None and held is not prev:
            opens += 1
        prev = held

    unpaid = 0.0 if prev is None else trade_cost(costs, info["chain"], prev.legs)
    return EpisodeAudit(
        seed=seed,
        pnl=info["equity"] - equity0,
        cost_charged=charged,
        n_crossings=int(info["n_trades"]),
        n_opens=opens,
        n_early_closes=early,
        n_settlements=settled,
        unpaid_exit=unpaid,
    )


@dataclass(frozen=True, slots=True)
class DragRow:
    seller: str
    structure: str
    n: int
    mean_pnl_free: float
    mean_pnl_costed: float
    measured_drag: float  # mean(pnl_free) - mean(pnl_costed): what the episodes lost
    predicted_drag: float  # cost to open and fully unwind: charged plus unpaid
    mean_settlements: float
    mean_unpaid_exit: float
    identity_max_dev: float  # max |pnl_costed - (pnl_free - cost)| over seeds
    settlement_crossings: int  # charged crossings attributable to a cash settlement

    @property
    def identity_holds(self) -> bool:
        return self.identity_max_dev <= TOL

    @property
    def settlement_is_free(self) -> bool:
        return self.settlement_crossings == 0

    @property
    def no_unpaid_exit(self) -> bool:
        return self.mean_unpaid_exit <= TOL

    @property
    def drag_gap(self) -> float:
        """Predicted minus measured: what the episode never paid for."""
        return self.predicted_drag - self.measured_drag

    @property
    def passed(self) -> bool:
        return self.identity_holds and self.settlement_is_free and self.no_unpaid_exit


def drag_row(
    base: RunConfig,
    seller: str,
    action_id: int,
    roll_dte: int | None,
    n: int,
    costs: CostModel | None = None,
) -> DragRow:
    """Run one seller with costs off and on over shared seeds; on is ``costs`` when given."""
    free = base.model_copy(update={"cost": CostConfig(enabled=False)})
    costed = base.model_copy(update={"cost": base.cost.model_copy(update={"enabled": True})})

    pnl_free: list[float] = []
    pnl_costed: list[float] = []
    devs: list[float] = []
    unpaid: list[float] = []
    settles: list[int] = []
    crossings = 0
    for seed in range(base.seed, base.seed + n):
        a = audit_episode(free, make_seller(action_id, roll_dte), seed)
        b = audit_episode(costed, make_seller(action_id, roll_dte), seed, costs)
        pnl_free.append(a.pnl)
        pnl_costed.append(b.pnl)
        devs.append(abs(b.pnl - (a.pnl - b.cost_charged)))
        unpaid.append(b.unpaid_exit)
        settles.append(b.n_settlements)
        crossings += b.settlement_crossings

    mean_free, mean_costed = float(np.mean(pnl_free)), float(np.mean(pnl_costed))
    measured = mean_free - mean_costed
    return DragRow(
        seller=seller,
        structure=ACTION_NAMES[action_id],
        n=n,
        mean_pnl_free=mean_free,
        mean_pnl_costed=mean_costed,
        measured_drag=measured,
        predicted_drag=measured + float(np.mean(unpaid)),
        mean_settlements=float(np.mean(settles)),
        mean_unpaid_exit=float(np.mean(unpaid)),
        identity_max_dev=float(max(devs)),
        settlement_crossings=crossings,
    )


@dataclass(frozen=True, slots=True)
class DragReport:
    rows: tuple[DragRow, ...]

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(r.passed for r in self.rows)

    def render(self) -> str:
        lines = [
            f"{'seller':<17}{'structure':<19}{'free':>9}{'costed':>9}"
            f"{'drag':>9}{'predicted':>11}{'gap':>9}{'settles':>9}"
        ]
        for r in self.rows:
            lines.append(
                f"{r.seller:<17}{r.structure:<19}{r.mean_pnl_free:>9,.0f}"
                f"{r.mean_pnl_costed:>9,.0f}{r.measured_drag:>9,.0f}"
                f"{r.predicted_drag:>11,.0f}{r.drag_gap:>9,.0f}{r.mean_settlements:>9.1f}"
            )
        checks = (
            ("cost identity: costed == free - charged", all(r.identity_holds for r in self.rows)),
            ("cash settlement charges nothing", all(r.settlement_is_free for r in self.rows)),
            ("no unpaid exit at the horizon", all(r.no_unpaid_exit for r in self.rows)),
        )
        lines += [f"  [{'PASS' if ok else 'FAIL'}] {label}" for label, ok in checks]
        lines.append(f"Cost drag check {'HOLDS' if self.passed else 'FAILS'}.")
        return "\n".join(lines)


def cost_drag_check(
    n_episodes: int = 40,
    seed: int = 2000,
    sellers: Sequence[str] | None = None,
    costs: CostModel | None = None,
) -> DragReport:
    """Run the scripted sellers against the drag the cost model predicts.

    ``costs`` is the model under test; the default is the one ``CostConfig`` builds."""
    wanted = set(sellers) if sellers is not None else {s[0] for s in SELLERS}
    base = RunConfig(seed=seed, n_episodes=n_episodes)
    return DragReport(
        tuple(
            drag_row(base, name, action_id, roll, n_episodes, costs)
            for name, action_id, roll in SELLERS
            if name in wanted
        )
    )
