"""The cash and position book. Pure accounting; no market logic.

Lane: Hemdutt Rao (market layer).

Invariant: a rebalance changes equity by exactly minus the modelled cost. Cash
settlement at expiry changes equity by nothing, because a leg at ``tau == 0`` is
already marked at intrinsic.
"""

from __future__ import annotations

from optspread.costs import CostModel, trade_cost
from optspread.margin import margin_requirement
from optspread.types import MULTIPLIER, Chain, Position


class Book:
    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position: Position | None = None
        self.n_trades = 0
        self.total_cost = 0.0

    @property
    def held_action(self) -> int:
        return 0 if self.position is None else self.position.action_id

    def equity(self, chain: Chain) -> float:
        return self.cash + (0.0 if self.position is None else self.position.value(chain))

    def margin_used(self, chain: Chain) -> float:
        return 0.0 if self.position is None else margin_requirement(self.position.legs, chain)

    def rebalance(self, target: Position | None, chain: Chain, costs: CostModel) -> float:
        """Close what is held and open ``target``. Returns the dollar cost charged."""
        cost = 0.0
        if self.position is not None:
            cost += trade_cost(costs, chain, self.position.legs)
            self.cash += self.position.value(chain)
            self.position = None
            self.n_trades += 1
        if target is not None:
            cost += trade_cost(costs, chain, target.legs)
            self.cash -= target.value(chain)
            self.position = target
            self.n_trades += 1
        self.cash -= cost
        self.total_cost += cost
        return cost

    def settle_expired(self, chain: Chain) -> bool:
        """Cash-settle at intrinsic once the first leg expires. SPX: no assignment."""
        if self.position is None or chain.t < self.position.first_expiry:
            return False
        for leg in self.position.legs:
            self.cash += leg.qty * leg.intrinsic(chain.spot) * MULTIPLIER
        self.position = None
        return True
