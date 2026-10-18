"""The cash and position book. Pure accounting; no market logic.

Lane: Hemdutt Rao (market layer).

Two ways a structure leaves the book, and they cost different amounts:

``settle_expired``  the legs reached expiry. SPX options are European and cash
                    settled, so this is the settlement value and no second trade.
                    Equity does not move, because a leg at ``tau == 0`` is already
                    marked at intrinsic.
``rebalance``       the structure was traded out of before expiry, which crosses
                    the spread on every leg. Equity moves by exactly minus the
                    modelled cost.

``liquidate`` is the second of those, applied at the episode horizon. A position
still open when the episode ends has to be closed to be worth its mark: leaving
it marked at mid reports a P&L the episode could not have realised.

Margin is checked before a structure is opened, not only after. ``rebalance``
opens ``target`` only if the equity left once what is held has been closed and
both crossings paid covers the target's requirement at the same chain; otherwise
the book stays flat and ``refused_opens`` counts it. That is the breach rule in
``optspread.env`` (``equity < margin_used`` forces a flat) applied at the moment of
the trade, so a structure the account could never carry is not opened for a day
and liquidated the next. A position that clears the check can still breach later
as the market moves; the env's per-step check is still what catches that.
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
        self.refused_opens = 0

    @property
    def held_action(self) -> int:
        return 0 if self.position is None else self.position.action_id

    def equity(self, chain: Chain) -> float:
        return self.cash + (0.0 if self.position is None else self.position.value(chain))

    def margin_used(self, chain: Chain) -> float:
        return 0.0 if self.position is None else margin_requirement(self.position.legs, chain)

    def covers(self, target: Position, chain: Chain, costs: CostModel, close_cost: float) -> bool:
        """True when a flat book that has paid ``close_cost`` can carry ``target`` here.

        Equity after the open is the cash less both crossings (the target enters at its
        mark, so its value adds back what its premium took out), and it must be at
        least the requirement: the same inequality the breach check applies daily.
        """
        open_cost = trade_cost(costs, chain, target.legs)
        return self.cash - close_cost - open_cost >= margin_requirement(target.legs, chain)

    def rebalance(self, target: Position | None, chain: Chain, costs: CostModel) -> float:
        """Close what is held and open ``target`` if it is covered. Returns the cost charged.

        A refused open charges nothing and leaves the book flat: what was held is
        still closed, because the policy asked to leave it.
        """
        cost = 0.0
        if self.position is not None:
            cost += trade_cost(costs, chain, self.position.legs)
            self.cash += self.position.value(chain)
            self.position = None
            self.n_trades += 1
        if target is not None and not self.covers(target, chain, costs, cost):
            self.refused_opens += 1
        elif target is not None:
            cost += trade_cost(costs, chain, target.legs)
            self.cash -= target.value(chain)
            self.position = target
            self.n_trades += 1
        self.cash -= cost
        self.total_cost += cost
        return cost

    def liquidate(self, chain: Chain, costs: CostModel) -> float:
        """Close whatever is still held, paying the exit. Returns the dollar cost.

        Called at the episode horizon. An expired structure has already left via
        ``settle_expired``, so this only ever charges for a live one -- which is the
        distinction the cost-drag check was short by.
        """
        if self.position is None:
            return 0.0
        return self.rebalance(None, chain, costs)

    def settle_expired(self, chain: Chain) -> bool:
        """Cash-settle at intrinsic once the first leg expires. SPX: no assignment."""
        if self.position is None or chain.t < self.position.first_expiry:
            return False
        for leg in self.position.legs:
            self.cash += leg.qty * leg.intrinsic(chain.spot) * MULTIPLIER
        self.position = None
        return True
