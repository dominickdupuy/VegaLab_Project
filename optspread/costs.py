"""Transaction costs: a quoted half-spread crossed on every leg, plus commission.

Lane: Hemdutt Rao (market layer).

The half-spread is a fraction of mid with a floor in index points, so far-OTM
wings pay proportionally more (the floor binds): a first-order version of the
moneyness widening in the project document. Slippage as a function of size is
a planned extension; the ``CostModel`` interface does not change for it.

Gate: a round trip in the null world loses exactly the modelled cost and
nothing else (``tests/core/test_market_layer.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from optspread.config import CostConfig
from optspread.types import MULTIPLIER, Chain, OptionLeg


class CostModel(Protocol):
    def leg_cost(self, chain: Chain, leg: OptionLeg) -> float:
        """Dollar cost of executing ``leg`` (all its contracts) at this chain."""
        ...


@dataclass(frozen=True, slots=True)
class ZeroCost:
    def leg_cost(self, chain: Chain, leg: OptionLeg) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class QuotedSpreadCost:
    config: CostConfig

    def leg_cost(self, chain: Chain, leg: OptionLeg) -> float:
        cfg = self.config
        half = max(cfg.min_half_spread, cfg.half_spread_frac * chain.mark(leg))
        return abs(leg.qty) * (half * MULTIPLIER + cfg.commission_per_contract)


def make_cost_model(config: CostConfig) -> CostModel:
    return QuotedSpreadCost(config) if config.enabled else ZeroCost()


def trade_cost(model: CostModel, chain: Chain, legs: tuple[OptionLeg, ...]) -> float:
    return sum(model.leg_cost(chain, leg) for leg in legs)
