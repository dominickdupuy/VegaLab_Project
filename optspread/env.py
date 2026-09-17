"""Gymnasium environment and the fixed observation schema.

Lane: Ilan Danial (machinery).

Step order (all causal):
    1. the policy's action is executed at today's chain (close prices);
    2. the path advances one day;
    3. an expired structure is cash-settled; a margin breach forces liquidation;
    4. reward = today's equity change / initial cash.

OBS_NAMES is a specification decision, not an implementation detail (project
document 10.3). It is APPEND-ONLY: a saved policy's input layout is fixed.
Changing it needs review from two members and a gate rerun.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from optspread.actions import N_ACTIONS, build_position
from optspread.config import EnvConfig
from optspread.costs import CostModel
from optspread.portfolio import Book
from optspread.rng import SeededStepRNG
from optspread.types import MULTIPLIER, Chain, ChainProvider

OBS_NAMES: tuple[str, ...] = (
    "time_frac",  # day / episode_days
    "has_position",  # 0/1
    "action_frac",  # held action id / N_ACTIONS
    "days_held_frac",  # days since open / episode_days
    "dte_frac",  # days to first expiry of held legs / episode_days
    "equity_frac",  # equity / initial cash
    "unrealized_frac",  # (position value - entry value) / initial cash
    "margin_frac",  # margin used / initial cash
    "atm_iv",  # near-tenor ATM implied vol on today's chain
    "realized_vol_20d",  # trailing 20-day annualised realised vol (causal)
    "ret_1d",  # today's log return of the underlying
    "delta_frac",  # $ delta of held legs per 100% move / initial cash
    "vega_frac",  # $ vega per vol point / initial cash
    "theta_frac",  # $ theta per trading day / initial cash
)
OBS_DIM = len(OBS_NAMES)
REALIZED_VOL_WINDOW = 20


class SpreadEnv(gym.Env[NDArray[np.float32], int]):
    """Every collaborator is injected; the env constructs none of them."""

    def __init__(self, provider: ChainProvider, costs: CostModel, config: EnvConfig) -> None:
        self.provider = provider
        self.costs = costs
        self.config = config
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self._chain: Chain | None = None
        self._book = Book(config.initial_cash)
        self._day = 0
        self._log_returns: list[float] = []

    # -- gymnasium API ------------------------------------------------------ #

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self._chain = self.provider.reset(SeededStepRNG(0 if seed is None else seed))
        self._book = Book(self.config.initial_cash)
        self._day = 0
        self._log_returns = []
        return self._obs(), self._info(pnl=0.0, cost=0.0)

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        chain = self._require_chain()
        book = self._book
        equity_before = book.equity(chain)

        cost = 0.0
        if int(action) != book.held_action:
            cost = book.rebalance(
                build_position(int(action), chain, self.config), chain, self.costs
            )

        prev_spot = chain.spot
        chain = self.provider.step()
        self._chain = chain
        self._day += 1
        self._log_returns.append(float(np.log(chain.spot / prev_spot)))

        book.settle_expired(chain)
        if book.position is not None and book.equity(chain) < book.margin_used(chain):
            cost += book.rebalance(None, chain, self.costs)  # margin breach: forced flat

        pnl = book.equity(chain) - equity_before
        done = self.provider.done or self._day >= self.config.episode_days
        return self._obs(), pnl / self.config.initial_cash, done, False, self._info(pnl, cost)

    # -- internals ----------------------------------------------------------- #

    def _require_chain(self) -> Chain:
        if self._chain is None:
            raise RuntimeError("step() before reset()")
        return self._chain

    def _info(self, pnl: float, cost: float) -> dict[str, Any]:
        chain = self._require_chain()
        return {
            "day": self._day,
            "chain": chain,
            "position": self._book.position,
            "spot": chain.spot,
            "equity": self._book.equity(chain),
            "pnl": pnl,
            "cost": cost,
            "n_trades": self._book.n_trades,
        }

    def _obs(self) -> NDArray[np.float32]:
        cfg, chain, book = self.config, self._require_chain(), self._book
        cash = cfg.initial_cash
        pos = book.position
        window = self._log_returns[-REALIZED_VOL_WINDOW:]
        rv = (
            float(np.std(window, ddof=1) * np.sqrt(chain.trading_days_per_year))
            if len(window) > 1
            else 0.0
        )

        delta = vega = theta = 0.0
        if pos is not None:
            for leg in pos.legs:
                g = chain.greeks(leg.right, leg.strike, leg.expiry_day)
                delta += leg.qty * g.delta * chain.spot * MULTIPLIER
                vega += leg.qty * g.vega * 0.01 * MULTIPLIER
                theta += leg.qty * g.theta / chain.trading_days_per_year * MULTIPLIER

        values = (
            self._day / cfg.episode_days,
            0.0 if pos is None else 1.0,
            book.held_action / N_ACTIONS,
            0.0 if pos is None else (self._day - pos.open_day) / cfg.episode_days,
            0.0 if pos is None else max(pos.first_expiry - self._day, 0) / cfg.episode_days,
            book.equity(chain) / cash,
            0.0 if pos is None else (pos.value(chain) - pos.entry_value()) / cash,
            book.margin_used(chain) / cash,
            chain.iv_at(int(chain.expiry_days[0]), chain.spot),
            rv,
            self._log_returns[-1] if self._log_returns else 0.0,
            delta / cash,
            vega / cash,
            theta / cash,
        )
        return np.asarray(values, dtype=np.float32)
