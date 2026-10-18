"""Episode runner and evaluator shared by every gate and, later, by training.

Lane: Ilan Danial (machinery).

``evaluate`` is the one place that turns a ``RunConfig`` into numbers, so a run
is reproducible from its config hash and seed alone. A learned arm (decision D3,
Stable-Baselines3 PPO; ``optspread.train``) plugs in here by supplying the trained
``Policy``, whose ``act`` reads ``obs``: a ppo config names no policy that can be
built from its name, so asking for one is an error rather than a silent default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from optspread.config import RunConfig
from optspread.costs import make_cost_model
from optspread.env import SpreadEnv
from optspread.market import SyntheticChainProvider
from optspread.policies import Policy, make_policy


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    pnl: float  # net of costs, dollars
    cost: float
    n_trades: int
    steps: int


@dataclass(frozen=True, slots=True)
class EvalSummary:
    n: int
    mean_pnl: float
    se_pnl: float
    mean_cost: float
    mean_trades: float
    pnls: tuple[float, ...]

    @property
    def t_stat(self) -> float:
        return self.mean_pnl / self.se_pnl if self.se_pnl > 0 else 0.0


def build_env(config: RunConfig) -> SpreadEnv:
    return SpreadEnv(
        SyntheticChainProvider(config.market), make_cost_model(config.cost), config.env
    )


def run_episode(env: SpreadEnv, policy: Policy, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    equity0, total_cost, steps, done = info["equity"], 0.0, 0, False
    while not done:
        obs, _, done, _, info = env.step(policy.act(obs, info))
        total_cost += info["cost"]
        steps += 1
    return EpisodeResult(seed, info["equity"] - equity0, total_cost, info["n_trades"], steps)


def summarize(results: Sequence[EpisodeResult]) -> EvalSummary:
    pnls = np.array([r.pnl for r in results], dtype=np.float64)
    n = len(pnls)
    se = float(pnls.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return EvalSummary(
        n=n,
        mean_pnl=float(pnls.mean()),
        se_pnl=se,
        mean_cost=float(np.mean([r.cost for r in results])),
        mean_trades=float(np.mean([r.n_trades for r in results])),
        pnls=tuple(float(p) for p in pnls),
    )


def run_episodes(config: RunConfig, policy: Policy | None = None) -> tuple[EpisodeResult, ...]:
    """Run ``config.n_episodes`` episodes with seeds ``seed, seed+1, ...``.

    The per-episode grain, kept separate from ``summarize`` so the artifact writer
    can persist every row while the gates only need the aggregate.
    """
    env = build_env(config)
    if policy is not None:
        pol = policy
    elif config.policy == "ppo":
        raise ValueError("a ppo run is evaluated with its trained policy; pass it in")
    else:
        pol = make_policy(config.policy, roll_dte=config.env.roll_dte)
    seeds = range(config.seed, config.seed + config.n_episodes)
    return tuple(run_episode(env, pol, s) for s in seeds)


def evaluate(config: RunConfig, policy: Policy | None = None) -> EvalSummary:
    return summarize(run_episodes(config, policy))
