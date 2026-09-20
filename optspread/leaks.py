"""Leak detection suite (project document 3.2, 10.4).

Lane: Dominick Dupuy (statistical spine).

Three layers, plus a floor for the detector itself:

1. Perturbation property: resample every RNG stream after step ``t`` and the
   chain at ``t`` must not move (``perturbation_check``). Needs the per-step
   stream contract in ``optspread.rng``.
2. Structural invariants: ordered strikes and expiries, positive vols,
   no-arbitrage bounds and put-call parity (``chain_violations``).
3. Poisoned environment: ``PoisonedChainProvider`` contaminates the spot shown
   at ``t`` with ``eps`` times the return into ``t+1``. The perturbation check
   catches it exactly (the chain at ``t`` now moves when stream ``t+1`` is
   resampled). ``detect_leak`` is the statistical detector: no observation
   feature at ``t`` may predict the path's TRUE return into ``t+1``. It reads
   the true return from the provider, so it is a simulator instrument, not a
   black-box one: at ``eps = 1`` the shown series is a clean GBM shifted by a
   day and observations alone cannot tell.

``leak_floor`` plants leaks of decreasing size and records the smallest one the
detector still catches. Passing this suite shows we catch the modes it is
built to catch; it does not prove the absence of unknown leakage.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from optspread.actions import FLAT
from optspread.config import MarketConfig
from optspread.env import OBS_NAMES, SpreadEnv
from optspread.market import SyntheticChainProvider
from optspread.pricing import bs_price
from optspread.rng import SeededStepRNG, SplicedStepRNG, StepRNG
from optspread.types import Chain, ChainProvider

# -- 1. perturbation property --------------------------------------------- #


def chains_equal(a: Chain, b: Chain) -> bool:
    return (
        a.t == b.t
        and a.spot == b.spot
        and a.r == b.r
        and np.array_equal(a.strikes, b.strikes)
        and np.array_equal(a.expiry_days, b.expiry_days)
        and np.array_equal(a.ivs, b.ivs)
    )


def run_to(provider: ChainProvider, rng: StepRNG, t: int) -> Chain:
    chain = provider.reset(rng)
    for _ in range(t):
        chain = provider.step()
    return chain


def perturbation_check(
    make_provider: Callable[[], ChainProvider], seed: int, other_seed: int, t: int
) -> tuple[bool, bool]:
    """(past unchanged at t, future actually resampled at t+1) for a split at ``t``."""
    base, spliced = (
        SeededStepRNG(seed),
        SplicedStepRNG(SeededStepRNG(seed), SeededStepRNG(other_seed), t),
    )
    past_same = chains_equal(run_to(make_provider(), base, t), run_to(make_provider(), spliced, t))
    provider = make_provider()
    if t >= provider.n_days:
        return past_same, True
    future_differs = not chains_equal(
        run_to(provider, base, t + 1), run_to(make_provider(), spliced, t + 1)
    )
    return past_same, future_differs


# -- 2. structural invariants ---------------------------------------------- #


def chain_violations(chain: Chain, tol: float = 1e-6) -> list[str]:
    out: list[str] = []
    if not chain.spot > 0:
        out.append("spot must be positive")
    if np.any(np.diff(chain.strikes) <= 0):
        out.append("strikes not strictly increasing")
    if np.any(chain.expiry_days <= chain.t) or np.any(np.diff(chain.expiry_days) <= 0):
        out.append("expiries must be ascending and after t")
    if not (np.isfinite(chain.ivs).all() and np.all(chain.ivs > 0)):
        out.append("ivs must be finite and positive")
    for row, expiry in enumerate(chain.expiry_days):
        T = chain.tau(int(expiry))
        disc = float(np.exp(-chain.r * T))
        C = bs_price("C", chain.spot, chain.strikes, chain.r, chain.ivs[row], T)
        P = bs_price("P", chain.spot, chain.strikes, chain.r, chain.ivs[row], T)
        lower_c, lower_p = (
            np.maximum(chain.spot - chain.strikes * disc, 0),
            np.maximum(chain.strikes * disc - chain.spot, 0),
        )
        if np.any(chain.spot + tol < C) or np.any(lower_c - tol > C):
            out.append(f"call bounds violated at expiry row {row}")
        if np.any(chain.strikes * disc + tol < P) or np.any(lower_p - tol > P):
            out.append(f"put bounds violated at expiry row {row}")
        if np.any(np.abs(C - P - (chain.spot - chain.strikes * disc)) > tol):
            out.append(f"put-call parity violated at expiry row {row}")
    return out


def sequence_violations(chains: Sequence[Chain]) -> list[str]:
    out = [v for c in chains for v in chain_violations(c)]
    ts = [c.t for c in chains]
    if any(b - a != 1 for a, b in pairwise(ts)):
        out.append("timestamps not consecutive")
    return out


# -- 3. poisoned environment and detector ---------------------------------- #


class PoisonedChainProvider(SyntheticChainProvider):
    """Deliberate one-step look-ahead: the spot shown at ``t`` carries ``eps`` times
    the log return into ``t+1``. ``eps = 0`` is the clean provider."""

    def __init__(self, config: MarketConfig, eps: float) -> None:
        super().__init__(config)
        self.eps = eps

    def _chain(self, spot: float) -> Chain:
        if self.eps != 0.0 and self.day < self.n_days:
            spot *= float(np.exp(self.eps * self.log_return(self.day + 1)))
        return super()._chain(spot)


@dataclass(frozen=True, slots=True)
class LeakReport:
    n_pairs: int
    n_features_tested: int
    worst_feature: str
    worst_t: float
    p_value: float
    threshold: float  # Bonferroni-adjusted alpha
    flagged: bool


def detect_leak(
    make_env: Callable[[], SpreadEnv], seeds: Sequence[int], alpha: float = 0.01
) -> LeakReport:
    """Does any observation feature at ``t`` predict the path's true return into ``t+1``?

    Runs a flat book so the observation is a pure function of the market. Pooled pairs
    are independent under GBM, so a per-feature Pearson t-test with a Bonferroni
    correction across features is valid at level ``alpha``.
    """
    xs: list[NDArray[np.float32]] = []
    ys: list[float] = []
    for seed in seeds:
        env = make_env()
        provider = env.provider
        if not isinstance(provider, SyntheticChainProvider):
            raise TypeError("detect_leak needs a provider that exposes the true log_return")
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            xs.append(obs)
            ys.append(provider.log_return(int(info["day"]) + 1))
            obs, _, done, _, info = env.step(FLAT)
    X, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    n = len(y)
    worst_t, worst_p, worst_name, tested = 0.0, 1.0, "", 0
    for j, name in enumerate(OBS_NAMES):
        if X[:, j].std() == 0.0:
            continue
        tested += 1
        r = float(np.corrcoef(X[:, j], y)[0, 1])
        t = r * np.sqrt((n - 2) / max(1.0 - r * r, 1e-12))
        p = float(2.0 * stats.t.sf(abs(t), df=n - 2))
        if p < worst_p:
            worst_t, worst_p, worst_name = float(t), p, name
    threshold = alpha / max(tested, 1)
    return LeakReport(n, tested, worst_name, worst_t, worst_p, threshold, worst_p < threshold)


@dataclass(frozen=True, slots=True)
class FloorReport:
    eps_grid: tuple[float, ...]
    flagged: tuple[bool, ...]
    floor: float | None  # smallest planted eps still detected; None if none were


def leak_floor(
    make_env: Callable[[float], SpreadEnv],
    eps_grid: Sequence[float],
    seeds: Sequence[int],
    alpha: float = 0.01,
) -> FloorReport:
    """Plant leaks of decreasing size and record the smallest the detector still catches."""
    grid = tuple(sorted(eps_grid, reverse=True))
    flags = tuple(detect_leak(partial(make_env, eps), seeds, alpha).flagged for eps in grid)
    caught = [eps for eps, f in zip(grid, flags, strict=True) if f]
    return FloorReport(grid, flags, min(caught) if caught else None)
