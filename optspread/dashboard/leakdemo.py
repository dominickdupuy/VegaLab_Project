"""The poisoned-environment demo: the leak detector run live on a clean and a poisoned env.

Lane: Ilan Danial (machinery).

The one deliberate exception to the dashboard reading only what is on disk: the
demo runs the detector live, because watching it fire is the point. Moved from
``optspread.dashboard_data``, which re-exports it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial

from optspread.config import MarketConfig
from optspread.leaks import (
    PASS_CRITERION,
    LeakReport,
    PoisonedChainProvider,
    default_env,
    detect_leak,
    perturbation_check,
)
from optspread.market import SyntheticChainProvider
from optspread.types import ChainProvider

# -- the poisoned-environment demo ------------------------------------------ #


@dataclass(frozen=True, slots=True)
class LeakDemo:
    """The detector run live on the clean env and on its spot-poisoned twin.

    Two layers, because they fail differently. The perturbation property is exact:
    resample the future and the poisoned chain at ``t`` moves. The statistical
    detector is what would catch a leak in a world we did not build ourselves.
    """

    eps: float
    seeds: tuple[int, ...]
    clean: LeakReport
    poisoned: LeakReport
    clean_past_unchanged: bool
    poisoned_past_unchanged: bool
    seconds: float
    criterion: str = field(default=PASS_CRITERION, repr=False)

    @property
    def caught(self) -> bool:
        """The demo's claim: the clean world is quiet and the poisoned one is caught twice."""
        return (
            not self.clean.flagged
            and self.clean_past_unchanged
            and self.poisoned.flagged
            and not self.poisoned_past_unchanged
        )


def _clean_provider(market: MarketConfig) -> ChainProvider:
    return SyntheticChainProvider(market)


def _poisoned_provider(market: MarketConfig, eps: float) -> ChainProvider:
    return PoisonedChainProvider(market, eps)


def run_leak_demo(
    eps: float = 0.5, n_seeds: int = 10, split_t: int = 10, market: MarketConfig | None = None
) -> LeakDemo:
    """Run both detectors on the clean and the poisoned env, on the same seeds."""
    if eps <= 0.0:
        raise ValueError(f"the poisoned variant needs eps > 0, got {eps}")
    cfg = market if market is not None else MarketConfig()
    seeds = tuple(range(n_seeds))
    start = time.perf_counter()
    clean = detect_leak(partial(default_env, 0.0, cfg), seeds)
    poisoned = detect_leak(partial(default_env, eps, cfg), seeds)
    clean_past, _ = perturbation_check(partial(_clean_provider, cfg), 1, 2, split_t)
    poisoned_past, _ = perturbation_check(partial(_poisoned_provider, cfg, eps), 1, 2, split_t)
    return LeakDemo(
        eps=eps,
        seeds=seeds,
        clean=clean,
        poisoned=poisoned,
        clean_past_unchanged=clean_past,
        poisoned_past_unchanged=poisoned_past,
        seconds=time.perf_counter() - start,
    )
