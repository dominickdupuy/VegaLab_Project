"""Bring your own: the four layers a researcher can replace, and the runner that carries them.

Lane: Hemdutt Rao (market layer and data adapter).

    layer            interface                                default
    data             optspread.data.ChainSource               SyntheticSource
    agent            optspread.policies.Policy, or            the scripted arms,
                     optspread.train.Trainer                  train_ppo
    planted effect   optspread.effects.PlantedEffect          ConstantPremium
    cost model       optspread.costs.CostModel                QuotedSpreadCost

``Components`` is a ``sweep.Runner``: config and policy in, episodes out, with the
market and the costs built from your factories rather than from the config. Every
orchestrator takes a runner (``sweep.run_sweep``, ``sweep.pilot_matrix``,
``orchestrate.run_grid``, ``replay.replay_run``), so your pieces run through the same
paired, resumable grid and the same paired test as the confirmatory design. A policy
enters as the policy a runner is handed (``pilot_matrix``) or as the trainer of the
learned arm (``run_grid``). docs/EXTENDING.md has a worked example of each.

None of this touches a config. The config hash of a run made with components of
your own is the hash of the default world at the same settings, so such runs go
under a root of their own, and replay through the same components. The manifest
records which (``Components.identity``, in ``DataRecord.components``), and every reuse
check compares it: a run made with your components is never reused as a default run,
nor a default run as yours, and one is not written over the other.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from optspread.config import CostConfig, MarketConfig, RunConfig
from optspread.costs import CostModel, make_cost_model
from optspread.data import ChainSource, SyntheticSource
from optspread.effects import PlantedEffect
from optspread.env import SpreadEnv
from optspread.harness import EpisodeResult, build_env, run_episodes
from optspread.manifest import callable_name
from optspread.policies import Policy

SourceFactory = Callable[[MarketConfig], ChainSource]
CostFactory = Callable[[CostConfig], CostModel]


def describe(factory: Any) -> str:
    """A factory as a manifest names it: a class or function by ``callable_name``, a
    ``functools.partial`` with its arguments (an effect by its fields), anything else by
    its type and ``repr``."""
    if isinstance(factory, partial):
        args = [repr(a) for a in factory.args]
        args += [f"{k}={v!r}" for k, v in sorted(factory.keywords.items())]
        return f"{describe(factory.func)}({', '.join(args)})"
    if isinstance(factory, type) or hasattr(factory, "__code__"):
        return callable_name(factory)
    return f"{callable_name(type(factory))}:{factory!r}"


def planted(effect: PlantedEffect) -> SourceFactory:
    """The synthetic world with ``effect`` planted at each cell's premium."""
    return partial(SyntheticSource, effect=effect)


@dataclass(frozen=True, slots=True)
class Components:
    """Your source and cost model, built per cell from the cell's own config.

    The factories see the cell's ``MarketConfig`` and ``CostConfig``, so the grid's
    premium axis and the cost switch still reach them. To run in parallel
    (``workers > 1``) both must pickle: a class, a module-level function, or a
    ``functools.partial`` of one, not a lambda.
    """

    source: SourceFactory = SyntheticSource
    costs: CostFactory = make_cost_model

    @property
    def identity(self) -> str | None:
        """What a run made with these components records (``DataRecord.components``): the
        factories by name and arguments, or None for the defaults, which run exactly
        what the harness runs and so are the default components."""
        if self.source is SyntheticSource and self.costs is make_cost_model:
            return None
        return f"Components(source={describe(self.source)}, costs={describe(self.costs)})"

    def env(self, config: RunConfig) -> SpreadEnv:
        """The environment ``config`` runs in with these components (for the leak suite)."""
        provider = self.source(config.market).provider()
        return build_env(config, provider=provider, costs=self.costs(config.cost))

    def __call__(
        self, config: RunConfig, policy: Policy | None = None
    ) -> tuple[EpisodeResult, ...]:
        provider = self.source(config.market).provider()
        return run_episodes(config, policy, provider=provider, costs=self.costs(config.cost))
