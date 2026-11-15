"""Replay a frozen synthetic dataset through the ordinary environment.

``DatasetChainProvider`` is a ``ChainProvider`` like the generator and the historical
adapter. An episode is one stored path: ``reset`` reads the path index off the seed
(``seed mod n_paths``), so evaluation walks seeds ``0..n-1`` once each, and training
(``optspread.train.TrainingPaths``, consecutive seeds) cycles the frozen train paths in
order. The agent sees those paths and no others.

Each chain is laid on the same strike and tenor grid ``SyntheticChainProvider`` uses, with
the stored spot and a flat surface at the stored ATM vol, so a ``noise`` dataset trades
exactly like the live null world on the same seeds.
"""

from __future__ import annotations

import numpy as np

from optspread.alpha.dataset import Split, SynthDataset, World
from optspread.config import CostConfig, EnvConfig, MarketConfig
from optspread.costs import make_cost_model
from optspread.env import SpreadEnv
from optspread.rng import SeededStepRNG, StepRNG
from optspread.types import Chain


class DatasetChainProvider:
    def __init__(self, ds: SynthDataset, world: World, split: Split) -> None:
        self.market: MarketConfig = ds.spec.market()
        self._spot = ds.spot(world, split)
        self._iv = ds.iv(world, split)
        self.n_paths = int(self._spot.shape[0])
        self.path_index = 0
        self._day = 0
        self._ready = False

    @property
    def n_days(self) -> int:
        return self.market.n_days

    @property
    def done(self) -> bool:
        return self._day >= self.market.n_days

    @staticmethod
    def path_for(rng: StepRNG, n_paths: int) -> int:
        if not isinstance(rng, SeededStepRNG):
            raise TypeError(
                f"a dataset path is chosen by seed; got {type(rng).__name__}, not SeededStepRNG"
            )
        return rng.seed % n_paths

    def reset(self, rng: StepRNG) -> Chain:
        self.path_index = self.path_for(rng, self.n_paths)
        self._day = 0
        self._ready = True
        return self._chain()

    def step(self) -> Chain:
        if not self._ready:
            raise RuntimeError("step() before reset()")
        if self.done:
            raise RuntimeError("step() past the horizon")
        self._day += 1
        return self._chain()

    def _chain(self) -> Chain:
        cfg = self.market
        spot = float(self._spot[self.path_index, self._day])
        centre = round(spot / cfg.strike_step) * cfg.strike_step
        offsets = np.arange(-cfg.n_strikes_each_side, cfg.n_strikes_each_side + 1)
        strikes = centre + cfg.strike_step * offsets.astype(np.float64)
        iv = float(self._iv[self.path_index, self._day])
        return Chain(
            t=self._day,
            spot=spot,
            r=cfg.r,
            strikes=strikes,
            expiry_days=self._day + np.asarray(cfg.tenor_days, dtype=np.int64),
            ivs=np.full((len(cfg.tenor_days), len(strikes)), iv, dtype=np.float64),
            trading_days_per_year=cfg.trading_days_per_year,
        )


def dataset_env(
    ds: SynthDataset,
    world: World,
    split: Split,
    cost: CostConfig | None = None,
    env: EnvConfig | None = None,
) -> SpreadEnv:
    """The ordinary environment over one world and split of a frozen dataset."""
    return SpreadEnv(
        DatasetChainProvider(ds, world, split),
        make_cost_model(cost if cost is not None else CostConfig()),
        env if env is not None else EnvConfig(episode_days=ds.spec.n_days),
    )
