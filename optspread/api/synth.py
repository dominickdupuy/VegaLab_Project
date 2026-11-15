"""API payloads for the Synth data page: sample paths of the simulated market.

Lane: Hemdutt Rao (market layer).

The page sets the market's knobs (the path vol and the research knobs of
``MarketConfig``: drift, vol of vol and skew) and sees what ``SyntheticChainProvider``
makes of them before a run is launched on that market: the index paths, the near-tenor
ATM implied vol along each, and the near-tenor smile half way through. The paths are
the ones a run's episodes see: path ``i`` is seed ``seed + i``, so a run whose first seed
is ``seed`` plays the first path in its first episode. Computed, not read; nothing is
written.
"""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict, Field

from optspread.api._base import Out
from optspread.config import MarketConfig
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG
from optspread.types import Chain

_MARKET = MarketConfig()


class SynthParams(Out):
    """The knobs the page sets, bounded as the launch form bounds them."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    sigma: float = Field(default=_MARKET.sigma, gt=0.0, le=1.5)
    alpha_bps: float = Field(default=0.0, ge=0.0, le=200.0)
    vol_of_vol: float = Field(default=0.0, ge=0.0, le=2.0)
    skew_slope: float = Field(default=0.0, ge=-1.0, le=1.0)
    n_days: int = Field(default=_MARKET.n_days, ge=5, le=504)
    n_paths: int = Field(default=5, ge=1, le=20)
    seed: int = Field(default=0, ge=0, lt=2**31)

    def market(self) -> MarketConfig:
        return MarketConfig(
            sigma=self.sigma,
            alpha_bps=self.alpha_bps,
            vol_of_vol=self.vol_of_vol,
            skew_slope=self.skew_slope,
            n_days=self.n_days,
        )


class SynthPathsOut(Out):
    params: SynthParams
    days: list[int]  # 0..n_days
    spots: list[list[float]]  # [path][day]
    atm_ivs: list[list[float]]  # [path][day], near tenor, decimal
    smile_day: int  # the day the smile is taken on: half way
    # [path][strike]: ln(K / spot) of each listed strike, and its near-tenor vol (decimal);
    # the strikes are centred on each path's own spot, so each path has its own
    smile_log_moneyness: list[list[float]]
    smile_ivs: list[list[float]]
    source: str


def _atm_iv(chain: Chain) -> float:
    """What the policy's ``atm_iv`` reads (``env.SpreadEnv._obs``): the near tenor at spot."""
    return chain.iv_at(int(chain.expiry_days[0]), chain.spot)


def synth_paths(params: SynthParams) -> SynthPathsOut:
    """``n_paths`` paths of the market ``params`` sets, on seeds ``seed, seed + 1, ...``."""
    config = params.market()
    mid = config.n_days // 2
    spots: list[list[float]] = []
    ivs: list[list[float]] = []
    smile_x: list[list[float]] = []
    smile_ivs: list[list[float]] = []
    for i in range(params.n_paths):
        provider = SyntheticChainProvider(config)
        chain = provider.reset(SeededStepRNG(params.seed + i))
        path, vol = [chain.spot], [_atm_iv(chain)]
        for _ in range(config.n_days):
            chain = provider.step()
            path.append(chain.spot)
            vol.append(_atm_iv(chain))
            if chain.t == mid:  # n_days >= 5, so every path passes it
                smile_x.append([float(x) for x in np.log(chain.strikes / chain.spot)])
                smile_ivs.append([float(v) for v in chain.ivs[0]])
        spots.append([float(s) for s in path])
        ivs.append([float(v) for v in vol])
    return SynthPathsOut(
        params=params,
        days=list(range(config.n_days + 1)),
        spots=spots,
        atm_ivs=ivs,
        smile_day=mid,
        smile_log_moneyness=smile_x,
        smile_ivs=smile_ivs,
        source="optspread.market.SyntheticChainProvider, the generator every synthetic run reads",
    )
