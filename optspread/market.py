"""Synthetic chain provider: GBM with the chain priced off the path vol.

Lane: Hemdutt Rao (market layer).

The path is a zero-drift GBM (spot is a martingale) and the book earns nothing on
cash. With ``r = 0`` (``MarketConfig``'s default, the fair world) and
``iv_premium_vol_pts == 0`` every contract is priced by Black-Scholes at the exact vol
that drives the underlying, on a forward equal to the spot, so at its expected payoff
under the path, and every strategy has zero expectancy before costs. At any other
``r`` the chain is priced on a forward the path does not follow: at the study's
``r = 0.04`` (``config.STUDY_R``) a put is priced below its expected payoff and a call
above it, so a seller of puts loses the carry before costs (pre-registration A11,
docs/DRIFT_SENSITIVITY.md). A positive premium prices the chain richer than the path
realises: the planted variance risk premium the power sweep calibrates against.

What is planted is a ``PlantedEffect`` (``optspread.effects``), applied to the fair
surface each time a chain is built, at the size the config gives. The default is
``ConstantPremium``, the whole surface shifted by the premium; another effect is
passed in Python (``SyntheticChainProvider(config, effect=...)``), never through
the config.

RNG: the return into day ``t`` is drawn from ``rng.stream(t)`` alone (the
per-step contract in ``optspread.rng``). The chain at ``t`` therefore depends on
no stream later than ``t``, which ``optspread.leaks`` tests directly.

**Research knobs** (``MarketConfig``; each 0.0 by default, which is the world above, the
same floats as before the knobs existed). None is part of the confirmatory design:

- ``alpha_bps``: a drift of that many basis points per trading day (not annualised), added
  to every log return, so a long position has an edge a policy can be shown to find.
- ``vol_of_vol``: the path vol follows a driftless log-normal walk from ``sigma``, its
  shock into day ``d`` drawn from ``stream(d, VOL_CHANNEL)``. The whole walk is computed
  at ``reset``; the vol of day ``d`` depends on the streams of days ``<= d`` only, so the
  chain at ``t`` still reads no later stream, and ``log_return`` stays a pure read (the
  leak detector calls it to look ahead without moving the path). The chain on day ``d``
  is priced at that day's vol, the vol of the return just realised.
- ``skew_slope``: the fair surface is tilted to ``vol + skew_slope * ln(S / K)`` at every
  tenor, floored at ``MIN_VOL``; positive is the equity skew, dearer downside strikes. The
  planted effect is laid on the tilted surface as on the flat one.
"""

from __future__ import annotations

import numpy as np

from optspread.config import MarketConfig
from optspread.effects import CONSTANT_PREMIUM, PlantedEffect, plant
from optspread.rng import PATH_CHANNEL, VOL_CHANNEL, StepRNG
from optspread.types import Chain

MIN_VOL = 0.001  # floor on the path vol and on the tilted surface, decimal


class SyntheticChainProvider:
    def __init__(self, config: MarketConfig, effect: PlantedEffect | None = None) -> None:
        self.config = config
        self.effect = effect if effect is not None else CONSTANT_PREMIUM
        self._rng: StepRNG | None = None
        self._day = 0
        self._spot = config.spot0
        self._vols: list[float] = []  # the path vol on each day, computed at reset

    @property
    def n_days(self) -> int:
        return self.config.n_days

    @property
    def done(self) -> bool:
        return self._day >= self.config.n_days

    @property
    def day(self) -> int:
        return self._day

    def reset(self, rng: StepRNG) -> Chain:
        self._rng = rng
        self._day = 0
        self._spot = self.config.spot0
        self._vols = self._vol_path(rng)
        return self._chain(self._spot)

    def _vol_path(self, rng: StepRNG) -> list[float]:
        """The path vol on days ``0..n_days``: ``sigma`` throughout without vol of vol."""
        cfg = self.config
        if cfg.vol_of_vol == 0.0:
            return [cfg.sigma] * (cfg.n_days + 1)
        step = cfg.vol_of_vol * float(np.sqrt(1.0 / cfg.trading_days_per_year))
        vols = [cfg.sigma]
        for day in range(1, cfg.n_days + 1):
            z = float(rng.stream(day, VOL_CHANNEL).standard_normal())
            vols.append(max(vols[-1] * float(np.exp(step * z)), MIN_VOL))
        return vols

    def step(self) -> Chain:
        if self._rng is None:
            raise RuntimeError("step() before reset()")
        if self.done:
            raise RuntimeError("step() past the horizon")
        self._day += 1
        self._spot *= float(np.exp(self.log_return(self._day)))
        return self._chain(self._spot)

    def log_return(self, day: int) -> float:
        """The path's log return INTO ``day``: a pure function of ``stream(day)`` and,
        with vol of vol, of the vol path drawn at ``reset``."""
        if self._rng is None:
            raise RuntimeError("log_return() before reset()")
        cfg = self.config
        dt = 1.0 / cfg.trading_days_per_year
        z = float(self._rng.stream(day, PATH_CHANNEL).standard_normal())
        # past the horizon (a look-ahead probe at the last day) the vol stays where it ended
        sigma = self._vols[min(day, len(self._vols) - 1)]
        diffusion = -0.5 * sigma * sigma * dt + sigma * float(np.sqrt(dt)) * z
        # per day, not annualised; at zero the return is the diffusion's, bit for bit
        return diffusion + cfg.alpha_bps / 10_000.0 if cfg.alpha_bps else diffusion

    def _chain(self, spot: float) -> Chain:
        """Chain at the current day for ``spot`` (the poisoned variant passes a shifted spot)."""
        cfg = self.config
        centre = round(spot / cfg.strike_step) * cfg.strike_step
        offsets = np.arange(-cfg.n_strikes_each_side, cfg.n_strikes_each_side + 1)
        strikes = centre + cfg.strike_step * offsets.astype(np.float64)
        expiry_days = self._day + np.asarray(cfg.tenor_days, dtype=np.int64)
        vol = self._vols[self._day] if self._vols else cfg.sigma
        shape = (len(cfg.tenor_days), len(strikes))
        if cfg.skew_slope == 0.0:
            ivs = np.full(shape, vol, dtype=np.float64)
        else:  # the same smile at every tenor
            smile = np.maximum(vol + cfg.skew_slope * np.log(spot / strikes), MIN_VOL)
            ivs = np.broadcast_to(smile, shape).astype(np.float64)
        fair = Chain(
            t=self._day,
            spot=spot,
            r=cfg.r,
            strikes=strikes,
            expiry_days=expiry_days,
            ivs=ivs,
            trading_days_per_year=cfg.trading_days_per_year,
        )
        return plant(self.effect, fair, cfg.iv_premium_vol_pts)
