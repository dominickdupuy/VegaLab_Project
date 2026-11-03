"""Synthetic chain provider: zero-drift GBM with the chain priced off the path vol.

Lane: Hemdutt Rao (market layer).

With ``iv_premium_vol_pts == 0`` every contract is priced at the exact vol that
drives the underlying, so every strategy has zero expectancy before costs. A
positive premium prices the chain richer than the path realises: the planted
variance risk premium the power sweep calibrates against.

RNG: the return into day ``t`` is drawn from ``rng.stream(t)`` alone (the
per-step contract in ``optspread.rng``). The chain at ``t`` therefore depends on
no stream later than ``t``, which ``optspread.leaks`` tests directly.
"""

from __future__ import annotations

import numpy as np

from optspread.config import MarketConfig
from optspread.rng import PATH_CHANNEL, VOL_CHANNEL, StepRNG
from optspread.types import Chain


class SyntheticChainProvider:
    def __init__(self, config: MarketConfig) -> None:
        self.config = config
        self._rng: StepRNG | None = None
        self._day = 0
        self._spot = config.spot0
        self._vols: list[float] = []  # precomputed in reset(); index d = vol entering day d

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
        # Precompute the vol path so log_return(day) is idempotent — safe for the leak detector.
        cfg = self.config
        dt = 1.0 / cfg.trading_days_per_year
        self._vols = [cfg.sigma]
        for d in range(1, cfg.n_days + 1):
            if cfg.vol_of_vol > 0.0:
                z_vol = float(rng.stream(d, VOL_CHANNEL).standard_normal())
                v = max(
                    self._vols[-1] * float(np.exp(cfg.vol_of_vol * float(np.sqrt(dt)) * z_vol)),
                    0.001,
                )
            else:
                v = cfg.sigma
            self._vols.append(v)
        return self._chain(self._spot)

    def step(self) -> Chain:
        if self._rng is None:
            raise RuntimeError("step() before reset()")
        if self.done:
            raise RuntimeError("step() past the horizon")
        self._day += 1
        self._spot *= float(np.exp(self.log_return(self._day)))
        return self._chain(self._spot)

    def log_return(self, day: int) -> float:
        """The path's log return INTO ``day``. Pure read — vol was precomputed in reset().

        Calling this multiple times for the same day always returns the same value,
        which is required by the leak detector (it calls log_return to look ahead
        without stepping, and must not disturb the path it is measuring).
        """
        if self._rng is None:
            raise RuntimeError("log_return() before reset()")
        cfg = self.config
        dt = 1.0 / cfg.trading_days_per_year
        z = float(self._rng.stream(day, PATH_CHANNEL).standard_normal())
        sigma = self._vols[day]
        drift = cfg.alpha_bps / 10_000.0  # per-day drift in bps (not annualised — no dt)
        return drift - 0.5 * sigma * sigma * dt + sigma * float(np.sqrt(dt)) * z

    def _chain(self, spot: float) -> Chain:
        """Chain at the current day for ``spot`` (the poisoned variant passes a shifted spot)."""
        cfg = self.config
        centre = round(spot / cfg.strike_step) * cfg.strike_step
        offsets = np.arange(-cfg.n_strikes_each_side, cfg.n_strikes_each_side + 1)
        strikes = centre + cfg.strike_step * offsets.astype(np.float64)
        expiry_days = self._day + np.asarray(cfg.tenor_days, dtype=np.int64)
        iv_atm = self._vols[self._day] + cfg.iv_premium_vol_pts / 100.0
        # log(S/K) > 0 for low strikes (OTM puts), < 0 for high strikes (OTM calls)
        log_moneyness = np.log(spot / strikes)  # shape: (n_strikes,)
        smile = np.maximum(
            0.001, iv_atm + cfg.skew_slope * log_moneyness
        )  # shape: (n_strikes,)
        ivs = np.tile(smile, (len(cfg.tenor_days), 1))  # same smile at every tenor
        return Chain(
            t=self._day,
            spot=spot,
            r=cfg.r,
            strikes=strikes,
            expiry_days=expiry_days,
            ivs=ivs,
            trading_days_per_year=cfg.trading_days_per_year,
        )
