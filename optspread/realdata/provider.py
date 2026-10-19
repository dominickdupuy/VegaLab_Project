"""Real chains on the simulator's contract: a historical window of the SPX surface.

Lane: Hemdutt Rao (data adapter).

A ``HistoricalChainProvider`` is a ``ChainProvider`` like the synthetic one. An
episode is a window of ``n_days`` consecutive trading days whose start is drawn
from ``rng.stream(0)``; day ``t`` of the episode is the market's close on the
window's ``t``-th day. Nothing about a later day is read before ``step`` reaches
it, and resampling the streams after ``t`` cannot change the chain at ``t``, so
the no-look-ahead contract holds by construction.

Each chain is laid on the same strike and tenor grid the synthetic provider
uses (``MarketConfig``), so the environment, the action library and the leak
suite see the same shapes. IVs come from the compact surface grid
(``panel/surface_grid.parquet``): interpolated in tenor (calendar days, from the
grid's trading days at 365/252) and in log-moneyness against the engine's own
forward, flat beyond the last quoted point. ``r`` is the day's 30-day carry, rate
less dividends, because the engine prices without a dividend yield; that puts its
forward where the market's was.

The planted premium (``iv_premium_vol_pts``) is not applied: a real surface
carries whatever premium the market priced, which is the point of using it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optspread.config import MarketConfig
from optspread.realdata.panel import K_GRID, TENOR_GRID
from optspread.rng import PATH_CHANNEL, StepRNG
from optspread.types import Chain

IV_FLOOR, IV_CAP = 0.01, 3.0

F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SurfaceStore:
    """The surface grid as dense arrays: ``ivs[day, tenor, k]`` with gaps filled."""

    dates: tuple[str, ...]
    spot: F64
    carry: F64
    ivs: F64  # (n_days, len(TENOR_GRID), len(K_GRID))

    @classmethod
    def from_frames(cls, grid: pd.DataFrame, daily: pd.DataFrame) -> SurfaceStore:
        cube = grid.pivot_table(index=["date", "tenor_days"], columns="k", values="iv")
        cube = cube.reindex(columns=K_GRID)
        days = sorted(set(cube.index.get_level_values("date")) & set(daily["date"]))
        d = daily.set_index("date").loc[days]
        ivs = np.full((len(days), len(TENOR_GRID), len(K_GRID)), np.nan)
        for i, day in enumerate(days):
            block = cube.xs(day, level="date").reindex(list(TENOR_GRID))
            ivs[i] = _fill(block.to_numpy(dtype=np.float64))
        spot = d["spot"].to_numpy(dtype=np.float64)
        carry = d["carry30"].fillna(d.get("rate30", 0.0)).fillna(0.0).to_numpy(dtype=np.float64)
        usable = np.isfinite(ivs).all(axis=(1, 2)) & np.isfinite(spot) & (spot > 0)
        return cls(
            dates=tuple(str(x) for x, ok in zip(days, usable, strict=True) if ok),
            spot=spot[usable],
            carry=carry[usable],
            ivs=ivs[usable],
        )

    @property
    def n(self) -> int:
        return len(self.dates)


def _fill(block: F64) -> F64:
    """Fill a (tenor, k) block: nearest quoted k along each tenor, then nearest tenor."""
    out = block.copy()
    for row in out:
        ok = np.isfinite(row)
        if ok.any():
            idx = np.arange(row.size)
            row[~ok] = np.interp(idx[~ok], idx[ok], row[ok])
    for col in out.T:
        ok = np.isfinite(col)
        if ok.any():
            idx = np.arange(col.size)
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    return out


@lru_cache(maxsize=2)
def load_store(panel_dir: Path) -> SurfaceStore:  # pragma: no cover - reads the shared folder
    grid = pd.read_parquet(panel_dir / "surface_grid.parquet")
    daily = pd.read_parquet(panel_dir / "spx_daily.parquet")
    return SurfaceStore.from_frames(grid, daily)


class HistoricalChainProvider:
    def __init__(self, config: MarketConfig, store: SurfaceStore) -> None:
        if store.n <= config.n_days:
            raise ValueError(f"surface has {store.n} days, an episode needs {config.n_days + 1}")
        self.config = config
        self.store = store
        self._start = 0
        self._day = 0
        self._reset = False

    @property
    def n_days(self) -> int:
        return self.config.n_days

    @property
    def done(self) -> bool:
        return self._day >= self.config.n_days

    @property
    def start_date(self) -> str:
        return self.store.dates[self._start]

    def reset(self, rng: StepRNG) -> Chain:
        n_starts = self.store.n - self.config.n_days
        self._start = int(rng.stream(0, PATH_CHANNEL).integers(n_starts))
        self._day = 0
        self._reset = True
        return self._chain()

    def step(self) -> Chain:
        if not self._reset:
            raise RuntimeError("step() before reset()")
        if self.done:
            raise RuntimeError("step() past the horizon")
        self._day += 1
        return self._chain()

    def _chain(self) -> Chain:
        cfg, i = self.config, self._start + self._day
        spot, r = float(self.store.spot[i]), float(self.store.carry[i])
        centre = round(spot / cfg.strike_step) * cfg.strike_step
        offsets = np.arange(-cfg.n_strikes_each_side, cfg.n_strikes_each_side + 1)
        strikes = centre + cfg.strike_step * offsets.astype(np.float64)
        tenors = np.asarray(cfg.tenor_days, dtype=np.int64)
        ivs = np.vstack([self._smile(i, int(td), spot, r, strikes) for td in tenors])
        return Chain(
            t=self._day,
            spot=spot,
            r=r,
            strikes=strikes,
            expiry_days=self._day + tenors,
            ivs=ivs,
            trading_days_per_year=cfg.trading_days_per_year,
        )

    def _smile(self, i: int, trading_days: int, spot: float, r: float, strikes: F64) -> F64:
        cal = trading_days * 365.0 / self.config.trading_days_per_year
        grid_t = np.asarray(TENOR_GRID, dtype=np.float64)
        cube = self.store.ivs[i]
        per_k = np.array([np.interp(cal, grid_t, cube[:, j]) for j in range(len(K_GRID))])
        T = trading_days / self.config.trading_days_per_year
        k = np.log(strikes / (spot * np.exp(r * T)))
        return np.asarray(np.clip(np.interp(k, K_GRID, per_k), IV_FLOOR, IV_CAP), dtype=np.float64)
