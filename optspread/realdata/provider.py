"""Real chains on the simulator's contract: a historical window of the SPX surface.

Lane: Hemdutt Rao (data adapter).

A ``HistoricalChainProvider`` is a ``ChainProvider`` like the synthetic one. An
episode is a window of ``n_days`` consecutive trading days whose start is drawn
from ``rng.stream(0)``; day ``t`` of the episode is the market's close on the
window's ``t``-th day. Nothing about a later day is read before ``step`` reaches
it, and resampling the streams after ``t`` cannot change the chain at ``t``, so
the no-look-ahead contract holds by construction.

The store a run reads is one split's dates (``optspread.realdata.SPLITS``, #28):
training and evaluation never share a day, and no split reaches past the
evaluation range. A start is drawn only where the whole window is consecutive on
the trading calendar, so an episode never runs across a day dropped as unusable.

Each chain is laid on the same strike and tenor grid the synthetic provider
uses (``MarketConfig``), so the environment, the action library and the leak
suite see the same shapes. IVs come from the compact surface grid
(``panel/surface_grid.parquet``): interpolated in tenor as total variance (calendar
days, from the grid's trading days at 365/252) and in log-moneyness against the
engine's own forward, flat beyond the last quoted point. ``r`` is the day's 30-day
carry, rate less dividends, because the engine prices without a dividend yield;
that puts its forward where the market's was.

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
from optspread.realdata import SPLITS, Split
from optspread.realdata.panel import K_GRID, TENOR_GRID
from optspread.rng import PATH_CHANNEL, StepRNG
from optspread.types import Chain

IV_FLOOR, IV_CAP = 0.01, 3.0

F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SurfaceStore:
    """The surface grid as dense arrays: ``ivs[day, tenor, k]`` with gaps filled.

    ``calendar`` is each kept day's position on the trading calendar (every date of
    the daily series in range). A day dropped as unusable leaves a hole in it, which
    is how a window that would run across the gap is found and refused. ``None``
    means the days are consecutive.
    """

    dates: tuple[str, ...]
    spot: F64
    carry: F64
    ivs: F64  # (n_days, len(TENOR_GRID), len(K_GRID))
    calendar: NDArray[np.int64] | None = None

    @classmethod
    def from_frames(
        cls, grid: pd.DataFrame, daily: pd.DataFrame, split: Split | None = None
    ) -> SurfaceStore:
        """The store over one split's dates (#28), or over every date when ``None``.

        The frames are cut to the split before anything else is read from them, so
        a store for a strategy run never holds a day outside its range.
        """
        if split is not None:
            grid, daily = _within(grid, split), _within(daily, split)
        cube = grid.pivot_table(index=["date", "tenor_days"], columns="k", values="iv")
        cube = cube.reindex(columns=K_GRID)
        trading = sorted(daily["date"])
        days = sorted(set(cube.index.get_level_values("date")) & set(trading))
        d = daily.set_index("date").loc[days]
        ivs = np.full((len(days), len(TENOR_GRID), len(K_GRID)), np.nan)
        for i, day in enumerate(days):
            block = cube.xs(day, level="date").reindex(list(TENOR_GRID))
            ivs[i] = _fill(block.to_numpy(dtype=np.float64))
        spot = d["spot"].to_numpy(dtype=np.float64)
        carry = d["carry30"].fillna(d.get("rate30", 0.0)).fillna(0.0).to_numpy(dtype=np.float64)
        usable = np.isfinite(ivs).all(axis=(1, 2)) & np.isfinite(spot) & (spot > 0)
        position = {day: i for i, day in enumerate(trading)}
        return cls(
            dates=tuple(str(x) for x, ok in zip(days, usable, strict=True) if ok),
            spot=spot[usable],
            carry=carry[usable],
            ivs=ivs[usable],
            calendar=np.asarray(
                [position[x] for x, ok in zip(days, usable, strict=True) if ok], dtype=np.int64
            ),
        )

    @property
    def n(self) -> int:
        return len(self.dates)

    def window_starts(self, n_days: int) -> NDArray[np.int64]:
        """Every start whose ``n_days + 1`` days are consecutive trading days."""
        starts = np.arange(max(self.n - n_days, 0), dtype=np.int64)
        if self.calendar is None:
            return starts
        whole = self.calendar[starts + n_days] - self.calendar[starts] == n_days
        return np.asarray(starts[whole], dtype=np.int64)


def _within(frame: pd.DataFrame, split: Split) -> pd.DataFrame:
    lo, hi = SPLITS[split]
    day = pd.to_datetime(frame["date"]).dt.date
    return frame[(day >= lo) & (day <= hi)]


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
def load_store(panel_dir: Path, split: Split) -> SurfaceStore:  # pragma: no cover - shared folder
    grid = pd.read_parquet(panel_dir / "surface_grid.parquet")
    daily = pd.read_parquet(panel_dir / "spx_daily.parquet")
    return SurfaceStore.from_frames(grid, daily, split)


class HistoricalChainProvider:
    def __init__(self, config: MarketConfig, store: SurfaceStore) -> None:
        if store.n <= config.n_days:
            raise ValueError(f"surface has {store.n} days, an episode needs {config.n_days + 1}")
        starts = store.window_starts(config.n_days)
        if starts.size == 0:
            raise ValueError(
                f"no {config.n_days + 1} consecutive trading days in the surface: every "
                "window would span a gap"
            )
        self.config = config
        self.store = store
        self._starts = starts
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
        draw = int(rng.stream(0, PATH_CHANNEL).integers(self._starts.size))
        self._start = int(self._starts[draw])
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
        # Total variance is what interpolates linearly in time without creating
        # calendar arbitrage; outside the grid the vol is held flat.
        t_c = min(max(cal, grid_t[0]), grid_t[-1])
        w = cube * cube * grid_t[:, None]
        w_at = np.array([np.interp(t_c, grid_t, w[:, j]) for j in range(len(K_GRID))])
        per_k = np.sqrt(np.maximum(w_at, 0.0) / t_c)
        T = trading_days / self.config.trading_days_per_year
        k = np.log(strikes / (spot * np.exp(r * T)))
        return np.asarray(np.clip(np.interp(k, K_GRID, per_k), IV_FLOOR, IV_CAP), dtype=np.float64)
