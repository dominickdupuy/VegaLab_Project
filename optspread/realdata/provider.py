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
A surface of your own may cut its splits on its own ``SplitRanges`` instead; the
source refuses them for the team's SPX panel (``optspread.data``).

Each chain is laid on the same strike and tenor grid the synthetic provider
uses (``MarketConfig``), so the environment, the action library and the leak
suite see the same shapes. That grid is fixed in index points (``strike_step``, 50,
and 16 strikes a side), so it is 4.5% of spot apart in 2010 and 1.1% in 2021: coarse
in most of the training range and fine in the evaluation range, a shift in the action
set between the splits. Scaling it to spot would change ``MarketConfig``, which is
hashed and frozen; docs/DATA.md lists it under Known limits.

IVs come from the compact surface grid (``panel/surface_grid.parquet``), made free
of calendar arbitrage per day (``calendar_free``, #29), then interpolated in tenor
as total variance (calendar days, from the grid's trading days at 365/252) and in
log-moneyness against the engine's own forward, flat beyond the grid's ends. Both
interpolations keep total variance non-decreasing in tenor at every forward
moneyness, so every chain clears ``leaks.calendar_violations``. The spot is the
parity spot and ``r`` the day's 30-day carry against it, rate less dividends,
because the engine prices without a dividend yield; that puts its forward where the
market's was.

A held leg is marked between the listed tenors, not on the nearest one. Each
chain is a ``TermStructureChain``, whose ``iv_at`` interpolates total variance
between the bracketing listed expiries (#27); the synthetic world keeps ``Chain``.

The planted premium (``iv_premium_vol_pts``) is not applied: a real surface
carries whatever premium the market priced, which is the point of using it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from optspread.config import MarketConfig
from optspread.realdata import (
    SPLITS,
    Split,
    SplitRanges,
    check_split,
    panel_digests,
    refuse_team_ranges,
    team_digests,
)
from optspread.realdata.panel import K_GRID, TENOR_GRID, nyse_trading_days
from optspread.rng import PATH_CHANNEL, StepRNG
from optspread.types import Chain

IV_FLOOR, IV_CAP = 0.01, 3.0

F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TermStructureChain(Chain):
    """A real chain: a held leg between two listed tenors takes their interpolated vol.

    ``Chain.iv_at`` reads the nearest listed expiry. On the flat synthetic surface
    that is exact; on a real term structure a held leg switches rows as it ages past
    the midpoint of two tenors, and the switch books a P&L step that is only the
    lookup. Here total variance ``sigma^2 * T`` is linear in ``T`` between the
    bracketing listed expiries and the vol is flat outside the listed range. On a
    listed expiry the listed vol comes back unchanged, so a structure opened on the
    grid is priced as before. The strike is the nearest listed strike, as in
    ``Chain``. ``price``, ``greeks`` and ``mark`` all read ``iv_at``, so the book,
    the costs and the margin see the same mark.
    """

    def iv_at(self, expiry_day: int, strike: float) -> float:
        col = self.strike_col(strike)
        days = self.expiry_days
        hi = int(np.searchsorted(days, expiry_day))
        if hi < days.shape[0] and int(days[hi]) == expiry_day:
            return float(self.ivs[hi, col])
        if hi == 0 or hi == days.shape[0]:
            return float(self.ivs[min(hi, days.shape[0] - 1), col])  # flat in vol outside
        # Times in trading days from today; the year length cancels in sigma^2 = w / T.
        t_lo, t_hi = float(days[hi - 1] - self.t), float(days[hi] - self.t)
        t = float(expiry_day - self.t)
        w_lo = float(self.ivs[hi - 1, col]) ** 2 * t_lo
        w_hi = float(self.ivs[hi, col]) ** 2 * t_hi
        w = w_lo + (t - t_lo) / (t_hi - t_lo) * (w_hi - w_lo)
        return float(np.sqrt(w / t))


@dataclass(frozen=True, slots=True)
class SurfaceStore:
    """One split's surface grid as dense arrays: ``ivs[day, tenor, k]``, gaps filled.

    ``from_frames`` (and ``load_store`` over it) builds it for one split of ``SPLITS``
    and nothing else: there is no "every date" store, because a provider serves a
    strategy run. Every date is ``market_surface``'s, a different type no provider takes.

    ``calendar`` is each kept day's position on the trading calendar (the NYSE calendar
    and every date of the daily series, ``trading_days``). A day dropped as unusable, or
    absent from both tables, leaves a hole in it, which is how a window that would run
    across the gap is found and refused. ``None`` means the days are consecutive.

    ``split`` and ``ranges`` say how the frames were cut, and ``panel`` holds the
    digests of the files ``load_store`` read (``panel/<file>`` to sha256), so the
    provider can refuse a store cut on ranges of its own over the team's panel. A store
    built from arrays of your own leaves all three empty.
    """

    dates: tuple[str, ...]
    spot: F64
    carry: F64
    ivs: F64  # (n_days, len(TENOR_GRID), len(K_GRID))
    calendar: NDArray[np.int64] | None = None
    split: Split | None = None
    ranges: SplitRanges | None = None
    panel: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_frames(
        cls,
        grid: pd.DataFrame,
        daily: pd.DataFrame,
        split: Split,
        ranges: SplitRanges | None = None,
    ) -> SurfaceStore:
        """The store over one split's dates (#28): ``split`` is required and checked.

        The frames are cut to the split before anything else is read from them, so
        a store for a strategy run never holds a day outside its range. The split's
        dates are ``SPLITS[split]``, or ``ranges[split]`` for a surface of your own.
        Each day's grid is made free of calendar arbitrage (``calendar_free``).

        The engine's spot is the parity spot and ``r`` the carry against it
        (``carry30_parity``, panel version 4), so the engine's forward is the option
        market's at every tenor; a daily table without them (an older panel, a surface
        of your own) gives the pair it has, the close and ``carry30``.
        """
        split = check_split(split)
        grid, daily = _dated(grid), _dated(daily)
        grid, daily = _within(grid, split, ranges), _within(daily, split, ranges)
        days, ivs, d = _cube(grid, daily)
        for i in range(len(days)):
            ivs[i] = calendar_free(ivs[i])
        spot, carry = _engine_spot_and_carry(d)
        usable = np.isfinite(ivs).all(axis=(1, 2)) & np.isfinite(spot) & (spot > 0)
        position = {day: i for i, day in enumerate(trading_days(d.index))}
        return cls(
            dates=tuple(str(x) for x, ok in zip(days, usable, strict=True) if ok),
            spot=spot[usable],
            carry=carry[usable],
            ivs=ivs[usable],
            calendar=np.asarray(
                [position[x] for x, ok in zip(days, usable, strict=True) if ok], dtype=np.int64
            ),
            split=split,
            ranges=ranges,
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


@dataclass(frozen=True, slots=True)
class MarketSurface:
    """Every date of the panel, as quoted: for market measurement only.

    The VRP series and the dashboard's data views measure the market and report the
    split; they are not strategy runs (#28). This is their read of the surface grid:
    ``ivs[day, tenor, k]`` with the holes kept (NaN), the vendor close as ``spot``. It
    is a different type from ``SurfaceStore`` on purpose, and ``HistoricalChainProvider``
    refuses it, so the holdout cannot reach a strategy run this way.
    """

    dates: tuple[str, ...]
    spot: F64
    ivs: F64


def market_surface(grid: pd.DataFrame, daily: pd.DataFrame) -> MarketSurface:
    """The grid over every date the two frames share, uncut and unfilled."""
    days, ivs, d = _cube(_dated(grid), _dated(daily))
    return MarketSurface(
        dates=tuple(str(x) for x in days), spot=d["spot"].to_numpy(dtype=np.float64), ivs=ivs
    )


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with its ``date`` column as ``datetime.date``, whatever it was stored as."""
    return frame.assign(date=pd.to_datetime(frame["date"]).dt.date)


def _cube(grid: pd.DataFrame, daily: pd.DataFrame) -> tuple[list[date], F64, pd.DataFrame]:
    """The days both frames hold, their raw (tenor, k) blocks, and their daily rows."""
    cube = grid.pivot_table(index=["date", "tenor_days"], columns="k", values="iv")
    cube = cube.reindex(columns=K_GRID)
    days = sorted(set(cube.index.get_level_values("date")) & set(daily["date"]))
    d = daily.drop_duplicates("date").set_index("date").loc[days]
    ivs = np.full((len(days), len(TENOR_GRID), len(K_GRID)), np.nan)
    for i, day in enumerate(days):
        ivs[i] = cube.xs(day, level="date").reindex(list(TENOR_GRID)).to_numpy(dtype=np.float64)
    return days, ivs, d


def _engine_spot_and_carry(d: pd.DataFrame) -> tuple[F64, F64]:
    """The spot the engine prices on and the carry that puts its forward at the market's.

    ``carry30_parity`` is measured against ``spot_parity`` (the prepaid forward of the
    shortest expiry, the spot the option quotes themselves imply), so the two are taken
    together: the engine's forward ``S exp(r T)`` is then the market's at 30 days, and
    at every other tenor off only by the curvature of the carry, not by the gap between
    the index close and the option snapshot (up to 140 bp on a stressed day, MINOR-3 of
    the data audit). A table without them (built before panel version 4) gives its own
    pair, the close and ``carry30``, which agree with each other.
    """
    if "carry30_parity" in d and "spot_parity" in d:
        parity = d["spot_parity"].to_numpy(dtype=np.float64)
        close = d["spot"].to_numpy(dtype=np.float64)
        spot = np.where(np.isfinite(parity) & (parity > 0), parity, close)
        carry_col = d["carry30_parity"]
    else:
        spot, carry_col = d["spot"].to_numpy(dtype=np.float64), d["carry30"]
    rate = d.get("rate30", 0.0)
    carry = carry_col.fillna(rate).fillna(0.0).to_numpy(dtype=np.float64)
    return np.asarray(spot, dtype=np.float64), carry


def trading_days(dates: Iterable[date]) -> list[date]:
    """The calendar a window must be consecutive on: every NYSE trading day from the
    first of ``dates`` to the last (``panel.nyse_trading_days``, where the holiday list
    reaches), together with ``dates`` themselves.

    The exchange's calendar, not the daily table's dates, is what finds a day absent
    from both tables (MINOR-8 of the data audit). A date the table has is a day someone
    quoted, so it is kept even where the list says closed: a surface of your own may
    trade on a calendar of its own.
    """
    have = sorted(set(dates))
    if not have:
        return []
    return sorted(set(have) | set(nyse_trading_days(have[0], have[-1])))


def _within(frame: pd.DataFrame, split: Split, ranges: SplitRanges | None) -> pd.DataFrame:
    lo, hi = SPLITS[split] if ranges is None else ranges[split]
    day = pd.to_datetime(frame["date"]).dt.date
    return frame[(day >= lo) & (day <= hi)]


def calendar_free(block: F64) -> F64:
    """One day's (tenor, k) grid, filled and free of calendar arbitrage (#29).

    Total variance ``w = sigma^2 T`` at a fixed log-moneyness must not fall with tenor;
    the grid's ``k`` is ``ln(K / F)``, so a fixed column is a fixed forward moneyness,
    which is what the calendar check compares (``leaks.calendar_violations``). Three
    steps, each keeping what the one before made monotone:

    1. **Project the quoted cells.** Per column, ``w`` over the quoted tenors is replaced
       by its least-squares projection onto the non-decreasing sequences (isotonic
       regression, pool-adjacent-violators, equal weights): a run of tenors that breaks
       the order is pooled to its mean variance. A column already in order is untouched:
       on the v3 panel the projection moves 904 of 1.15M quoted cells (0.08%), on the
       251 days the audit found, by at most 5.3 vol points.
    2. **Fill along tenor, in total variance.** An unquoted tenor between two quoted ones
       takes ``w`` linear in ``T`` between them; before the first or past the last quoted
       tenor the vol is held flat, so ``w`` grows with ``T``. Each column is now complete
       and still non-decreasing.
    3. **Fill along k.** A column no tenor quoted takes the vol linear in ``k`` between
       the nearest filled columns, flat beyond the last. ``sigma sqrt(T)`` is then a
       convex combination of two non-decreasing sequences, so it stays non-decreasing.

    The fill used to run the other way, along ``k`` first with each tenor held flat past
    its own last quoted strike. Tenors stop at different strikes, so a wing quoted high on
    a short tenor was carried out under a longer tenor's lower one, which made a calendar
    arbitrage the quotes did not (data audit MINOR-4 ii).
    """
    t = np.asarray(TENOR_GRID, dtype=np.float64)[:, None]
    w = block * block * t
    for j in range(w.shape[1]):
        col = w[:, j]
        ok = np.isfinite(col)
        if ok.sum() > 1 and np.any(np.diff(col[ok]) < 0.0):
            col[ok] = _isotonic(col[ok])
        if ok.any() and not ok.all():
            tq, wq = t[ok, 0], col[ok]
            tt = t[~ok, 0]
            inside = np.interp(tt, tq, wq)
            before = wq[0] * tt / tq[0]
            after = wq[-1] * tt / tq[-1]
            col[~ok] = np.where(tt < tq[0], before, np.where(tt > tq[-1], after, inside))
    out = np.sqrt(np.maximum(w, 0.0) / t)
    filled = np.isfinite(out).all(axis=0)
    if filled.any() and not filled.all():
        idx = np.arange(out.shape[1])
        for row in out:
            row[~filled] = np.interp(idx[~filled], idx[filled], row[filled])
    return out


def _isotonic(y: F64) -> F64:
    """The non-decreasing sequence nearest ``y`` in least squares (pool adjacent violators)."""
    means: list[float] = []
    sizes: list[int] = []
    for v in y.tolist():
        means.append(v)
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            n = sizes[-2] + sizes[-1]
            means[-2] = (means[-2] * sizes[-2] + means[-1] * sizes[-1]) / n
            sizes[-2] = n
            means.pop()
            sizes.pop()
    return np.repeat(np.asarray(means, dtype=np.float64), sizes)


@lru_cache(maxsize=2)
def load_store(panel_dir: Path, split: Split, ranges: SplitRanges | None = None) -> SurfaceStore:
    """One split of the panel under ``panel_dir``. ``split`` is checked, and ranges of your
    own are refused over the team's panel, known by path or by content
    (``optspread.realdata.is_team_panel``), before a byte of it is read."""
    split = check_split(split)
    refuse_team_ranges(panel_dir, ranges)
    digests = panel_digests(panel_dir)
    grid = pd.read_parquet(panel_dir / "surface_grid.parquet")
    daily = pd.read_parquet(panel_dir / "spx_daily.parquet")
    store = SurfaceStore.from_frames(grid, daily, split, ranges)
    return replace(store, panel=tuple(sorted(digests.items())))


class HistoricalChainProvider:
    def __init__(self, config: MarketConfig, store: SurfaceStore) -> None:
        if not isinstance(store, SurfaceStore):
            raise TypeError(
                f"a provider serves a SurfaceStore cut to one split, not a "
                f"{type(store).__name__}: every date is for market measurement only (#28)"
            )
        if store.split is not None:
            check_split(store.split)
        if store.ranges is not None and not team_digests().isdisjoint(
            digest for _, digest in store.panel
        ):
            raise ValueError(
                f"a store cut on ranges of its own ({store.ranges}) was read from the team's "
                "SPX panel, which serves only the splits pre-registration A8 fixed"
            )
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

    @property
    def n_starts(self) -> int:
        """How many windows this provider can serve."""
        return int(self._starts.size)

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
        return TermStructureChain(
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
