"""API payloads for the power lab and the power surface, computed by ``optspread.power``.

Lane: Ilan Danial (machinery).

These compute rather than read, so they name the module and formula their numbers came
from. Moved from ``optspread.webapi``, which re-exports it.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from pydantic import ConfigDict, Field

from optspread import power as pw
from optspread.api._base import Out, _f

# -- the power lab ---------------------------------------------------------------- #

# Illustrative until the variance-components pilot reports; the pilot_budget script's
# own example values, in dollars of mean P&L per episode.
POWER_DEFAULTS = {"tau2": 250_000.0, "sigma2": 1.3e8, "delta": 1500.0}
FLOOR_SEEDS = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100)
FRONTIER_SEEDS = (2, 3, 5, 8, 10, 16, 20, 32, 50, 64)


class PowerParams(Out):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tau2: float = Field(default=POWER_DEFAULTS["tau2"], ge=0.0)
    sigma2: float = Field(default=POWER_DEFAULTS["sigma2"], ge=0.0)
    n_seeds: int = Field(default=10, ge=2, le=10_000)  # the t test needs two seed means
    n_episodes: int = Field(default=100, ge=1, le=1_000_000)
    delta: float = Field(default=POWER_DEFAULTS["delta"], gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    target_power: float = Field(default=0.80, gt=0.5, lt=1.0)
    train_steps: int = Field(default=1_000_000, ge=0)
    episode_days: int = Field(default=63, ge=1)
    train_rate: float = Field(default=1_000.0, gt=0.0)
    eval_rate: float = Field(default=1_800.0, gt=0.0)
    cores: int = Field(default=1, ge=1, le=4096)


class CurveOut(Out):
    delta: list[float]
    power: list[float]
    power_s5: list[float]
    power_s50: list[float]


class FloorOut(Out):
    seeds: list[int]
    floor: list[float]
    floor_s5: float
    floor_s50: float
    ratio: float  # floor at S=5 over S=50: 3.8 when tau^2 dominates, sqrt(10) by the normal form


class DesignOut(Out):
    n_seeds: int
    n_episodes: int | None
    power: float | None
    train_hours: float | None
    eval_hours: float | None
    total_hours: float | None


class PowerOut(Out):
    params: PowerParams
    se: float
    var_grand_mean: float
    variance_budget: float
    power: float
    floor: float  # smallest effect detected at the target power with this design
    episodes_needed: int | None
    seeds_needed: int
    curve: CurveOut
    floor_vs_seeds: FloorOut
    frontier: list[DesignOut]
    design: DesignOut
    source: str


def _floor(p: PowerParams, n_seeds: int) -> float:
    return pw.detectable_effect_t(p.tau2, p.sigma2, n_seeds, p.n_episodes, p.alpha, p.target_power)


def _design(d: pw.Design) -> DesignOut:
    b = d.budget
    return DesignOut(
        n_seeds=d.n_seeds,
        n_episodes=d.n_episodes,
        power=_f(d.power),
        train_hours=_f(b.train_hours) if b else None,
        eval_hours=_f(b.eval_hours) if b else None,
        total_hours=_f(b.total_hours) if b else None,
    )


def power_lab(p: PowerParams) -> PowerOut:
    """Power, floor and frontier for one design, from ``optspread.power``. Instant.

    Every figure is the exact noncentral t of the committed test (#20), the frontier
    and the episodes and seeds needed included; nothing here is the normal form.
    """
    var = pw.var_grand_mean(p.tau2, p.sigma2, p.n_seeds, p.n_episodes)
    floor = _floor(p, p.n_seeds)
    top = max(2.0 * p.delta, 2.5 * max(floor, _floor(p, 5) / 2.0), 1.0)
    grid = np.linspace(0.0, top, 81)

    def curve(n_seeds: int) -> list[float]:
        return [
            pw.power_t(float(d), p.tau2, p.sigma2, n_seeds, p.n_episodes, p.alpha) for d in grid
        ]

    floors = [_floor(p, s) for s in FLOOR_SEEDS]
    designs = pw.frontier(
        p.delta,
        p.tau2,
        p.sigma2,
        FRONTIER_SEEDS,
        alpha=p.alpha,
        target_power=p.target_power,
        train_steps_per_seed=p.train_steps,
        episode_days=p.episode_days,
        train_rate=p.train_rate,
        eval_rate=p.eval_rate,
        cores=p.cores,
    )
    now = pw.budget(
        p.n_seeds,
        p.n_episodes,
        train_steps_per_seed=p.train_steps,
        episode_days=p.episode_days,
        train_rate=p.train_rate,
        eval_rate=p.eval_rate,
        cores=p.cores,
    )
    power_now = pw.power_t(p.delta, p.tau2, p.sigma2, p.n_seeds, p.n_episodes, p.alpha)
    return PowerOut(
        params=p,
        se=math.sqrt(var),
        var_grand_mean=var,
        variance_budget=pw.variance_budget_t(p.delta, p.n_seeds, p.alpha, p.target_power),
        power=power_now,
        floor=floor,
        episodes_needed=pw.episodes_for(
            p.delta, p.tau2, p.sigma2, p.n_seeds, p.alpha, p.target_power
        ),
        seeds_needed=pw.seeds_for(p.delta, p.tau2, p.sigma2, p.n_episodes, p.alpha, p.target_power),
        curve=CurveOut(
            delta=[float(d) for d in grid],
            power=curve(p.n_seeds),
            power_s5=curve(5),
            power_s50=curve(50),
        ),
        floor_vs_seeds=FloorOut(
            seeds=list(FLOOR_SEEDS),
            floor=floors,
            floor_s5=_floor(p, 5),
            floor_s50=_floor(p, 50),
            ratio=_floor(p, 5) / _floor(p, 50) if _floor(p, 50) > 0 else float("nan"),
        ),
        frontier=[_design(d) for d in designs],
        design=DesignOut(
            n_seeds=p.n_seeds,
            n_episodes=p.n_episodes,
            power=power_now,
            train_hours=now.train_hours,
            eval_hours=now.eval_hours,
            total_hours=now.total_hours,
        ),
        source=(
            "optspread.power: Var(grand mean) = tau^2/S + sigma^2/(S N), one-sided t test "
            "on the S seed means, exact noncentral t"
        ),
    )


# -- the power surface ------------------------------------------------------------ #

SURFACE_PREMIUMS = 41  # points along the premium axis


class PowerSurfaceParams(Out):
    """The power lab's variance components and N, and the premium axis to draw.

    ``delta`` is the effect, in dollars of mean P&L per episode, at one planted vol
    point (A6 takes the fixed-delta seller's contrast at 1 vol point); the surface
    scales it linearly with the premium and holds tau^2 and sigma^2 fixed.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    tau2: float = Field(default=POWER_DEFAULTS["tau2"], ge=0.0)
    sigma2: float = Field(default=POWER_DEFAULTS["sigma2"], ge=0.0)
    n_episodes: int = Field(default=100, ge=1, le=1_000_000)
    delta: float = Field(default=POWER_DEFAULTS["delta"], gt=0.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    target_power: float = Field(default=0.80, gt=0.5, lt=1.0)
    max_premium: float = Field(default=4.0, gt=0.0, le=20.0)
    max_seeds: int = Field(default=100, ge=3, le=200)


class PowerSurfaceOut(Out):
    params: PowerSurfaceParams
    premium: list[float]  # (P,) vol points
    seeds: list[int]  # (S,)
    power: list[list[float | None]]  # (S, P) exact power of the one-sided t test
    contour_premium: list[float | None]  # (S,) the premium detected at the target power
    source: str


def surface_seeds(max_seeds: int) -> list[int]:
    """Every S to 30, then every fourth: the curvature is at small S."""
    seeds = list(range(2, min(30, max_seeds) + 1))
    seeds += list(range(32, max_seeds + 1, 4))
    if seeds[-1] != max_seeds:
        seeds.append(max_seeds)
    return seeds


@lru_cache(maxsize=64)  # about a second each; sliders come back to the same designs
def power_surface(p: PowerSurfaceParams) -> PowerSurfaceOut:
    """Exact detection probability over planted premium x seeds, at the page's N.

    Each cell is ``optspread.power.power_t`` at effect ``delta * premium``; the
    contour is ``detectable_effect_t`` over ``delta``, so it is exact rather than
    read off the grid.
    """
    premiums = np.linspace(0.0, p.max_premium, SURFACE_PREMIUMS)
    seeds = surface_seeds(p.max_seeds)
    grid = [
        [
            _f(pw.power_t(p.delta * float(x), p.tau2, p.sigma2, s, p.n_episodes, p.alpha))
            for x in premiums
        ]
        for s in seeds
    ]
    contour = [
        pw.detectable_effect_t(p.tau2, p.sigma2, s, p.n_episodes, p.alpha, p.target_power) / p.delta
        for s in seeds
    ]
    return PowerSurfaceOut(
        params=p,
        premium=[float(x) for x in premiums],
        seeds=seeds,
        power=grid,
        contour_premium=contour,
        source="optspread.power.power_t and detectable_effect_t: noncentral t, S - 1 df",
    )
