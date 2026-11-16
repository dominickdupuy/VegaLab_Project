"""How big must a planted edge be before an arm finds it reliably? Its detection floor.

Lane: Dominick Dupuy (the machinery is the study's; this module only feeds it).

An edge-size sweep (``optspread.alpha.sweep``, axis ``edge``) benchmarks one arm at several
planted sizes. At each size the benchmark gives the captured edge, the arm on held-out
signal paths minus the same arm on the twin's, per training seed. **Detected** is the
study's own test on those figures (``optspread.analysis.decide_contrast``): one-sided
Student t on the S per-seed figures, ``t = mean / (sd / sqrt S)``, ``S - 1`` degrees of
freedom, against ``t_{0.95, S-1}``. The study's se, ``MS_between / (S N)``, is exactly the
sample variance of the seed means over S, so the figures are all the test reads. A
deterministic arm (scripted or a plug-in rule) has no training seeds; it is read on
``benchmark.BLOCKS`` disjoint blocks of held-out paths, the study's seed-block design
(each block its own paths), so its S is the number of blocks.

The curve and the floor are ``optspread.m4analysis``'s, unchanged: lambda (the
noncentrality) is estimated at each size by inverting the observed t
(``noncentrality_interval``: median-unbiased, with its exact 95% band), lambda is linear in
edge size between the sizes through an anchor at (0, 0), the power at a lambda is
``power_at``, and the floor is where lambda first reaches the noncentrality of 80% power
(``detection_floor``). Its status is the study's: ``reached`` (between two swept sizes),
``below_grid`` (under the smallest positive size, resting on the line to zero) or
``not_reached`` (above the largest size swept). Size zero, if swept, is the null: the
signal world is then its own twin, so the captured edge is zero by construction and is
reported as the false-alarm check, not as a knot of the curve. That anchor is drawn only
where size zero is the null (``zero_is_null``; the sweep says whether it is and refuses a
grid where it is not), never assumed.

**With few seeds the floor is rough.** At S = 3 a t test has 2 degrees of freedom, the
noncentrality band at one size spans several units, and the study found that even at
S = 5 a floor read off a noisy curve sits about a tenth low in median terms. ``ROUGH``
says so in every output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from optspread.analysis import ALPHA
from optspread.m4analysis import (
    NONCENTRALITY_CAP,
    TARGET_POWER,
    CurvePoint,
    detection_floor,
    interpolate,
    noncentrality_interval,
    power_at,
)
from optspread.power import t_critical

Unit = Literal["seeds", "path blocks"]
FloorStatus = Literal["reached", "below_grid", "not_reached", "untestable"]
CURVE_POINTS = 60  # interpolated points from zero to the largest size, for the chart
ROUGH = (
    "With few seeds the floor is rough: each size's test has S - 1 degrees of freedom, so its "
    "band is wide, and a floor read off a noisy curve tends to sit low (by about a tenth at "
    "S = 5 in the study's simulations). Read it as a bracket between swept sizes, not a number."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class EdgeTest(_Frozen):
    """The one-sided t test on one size's per-seed (or per-block) captured edge."""

    edge: float
    unit: Unit
    n: int
    estimate: float
    se: float
    t: float | None  # None when undefined: no spread and a zero estimate is 0, kept as 0
    df: int
    critical: float
    detected: bool
    lower_bound: float
    noncentrality: tuple[float, float, float]  # lo, median-unbiased, hi
    power: float  # at the median-unbiased lambda
    power_lo: float
    power_hi: float


class EdgeFloor(_Frozen):
    """An arm's detection floor in edge-size units, the study's way, with its curve."""

    unit: Unit
    n: int
    target_power: float
    alpha: float
    status: FloorStatus
    edge: float | None
    lo: float | None
    hi: float | None
    bracket: tuple[float, float | None] | None
    max_power: float | None
    recrossings: tuple[float, ...]
    curve: tuple[tuple[float, float, float, float], ...]  # (edge, power, lo, hi)
    null_detected: bool | None  # the size-zero point's verdict, if zero was swept
    reading: str
    note: str = ROUGH


def seed_test(edge: float, figures: Sequence[float], unit: Unit) -> EdgeTest:
    """``analysis.decide_contrast``'s test on S per-seed figures, with lambda's interval."""
    x = np.asarray(figures, dtype=np.float64)
    s = len(x)
    if s < 2:
        raise ValueError(f"a t test on seed figures needs at least 2 of them, got {s}")
    estimate = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(s))
    if se <= 1e-12 * abs(estimate):
        se = 0.0  # identical figures: their spread is rounding, not a measurement
    critical = t_critical(s, ALPHA)
    if se > 0.0:
        t: float = estimate / se
    else:
        t = math.copysign(math.inf, estimate) if estimate != 0.0 else 0.0
    if abs(t) > NONCENTRALITY_CAP:
        # past the study's cap the t inversion walks into scipy's NaNs; such a t is a sure
        # verdict either way, held at the cap as m4analysis holds an infinite one
        t = math.copysign(math.inf, t)
    lo, mid, hi = noncentrality_interval(t, s)
    return EdgeTest(
        edge=edge,
        unit=unit,
        n=s,
        estimate=estimate,
        se=se,
        t=t if math.isfinite(t) else None,
        df=s - 1,
        critical=critical,
        detected=t > critical,
        lower_bound=estimate - critical * se,
        noncentrality=(lo, mid, hi),
        power=power_at(mid, s, ALPHA),
        power_lo=power_at(lo, s, ALPHA),
        power_hi=power_at(hi, s, ALPHA),
    )


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def _reading(
    status: FloorStatus,
    edge: float | None,
    bracket: tuple[float, float | None] | None,
    max_power: float | None,
    param: str,
) -> str:
    if status == "untestable":
        return "No floor: the test needs at least two seeds (or path blocks) at a planted size."
    if status == "not_reached":
        top = bracket[0] if bracket else 0.0
        return (
            f"Not found reliably anywhere on this grid: even at {_fmt(top)} {param} the power "
            f"is {max_power:.0%}, under 80%, so the floor is above the largest size swept."
        )
    assert edge is not None
    if status == "below_grid":
        return (
            f"Found reliably at every size swept: the floor, about {_fmt(edge)} {param}, is "
            "below the smallest positive size and rests on the line drawn to zero."
        )
    assert bracket is not None and bracket[1] is not None
    return (
        f"Found reliably (80% power) from about {_fmt(edge)} {param}, between the swept "
        f"sizes {_fmt(bracket[0])} and {_fmt(bracket[1])}."
    )


def edge_floor(
    tests: Sequence[EdgeTest], param: str = "units", *, zero_is_null: bool = True
) -> EdgeFloor:
    """The power curve and the 80% floor over the sizes' tests, one arm and one grid.

    ``param`` names the edge's unit in the reading (``vol points``, ``bps a day``).
    ``zero_is_null`` says size zero plants nothing; the curve runs through (0, 0) only
    then, and a ValueError says so otherwise.
    """
    if not zero_is_null:
        raise ValueError(
            "no detection floor: size 0 is not the null on this grid, so the power curve "
            "cannot be drawn through (0, 0)"
        )
    null = next((t for t in tests if t.edge == 0.0), None)
    positive = sorted((t for t in tests if t.edge > 0.0), key=lambda t: t.edge)
    unit: Unit = tests[0].unit if tests else "seeds"
    n = min((t.n for t in tests), default=0)
    if not positive or n < 2:
        return EdgeFloor(
            unit=unit,
            n=n,
            target_power=TARGET_POWER,
            alpha=ALPHA,
            status="untestable",
            edge=None,
            lo=None,
            hi=None,
            bracket=None,
            max_power=None,
            recrossings=(),
            curve=(),
            null_detected=null.detected if null else None,
            reading=_reading("untestable", None, None, None, param),
        )
    curve = [CurvePoint(0.0, False, 0.0, 0.0, 0.0, ALPHA, ALPHA, ALPHA, None)]
    for t in positive:
        lo, mid, hi = t.noncentrality
        curve.append(CurvePoint(t.edge, True, mid, lo, hi, t.power, t.power_lo, t.power_hi, None))
    floor = detection_floor(curve, n, TARGET_POWER, ALPHA)
    step = positive[-1].edge / CURVE_POINTS
    status: FloorStatus = floor.status  # type: ignore[assignment]
    return EdgeFloor(
        unit=unit,
        n=n,
        target_power=TARGET_POWER,
        alpha=ALPHA,
        status=status,
        edge=floor.premium,
        lo=floor.lo,
        hi=floor.hi,
        bracket=floor.bracket,
        max_power=floor.max_power,
        recrossings=floor.recrossings,
        curve=tuple(interpolate(curve, n, step, ALPHA)),
        null_detected=null.detected if null else None,
        reading=_reading(status, floor.premium, floor.bracket, floor.max_power, param),
    )
