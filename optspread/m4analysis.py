"""The M4 analysis of the confirmatory sweep: power curves, floors, the gap, tau^2, bootstrap.

Lane: Dominick Dupuy (statistical spine).

Issues #12, #13 and #14, on the sweep of A7 (``CONFIRMATORY_GRID``: premiums 0, 0.25,
0.5, 1, 2 and 4 vol points by the fixed-delta seller and PPO by S = 5 seed blocks of
N = 500 episodes at 1,000,000 training steps). ``scripts/m4_analysis.py`` runs it on a
runs root and writes ``docs/M4_ANALYSIS.md`` and ``.json``. It only reads the cells a
``GridRecord`` lists; nothing here changes how a run is produced. ``analyse`` refuses a
grid that is not the confirmatory design or has a missing or invalid cell unless it is
told the analysis is exploratory, and then says so in every output.

**When these rules were written (S1).** The decisions are A1 and A2's, pre-registered
before the pilot. Everything else here, the curve's interpolation and its anchor at
zero, the floor, the gap, Brown-Forsythe on seed means, the bootstrap agreement
tolerance with its Bonferroni split and the switch to the bootstrap rule, was written
on 2026-10-28: after the sweep ran (2026-10-26) and after its raw arm means were posted
in issue #9, and before this code read the sweep's files. These rules are not
pre-registered. They enter the pre-registration only when A9 (issue #37) records them,
and A9 is to say they were written after the sweep. ``RULES_TIMING`` is that statement;
the report and the sidecar carry it.

**The decisions (#12; A1, A2).** For each arm a and premium p > 0, the pre-registered
event: ``analysis.decide_premium`` on a at p against a at zero, block by block (the
contrast ``GridRecord.contrast`` reads). Reported with the estimate, se, t, one-sided
95% lower bound and verdict; the contrast's variance components
(``power.variance_components``); the two-sided 95% t interval; and the plug-in exact
power ``power_t(estimate, tau^2, sigma^2, S, N)`` at the measured components (A3).

**The power curve (#12).** The test's power depends on the world only through the
noncentrality ``lambda = delta / sqrt(Var(grand mean))``: it is
``P(nct(S - 1, lambda) > t_{0.95, S-1})`` (``power_at``; ``power_t`` is the same number
written in delta and the components). At each premium the decision's t statistic is a
draw from that noncentral t, so lambda is estimated by inverting it
(``noncentrality_interval``). The curve point is the power at the median-unbiased
lambda (the one under which the observed t is the median) and the band is the power
at the exact ``CURVE_LEVEL`` interval for lambda (the lambdas under which the observed
t is the 97.5th and the 2.5th percentile). The band is exact under normal seed means
and pointwise, not simultaneous. Median-unbiased rather than plugging t in, because at
S = 5 the t statistic overstates lambda by a quarter on average (``E[t] / lambda`` is
1.25 at 4 df), and so the point always lies inside its band. A point is the
probability that a fresh experiment at the grid's own S and N (5 x 500 confirmatory)
detects the premium, were the world the one measured there.

Between sweep points each of the three lambdas is linear in premium, through an anchor
at (0, 0): at zero premium there is nothing to detect and the power is alpha (the size
of the event is the section 8 null-world case's to show, A1). A linear lambda is what
an effect growing linearly at a constant variance gives. Below the first sweep point
this holds the first point's variance, so a floor under 0.25 vol points rests on that
assumption and reads ``below_grid``. ``interpolated`` lists the curve every
``CURVE_STEP`` vol points, so the dashboard never computes a noncentral t.

**The floor (#12).** Per arm, the smallest premium at which the curve reaches
``TARGET_POWER``: where lambda first reaches ``lambda_80``, the noncentrality of 80%
power (``noncentrality_for_power``, which is ``power.t_noncentrality``, so the planner
and the analysis use one number), solved exactly on the piecewise-linear lambda. Its
interval is where the band's upper and lower edges reach 0.80. Its bracket is the pair
of sweep premiums the crossing lies between, which needs no interpolation. If the curve
never reaches 0.80 on the grid the floor is None, the status ``not_reached`` and the
bracket (largest premium, None): the floor is above the sweep, and the report gives the
largest power reached. A curve that falls back under 0.80 after its floor lists those
premiums as ``recrossings``. The first crossing of a noisy curve comes early: on
linear truths at S = 5, simulated, the median floor estimate is 8 to 11% below the true
floor and it lies below it in 57 to 61% of draws, so a floor reads about a tenth low in
median terms, while the interval [lo, hi] covers the true floor in 97 to 99% of draws.
A ``below_grid`` floor holds the first point's noncentrality per vol point down to zero,
which nothing measured supports when the measured noncentralities are not proportional
to premium; the report lists them per vol point beside such a floor.

**The gap (#12, section 4.4).** The PPO floor minus the oracle's (the fixed-delta
seller): the horizontal distance between the two curves at 0.80. If PPO never reaches
0.80 and the oracle does, it is a lower bound, the largest premium minus the oracle
floor. ``band_range`` pairs PPO's floor interval with the oracle's opposite ends and
``bracket`` does the same with the brackets; both are ranges, not confidence intervals.
A gap from a ``below_grid`` oracle floor inherits that floor's interpolation.

**Homogeneity of tau^2 (#13, section 5.4).** Brown-Forsythe (Levene centred on the
group median) across the premiums, on seed-level means: section 5.4 names Levene or
Brown-Forsythe, and with five means a group the median-centred form keeps its size
under the skew of short-option P&L where the mean-centred one does not. Run twice per
arm. ``contrast``: the seed means of each contrast, p > 0, whose tau^2 is the one the
curve uses (A5); zero is identically zero and left out. ``arm``: the arm's own seed
means at every premium including zero, which is 5.4's question as written (are
policies more consistent where there is an edge than in the null world). Either way the
test compares the variance of seed means, ``tau^2 + sigma^2 / N``, which is tau^2
only where sigma^2 / N is equal or small, so each level's split is reported beside
it. The groups share blocks (one training seed and one set of paths at every premium),
which makes the F reference conservative, and five means a group gives the test little
power: a rejection is a finding, a non-rejection is weak evidence. How little is
measured, not asserted: ``brown_forsythe_calibration`` simulates the test on normal
seed means at the analysis's own groups and S, its size with equal variances and its
power where the largest variance is ``HOMOGENEITY_RATIOS`` times the smallest (spread
geometrically across the groups). At 5 groups of 5 its size is about 0.003 at alpha
0.05 and its power under a half at a 25-fold spread, so "not rejected" does not show
that tau^2 is constant across premiums.

**The bootstrap (#14, section 5.3).** At each level ``validation.bootstrap_detection``
resamples the S seed means of the contrast ``N_BOOT`` times and compares the rate at
which the t test fires with the analytic rate at the observed effect. The agreement
tolerance (written after the sweep, see above): the gap (bootstrap minus analytic) agrees
when its two-sided reference p-value is at least ``AGREEMENT_ALPHA / k``, k the arm's
number of levels (Bonferroni within an arm), against ``REFERENCE_REPLICATIONS`` sets of
S normal seed means with the observed mean and spread run through the same
comparison. A fixed tolerance would mean nothing at S = 5, where normal seed means
alone give gaps from about -0.09 to +0.21. An arm whose levels all agree keeps the t
test; one that diverges anywhere has the bootstrap rule in force (one-sided 95%
percentile lower bound of the mean above zero), and the report says where and why.
The rule in force changes the verdict only: the curve, the floor and the gap are t-based
whichever rule is in force.
Beside it: the 95% percentile interval of the mean against the t interval, their width
ratio and the ratio normal seed means give, and the bootstrap rule's own size at S by
simulation. At S = 5 the percentile interval is narrower than t by construction, so the
bootstrap cannot by itself vindicate the t test there; agreement says only that the
seed means show nothing normal data would not. Nor can it see much: by simulation at
S = 5 (80 arms of five levels, 1,000 resamples, 300 reference sets, MC se about 0.025)
an arm switched rules at 0.025 of normal seed means, 0.050 of t(3) ones and 0.062 of
lognormal ones. And the rule it switches to is anticonservative there: its size on five
normal seed means is about 0.10 against 0.05 (``bootstrap_rule_size``, in the report).
Each level resamples with its own seed (``level_seed``: the base seed, the arm and the
premium), so the levels' gaps are not correlated by shared resample indices.

**The design's power, measured (S3).** A7 chose S = 5, N = 500 on the pilot's PPO
variances: power ``A7_POWER`` at the seller's pilot effect ``A7_DELTA`` and an exact-t
floor of ``A7_FLOOR`` dollars per episode. Those are estimates from the pilot (its
tau^2 had an se six times itself). ``design_check`` recomputes both at every arm and
premium with the sweep's own variance components, ``power_t(A7_DELTA, tau^2, sigma^2,
S, N)`` and ``detectable_effect_t(tau^2, sigma^2, S, N)``, so the report can say what
power the confirmatory design actually had; 0.898 and 698.3 are the plan, not what the
sweep achieved.

**The JSON sidecar** (``M4Analysis.to_dict``), built for the dashboard. Premiums are in
vol points, increasing; dollar figures are mean net P&L per episode of the contrast;
floats that are not finite are written as null. Change this, the sidecar and the
dashboard together::

    {"schema_version": 1, "milestone": "M4", "report_date": "YYYY-MM-DD",
     "confirmatory": bool, "deviations": [str],
     "rules": {"written", "sweep_ran", "written_after_sweep": true, "statement"},
     "provenance": {"root", "grid_file", "grid_hash", "confirmatory_grid_hash",
                    "analysis_git_sha", "analysis_git_dirty", "sweep_git_shas": [str],
                    "sweep_git_dirty", "cells_total", "cells_used", "versions": {}},
     "design": {"n_seeds", "n_episodes", "blocks": [int], "premiums": [float],
                "arms": [str], "total_timesteps", "first_path_seed",
                "first_train_seed", "costs"},
     "settings": {"alpha", "df", "critical", "target_power", "noncentrality_at_target",
                  "curve_level", "curve_step", "n_boot", "boot_seed",
                  "reference_replications", "agreement_alpha", "homogeneity_alpha"},
     "arm_order": [str],
     "arms": {<arm>: {
        "role": "oracle" | "learned" | "other",
        "decisions": [{"premium", "estimate", "se", "t", "df", "critical",
                       "lower_bound", "detected", "t_interval": [lo, hi],
                       "power_plugin", "seed_means": [float],
                       "components": {"tau2", "sigma2", "se_tau2", "se_sigma2",
                                      "ms_between", "ms_within", "clamped",
                                      "var_grand_mean", "seed_mean_variance"}}],
        "curve": {"points": [{"premium", "measured", "noncentrality",
                              "noncentrality_lo", "noncentrality_hi", "power",
                              "power_lo", "power_hi", "power_plugin"}],
                  "interpolated": [{"premium", "power", "power_lo", "power_hi"}]},
        "floor": {"premium", "lo", "hi", "status": "reached" | "below_grid" |
                  "not_reached", "bracket": [lo, hi], "recrossings": [float],
                  "max_power"},
        "homogeneity": {"contrast" | "arm": {"test", "premiums": [float],
                        "statistic", "df": [between, within], "p_value", "rejected",
                        "levels": [{"premium", "tau2", "sigma2_over_n",
                                    "seed_mean_variance"}]}},
        "bootstrap": {"rule_in_force": "t" | "bootstrap", "diverged_at": [float],
                      "levels": [{"premium", "rate", "analytic", "gap",
                                  "reference": [lo, hi], "p_value", "threshold",
                                  "agrees", "seed", "t_interval": [lo, hi],
                                  "boot_interval": [lo, hi], "width_ratio",
                                  "reference_width_ratio", "t_lower", "boot_lower",
                                  "boot_detected"}]}}},
     "gap": null | {"oracle", "learned", "value", "kind": "point" | "lower_bound" |
                    "undefined", "band_range": [lo, hi], "bracket": [lo, hi]},
     "bootstrap_rule_size": {"n_seeds", "replications", "size", "mc_se"},
     "homogeneity_calibration": [{"groups", "per_group", "replications", "alpha",
                                  "size", "mc_se", "power": [{"ratio", "rate"}]}],
     "design_check": {"planned": {"source": "A7", "delta", "tau2", "sigma2", "n_seeds",
                                  "n_episodes", "power", "floor"},
                      "reference_premium", "arms": {<arm>: [{"premium", "tau2",
                      "sigma2", "power", "floor"}]}}}

The keys after ``bootstrap_rule_size`` were added after schema 1 was first written; a
reader of schema 1 must not require them.
"""

from __future__ import annotations

import json
import math
import warnings
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import levene, nct
from scipy.stats import t as student_t

from optspread.analysis import ALPHA, NULL_PREMIUM, Decision, decide_premium
from optspread.config import ArmName, CostConfig, RunConfig, TrainConfig, config_hash
from optspread.experiment.grid import GridConfig, GridRecord, grid_cells, grid_filename
from optspread.manifest import RunManifest
from optspread.power import (
    VarianceComponents,
    detectable_effect_t,
    power_t,
    t_critical,
    t_noncentrality,
    variance_components,
)
from optspread.runs.rundir import valid_run
from optspread.validation import BootstrapCheck, bootstrap_detection

SCHEMA_VERSION = 1
ORACLE_ARM: ArmName = "delta_seller"
LEARNED_ARM: ArmName = "ppo"

TARGET_POWER = 0.80  # D1: the floor is where the curve reaches 80% power
CURVE_LEVEL = 0.95  # two-sided level of the noncentrality band
CURVE_STEP = 0.05  # vol points between the dashboard's interpolated points
N_BOOT = 10_000  # section 5.3
BOOT_SEED = 0
REFERENCE_REPLICATIONS = 1_000  # normal seed-mean sets behind the agreement tolerance
AGREEMENT_ALPHA = 0.05  # per arm, split over its levels
HOMOGENEITY_ALPHA = 0.05
NONCENTRALITY_CAP = 1e3  # stands in for an infinite t (seed means with no spread)
HOMOGENEITY_RATIOS = (4.0, 10.0, 25.0)  # largest over smallest seed-mean variance

# S1: when the rules below A1 and A2 were written. The report and the sidecar carry it.
RULES_WRITTEN = date(2026, 10, 28)
SWEEP_RAN = date(2026, 10, 26)
RULES_TIMING = (
    "The decisions are A1 and A2's, pre-registered. The rest of this analysis (the "
    "curve's interpolation and its anchor at zero, the floor, the gap, Brown-Forsythe on "
    "seed means, the bootstrap agreement tolerance with its Bonferroni split, and the "
    f"switch to the bootstrap rule) was written on {RULES_WRITTEN.isoformat()}, after the "
    f"sweep ran ({SWEEP_RAN.isoformat()}) and after its raw arm means were posted in issue "
    "#9, and before the analysis code read the sweep's files. These rules are not "
    "pre-registered: they enter the pre-registration only when A9 (issue #37) records "
    "them, and A9 says they were written after the sweep."
)

# A7 (docs/PRE_REGISTRATION.md): the plan the confirmatory design was chosen on. The
# seller's pilot effect at 1 vol point, and PPO's pilot contrast variances.
A7_DELTA = 824.3
A7_TAU2 = 51_838.0
A7_SIGMA2 = 105_996_836.0
A7_POWER = 0.898  # as A7 states them, to three figures and to one decimal
A7_FLOOR = 698.3
A7_PREMIUM = 1.0  # the premium the pilot measured A7_DELTA at

# A7: the confirmatory sweep as `python -m scripts.sweep --seeds 5 --episodes 500
# --timesteps 1000000` builds it (paths from 100,000, training seeds from 1,000).
CONFIRMATORY_GRID = GridConfig(
    base=RunConfig(seed=100_000, n_episodes=500, cost=CostConfig(enabled=True)),
    n_seeds=5,
    train=TrainConfig(total_timesteps=1_000_000, seed=1000),
)
CONFIRMATORY_GRID_HASH = "7d9d53d340006f02"


class RefusedGridError(ValueError):
    """The grid is not one this analysis reads as asked; ``reasons`` says why."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


# -- the noncentral t ------------------------------------------------------- #


def power_at(noncentrality: float, n_seeds: int, alpha: float = ALPHA) -> float:
    """``P(nct(S - 1, lambda) > t_{1-alpha, S-1})``: ``power_t`` written in lambda."""
    if math.isinf(noncentrality):
        return 1.0 if noncentrality > 0.0 else 0.0
    if noncentrality == 0.0:
        return alpha
    return float(nct.sf(t_critical(n_seeds, alpha), n_seeds - 1, noncentrality))


def noncentrality_for_power(target: float, n_seeds: int, alpha: float = ALPHA) -> float:
    """The lambda at which the test's power is ``target``: ``lambda_80`` at 0.80.

    ``power.t_noncentrality``, the planner's own solver, so a curve point the planner
    puts exactly at the target is exactly at it here.
    """
    if not alpha < target < 1.0:
        raise ValueError(f"target power must lie in (alpha, 1), got {target}")
    return t_noncentrality(n_seeds, alpha, target)


def _invert(t_obs: float, df: int, q: float) -> float:
    """The lambda under which ``t_obs`` is the q-th quantile of ``nct(df, lambda)``."""

    def f(x: float) -> float:
        return float(nct.cdf(t_obs, df, x)) - q

    # The cdf falls in lambda. Steps double outward from t_obs; doubling |lambda| itself
    # walks into the region where scipy's nct returns NaN for a large |t|.
    step, lo = 1.0, t_obs - 1.0
    while not f(lo) > 0.0:
        step *= 2.0
        lo = t_obs - step
    step, hi = 1.0, t_obs + 1.0
    while not f(hi) < 0.0:
        step *= 2.0
        hi = t_obs + step
    return float(brentq(f, lo, hi, xtol=1e-10))


def noncentrality_interval(
    t_obs: float, n_seeds: int, level: float = CURVE_LEVEL
) -> tuple[float, float, float]:
    """``(lo, median-unbiased, hi)`` for lambda from a t observed on S - 1 df.

    Exact under normal seed means. An infinite t (seed means with no spread) has no
    interval and is held at ``NONCENTRALITY_CAP`` with its sign.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if math.isinf(t_obs):
        cap = math.copysign(NONCENTRALITY_CAP, t_obs)
        return cap, cap, cap
    df, tail = n_seeds - 1, (1.0 - level) / 2.0
    return _invert(t_obs, df, 1.0 - tail), _invert(t_obs, df, 0.5), _invert(t_obs, df, tail)


def crossing(premiums: Sequence[float], lams: Sequence[float], target: float) -> float | None:
    """The first premium at which the piecewise-linear lambda reaches ``target``, or None."""
    for k in range(1, len(premiums)):
        if lams[k] >= target:
            if lams[k - 1] >= target:
                return premiums[k - 1]
            if lams[k] == target:  # on the knot: no division to round
                return premiums[k]
            frac = (target - lams[k - 1]) / (lams[k] - lams[k - 1])
            return premiums[k - 1] + frac * (premiums[k] - premiums[k - 1])
    return None


# -- one premium -------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PremiumResult:
    """The decision at one premium and the noncentrality it implies."""

    premium: float
    decision: Decision
    seed_means: tuple[float, ...]
    power_plugin: float
    noncentrality: tuple[float, float, float]  # lo, median-unbiased, hi

    @property
    def t_interval(self) -> tuple[float, float]:
        """The two-sided 95% t interval of the contrast's grand mean."""
        half = float(student_t.ppf(0.975, self.decision.df)) * self.decision.se
        return self.decision.estimate - half, self.decision.estimate + half


def premium_result(
    premium: float,
    at_premium: NDArray[np.float64],
    at_null: NDArray[np.float64],
    alpha: float = ALPHA,
) -> PremiumResult:
    decision = decide_premium(at_premium, at_null, alpha)
    vc = decision.components
    return PremiumResult(
        premium=premium,
        decision=decision,
        seed_means=tuple(float(m) for m in (at_premium - at_null).mean(axis=1)),
        power_plugin=power_t(
            decision.estimate, vc.tau2, vc.sigma2, vc.n_seeds, vc.n_episodes, alpha
        ),
        noncentrality=noncentrality_interval(decision.t_stat, vc.n_seeds),
    )


# -- the curve and the floor ------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One knot of the curve. ``measured`` is False only for the anchor at zero."""

    premium: float
    measured: bool
    noncentrality: float
    noncentrality_lo: float
    noncentrality_hi: float
    power: float
    power_lo: float
    power_hi: float
    power_plugin: float | None


def power_curve(
    results: Sequence[PremiumResult], n_seeds: int, alpha: float = ALPHA
) -> tuple[CurvePoint, ...]:
    """The anchor at zero, then one knot per measured premium."""
    points = [CurvePoint(NULL_PREMIUM, False, 0.0, 0.0, 0.0, alpha, alpha, alpha, None)]
    for r in results:
        lo, mid, hi = r.noncentrality
        points.append(
            CurvePoint(
                r.premium,
                True,
                mid,
                lo,
                hi,
                power_at(mid, n_seeds, alpha),
                power_at(lo, n_seeds, alpha),
                power_at(hi, n_seeds, alpha),
                r.power_plugin,
            )
        )
    return tuple(points)


def interpolate(
    curve: Sequence[CurvePoint], n_seeds: int, step: float = CURVE_STEP, alpha: float = ALPHA
) -> list[tuple[float, float, float, float]]:
    """``(premium, power, lo, hi)`` every ``step`` vol points from zero to the last knot."""
    p = [c.premium for c in curve]
    knots = [
        [c.noncentrality for c in curve],
        [c.noncentrality_lo for c in curve],
        [c.noncentrality_hi for c in curve],
    ]
    out = []
    for x in np.round(np.arange(0.0, p[-1] + step / 2.0, step), 10):
        mid, lo, hi = (float(np.interp(x, p, k)) for k in knots)
        out.append(
            (
                float(x),
                power_at(mid, n_seeds, alpha),
                power_at(lo, n_seeds, alpha),
                power_at(hi, n_seeds, alpha),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Floor:
    premium: float | None
    lo: float | None  # where the band's upper edge reaches the target
    hi: float | None  # where its lower edge does; None if that never happens on the grid
    status: str  # reached | below_grid | not_reached
    bracket: tuple[float, float | None]
    recrossings: tuple[float, ...]
    max_power: float


def detection_floor(
    curve: Sequence[CurvePoint], n_seeds: int, target: float = TARGET_POWER, alpha: float = ALPHA
) -> Floor:
    measured = [c for c in curve if c.measured]
    if not measured:
        raise ValueError("a floor needs at least one measured premium")
    p = [c.premium for c in curve]
    lam = noncentrality_for_power(target, n_seeds, alpha)
    at = crossing(p, [c.noncentrality for c in curve], lam)
    lo = crossing(p, [c.noncentrality_hi for c in curve], lam)
    hi = crossing(p, [c.noncentrality_lo for c in curve], lam)
    best = max(c.power for c in measured)
    if at is None:
        return Floor(None, lo, hi, "not_reached", (p[-1], None), (), best)
    k = next(i for i, x in enumerate(p) if x >= at and i > 0)
    status = "below_grid" if at <= measured[0].premium else "reached"
    recross = tuple(c.premium for c in measured if c.premium > at and c.noncentrality < lam)
    return Floor(at, lo, hi, status, (p[k - 1], p[k]), recross, best)


# -- the gap ----------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Gap:
    oracle: ArmName
    learned: ArmName
    value: float | None
    kind: str  # point | lower_bound | undefined
    band_range: tuple[float | None, float | None]
    bracket: tuple[float | None, float | None]


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def gap_between(
    oracle: ArmName, oracle_floor: Floor, learned: ArmName, learned_floor: Floor, last: float
) -> Gap:
    """The learned arm's floor minus the oracle's, at the target power (section 4.4)."""
    o, lf = oracle_floor, learned_floor
    value: float | None
    if o.premium is not None and lf.premium is not None:
        value, kind = lf.premium - o.premium, "point"
    elif o.premium is not None:
        value, kind = last - o.premium, "lower_bound"
    else:
        value, kind = None, "undefined"
    return Gap(
        oracle,
        learned,
        value,
        kind,
        (_minus(lf.lo, o.hi), _minus(lf.hi, o.lo)),
        (_minus(lf.bracket[0], o.bracket[1]), _minus(lf.bracket[1], o.bracket[0])),
    )


# -- homogeneity (#13) ------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class Homogeneity:
    """Brown-Forsythe across premiums on one set of seed means, with each level's split."""

    on: str  # "contrast" or "arm"
    premiums: tuple[float, ...]
    statistic: float | None
    df: tuple[int, int]
    p_value: float | None
    levels: tuple[VarianceComponents, ...]

    @property
    def rejected(self) -> bool | None:
        return None if self.p_value is None else self.p_value < HOMOGENEITY_ALPHA


def brown_forsythe(
    on: str, premiums: Sequence[float], matrices: Sequence[NDArray[np.float64]]
) -> Homogeneity:
    """Levene centred on the median, across the groups of seed means of ``matrices``.

    None for the statistic and p-value when it is undefined: fewer than two groups, or
    no spread about any group's median (scipy then divides zero by zero).
    """
    groups = [m.mean(axis=1) for m in matrices]
    k, n = len(groups), sum(len(g) for g in groups)
    stat: float | None = None
    p: float | None = None
    if k >= 2:
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")
            result = levene(*groups, center="median")
        if math.isfinite(result.statistic) and math.isfinite(result.pvalue):
            stat, p = float(result.statistic), float(result.pvalue)
    return Homogeneity(
        on,
        tuple(premiums),
        stat,
        (k - 1, n - k),
        p,
        tuple(variance_components(m) for m in matrices),
    )


@dataclass(frozen=True, slots=True)
class HomogeneityCalibration:
    """Brown-Forsythe's size and power at ``groups`` groups of ``per_group`` normal seed means."""

    groups: int
    per_group: int
    replications: int
    alpha: float
    size: float
    power: tuple[tuple[float, float], ...]  # (largest over smallest variance, rejection rate)

    @property
    def mc_se(self) -> float:
        q = max(self.size, 1.0 / self.replications)
        return math.sqrt(q * (1.0 - q) / self.replications)


def brown_forsythe_calibration(
    groups: int,
    per_group: int,
    *,
    ratios: Sequence[float] = HOMOGENEITY_RATIOS,
    replications: int = REFERENCE_REPLICATIONS,
    seed: int = BOOT_SEED,
    alpha: float = HOMOGENEITY_ALPHA,
) -> HomogeneityCalibration:
    """How often ``brown_forsythe`` rejects on normal seed means, equal and unequal.

    Under the null every group has variance 1; under each ratio r the variances run
    geometrically from 1 to r across the groups. ``mc_se`` is floored at one rejection,
    so a size of zero still carries its uncertainty.
    """
    if groups < 2 or per_group < 2:
        raise ValueError(f"need at least 2 groups of 2, got {groups} of {per_group}")
    rng = np.random.default_rng((seed, groups, per_group, 54))

    def rate(ratio: float) -> float:
        sd = np.sqrt(np.geomspace(1.0, ratio, groups))[:, None]
        hits = 0
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore")  # two a group: no spread about the median
            for _ in range(replications):
                x = rng.normal(0.0, 1.0, size=(groups, per_group)) * sd
                hits += float(levene(*x, center="median").pvalue) < alpha  # NaN never rejects
        return hits / replications

    size = rate(1.0)
    return HomogeneityCalibration(
        groups, per_group, replications, alpha, size, tuple((r, rate(r)) for r in ratios)
    )


# -- the bootstrap (#14) ------------------------------------------------------ #


def resampled_means(seed_means: NDArray[np.float64], n_boot: int, seed: int) -> NDArray[np.float64]:
    """The bootstrap means of the resamples ``bootstrap_detection`` draws at ``seed``."""
    rng = np.random.default_rng(seed)
    S = int(seed_means.size)
    return np.asarray(seed_means[rng.integers(0, S, size=(n_boot, S))].mean(axis=1))


def level_seed(seed: int, arm: str, premium: float) -> int:
    """The bootstrap seed of one level: the base seed, the arm and the premium, mixed.

    So no two levels or arms share resample indices (which would correlate their gaps).
    """
    words = [seed, zlib.crc32(arm.encode()), round(premium * 1e6)]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


@dataclass(frozen=True, slots=True)
class BootstrapLevel:
    premium: float
    seed: int  # the seed this level resampled with
    check: BootstrapCheck
    reference: tuple[float, float]  # central 95% of the gap under normal seed means
    p_value: float  # of the observed gap against that reference, two-sided
    threshold: float  # AGREEMENT_ALPHA over the arm's levels
    t_interval: tuple[float, float]
    boot_interval: tuple[float, float]
    reference_width_ratio: float  # median width ratio under normal seed means
    t_lower: float
    boot_lower: float

    @property
    def agrees(self) -> bool:
        return self.p_value >= self.threshold

    @property
    def width_ratio(self) -> float:
        return _width_ratio(self.boot_interval, self.t_interval)

    @property
    def boot_detected(self) -> bool:
        return self.boot_lower > 0.0


def _width_ratio(boot: tuple[float, float], t: tuple[float, float]) -> float:
    width = t[1] - t[0]
    return (boot[1] - boot[0]) / width if width > 0.0 else math.nan


def _t_interval(mean: float, se: float, n_seeds: int) -> tuple[float, float]:
    half = float(student_t.ppf(0.975, n_seeds - 1)) * se
    return mean - half, mean + half


def _quantiles(x: NDArray[np.float64], qs: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in np.quantile(x, qs))


def bootstrap_level(
    premium: float,
    seed_means: Sequence[float],
    threshold: float,
    *,
    alpha: float = ALPHA,
    n_boot: int = N_BOOT,
    replications: int = REFERENCE_REPLICATIONS,
    seed: int = BOOT_SEED,
) -> BootstrapLevel:
    """Section 5.3 at one level, graded against normal seed means of the same mean and spread."""
    m = np.asarray(seed_means, dtype=np.float64)
    S = int(m.size)
    check = bootstrap_detection(m, alpha, n_boot, seed)
    means = resampled_means(m, n_boot, seed)
    lo, hi, boot_lower = _quantiles(means, (0.025, 0.975, alpha))
    sd = float(m.std(ddof=1))
    rng = np.random.default_rng((seed, round(premium * 1e6)))
    gaps = np.empty(replications)
    ratios = np.empty(replications)
    for r in range(replications):
        x = rng.normal(check.estimate, sd, size=S)
        r_seed = int(rng.integers(0, 2**32))
        gaps[r] = bootstrap_detection(x, alpha, n_boot, r_seed).gap
        q = _quantiles(resampled_means(x, n_boot, r_seed), (0.025, 0.975))
        se_x = float(x.std(ddof=1)) / math.sqrt(S)
        ratios[r] = _width_ratio((q[0], q[1]), _t_interval(float(x.mean()), se_x, S))
    eps = 1e-12  # rates are multiples of 1 / n_boot; ties count on both sides
    below = (int(np.sum(gaps <= check.gap + eps)) + 1) / (replications + 1)
    above = (int(np.sum(gaps >= check.gap - eps)) + 1) / (replications + 1)
    ref_lo, ref_hi = _quantiles(gaps, (0.025, 0.975))
    finite = ratios[np.isfinite(ratios)]
    return BootstrapLevel(
        premium=premium,
        seed=seed,
        check=check,
        reference=(ref_lo, ref_hi),
        p_value=min(1.0, 2.0 * min(below, above)),
        threshold=threshold,
        t_interval=_t_interval(check.estimate, check.se, S),
        boot_interval=(lo, hi),
        reference_width_ratio=float(np.median(finite)) if finite.size else math.nan,
        t_lower=check.estimate - t_critical(S, alpha) * check.se,
        boot_lower=boot_lower,
    )


@dataclass(frozen=True, slots=True)
class RuleSize:
    """How often the bootstrap rule fires on S normal seed means with nothing there."""

    n_seeds: int
    replications: int
    size: float

    @property
    def mc_se(self) -> float:
        return math.sqrt(self.size * (1.0 - self.size) / self.replications)


def bootstrap_rule_size(
    n_seeds: int,
    *,
    alpha: float = ALPHA,
    n_boot: int = N_BOOT,
    replications: int = REFERENCE_REPLICATIONS,
    seed: int = BOOT_SEED,
) -> RuleSize:
    rng = np.random.default_rng((seed, n_seeds, 53))
    fired = 0
    for _ in range(replications):
        x = rng.normal(0.0, 1.0, size=n_seeds)
        means = resampled_means(x, n_boot, int(rng.integers(0, 2**32)))
        fired += float(np.quantile(means, alpha)) > 0.0
    return RuleSize(n_seeds, replications, fired / replications)


# -- an arm, and the whole analysis ----------------------------------------- #


@dataclass(frozen=True, slots=True)
class ArmAnalysis:
    arm: ArmName
    n_seeds: int
    results: tuple[PremiumResult, ...]
    curve: tuple[CurvePoint, ...]
    floor: Floor
    homogeneity: tuple[Homogeneity, Homogeneity]  # on the contrasts, on the arm itself
    bootstrap: tuple[BootstrapLevel, ...]

    @property
    def role(self) -> str:
        return {ORACLE_ARM: "oracle", LEARNED_ARM: "learned"}.get(self.arm, "other")

    @property
    def diverged_at(self) -> tuple[float, ...]:
        return tuple(b.premium for b in self.bootstrap if not b.agrees)

    @property
    def rule_in_force(self) -> str:
        return "bootstrap" if self.diverged_at else "t"


def analyse_arm(
    arm: ArmName,
    premiums: Sequence[float],
    matrices: dict[float, NDArray[np.float64]],
    *,
    alpha: float = ALPHA,
    n_boot: int = N_BOOT,
    replications: int = REFERENCE_REPLICATIONS,
    seed: int = BOOT_SEED,
) -> ArmAnalysis:
    """Everything M4 reports for one arm, from its ``(S, N)`` matrix at every premium."""
    null = matrices[NULL_PREMIUM]
    n_seeds = int(null.shape[0])
    positive = [p for p in sorted(premiums) if p > NULL_PREMIUM]
    results = tuple(premium_result(p, matrices[p], null, alpha) for p in positive)
    curve = power_curve(results, n_seeds, alpha)
    threshold = AGREEMENT_ALPHA / len(results)
    return ArmAnalysis(
        arm=arm,
        n_seeds=n_seeds,
        results=results,
        curve=curve,
        floor=detection_floor(curve, n_seeds, TARGET_POWER, alpha),
        homogeneity=(
            brown_forsythe("contrast", positive, [matrices[p] - null for p in positive]),
            brown_forsythe("arm", sorted(premiums), [matrices[p] for p in sorted(premiums)]),
        ),
        bootstrap=tuple(
            bootstrap_level(
                r.premium,
                r.seed_means,
                threshold,
                alpha=alpha,
                n_boot=n_boot,
                replications=replications,
                seed=level_seed(seed, arm, r.premium),
            )
            for r in results
        ),
    )


@dataclass(frozen=True, slots=True)
class DesignPower:
    """The design's power and floor at one arm and premium, on the sweep's own variances."""

    premium: float
    tau2: float
    sigma2: float
    power: float  # power_t at A7_DELTA
    floor: float  # detectable_effect_t, dollars per episode


def planned_design(n_seeds: int, n_episodes: int, alpha: float = ALPHA) -> DesignPower:
    """A7's plan: PPO's pilot variances at the seller's pilot effect."""
    return DesignPower(
        A7_PREMIUM,
        A7_TAU2,
        A7_SIGMA2,
        power_t(A7_DELTA, A7_TAU2, A7_SIGMA2, n_seeds, n_episodes, alpha),
        detectable_effect_t(A7_TAU2, A7_SIGMA2, n_seeds, n_episodes, alpha, TARGET_POWER),
    )


def design_power(
    arm: ArmAnalysis, n_episodes: int, alpha: float = ALPHA
) -> tuple[DesignPower, ...]:
    """At every premium of ``arm``, A7's power and floor with its contrast's own variances."""
    out = []
    for r in arm.results:
        vc = r.decision.components
        out.append(
            DesignPower(
                r.premium,
                vc.tau2,
                vc.sigma2,
                power_t(A7_DELTA, vc.tau2, vc.sigma2, arm.n_seeds, n_episodes, alpha),
                detectable_effect_t(
                    vc.tau2, vc.sigma2, arm.n_seeds, n_episodes, alpha, TARGET_POWER
                ),
            )
        )
    return tuple(out)


def design_problems(record: GridRecord) -> tuple[list[str], list[str]]:
    """``(refused whatever the flag, refused unless exploratory)`` for a grid record."""
    hard: list[str] = []
    soft: list[str] = []
    grid = record.grid
    own = config_hash(grid)
    if record.grid_hash != own:
        hard.append(f"the record's grid hash {record.grid_hash} is not its config's ({own})")
    if NULL_PREMIUM not in grid.premiums:
        hard.append(f"the grid has no {NULL_PREMIUM:g} premium for the contrast (A1)")
    if not any(p > NULL_PREMIUM for p in grid.premiums):
        hard.append("the grid has no positive premium to test")
    expected = {(c.premium, c.arm, c.block): c.run_id for c in grid_cells(grid)}
    listed = {(c.premium, c.arm, c.block): c.run_id for c in record.cells}
    if listed != expected:
        hard.append("the record does not list exactly its grid's cells")
    if own != CONFIRMATORY_GRID_HASH:
        soft.append(
            f"not the confirmatory design: grid {own}, the A7 design is "
            f"{CONFIRMATORY_GRID_HASH} ({grid_filename(CONFIRMATORY_GRID)})"
        )
    return hard, soft


def invalid_cells(record: GridRecord, root: Path) -> list[tuple[int, str]]:
    """``(block, label)`` of every cell not valid under ``root`` (``sweep.valid_run``)."""
    return [
        (c.block, f"{c.label.strip()} ({c.run_id})")
        for c in grid_cells(record.grid)
        if valid_run(root / c.run_id, c.config) is None
    ]


def _blocks_only(record: GridRecord, blocks: Sequence[int]) -> GridRecord:
    """The record restricted to ``blocks``, renumbered 0.. so ``GridRecord.matrix`` reads it."""
    renumber = {b: i for i, b in enumerate(blocks)}
    cells = tuple(
        c.model_copy(update={"block": renumber[c.block]})
        for c in record.cells
        if c.block in renumber
    )
    grid = record.grid.model_copy(update={"n_seeds": len(blocks)})
    return record.model_copy(update={"grid": grid, "cells": cells})


@dataclass(frozen=True, slots=True)
class Provenance:
    """What the outputs name besides the numbers: when, from where, at which commit."""

    report_date: date
    root: str  # portable: $VEGALAB_DATA or ~, never one machine's home
    grid_file: str
    analysis_git_sha: str
    analysis_git_dirty: bool
    versions: dict[str, str]


@dataclass(frozen=True, slots=True)
class M4Analysis:
    record: GridRecord
    deviations: tuple[str, ...]
    blocks: tuple[int, ...]
    arms: tuple[ArmAnalysis, ...]
    gap: Gap | None
    rule_size: RuleSize
    homogeneity_calibration: tuple[HomogeneityCalibration, ...]
    sweep_git_shas: tuple[str, ...]
    sweep_git_dirty: bool
    n_boot: int
    replications: int
    seed: int
    alpha: float

    @property
    def confirmatory(self) -> bool:
        return not self.deviations

    @property
    def n_seeds(self) -> int:
        return len(self.blocks)

    @property
    def n_episodes(self) -> int:
        return self.record.grid.base.n_episodes

    @property
    def lambda_target(self) -> float:
        return noncentrality_for_power(TARGET_POWER, self.n_seeds, self.alpha)

    def arm(self, name: ArmName) -> ArmAnalysis:
        return next(a for a in self.arms if a.arm == name)

    @property
    def planned(self) -> DesignPower:
        return planned_design(self.n_seeds, self.n_episodes, self.alpha)

    def design_power(self, arm: ArmAnalysis) -> tuple[DesignPower, ...]:
        return design_power(arm, self.n_episodes, self.alpha)

    def to_dict(self, provenance: Provenance) -> dict[str, Any]:
        """The sidecar; the schema is in the module docstring."""
        grid = self.record.grid
        out = _finite(
            {
                "schema_version": SCHEMA_VERSION,
                "milestone": "M4",
                "report_date": provenance.report_date.isoformat(),
                "confirmatory": self.confirmatory,
                "deviations": list(self.deviations),
                "rules": {
                    "written": RULES_WRITTEN.isoformat(),
                    "sweep_ran": SWEEP_RAN.isoformat(),
                    "written_after_sweep": True,
                    "statement": RULES_TIMING,
                },
                "provenance": {
                    "root": provenance.root,
                    "grid_file": provenance.grid_file,
                    "grid_hash": self.record.grid_hash,
                    "confirmatory_grid_hash": CONFIRMATORY_GRID_HASH,
                    "analysis_git_sha": provenance.analysis_git_sha,
                    "analysis_git_dirty": provenance.analysis_git_dirty,
                    "sweep_git_shas": list(self.sweep_git_shas),
                    "sweep_git_dirty": self.sweep_git_dirty,
                    "cells_total": len(self.record.cells),
                    "cells_used": self.n_seeds * len(grid.premiums) * len(grid.arms),
                    "versions": dict(provenance.versions),
                },
                "design": {
                    "n_seeds": self.n_seeds,
                    "n_episodes": self.n_episodes,
                    "blocks": list(self.blocks),
                    "premiums": sorted(grid.premiums),
                    "arms": list(grid.arms),
                    "total_timesteps": None if grid.train is None else grid.train.total_timesteps,
                    "first_path_seed": grid.base.seed,
                    "first_train_seed": None if grid.train is None else grid.train.seed,
                    "costs": grid.base.cost.enabled,
                },
                "settings": {
                    "alpha": self.alpha,
                    "df": self.n_seeds - 1,
                    "critical": t_critical(self.n_seeds, self.alpha),
                    "target_power": TARGET_POWER,
                    "noncentrality_at_target": self.lambda_target,
                    "curve_level": CURVE_LEVEL,
                    "curve_step": CURVE_STEP,
                    "n_boot": self.n_boot,
                    "boot_seed": self.seed,
                    "reference_replications": self.replications,
                    "agreement_alpha": AGREEMENT_ALPHA,
                    "homogeneity_alpha": HOMOGENEITY_ALPHA,
                },
                "arm_order": [a.arm for a in self.arms],
                "arms": {a.arm: _arm_dict(a, self.alpha) for a in self.arms},
                "gap": None if self.gap is None else _gap_dict(self.gap),
                "bootstrap_rule_size": {
                    "n_seeds": self.rule_size.n_seeds,
                    "replications": self.rule_size.replications,
                    "size": self.rule_size.size,
                    "mc_se": self.rule_size.mc_se,
                },
                "homogeneity_calibration": [
                    {
                        "groups": c.groups,
                        "per_group": c.per_group,
                        "replications": c.replications,
                        "alpha": c.alpha,
                        "size": c.size,
                        "mc_se": c.mc_se,
                        "power": [{"ratio": r, "rate": w} for r, w in c.power],
                    }
                    for c in self.homogeneity_calibration
                ],
                "design_check": {
                    "planned": {
                        "source": "A7",
                        "delta": A7_DELTA,
                        "tau2": A7_TAU2,
                        "sigma2": A7_SIGMA2,
                        "n_seeds": self.n_seeds,
                        "n_episodes": self.n_episodes,
                        "power": self.planned.power,
                        "floor": self.planned.floor,
                    },
                    "reference_premium": A7_PREMIUM,
                    "arms": {
                        a.arm: [
                            {
                                "premium": d.premium,
                                "tau2": d.tau2,
                                "sigma2": d.sigma2,
                                "power": d.power,
                                "floor": d.floor,
                            }
                            for d in self.design_power(a)
                        ]
                        for a in self.arms
                    },
                },
            }
        )
        assert isinstance(out, dict)
        return out

    def render_json(self, provenance: Provenance) -> str:
        return json.dumps(self.to_dict(provenance), indent=2, allow_nan=False) + "\n"

    def render(self, provenance: Provenance) -> str:
        return _render(self, provenance)


def analyse(
    record: GridRecord,
    root: Path,
    *,
    exploratory: bool = False,
    alpha: float = ALPHA,
    n_boot: int = N_BOOT,
    replications: int = REFERENCE_REPLICATIONS,
    seed: int = BOOT_SEED,
) -> M4Analysis:
    """The M4 analysis of the grid ``record`` lists, its cells read from ``root``.

    Raises ``RefusedGridError`` for a grid that cannot be analysed at all, and for one that
    is not the confirmatory design or has a missing or invalid cell unless
    ``exploratory``; an exploratory analysis drops every block with a bad cell and
    records each deviation.
    """
    hard, deviations = design_problems(record)
    if hard:
        raise RefusedGridError(hard)
    grid = record.grid
    bad = invalid_cells(record, root)
    blocks = tuple(b for b in range(grid.n_seeds) if b not in {blk for blk, _ in bad})
    if bad:
        shown = "; ".join(label for _, label in bad[:4]) + ("; ..." if len(bad) > 4 else "")
        deviations.append(
            f"incomplete: {len(bad)} of {len(record.cells)} cells missing or invalid under "
            f"the root ({shown}); analysed on blocks {list(blocks)} of {grid.n_seeds}"
        )
    if deviations and not exploratory:
        raise RefusedGridError(deviations)
    if len(blocks) < 2:
        raise RefusedGridError([f"{len(blocks)} complete block(s); the test needs at least 2"])
    used = record if len(blocks) == grid.n_seeds else _blocks_only(record, blocks)

    premiums = sorted(grid.premiums)
    arms = tuple(
        analyse_arm(
            arm,
            premiums,
            {p: used.matrix(root, p, arm) for p in premiums},
            alpha=alpha,
            n_boot=n_boot,
            replications=replications,
            seed=seed,
        )
        for arm in grid.arms
    )
    gap = None
    names = [a.arm for a in arms]
    if ORACLE_ARM in names and LEARNED_ARM in names:
        o, lf = arms[names.index(ORACLE_ARM)], arms[names.index(LEARNED_ARM)]
        gap = gap_between(o.arm, o.floor, lf.arm, lf.floor, premiums[-1])
    manifests = [RunManifest.read(root / c.run_id) for c in used.cells]
    return M4Analysis(
        record=record,
        deviations=tuple(deviations),
        blocks=blocks,
        arms=arms,
        gap=gap,
        rule_size=bootstrap_rule_size(
            len(blocks), alpha=alpha, n_boot=n_boot, replications=replications, seed=seed
        ),
        homogeneity_calibration=tuple(
            brown_forsythe_calibration(k, len(blocks), replications=replications, seed=seed)
            for k in sorted({len(premiums) - 1, len(premiums)})
        ),
        sweep_git_shas=tuple(sorted({m.git_sha for m in manifests})),
        sweep_git_dirty=any(m.git_dirty for m in manifests),
        n_boot=n_boot,
        replications=replications,
        seed=seed,
        alpha=alpha,
    )


# -- the sidecar -------------------------------------------------------------- #


def _finite(value: Any) -> Any:
    """Every float that is not finite becomes None, so the sidecar is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _components(vc: VarianceComponents) -> dict[str, Any]:
    return {
        "tau2": vc.tau2,
        "sigma2": vc.sigma2,
        "se_tau2": vc.se_tau2,
        "se_sigma2": vc.se_sigma2,
        "ms_between": vc.ms_between,
        "ms_within": vc.ms_within,
        "clamped": vc.clamped,
        "var_grand_mean": vc.var_grand_mean,
        "seed_mean_variance": vc.seed_mean_variance,
    }


def _homogeneity_dict(h: Homogeneity) -> dict[str, Any]:
    return {
        "test": "brown-forsythe (levene, median-centred) on seed means",
        "premiums": list(h.premiums),
        "statistic": h.statistic,
        "df": list(h.df),
        "p_value": h.p_value,
        "rejected": h.rejected,
        "levels": [
            {
                "premium": p,
                "tau2": vc.tau2,
                "sigma2_over_n": vc.sigma2 / vc.n_episodes,
                "seed_mean_variance": vc.seed_mean_variance,
            }
            for p, vc in zip(h.premiums, h.levels, strict=True)
        ],
    }


def _arm_dict(a: ArmAnalysis, alpha: float) -> dict[str, Any]:
    return {
        "role": a.role,
        "decisions": [
            {
                "premium": r.premium,
                "estimate": r.decision.estimate,
                "se": r.decision.se,
                "t": r.decision.t_stat,
                "df": r.decision.df,
                "critical": r.decision.critical,
                "lower_bound": r.decision.lower_bound,
                "detected": r.decision.rejected,
                "t_interval": list(r.t_interval),
                "power_plugin": r.power_plugin,
                "seed_means": list(r.seed_means),
                "components": _components(r.decision.components),
            }
            for r in a.results
        ],
        "curve": {
            "points": [
                {
                    "premium": c.premium,
                    "measured": c.measured,
                    "noncentrality": c.noncentrality,
                    "noncentrality_lo": c.noncentrality_lo,
                    "noncentrality_hi": c.noncentrality_hi,
                    "power": c.power,
                    "power_lo": c.power_lo,
                    "power_hi": c.power_hi,
                    "power_plugin": c.power_plugin,
                }
                for c in a.curve
            ],
            "interpolated": [
                {"premium": p, "power": w, "power_lo": lo, "power_hi": hi}
                for p, w, lo, hi in interpolate(a.curve, a.n_seeds, CURVE_STEP, alpha)
            ],
        },
        "floor": {
            "premium": a.floor.premium,
            "lo": a.floor.lo,
            "hi": a.floor.hi,
            "status": a.floor.status,
            "bracket": list(a.floor.bracket),
            "recrossings": list(a.floor.recrossings),
            "max_power": a.floor.max_power,
        },
        "homogeneity": {h.on: _homogeneity_dict(h) for h in a.homogeneity},
        "bootstrap": {
            "rule_in_force": a.rule_in_force,
            "diverged_at": list(a.diverged_at),
            "levels": [
                {
                    "premium": b.premium,
                    "rate": b.check.rate,
                    "analytic": b.check.analytic,
                    "gap": b.check.gap,
                    "reference": list(b.reference),
                    "p_value": b.p_value,
                    "threshold": b.threshold,
                    "agrees": b.agrees,
                    "seed": b.seed,
                    "t_interval": list(b.t_interval),
                    "boot_interval": list(b.boot_interval),
                    "width_ratio": b.width_ratio,
                    "reference_width_ratio": b.reference_width_ratio,
                    "t_lower": b.t_lower,
                    "boot_lower": b.boot_lower,
                    "boot_detected": b.boot_detected,
                }
                for b in a.bootstrap
            ],
        },
    }


def _gap_dict(g: Gap) -> dict[str, Any]:
    return {
        "oracle": g.oracle,
        "learned": g.learned,
        "value": g.value,
        "kind": g.kind,
        "band_range": list(g.band_range),
        "bracket": list(g.bracket),
    }


# -- the Markdown ------------------------------------------------------------- #


def _vp(x: float | None) -> str:
    return "--" if x is None else f"{x:.3g}"


def _usd(x: float) -> str:
    return f"{x:,.0f}" if math.isfinite(x) else str(x)


def _pw(x: float | None) -> str:
    return "--" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _floor_line(a: ArmAnalysis) -> str:
    f = a.floor
    if f.premium is None:
        return (
            f"- **{a.arm}**: not reached on the grid; the floor is above {f.bracket[0]:g} vol "
            f"points (largest power {f.max_power:.3f}"
            + ("" if f.lo is None else f"; the band's upper edge reaches it at {f.lo:.3g}")
            + ")."
        )
    where = {
        "reached": f"between the sweep points {f.bracket[0]:g} and {f.bracket[1]:g}",
        "below_grid": (
            f"below the first sweep point ({f.bracket[1]:g}), so it rests on the "
            "interpolation from the anchor at zero"
        ),
    }[f.status]
    if f.status == "below_grid":
        per = ", ".join(f"{c.noncentrality / c.premium:.3g}" for c in a.curve if c.measured)
        where += (
            f"; the measured noncentrality per vol point is {per} at "
            + ", ".join(f"{c.premium:g}" for c in a.curve if c.measured)
            + ", and the floor holds the first of these down to zero"
        )
    band = f"band {_vp(f.lo)} to {_vp(f.hi) if f.hi is not None else 'above the grid'}"
    recross = (
        f"; the curve falls back under {TARGET_POWER:.0%} at "
        + ", ".join(f"{p:g}" for p in f.recrossings)
        if f.recrossings
        else ""
    )
    return f"- **{a.arm}**: {f.premium:.3g} vol points ({band}), {where}{recross}."


def _gap_line(g: Gap | None, oracle_status: str | None = None) -> str:
    if g is None:
        return f"The gap needs both {ORACLE_ARM} and {LEARNED_ARM} in the grid; it has not."
    ranges = (
        f" Band range {_vp(g.band_range[0])} to {_vp(g.band_range[1])}; between sweep points "
        f"{_vp(g.bracket[0])} to {_vp(g.bracket[1])} (ranges, not confidence intervals)."
    )
    if oracle_status == "below_grid":
        ranges += (
            f" The {g.oracle} floor lies below the first sweep point, so the gap inherits its "
            "interpolation from the anchor at zero."
        )
    if g.kind == "point" and g.value is not None:
        return (
            f"**Gap** ({g.learned} floor minus {g.oracle} floor): {g.value:.3g} vol points."
            + ranges
        )
    if g.kind == "lower_bound" and g.value is not None:
        return (
            f"**Gap**: more than {g.value:.3g} vol points; {g.learned} does not reach "
            f"{TARGET_POWER:.0%} on the grid, so only a lower bound is measured." + ranges
        )
    return f"**Gap**: undefined; {g.oracle} does not reach {TARGET_POWER:.0%} on the grid."


def _render(m: M4Analysis, prov: Provenance) -> str:
    grid = m.record.grid
    S, N = m.n_seeds, m.n_episodes
    steps = (
        "scripted only" if grid.train is None else f"{grid.train.total_timesteps:,} training steps"
    )
    shas = ", ".join(s[:8] for s in m.sweep_git_shas) + (" (dirty)" if m.sweep_git_dirty else "")
    head = [
        "# M4 analysis: the power curve, the detection floor and the gap (#12, #13, #14)",
        "",
        f"Run {prov.report_date.isoformat()} by `python -m scripts.m4_analysis`; "
        "`optspread/m4analysis.py` defines every number. Synthetic, costs "
        f"{'on' if grid.base.cost.enabled else 'off'}.",
        "",
        f"Grid `{prov.grid_file}` (hash {m.record.grid_hash}) under `{prov.root}`: S = {S} "
        f"blocks x N = {N} episodes, premiums {', '.join(f'{p:g}' for p in sorted(grid.premiums))} "
        f"vol points, arms {', '.join(grid.arms)}, {steps}. Analysis code "
        f"{prov.analysis_git_sha[:8]}{' (dirty)' if prov.analysis_git_dirty else ''}; "
        f"sweep cells written at {shas}.",
        "",
    ]
    if m.confirmatory:
        head += [
            f"**Confirmatory**: the A7 design (grid {CONFIRMATORY_GRID_HASH}), every cell valid.",
            "",
        ]
    else:
        head += ["**EXPLORATORY: not the confirmatory analysis.**", ""]
        head += [f"- {d}" for d in m.deviations] + [""]
    head += [f"**When these rules were written.** {RULES_TIMING}", ""]

    crit = t_critical(S, m.alpha)
    dec = [
        "## The decisions (A1, A2)",
        "",
        f"Each arm at p against itself at zero, one-sided t on the {S} seed means, "
        f"{S - 1} df, critical {crit:.3f}. Dollars per episode.",
        "",
        "| arm | premium | estimate | se | t | 95% lower bound | detected | tau^2 | sigma^2 "
        "| power (plug-in) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        for r in a.results:
            d, vc = r.decision, r.decision.components
            clamp = " (clamped)" if vc.clamped else ""
            dec.append(
                f"| {a.arm} | {r.premium:g} | {_usd(d.estimate)} | {_usd(d.se)} | {d.t_stat:.2f} "
                f"| {_usd(d.lower_bound)} | {'yes' if d.rejected else 'no'} "
                f"| {vc.tau2:,.4g}{clamp} | {vc.sigma2:,.4g} | {r.power_plugin:.3f} |"
            )

    curve = [
        "",
        "## The power curve (#12)",
        "",
        f"Power of a fresh {S} x {N} experiment at each premium: the noncentral t at the "
        f"median-unbiased noncentrality from the observed t, with the exact "
        f"{CURVE_LEVEL:.0%} band; linear in noncentrality between sweep points, from "
        f"power {m.alpha:g} at zero. Power reaches {TARGET_POWER:.0%} at noncentrality "
        f"{m.lambda_target:.3f}.",
        "",
        "| arm | premium | noncentrality (band) | power | band | plug-in |",
        "|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        for c in a.curve[1:]:
            curve.append(
                f"| {a.arm} | {c.premium:g} | {c.noncentrality:.2f} ({c.noncentrality_lo:.2f} to "
                f"{c.noncentrality_hi:.2f}) | {c.power:.3f} | {c.power_lo:.3f} to "
                f"{c.power_hi:.3f} | {_pw(c.power_plugin)} |"
            )

    floors = [
        "",
        "## The detection floor and the gap (#12)",
        "",
        f"The smallest premium at which the curve reaches {TARGET_POWER:.0%}, in vol points:",
        "",
        *(_floor_line(a) for a in m.arms),
        "",
        "A floor taken where a noisy curve first crosses 80% comes early: on linear truths "
        "at S = 5, simulated, its median is 8 to 11% below the true floor and it lies below "
        "it in 57 to 61% of draws, so read each floor as about a tenth low; its band [lo, hi] "
        "covered the true floor in 97 to 99% of draws.",
        "",
        _gap_line(
            m.gap,
            next((a.floor.status for a in m.arms if m.gap and a.arm == m.gap.oracle), None),
        ),
    ]

    plan = m.planned
    design = [
        "",
        "## The power the design had (A7's plan against the sweep's variances)",
        "",
        f"**Planned (A7, on the pilot's PPO variances tau^2 = {A7_TAU2:,.0f}, sigma^2 = "
        f"{A7_SIGMA2:,.0f}):** power {plan.power:.3f} at delta = {A7_DELTA:g} (the seller's "
        f"pilot effect at {A7_PREMIUM:g} vol point), exact-t floor {plan.floor:,.1f} dollars "
        f"per episode, at S = {S} x N = {N}. These are pilot estimates, not what the sweep "
        "achieved.",
        "",
        f"**Measured (the sweep's own variances):** the same S = {S} x N = {N} design's power "
        f"at delta = {A7_DELTA:g} and its exact-t floor, with each contrast's tau^2 and "
        "sigma^2 from the sweep.",
        "",
        f"| arm | premium | tau^2 | sigma^2 | power at delta = {A7_DELTA:g} | floor (dollars) |",
        "|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        for dp in m.design_power(a):
            mark = " (A6's premium)" if dp.premium == A7_PREMIUM else ""
            design.append(
                f"| {a.arm} | {dp.premium:g}{mark} | {dp.tau2:,.4g} | {dp.sigma2:,.4g} "
                f"| {dp.power:.3f} | {dp.floor:,.0f} |"
            )
    if LEARNED_ARM in [a.arm for a in m.arms]:
        ref = [x for x in m.design_power(m.arm(LEARNED_ARM)) if x.premium == A7_PREMIUM]
        if ref:
            design += [
                "",
                f"At {A7_PREMIUM:g} vol point the {LEARNED_ARM} contrast's measured variances give "
                f"the design power {ref[0].power:.3f} and a floor of {ref[0].floor:,.0f} dollars, "
                f"against the planned {plan.power:.3f} and {plan.floor:,.1f}: quote these, not "
                "the plan, as the power the confirmatory design had.",
            ]

    homog = [
        "",
        "## Is tau^2 the same across the sweep? (#13)",
        "",
        "Brown-Forsythe across premiums on seed means, alpha "
        f"{HOMOGENEITY_ALPHA:g}: on each contrast's seed means (p > 0), and on the arm's own "
        "seed means (every premium). It tests the variance of seed means, tau^2 + sigma^2 / N; "
        "the split is below. Five means a group: a rejection is a finding, a non-rejection "
        "is weak evidence.",
        "",
        "| arm | seed means of | W | df | p | equal variance rejected |",
        "|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        for h in a.homogeneity:
            verdict = {None: "undefined", True: "yes", False: "no"}[h.rejected]
            stat = "--" if h.statistic is None else f"{h.statistic:.3f}"
            homog.append(
                f"| {a.arm} | {'the contrast' if h.on == 'contrast' else 'the arm'} | {stat} "
                f"| {h.df[0]}, {h.df[1]} | {_pw(h.p_value)} | {verdict} |"
            )
    homog += [""]
    for cal in m.homogeneity_calibration:
        rates = " / ".join(f"{w:.3f}" for _, w in cal.power)
        ratios = " / ".join(f"{r:g}" for r, _ in cal.power)
        homog.append(
            f"- **The test's own size and power at {cal.groups} groups of {cal.per_group}** "
            f"(normal seed means, {cal.replications:,} replications): size {cal.size:.3f} (MC "
            f"se {cal.mc_se:.3f}) at alpha {cal.alpha:g}; power {rates} when the largest "
            "variance is "
            f"{ratios} times the smallest, spread geometrically."
        )
    for a in m.arms:
        v = [vc.seed_mean_variance for vc in a.homogeneity[0].levels]
        if v and min(v) > 0.0:
            homog.append(
                f"- **{a.arm}**: its contrasts' seed-mean variance spans "
                f"{max(v) / min(v):,.3g}-fold across the premiums."
            )
    homog += [
        "",
        "So a non-rejection here does not show that tau^2 is the same across premiums.",
        "",
        "| arm | premium | contrast tau^2 | contrast sigma^2 / N | arm tau^2 | arm sigma^2 / N |",
        "|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        contrast, own = a.homogeneity
        by_p = dict(zip(contrast.premiums, contrast.levels, strict=True))
        for p, vc in zip(own.premiums, own.levels, strict=True):
            cv = by_p.get(p)
            cells = (
                "-- | --" if cv is None else f"{cv.tau2:,.4g} | {cv.sigma2 / cv.n_episodes:,.4g}"
            )
            homog.append(
                f"| {a.arm} | {p:g} | {cells} | {vc.tau2:,.4g} | {vc.sigma2 / vc.n_episodes:,.4g} |"
            )

    boot = [
        "",
        "## Bootstrap validation (#14)",
        "",
        f"{m.n_boot:,} resamples of the {S} seed means at each level; the t test's detection "
        "rate on them against the analytic rate (`bootstrap_detection`'s: the power with the "
        "observed t taken as the noncentrality). The tolerance (written after the sweep; see "
        "the top): a level agrees when its gap lies inside what normal seed "
        f"means give, two-sided reference p at least {AGREEMENT_ALPHA:g} over the arm's "
        f"levels ({m.replications:,} normal replications).",
        "",
        "| arm | premium | bootstrap | analytic | gap | normal 95% range | p | agrees "
        "| t 95% interval | bootstrap 95% interval | width ratio (normal) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        for b in a.bootstrap:
            boot.append(
                f"| {a.arm} | {b.premium:g} | {b.check.rate:.3f} | {b.check.analytic:.3f} "
                f"| {b.check.gap:+.3f} | {b.reference[0]:+.3f} to {b.reference[1]:+.3f} "
                f"| {b.p_value:.3f} | {'yes' if b.agrees else 'no'} "
                f"| {_usd(b.t_interval[0])} to {_usd(b.t_interval[1])} "
                f"| {_usd(b.boot_interval[0])} to {_usd(b.boot_interval[1])} "
                f"| {_pw(b.width_ratio)} ({_pw(b.reference_width_ratio)}) |"
            )
    boot.append("")
    for a in m.arms:
        if a.rule_in_force == "t":
            boot.append(f"- **{a.arm}**: every level agrees; the t test stands.")
        else:
            at = ", ".join(f"{p:g}" for p in a.diverged_at)
            fired = ", ".join(
                f"{b.premium:g}: {'detected' if b.boot_detected else 'not detected'}"
                for b in a.bootstrap
            )
            boot.append(
                f"- **{a.arm}**: diverges at {at} vol points, more than normal seed means "
                "explain, so the bootstrap rule (one-sided 95% percentile lower bound above "
                f"zero) is in force for this arm (section 5.3). Under it: {fired}."
            )
    rs = m.rule_size
    boot += [
        "",
        f"At S = {S} the percentile interval is narrower than t by construction (the "
        "normal ratio above), and the bootstrap rule itself fires at "
        f"{rs.size:.3f} (MC se {rs.mc_se:.3f}) on {S} normal seed means with nothing "
        f"planted, against {m.alpha:g}. Agreement says the seed means show nothing normal "
        "data would not; at this S it cannot say more.",
        "",
        "Its limits at this S. The check barely sees non-normal seed means: by simulation at "
        "S = 5 (80 arms of five levels, 1,000 resamples, 300 reference sets, MC se about "
        "0.025) an arm switched rules at 0.025 of normal seed means, 0.050 of t(3) ones and "
        "0.062 of lognormal ones. The rule it switches to is anticonservative, about 0.10 "
        "against 0.05 (above). And a switch changes the verdict only: the curve, the floor "
        "and the gap above are t-based whichever rule is in force.",
        "",
    ]
    return "\n".join([*head, *dec, *curve, *floors, *design, *homog, *boot])
