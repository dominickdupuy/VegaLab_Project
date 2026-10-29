"""Variance components and the power arithmetic behind the M2 sample size.

Lane: Dominick Dupuy (statistical spine).

Pre-registration D1 fixes the decision rule: the primary contrast is tested once,
one-sided at alpha = 0.05, with the sample size set in advance at 80% power from

    Var(grand mean) = tau^2 / S + sigma^2 / (S * N)

S is the number of seeds (independently trained policies), N the number of
evaluation episodes per seed, tau^2 the between-seed variance of a seed's mean
outcome and sigma^2 the within-seed, across-episode variance. The pilot measures
tau^2 and sigma^2; this module turns an S x N matrix of outcomes into those two
numbers, and the numbers into a sample size and a compute budget. Nothing here
decides what the pre-registration left open: the effect size is an input, and a
design the components cannot support is reported as infeasible rather than rounded
to a number.

The estimator is the balanced one-way random-effects ANOVA (method of moments):

    MS_within  = sum_ij (y_ij - ybar_i)^2 / (S (N - 1))     E[MS_within]  = sigma^2
    MS_between = N sum_i (ybar_i - ybar)^2 / (S - 1)        E[MS_between] = sigma^2 + N tau^2

so sigma^2 = MS_within and tau^2 = max(0, (MS_between - MS_within) / N). The clamp
is recorded: a clamped tau^2 means the pilot saw no seed effect at all, which is a
result in its own right.

The design is nested: each seed block is evaluated on its own paths, so the e_ij
are independent across blocks. If every policy were evaluated on the same paired
paths the design would be crossed and carry a shared-path term the formula above
does not; that is a pre-registration question for the lane owner, not something
this module settles.

Two power functions, kept apart on purpose. ``power_t`` is the exact detection
probability of the confirmatory test in ``optspread.analysis``, a one-sided Student t
on the S seed-level means: under normal seed means its statistic is noncentral t
with S - 1 degrees of freedom, so the probability is a closed form rather than an
approximation. The planning functions (``episodes_for``, ``seeds_for``, ``frontier``,
``detectable_effect_t``, ``variance_budget_t``) are built on it, so a planned design
has the power the committed test will have (#20). ``power``, ``detectable_effect``
and ``variance_budget`` are the textbook normal approximation, kept for orientation:
it ignores that the test estimates its variance from S seed means, and at the seed
counts published work uses that matters. At S = 5 the normal floor is 18% below the
exact one and the test has 66% power there; at S = 3 the gap is 38%. The section 8
gate in ``optspread.validation`` checks the test against ``power_t`` and reports
``power`` beside it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import nct, norm
from scipy.stats import t as student_t


def _check_variances(tau2: float, sigma2: float) -> None:
    if tau2 < 0.0 or sigma2 < 0.0:
        raise ValueError(f"variance components cannot be negative: tau2={tau2}, sigma2={sigma2}")


def _check_design(n_seeds: int, n_episodes: int) -> None:
    if n_seeds < 1 or n_episodes < 1:
        raise ValueError(
            f"a design needs at least one seed and one episode, got {n_seeds} x {n_episodes}"
        )


def var_grand_mean(tau2: float, sigma2: float, n_seeds: int, n_episodes: int) -> float:
    """The pre-registered formula: ``tau^2 / S + sigma^2 / (S * N)``."""
    _check_variances(tau2, sigma2)
    _check_design(n_seeds, n_episodes)
    return tau2 / n_seeds + sigma2 / (n_seeds * n_episodes)


# -- the estimator ---------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VarianceComponents:
    """What the pilot measures. ``clamped`` records that MS_between fell below MS_within."""

    n_seeds: int
    n_episodes: int
    grand_mean: float
    ms_between: float
    ms_within: float
    tau2: float
    sigma2: float
    clamped: bool

    @property
    def var_grand_mean(self) -> float:
        return var_grand_mean(self.tau2, self.sigma2, self.n_seeds, self.n_episodes)

    @property
    def se_grand_mean(self) -> float:
        return math.sqrt(self.var_grand_mean)

    @property
    def seed_mean_variance(self) -> float:
        """``MS_between / N``: the sample variance of the S seed means, clamp or no clamp.

        ``E = tau^2 + sigma^2 / N``, so divided by S it is an unbiased estimate of the
        pre-registered ``Var(grand mean)``. The confirmatory test uses this rather than
        the clamped plug-in, which overstates the variance whenever the clamp engages.
        """
        return self.ms_between / self.n_episodes

    @property
    def se_sigma2(self) -> float:
        """Normal-theory standard error of ``sigma^2``: ``Var(MS_within) = 2 sigma^4/(S(N-1))``."""
        return math.sqrt(2.0 * self.ms_within**2 / (self.n_seeds * (self.n_episodes - 1)))

    @property
    def se_tau2(self) -> float:
        """Normal-theory standard error of the unclamped ``tau^2``.

        ``Var(MS_between) = 2 (sigma^2 + N tau^2)^2 / (S - 1)``; the two mean squares are
        independent, so ``Var(tau^2) = (Var(MS_between) + Var(MS_within)) / N^2``, with
        the observed mean squares plugged in. At pilot seed counts this is wide, and the
        allocation built on it inherits that width (project document 7.3).
        """
        var_between = 2.0 * self.ms_between**2 / (self.n_seeds - 1)
        return math.sqrt(var_between + self.se_sigma2**2) / self.n_episodes

    def render(self) -> str:
        clamp = (
            "  (MS_between < MS_within: floored at 0, no seed effect seen)" if self.clamped else ""
        )
        return "\n".join(
            [
                f"S = {self.n_seeds} seeds x N = {self.n_episodes} episodes",
                f"grand mean      {self.grand_mean:,.4f}",
                f"MS_between      {self.ms_between:,.4f}",
                f"MS_within       {self.ms_within:,.4f}",
                f"tau^2           {self.tau2:,.4f} (se {self.se_tau2:,.4f}){clamp}",
                f"sigma^2         {self.sigma2:,.4f} (se {self.se_sigma2:,.4f})",
                f"Var(grand mean) {self.var_grand_mean:,.6f} = tau^2/S + sigma^2/(S N)",
                f"se(grand mean)  {self.se_grand_mean:,.4f}",
            ]
        )


def variance_components(outcomes: ArrayLike) -> VarianceComponents:
    """Balanced one-way random-effects ANOVA on an ``(S, N)`` matrix of outcomes.

    Row i holds the N episode outcomes of seed i. Needs S >= 2 to separate the seed
    component from the episode component and N >= 2 to estimate the episode
    component at all.
    """
    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(
            f"outcomes must be two-dimensional (seeds x episodes), got shape {y.shape}"
        )
    n_seeds, n_episodes = int(y.shape[0]), int(y.shape[1])
    if n_seeds < 2:
        raise ValueError(f"need at least 2 seeds to separate tau^2 from sigma^2, got {n_seeds}")
    if n_episodes < 2:
        raise ValueError(f"need at least 2 episodes per seed to estimate sigma^2, got {n_episodes}")
    seed_means = y.mean(axis=1)
    grand = float(seed_means.mean())
    ms_within = float(((y - seed_means[:, None]) ** 2).sum() / (n_seeds * (n_episodes - 1)))
    ms_between = float(n_episodes * ((seed_means - grand) ** 2).sum() / (n_seeds - 1))
    raw_tau2 = (ms_between - ms_within) / n_episodes
    return VarianceComponents(
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        grand_mean=grand,
        ms_between=ms_between,
        ms_within=ms_within,
        tau2=max(raw_tau2, 0.0),
        sigma2=ms_within,
        clamped=raw_tau2 < 0.0,
    )


# -- power and sample size -------------------------------------------------- #


def _z(p: float, name: str) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {p}")
    return float(norm.ppf(p))


def _degenerate_power(delta: float, alpha: float) -> float:
    """Power with no variance at all: the noncentrality is +inf, undefined or -inf (#31)."""
    if delta > 0.0:
        return 1.0
    return alpha if delta == 0.0 else 0.0


# The normal approximation: for orientation only (pre-registration A3). It ignores that
# the test estimates its variance from S seed means, so it overstates power at small S.


def variance_budget(delta: float, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """Normal approximation: the largest ``Var(grand mean)`` with the target power, any S.

    ``(delta / (z_{1-alpha} + z_{power}))^2``, the limit of ``variance_budget_t`` as S
    grows. The exact budget at a seed count is smaller; the planning functions use that.
    """
    if not delta > 0.0:
        raise ValueError(f"delta must be positive to size a one-sided test, got {delta}")
    z = _z(1.0 - alpha, "alpha") + _z(target_power, "target_power")
    return (delta / z) ** 2


def power(
    delta: float, tau2: float, sigma2: float, n_seeds: int, n_episodes: int, alpha: float = 0.05
) -> float:
    """Normal approximation to the power of the one-sided test at effect ``delta``.

    ``1 - Phi(z_{1-alpha} - delta / se)``: equals ``alpha`` at zero effect and rises
    to one as the standard error shrinks or the effect grows. At zero variance it is
    the limit as ``se`` falls to 0: one, ``alpha`` or zero as the effect is positive,
    zero or negative. Above ``power_t`` at every S; the section 8 gate reports it
    beside the exact figure, and nothing is planned on it.
    """
    variance = var_grand_mean(tau2, sigma2, n_seeds, n_episodes)
    z_alpha = _z(1.0 - alpha, "alpha")
    if variance == 0.0:
        return _degenerate_power(delta, alpha)
    return float(norm.sf(z_alpha - delta / math.sqrt(variance)))


def detectable_effect(
    tau2: float,
    sigma2: float,
    n_seeds: int,
    n_episodes: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """Normal approximation to the floor: ``(z_{1-alpha} + z_power) se``.

    The inverse of ``variance_budget``. Below ``detectable_effect_t`` at every S, by
    18% at five seeds and 38% at three; kept to show that gap, not to plan with.
    """
    z = _z(1.0 - alpha, "alpha") + _z(target_power, "target_power")
    return z * math.sqrt(var_grand_mean(tau2, sigma2, n_seeds, n_episodes))


# -- the exact power of the confirmatory test, and the plan built on it ----- #


@lru_cache(maxsize=256)
def t_critical(n_seeds: int, alpha: float = 0.05) -> float:
    """``t_{1-alpha, S-1}``: the one-sided critical value of the test on S seed means."""
    if n_seeds < 2:
        raise ValueError(f"a t test on seed means needs at least 2 seeds, got {n_seeds}")
    _z(1.0 - alpha, "alpha")  # range check only
    return float(student_t.ppf(1.0 - alpha, n_seeds - 1))


def power_t(
    delta: float, tau2: float, sigma2: float, n_seeds: int, n_episodes: int, alpha: float = 0.05
) -> float:
    """Exact power of the one-sided t test on the S seed means at effect ``delta``.

    Seed means are ``N(delta, tau^2 + sigma^2 / N)``, so the statistic is noncentral t
    with ``S - 1`` degrees of freedom and noncentrality ``delta / se``,
    ``se^2 = tau^2 / S + sigma^2 / (S N)``. Exactly ``alpha`` at zero effect; at zero
    variance the limit, as for ``power``.
    """
    variance = var_grand_mean(tau2, sigma2, n_seeds, n_episodes)
    critical = t_critical(n_seeds, alpha)
    if variance == 0.0:
        return _degenerate_power(delta, alpha)
    if delta == 0.0:
        return alpha
    return float(nct.sf(critical, n_seeds - 1, delta / math.sqrt(variance)))


@lru_cache(maxsize=1024)
def t_noncentrality(n_seeds: int, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """The ``delta / se`` at which the t test on S seed means has the target power.

    ``power_t`` depends on the design only through S and ``delta / se``, and rises with
    the latter, so this one number per seed count is what exact planning needs: the
    floor is it times ``se`` and the variance budget ``(delta / it)^2``. It plays the
    part ``z_{1-alpha} + z_power`` plays in the normal form and falls to it as S grows
    (3.04 at S = 5 against 2.49). Bisection between zero and a doubled upper bracket.
    """
    critical = t_critical(n_seeds, alpha)
    _z(target_power, "target_power")  # range check only
    if target_power <= alpha:
        raise ValueError(f"target_power must exceed alpha, got {target_power} <= {alpha}")
    df = n_seeds - 1
    lo, hi = 0.0, _z(1.0 - alpha, "alpha") + _z(target_power, "target_power")
    while nct.sf(critical, df, hi) < target_power:
        lo, hi = hi, 2.0 * hi
    while hi - lo > 1e-12 * hi:
        mid = 0.5 * (lo + hi)
        if nct.sf(critical, df, mid) < target_power:
            lo = mid
        else:
            hi = mid
    return hi


def variance_budget_t(
    delta: float, n_seeds: int, alpha: float = 0.05, target_power: float = 0.80
) -> float:
    """The largest ``Var(grand mean)`` at which the t test on S seeds has the target power.

    ``(delta / t_noncentrality(S))^2``: a design at S seeds is feasible exactly when its
    variance fits under this number. It grows with S toward ``variance_budget``.
    """
    if not delta > 0.0:
        raise ValueError(f"delta must be positive to size a one-sided test, got {delta}")
    return (delta / t_noncentrality(n_seeds, alpha, target_power)) ** 2


def detectable_effect_t(
    tau2: float,
    sigma2: float,
    n_seeds: int,
    n_episodes: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """The smallest effect the exact t test detects at the target power: ``power_t`` inverted.

    ``t_noncentrality(S) * se``. It needs no effect size as input, which is why the
    pilot reports it: the floor in outcome units at a stated design, before anyone has
    argued for a particular delta. Above ``detectable_effect`` at every S, by most at
    the small seed counts where the normal form overstates power; the two meet as S grows.
    """
    k = t_noncentrality(n_seeds, alpha, target_power)
    return k * math.sqrt(var_grand_mean(tau2, sigma2, n_seeds, n_episodes))


def episodes_for(
    delta: float,
    tau2: float,
    sigma2: float,
    n_seeds: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int | None:
    """Fewest episodes per seed at which the t test on ``n_seeds`` seeds has the target power.

    ``None`` when ``tau^2 / S`` alone exceeds the exact variance budget at S: more
    episodes cannot shrink the seed term, so no N makes the design feasible. That is
    reported rather than rounded, because it is the answer the pilot exists to give.
    Needs S >= 2, as the test does.
    """
    budget_var = variance_budget_t(delta, n_seeds, alpha, target_power)
    _check_variances(tau2, sigma2)
    room = budget_var - tau2 / n_seeds
    if room <= 0.0:
        return None
    if sigma2 == 0.0:
        return 1
    n = max(math.ceil(sigma2 / (n_seeds * room)), 1)
    # The closed form is exact up to the noncentrality's 1e-12, which can carry the
    # boundary across an integer; settle that integer on power_t itself, so N is the
    # fewest episodes the test powers rather than the formula's rounding of it.
    if n > 1 and power_t(delta, tau2, sigma2, n_seeds, n - 1, alpha) >= target_power:
        return n - 1
    if power_t(delta, tau2, sigma2, n_seeds, n, alpha) < target_power:
        return n + 1
    return n


# Where seeds_for gives up: the degrees of freedom stay within what scipy's t takes.
MAX_PLANNED_SEEDS = 10**18


def seeds_for(
    delta: float,
    tau2: float,
    sigma2: float,
    n_episodes: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Fewest seeds (at least 2) at which the t test has the target power at ``n_episodes``.

    Always feasible in principle, since ``power_t`` rises with S to one: both the
    noncentrality and the degrees of freedom grow. Doubling from S = 2 brackets the
    answer and bisection on the integers finds it, so the result does not lean on the
    normal form at all. An effect so small that no S up to ``MAX_PLANNED_SEEDS`` reaches
    the target raises ``OverflowError``: that is no design, not a number to report.
    """
    if not delta > 0.0:
        raise ValueError(f"delta must be positive to size a one-sided test, got {delta}")
    _check_variances(tau2, sigma2)
    _check_design(2, n_episodes)
    t_noncentrality(2, alpha, target_power)  # range checks only

    def powered(n_seeds: int) -> bool:
        return power_t(delta, tau2, sigma2, n_seeds, n_episodes, alpha) >= target_power

    if powered(2):
        return 2
    lo, hi = 2, 4
    while not powered(hi):
        if hi >= MAX_PLANNED_SEEDS:
            raise OverflowError(
                f"no seed count up to {MAX_PLANNED_SEEDS:.0e} has power {target_power} "
                f"at delta = {delta:g}"
            )
        lo, hi = hi, min(2 * hi, MAX_PLANNED_SEEDS)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if powered(mid):
            hi = mid
        else:
            lo = mid
    return hi


# -- the compute budget ----------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Budget:
    """Steps and wall-clock for a design at measured rates."""

    n_seeds: int
    n_episodes: int
    train_steps: int
    eval_steps: int
    train_hours: float
    eval_hours: float

    @property
    def total_steps(self) -> int:
        return self.train_steps + self.eval_steps

    @property
    def total_hours(self) -> float:
        return self.train_hours + self.eval_hours


def budget(
    n_seeds: int,
    n_episodes: int,
    *,
    train_steps_per_seed: int,
    episode_days: int,
    train_rate: float,
    eval_rate: float,
    cores: int = 1,
) -> Budget:
    """Cost a design: ``S`` trainings plus ``S * N`` evaluation episodes.

    Rates are steps per second per core, measured (``make bench``), never assumed; the
    README's planning figures are about 1,000 with the training update and about
    1,800 for a scripted steady state.
    """
    _check_design(n_seeds, n_episodes)
    if train_steps_per_seed < 0 or episode_days < 1:
        raise ValueError("train_steps_per_seed must be >= 0 and episode_days >= 1")
    if train_rate <= 0.0 or eval_rate <= 0.0 or cores < 1:
        raise ValueError("rates must be positive steps/s/core and cores >= 1")
    train_steps = n_seeds * train_steps_per_seed
    eval_steps = n_seeds * n_episodes * episode_days
    return Budget(
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        train_steps=train_steps,
        eval_steps=eval_steps,
        train_hours=train_steps / train_rate / 3600.0 / cores,
        eval_hours=eval_steps / eval_rate / 3600.0 / cores,
    )


@dataclass(frozen=True, slots=True)
class Design:
    """One row of the frontier: the cheapest feasible N at a seed count, or infeasible.

    ``power`` is the exact ``power_t`` of the design, at least the target when feasible.
    """

    n_seeds: int
    n_episodes: int | None
    power: float | None
    budget: Budget | None

    @property
    def feasible(self) -> bool:
        return self.n_episodes is not None


def frontier(
    delta: float,
    tau2: float,
    sigma2: float,
    seeds: Sequence[int],
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    train_steps_per_seed: int,
    episode_days: int,
    train_rate: float,
    eval_rate: float,
    cores: int = 1,
) -> tuple[Design, ...]:
    """For each seed count, the fewest episodes the t test needs and what the design costs.

    Seed counts below 2 have no t test and are refused, as ``power_t`` refuses them.
    """
    designs: list[Design] = []
    for n_seeds in seeds:
        n_episodes = episodes_for(delta, tau2, sigma2, n_seeds, alpha, target_power)
        if n_episodes is None:
            designs.append(Design(n_seeds, None, None, None))
            continue
        designs.append(
            Design(
                n_seeds,
                n_episodes,
                power_t(delta, tau2, sigma2, n_seeds, n_episodes, alpha),
                budget(
                    n_seeds,
                    n_episodes,
                    train_steps_per_seed=train_steps_per_seed,
                    episode_days=episode_days,
                    train_rate=train_rate,
                    eval_rate=eval_rate,
                    cores=cores,
                ),
            )
        )
    return tuple(designs)


def render_frontier(designs: Sequence[Design]) -> str:
    lines = [f"{'seeds':>6}{'episodes':>10}{'power':>8}{'train h':>10}{'eval h':>9}{'total h':>10}"]
    for d in designs:
        if d.n_episodes is None or d.power is None or d.budget is None:
            lines.append(
                f"{d.n_seeds:>6}{'--':>10}{'--':>8}   infeasible: tau^2/S exceeds the budget"
            )
            continue
        b = d.budget
        lines.append(
            f"{d.n_seeds:>6}{d.n_episodes:>10}{d.power:>8.3f}{b.train_hours:>10.2f}"
            f"{b.eval_hours:>9.2f}{b.total_hours:>10.2f}"
        )
    return "\n".join(lines)
