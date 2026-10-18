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

Two power functions, kept apart on purpose. ``power`` is the normal approximation a
sample-size calculation at the planning stage uses, and ``episodes_for``,
``seeds_for`` and ``frontier`` are built on it. ``power_t`` is the exact detection
probability of the confirmatory test in ``optspread.analysis``, a one-sided Student t
on the S seed-level means: under normal seed means its statistic is noncentral t
with S - 1 degrees of freedom, so the probability is a closed form rather than an
approximation. At the seed counts published work uses the two differ by enough to
matter (the normal form overstates power), which is why the section 8 gate in
``optspread.validation`` checks the test against ``power_t`` and reports ``power``
beside it.
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


def variance_budget(delta: float, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """The largest ``Var(grand mean)`` at which a one-sided test still has the target power.

    ``(delta / (z_{1-alpha} + z_{power}))^2``: the design is feasible exactly when its
    variance fits under this number.
    """
    if not delta > 0.0:
        raise ValueError(f"delta must be positive to size a one-sided test, got {delta}")
    z = _z(1.0 - alpha, "alpha") + _z(target_power, "target_power")
    return (delta / z) ** 2


def power(
    delta: float, tau2: float, sigma2: float, n_seeds: int, n_episodes: int, alpha: float = 0.05
) -> float:
    """Power of the one-sided test of the grand mean against zero at effect ``delta``.

    ``1 - Phi(z_{1-alpha} - delta / se)``: equals ``alpha`` at zero effect and rises
    to one as the standard error shrinks or the effect grows.
    """
    variance = var_grand_mean(tau2, sigma2, n_seeds, n_episodes)
    z_alpha = _z(1.0 - alpha, "alpha")
    if variance == 0.0:
        return 1.0 if delta > 0.0 else alpha
    return float(norm.sf(z_alpha - delta / math.sqrt(variance)))


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
    ``se^2 = tau^2 / S + sigma^2 / (S N)``. Exactly ``alpha`` at zero effect.
    """
    variance = var_grand_mean(tau2, sigma2, n_seeds, n_episodes)
    critical = t_critical(n_seeds, alpha)
    if variance == 0.0:
        return 1.0 if delta > 0.0 else alpha
    if delta == 0.0:
        return alpha
    return float(nct.sf(critical, n_seeds - 1, delta / math.sqrt(variance)))


def detectable_effect(
    tau2: float,
    sigma2: float,
    n_seeds: int,
    n_episodes: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """The smallest effect a design detects at the target power: ``(z_{1-alpha} + z_power) se``.

    The inverse of ``variance_budget``, on the same normal approximation as the rest of
    the planning functions. It needs no effect size as input, which is why the pilot
    reports it: the floor in outcome units at a stated design, before anyone has
    argued for a particular delta.
    """
    z = _z(1.0 - alpha, "alpha") + _z(target_power, "target_power")
    return z * math.sqrt(var_grand_mean(tau2, sigma2, n_seeds, n_episodes))


def episodes_for(
    delta: float,
    tau2: float,
    sigma2: float,
    n_seeds: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int | None:
    """Fewest episodes per seed that reach the target power at ``n_seeds`` seeds.

    ``None`` when ``tau^2 / S`` alone exceeds the variance budget: more episodes
    cannot shrink the seed term, so no N makes the design feasible. That is reported
    rather than rounded, because it is the answer the pilot exists to give.
    """
    budget_var = variance_budget(delta, alpha, target_power)
    _check_variances(tau2, sigma2)
    _check_design(n_seeds, 1)
    room = budget_var - tau2 / n_seeds
    if room <= 0.0:
        return None
    if sigma2 == 0.0:
        return 1
    return max(math.ceil(sigma2 / (n_seeds * room)), 1)


def seeds_for(
    delta: float,
    tau2: float,
    sigma2: float,
    n_episodes: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Fewest seeds that reach the target power at ``n_episodes`` episodes per seed.

    Always feasible: ``S >= (tau^2 + sigma^2 / N) / budget``.
    """
    budget_var = variance_budget(delta, alpha, target_power)
    _check_variances(tau2, sigma2)
    _check_design(1, n_episodes)
    return max(math.ceil((tau2 + sigma2 / n_episodes) / budget_var), 1)


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
    """One row of the frontier: the cheapest feasible N at a seed count, or infeasible."""

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
    """For each seed count, the fewest episodes that reach the target power and what it costs."""
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
                power(delta, tau2, sigma2, n_seeds, n_episodes, alpha),
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
