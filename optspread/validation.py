"""Validating the analysis itself: the pipeline against a case whose answer is known.

Lane: Dominick Dupuy (statistical spine).

Project document section 8: before any sweep output is trusted, the full analysis
pipeline runs end to end on synthetic data with hand-specified tau^2, sigma^2 and
effect size, where the correct detection probability is a closed form. If the
pipeline does not recover it, no sweep result is interpreted. It is the poisoned
environment of section 3.2 applied to the analysis layer: plotting code always
produces a plot, so the plot is not evidence.

The synthetic world is the pre-registered design (D1) with every ingredient set by
hand. For seed block i and episode path j,

    control_ij = p_ij + c_ij                    the scripted seller: no seed term
    treated_ij = delta + a_i + p_ij + e_ij      the learner: seed term a_i

    p_ij ~ N(0, path_sigma2)   the path, shared by both arms (paired seeds)
    a_i  ~ N(0, tau2)          between-seed variance of the contrast
    e_ij, c_ij ~ N(0, sigma2 / 2) each, so the contrast's episode variance is sigma2

so the contrast has exactly the components ``tau2`` and ``sigma2``, the path term
cancels if and only if the pipeline pairs, and the detection probability of the
one-sided t test on seed means is ``optspread.power.power_t`` at the truth: a
noncentral t, exact rather than approximate.

``validate_pipeline`` draws ``replications`` such experiments at each effect on a
grid that includes zero, runs the pipeline on each, and compares the rejection rate
with the closed form. ``PASS_CRITERION`` states the rule. The pipeline is an
argument, so the negative controls in the tests hand it deliberately wrong rules
(normal critical values, unpaired arms, episodes treated as independent) and
require the gate to refuse each one.

``bootstrap_detection`` is the section 5.3 check (an M4 item, cheap enough to land
with the rule): resample the S seed means with replacement, apply the same test,
and compare the detection rate with the analytic figure at the observed effect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from optspread.analysis import ALPHA, Decision, decide
from optspread.power import power, power_t, t_critical, var_grand_mean

K_SE = 4.0  # tolerance, in Monte Carlo standard errors, at every check

PASS_CRITERION = (
    "at every effect on the grid, including delta = 0 where the answer is alpha, the "
    "decision rule's rejection rate over R paired synthetic replications lies within "
    f"{K_SE:g} Monte Carlo standard errors, sqrt(p (1 - p) / R), of the closed-form "
    "noncentral-t power at the hand-specified tau^2, sigma^2 and delta; and the mean "
    f"estimated sigma^2 and Var(grand mean) lie within {K_SE:g} standard errors of "
    "their true values"
)

# The rule under test: two (S, N) arms and alpha in, a Decision out.
Pipeline = Callable[[NDArray[np.float64], NDArray[np.float64], float], Decision]


@dataclass(frozen=True, slots=True)
class AnalyticCase:
    """The hand-specified world. ``tau2`` and ``sigma2`` are the contrast's components."""

    tau2: float
    sigma2: float
    n_seeds: int
    n_episodes: int
    path_sigma2: float = 0.0
    alpha: float = ALPHA

    def __post_init__(self) -> None:
        if min(self.tau2, self.sigma2, self.path_sigma2) < 0.0:
            raise ValueError("variances in an analytic case cannot be negative")
        if self.n_seeds < 2 or self.n_episodes < 2:
            raise ValueError(
                f"the pipeline needs at least 2 seeds and 2 episodes, got "
                f"{self.n_seeds} x {self.n_episodes}"
            )

    @property
    def var_grand_mean(self) -> float:
        return var_grand_mean(self.tau2, self.sigma2, self.n_seeds, self.n_episodes)

    def analytic_power(self, delta: float) -> float:
        return power_t(delta, self.tau2, self.sigma2, self.n_seeds, self.n_episodes, self.alpha)

    def normal_power(self, delta: float) -> float:
        """The planning approximation, reported beside the closed form, never tested against."""
        return power(delta, self.tau2, self.sigma2, self.n_seeds, self.n_episodes, self.alpha)

    def describe(self) -> str:
        return (
            f"tau^2 = {self.tau2:g}, sigma^2 = {self.sigma2:g}, shared path variance "
            f"{self.path_sigma2:g}, S = {self.n_seeds} x N = {self.n_episodes}, one-sided "
            f"alpha = {self.alpha:g}"
        )


# The M2 case. Var(grand mean) = 4/10 + 100/250 = 0.8, so half the variance is the seed
# term; the path variance is four times the contrast's episode variance, so an unpaired
# rule would see a very different world. The grid spans power from alpha to about 0.93.
DEFAULT_CASE = AnalyticCase(tau2=4.0, sigma2=100.0, n_seeds=10, n_episodes=25, path_sigma2=400.0)
DEFAULT_DELTAS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0)


def simulate_arms(
    case: AnalyticCase, delta: float, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One paired experiment: ``(treated, control)``, each ``(S, N)``."""
    shape = (case.n_seeds, case.n_episodes)
    path = rng.normal(0.0, math.sqrt(case.path_sigma2), size=shape)
    seed_term = rng.normal(0.0, math.sqrt(case.tau2), size=(case.n_seeds, 1))
    half = math.sqrt(case.sigma2 / 2.0)
    treated = delta + seed_term + path + rng.normal(0.0, half, size=shape)
    control = path + rng.normal(0.0, half, size=shape)
    return np.asarray(treated, dtype=np.float64), np.asarray(control, dtype=np.float64)


def _default_pipeline(
    treated: NDArray[np.float64], control: NDArray[np.float64], alpha: float
) -> Decision:
    return decide(treated, control, alpha)


# -- negative controls: rules the gate must refuse -------------------------- #


def _normal_critical(
    treated: NDArray[np.float64], control: NDArray[np.float64], alpha: float
) -> Decision:
    """z_{1-alpha} in place of t_{1-alpha, S-1}: anticonservative at every seed count used."""
    return replace(decide(treated, control, alpha), critical=float(norm.ppf(1.0 - alpha)))


def _unpaired(treated: NDArray[np.float64], control: NDArray[np.float64], alpha: float) -> Decision:
    """The arms' standard errors added as if their paths were independent draws."""
    S = treated.shape[0]
    se = math.sqrt(
        (float(treated.mean(axis=1).var(ddof=1)) + float(control.mean(axis=1).var(ddof=1))) / S
    )
    return replace(decide(treated, control, alpha), se=se)


def _episodes_as_independent(
    treated: NDArray[np.float64], control: NDArray[np.float64], alpha: float
) -> Decision:
    """Every episode as its own observation: the tau^2 / S term silently dropped."""
    contrast = treated - control
    se = float(contrast.std(ddof=1)) / math.sqrt(contrast.size)
    return replace(decide(treated, control, alpha), se=se)


# Each is a mistake the analysis could plausibly make. On the default case every one of
# them fails the gate by more than five Monte Carlo standard errors at some grid point.
POISONED_PIPELINES: dict[str, Pipeline] = {
    "normal critical values": _normal_critical,
    "unpaired arms": _unpaired,
    "episodes as independent": _episodes_as_independent,
}


@dataclass(frozen=True, slots=True)
class Check:
    """An observed mean against its known value, with the Monte Carlo standard error."""

    name: str
    observed: float
    expected: float
    mc_se: float

    @property
    def z(self) -> float:
        gap = self.observed - self.expected
        if self.mc_se > 0.0:
            return gap / self.mc_se
        return 0.0 if gap == 0.0 else math.copysign(math.inf, gap)

    def within(self, k: float = K_SE) -> bool:
        return abs(self.z) <= k


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One effect size: what the pipeline detected against what it should have."""

    delta: float
    replications: int
    rejections: int
    analytic: float
    normal: float

    @property
    def empirical(self) -> float:
        return self.rejections / self.replications

    @property
    def check(self) -> Check:
        p = self.analytic
        return Check(
            f"rejection rate at delta = {self.delta:g}",
            self.empirical,
            p,
            math.sqrt(p * (1.0 - p) / self.replications),
        )


@dataclass(frozen=True, slots=True)
class PipelineValidation:
    case: AnalyticCase
    points: tuple[GridPoint, ...]
    sigma2: Check
    var_grand_mean: Check
    k: float = K_SE

    @property
    def checks(self) -> tuple[Check, ...]:
        return (*(p.check for p in self.points), self.sigma2, self.var_grand_mean)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.within(self.k))

    @property
    def passed(self) -> bool:
        return bool(self.points) and not self.failures

    @property
    def worst(self) -> Check:
        return max(self.checks, key=lambda c: abs(c.z))

    def render(self) -> str:
        lines = [
            f"case: {self.case.describe()}",
            f"{'delta':>7}{'analytic':>10}{'empirical':>11}{'MC se':>8}{'z':>7}{'normal':>9}",
        ]
        for p in self.points:
            c = p.check
            lines.append(
                f"{p.delta:>7g}{p.analytic:>10.4f}{p.empirical:>11.4f}{c.mc_se:>8.4f}"
                f"{c.z:>7.2f}{p.normal:>9.4f}"
            )
        for c in (self.sigma2, self.var_grand_mean):
            lines.append(
                f"{c.name}: mean {c.observed:.4f} against {c.expected:.4f} (z = {c.z:.2f})"
            )
        verdict = "PASS" if self.passed else "FAIL: " + "; ".join(c.name for c in self.failures)
        lines.append(f"{verdict} (every check within {self.k:g} Monte Carlo standard errors)")
        return "\n".join(lines)


def validate_pipeline(
    case: AnalyticCase = DEFAULT_CASE,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    replications: int = 10_000,
    seed: int = 0,
    pipeline: Pipeline = _default_pipeline,
    k: float = K_SE,
) -> PipelineValidation:
    """Run ``pipeline`` on ``replications`` synthetic experiments per effect and grade it."""
    if replications < 2:
        raise ValueError(f"need at least 2 replications per effect, got {replications}")
    if not deltas:
        raise ValueError("the grid needs at least one effect size")
    rng = np.random.default_rng(seed)
    points: list[GridPoint] = []
    sigma2_hat: list[float] = []
    var_hat: list[float] = []
    for delta in deltas:
        rejections = 0
        for _ in range(replications):
            decision = pipeline(*simulate_arms(case, delta, rng), case.alpha)
            rejections += decision.rejected
            sigma2_hat.append(decision.components.sigma2)
            var_hat.append(decision.se**2)
        points.append(
            GridPoint(
                delta,
                replications,
                rejections,
                case.analytic_power(delta),
                case.normal_power(delta),
            )
        )
    total = len(sigma2_hat)
    S, N = case.n_seeds, case.n_episodes
    # Normal theory: MS_within ~ sigma2 chi2_{S(N-1)} / (S(N-1)), and MS_between/(S N) ~
    # Var(grand mean) chi2_{S-1} / (S-1); the means over `total` draws shrink by sqrt(total).
    sd_sigma2 = case.sigma2 * math.sqrt(2.0 / (S * (N - 1)))
    sd_var = case.var_grand_mean * math.sqrt(2.0 / (S - 1))
    return PipelineValidation(
        case=case,
        points=tuple(points),
        sigma2=Check(
            "estimated sigma^2",
            float(np.mean(sigma2_hat)),
            case.sigma2,
            sd_sigma2 / math.sqrt(total),
        ),
        var_grand_mean=Check(
            "estimated Var(grand mean)",
            float(np.mean(var_hat)),
            case.var_grand_mean,
            sd_var / math.sqrt(total),
        ),
        k=k,
    )


# -- section 5.3: the bootstrap check on seed means -------------------------- #


@dataclass(frozen=True, slots=True)
class BootstrapCheck:
    """Bootstrap detection rate against the analytic one at the observed effect.

    Agreement is the evidence that the t assumption held at the level the test uses;
    the tolerance is the M4 report's to state, so ``agrees`` takes it as an argument.
    """

    n_seeds: int
    n_boot: int
    estimate: float
    se: float
    rate: float
    analytic: float

    @property
    def gap(self) -> float:
        return self.rate - self.analytic

    @property
    def mc_se(self) -> float:
        p = self.analytic
        return math.sqrt(p * (1.0 - p) / self.n_boot)

    def agrees(self, tolerance: float) -> bool:
        return abs(self.gap) <= tolerance


def bootstrap_detection(
    seed_means: ArrayLike, alpha: float = ALPHA, n_boot: int = 10_000, seed: int = 0
) -> BootstrapCheck:
    """Resample the S seed means with replacement, apply the t test, count detections.

    The analytic comparator is ``power_t`` with the observed mean as the effect and the
    observed seed-mean variance as the variance (``tau^2`` = that variance, ``sigma^2``
    = 0, one episode), which is what the analytic route reports from the same data.
    """
    m = np.asarray(seed_means, dtype=np.float64)
    if m.ndim != 1 or m.size < 2:
        raise ValueError(f"need a vector of at least 2 seed means, got shape {m.shape}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    S = int(m.size)
    critical = t_critical(S, alpha)
    rng = np.random.default_rng(seed)
    resamples = m[rng.integers(0, S, size=(n_boot, S))]
    sd = resamples.std(axis=1, ddof=1)
    mean = resamples.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(sd > 0.0, mean / (sd / math.sqrt(S)), np.sign(mean) * np.inf)
    rate = float(np.mean(t_stat > critical))
    estimate = float(m.mean())
    seed_mean_var = float(m.var(ddof=1))
    return BootstrapCheck(
        n_seeds=S,
        n_boot=n_boot,
        estimate=estimate,
        se=math.sqrt(seed_mean_var / S),
        rate=rate,
        analytic=power_t(estimate, seed_mean_var, 0.0, S, 1, alpha),
    )
