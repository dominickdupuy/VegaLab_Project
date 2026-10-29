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

    control_ij = p_ij + c_ij                    the arm at zero premium
    treated_ij = delta + a_i + p_ij + e_ij      the same arm at p: seed term a_i

    p_ij ~ N(0, path_sigma2)   the path, shared by both arms (paired seeds)
    a_i  ~ N(0, tau2)          between-seed variance of the contrast
    e_ij, c_ij ~ N(0, sigma2 / 2) each, so the contrast's episode variance is sigma2

so the contrast has exactly the components ``tau2`` and ``sigma2``, the path term
cancels if and only if the pipeline pairs (a seed level common to both worlds cancels
the same way, so the case leaves it out; the null-world case below keeps it), and the
detection probability of the one-sided t test on seed means is
``optspread.power.power_t`` at the truth: a noncentral t, exact rather than
approximate.

``validate_pipeline`` draws ``replications`` such experiments at each effect on a
grid that includes zero, runs the pipeline on each, and compares the rejection rate
with the closed form. ``PASS_CRITERION`` states the rule. The pipeline is an
argument, so the negative controls in the tests hand it deliberately wrong rules
(normal critical values, unpaired arms, episodes treated as independent) and
require the gate to refuse each one.

``validate_null_world`` is the null-world case (#26): the learner flat, the scripted
control paying a cost drag, nothing planted. The pre-registered event (each arm
against itself at zero premium) must fire at alpha there; the event it replaced (the
learner against the control) fires at nearly 1 and is the gate's negative control.

``validate_grid_decision`` takes the rule off synthetic arrays (#31): one arm at one
premium of a real grid record on disk, decided through ``GridRecord.matrix`` (what the
M4 analysis reads), ``GridRecord.contrast`` and the ``.npy`` files the pilot writes,
each of which must agree with the others and with the estimate and se the cells' own
headlines give.

``bootstrap_detection`` is the section 5.3 check (an M4 item, cheap enough to land
with the rule): resample the S seed means with replacement, apply the same test,
and compare the detection rate with the analytic figure at the observed effect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from optspread.analysis import (
    ALPHA,
    NULL_PREMIUM,
    Decision,
    decide,
    decide_contrast,
    decide_premium,
)
from optspread.power import power, power_t, t_critical, var_grand_mean

if TYPE_CHECKING:  # orchestrate pulls in the training stack; only the grid check needs it
    from optspread.config import ArmName
    from optspread.orchestrate import GridRecord

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
        """The normal approximation, reported beside the closed form, never tested against."""
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


# -- the null world: the detection event must not fire with nothing planted -- #

# The arms of one null-world experiment, keyed by (arm, world): the learner and the
# control, each at the premium under test and at zero. Nothing is planted, so each
# arm's two worlds have the same mean; the control's is minus its cost drag.
NullArms = dict[tuple[str, str], NDArray[np.float64]]
# Which two matrices an event tests, treated first: the pairing is what is under test.
Event = Callable[[NullArms], tuple[NDArray[np.float64], NDArray[np.float64]]]

# The section 8 case's units. The pre-registered self-contrast has Var(grand mean)
# tau^2 / S + sigma^2 / (S N) = 0.8; the replaced event's contrast (the learner against
# the control) carries both arms' seed terms, 2 tau^2 / S + sigma^2 / (S N) = 1.2, so a
# drag of 5 is 4.56 of its standard errors and that event fires almost every time (0.995).
NULL_WORLD_DRAG = 5.0


def simulate_null_world(case: AnalyticCase, drag: float, rng: np.random.Generator) -> NullArms:
    """One null-world experiment on the design of ``case``; no effect is planted.

    For seed block i and path j, with the path p_ij shared by every arm and world:

        learner at p      a_i + b_i + p_ij + e_ij     a_i the training seed's own level,
        learner at zero   a_i       + p_ij + e'_ij    b_i ~ N(0, tau2) its change across
        control at p      -drag     + p_ij + c_ij     the two worlds; e, e', c, c' each
        control at zero   -drag     + p_ij + c'_ij    N(0, sigma2 / 2)

    so the learner's self-contrast has exactly the case's tau2 and sigma2 and mean 0,
    and the learner against the control has mean ``drag``: the flat learner beating a
    seller that pays its costs, which is what #26 found the old contrast detecting.
    """
    shape = (case.n_seeds, case.n_episodes)
    path = rng.normal(0.0, math.sqrt(case.path_sigma2), size=shape)
    level = rng.normal(0.0, math.sqrt(case.tau2), size=(case.n_seeds, 1))
    change = rng.normal(0.0, math.sqrt(case.tau2), size=(case.n_seeds, 1))
    half = math.sqrt(case.sigma2 / 2.0)

    def noise() -> NDArray[np.float64]:
        return rng.normal(0.0, half, size=shape)

    arms = {
        ("learner", "premium"): level + change + path + noise(),
        ("learner", "null"): level + path + noise(),
        ("control", "premium"): -drag + path + noise(),
        ("control", "null"): -drag + path + noise(),
    }
    return {k: np.asarray(v, dtype=np.float64) for k, v in arms.items()}


def _against_itself(arms: NullArms) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return arms[("learner", "premium")], arms[("learner", "null")]


def _against_the_control(arms: NullArms) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return arms[("learner", "premium")], arms[("control", "premium")]


# The pre-registered event (#26), and the one it replaced, which the gate must refuse.
PREREGISTERED_EVENT: tuple[str, Event] = (
    "each arm against itself at zero premium",
    _against_itself,
)
REJECTED_EVENTS: dict[str, Event] = {
    "the learner against the fixed-delta seller at one premium": _against_the_control,
}


def validate_null_world(
    case: AnalyticCase = DEFAULT_CASE,
    drag: float = NULL_WORLD_DRAG,
    replications: int = 10_000,
    seed: int = 1,
    event: Event = _against_itself,
    k: float = K_SE,
) -> Check:
    """The event's rejection rate with nothing planted, against alpha.

    ``Check.within`` is the pass: a correct event fires at alpha (the learner's
    self-contrast is an exact t case with mean 0), a wrong one at its power against
    the drag. The rate is compared with alpha whichever event is passed, so the gate
    reads the same number for the event it accepts and the ones it refuses.
    """
    if replications < 2:
        raise ValueError(f"need at least 2 replications, got {replications}")
    rng = np.random.default_rng(seed)
    rejections = sum(
        decide(*event(simulate_null_world(case, drag, rng)), case.alpha).rejected
        for _ in range(replications)
    )
    p = case.alpha
    return Check(
        f"null-world rejection rate (drag {drag:g})",
        rejections / replications,
        p,
        math.sqrt(p * (1.0 - p) / replications),
    )


# -- the rule through a real grid record (#31) -------------------------------- #

# Routes that sum in a different order agree to rounding, not bit for bit.
GRID_RTOL = 1e-9


@dataclass(frozen=True, slots=True)
class GridRoute:
    """One way from the files on disk to the decision."""

    name: str
    estimate: float
    se: float
    rejected: bool


def _route(name: str, decision: Decision) -> GridRoute:
    return GridRoute(name, decision.estimate, decision.se, decision.rejected)


@dataclass(frozen=True, slots=True)
class GridDecisionCheck:
    """One arm at one premium of a grid on disk, decided by every route the results take.

    The first route is the reference. Every other must reach the same decision with
    the same estimate and se, within ``GRID_RTOL``, and each seed block must have run
    the same paths at the premium and at zero; ``failures`` names what did not.
    """

    premium: float
    arm: str
    routes: tuple[GridRoute, ...]
    unpaired: tuple[int, ...]  # blocks whose two worlds started on different path seeds

    @property
    def failures(self) -> tuple[str, ...]:
        ref = self.routes[0]
        scale = max(abs(ref.estimate), ref.se, 1.0)
        out = [
            f"block {b} ran different paths at {self.premium:g} and at {NULL_PREMIUM:g}"
            for b in self.unpaired
        ]
        for r in self.routes[1:]:
            same = all(
                math.isclose(a, b, rel_tol=GRID_RTOL, abs_tol=GRID_RTOL * scale)
                for a, b in ((r.estimate, ref.estimate), (r.se, ref.se))
            )
            if not same or r.rejected != ref.rejected:
                out.append(
                    f"{r.name}: estimate {r.estimate:.6g} (se {r.se:.6g}, "
                    f"{'detected' if r.rejected else 'not detected'}) against {ref.name}: "
                    f"{ref.estimate:.6g} (se {ref.se:.6g}, "
                    f"{'detected' if ref.rejected else 'not detected'})"
                )
        return tuple(out)

    @property
    def passed(self) -> bool:
        return not self.failures


def validate_grid_decision(
    record: GridRecord,
    root: Path,
    premium: float,
    arm: ArmName,
    alpha: float = ALPHA,
    pilot_files: bool = False,
) -> GridDecisionCheck:
    """The detection event for ``arm`` at ``premium``, read from a grid under ``root``.

    The routes, the first the reference:

    - ``GridRecord.matrix``: ``decide_premium`` on the arm's matrices at the premium
      and at zero, each row read from its block's ``run.json`` (the M4 analysis);
    - ``GridRecord.contrast``: ``decide_contrast`` on the record's contrast;
    - with ``pilot_files`` (``record`` is a pilot's and ``premium`` its premium), the
      files ``write_pilot_matrices`` left: the contrast matrix (``matrix.npy`` for
      PPO, ``matrix_oracle.npy`` for the seller) and the raw ``matrix_<arm>_p<p>.npy``
      at both premiums, as ``scripts/pilot_budget.py`` and a reader of the files see
      them;
    - the cells' headlines: each block's ``mean_pnl`` at the premium less at zero,
      averaged over blocks, with the se of those differences. The harness computed
      them from the same episodes, not the analysis, so they are an independent
      sum; ``critical`` is the same ``t_{1-alpha, S-1}``.
    """
    from optspread.orchestrate import PILOT_MATRIX, PILOT_ORACLE_MATRIX

    at_p = record.matrix(root, premium, arm)
    at_0 = record.matrix(root, NULL_PREMIUM, arm)
    routes = [
        _route("GridRecord.matrix", decide_premium(at_p, at_0, alpha)),
        _route("GridRecord.contrast", decide_contrast(record.contrast(root, premium, arm), alpha)),
    ]
    if pilot_files:
        names = {"ppo": PILOT_MATRIX, "delta_seller": PILOT_ORACLE_MATRIX}
        contrast_file = names.get(arm, f"matrix_{arm}.npy")
        routes.append(_route(contrast_file, decide_contrast(np.load(root / contrast_file), alpha)))
        raw = [np.load(root / f"matrix_{arm}_p{p:g}.npy") for p in (premium, NULL_PREMIUM)]
        routes.append(_route(f"matrix_{arm}_p<p>.npy", decide_premium(raw[0], raw[1], alpha)))
    S = record.grid.n_seeds
    pairs = [(record.cell(premium, arm, b), record.cell(NULL_PREMIUM, arm, b)) for b in range(S)]
    diffs = np.array([p.mean_pnl - z.mean_pnl for p, z in pairs], dtype=np.float64)
    estimate = float(diffs.mean())
    se = float(diffs.std(ddof=1)) / math.sqrt(S)
    t_stat = estimate / se if se > 0.0 else (math.copysign(math.inf, estimate) if estimate else 0.0)
    routes.append(GridRoute("cell headlines", estimate, se, t_stat > t_critical(S, alpha)))
    unpaired = tuple(
        b for b, (p, z) in enumerate(pairs) if (p.seed, p.n_episodes) != (z.seed, z.n_episodes)
    )
    return GridDecisionCheck(premium, str(arm), tuple(routes), unpaired)


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
