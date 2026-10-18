"""The confirmatory decision rule: the one test the primary contrast gets.

Lane: Dominick Dupuy (statistical spine).

Pre-registration D1 fixes the rule's shape: the primary contrast (the trained policy
against the scripted fixed-delta seller, on the same seeds) is tested once,
one-sided at alpha = 0.05, and the result is reported as the estimate with its
interval. This module is that rule as code, so the real analysis and the section 8
validation in ``optspread.validation`` run the same function rather than two
descriptions of it.

    treated, control   (S, N) outcome matrices; row i of both arms is seed block i,
                       column j the same episode path in both
    contrast           treated - control, cell by cell. The two arms share paths, so
                       the path term cancels here and only the arms' own noise is left
    components         ``optspread.power.variance_components`` on the contrast
    test               one-sided Student t on the S seed-level means of the contrast,
                       ``t = grand mean / se`` with ``se^2 = MS_between / (S N)`` and
                       ``S - 1`` degrees of freedom (project document 5.2: the test
                       operates on seed means, not on per-episode P&L)

``se^2 = MS_between / (S N)`` is the sample variance of the seed means over S, an
unbiased estimate of the pre-registered ``tau^2 / S + sigma^2 / (S N)``. It is not
the clamped plug-in ``VarianceComponents.var_grand_mean``: the clamp is right for
reporting the components and wrong inside the test, where it inflates the variance
exactly when the seed means agree more than the episodes do, making the test
conservative in a way nobody chose.

Pairing within a block keeps the design nested across blocks: each block's paths
are its own (``optspread.sweep.pilot_matrix``), so the contrast's seed means are
independent and the t distribution is the exact reference under normal seed means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from optspread.power import VarianceComponents, t_critical, variance_components

ALPHA = 0.05  # D1: one-sided


@dataclass(frozen=True, slots=True)
class Decision:
    """What the rule saw and what it decided. ``rejected`` is the detection event."""

    components: VarianceComponents
    alpha: float
    estimate: float  # grand mean of the contrast
    se: float
    critical: float  # t_{1-alpha, S-1}

    @property
    def df(self) -> int:
        return self.components.n_seeds - 1

    @property
    def t_stat(self) -> float:
        if self.se > 0.0:
            return self.estimate / self.se
        return math.copysign(math.inf, self.estimate) if self.estimate != 0.0 else 0.0

    @property
    def rejected(self) -> bool:
        return self.t_stat > self.critical

    @property
    def lower_bound(self) -> float:
        """One-sided ``1 - alpha`` lower confidence bound: positive exactly when rejected."""
        return self.estimate - self.critical * self.se

    def render(self) -> str:
        verdict = "detected" if self.rejected else "not detected"
        return (
            f"estimate {self.estimate:,.4g} (se {self.se:,.4g}), t = {self.t_stat:.3f} on "
            f"{self.df} df against {self.critical:.3f}: {verdict}; one-sided "
            f"{1.0 - self.alpha:.0%} lower bound {self.lower_bound:,.4g}"
        )


def paired_contrast(treated: ArrayLike, control: ArrayLike) -> NDArray[np.float64]:
    """``treated - control`` cell by cell, refusing anything that is not paired."""
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.ndim != 2 or t.shape != c.shape:
        raise ValueError(
            f"paired arms must be two (seeds x episodes) matrices of one shape, got "
            f"{t.shape} and {c.shape}"
        )
    return np.asarray(t - c, dtype=np.float64)


def decide_contrast(contrast: ArrayLike, alpha: float = ALPHA) -> Decision:
    """The rule on an already-formed ``(S, N)`` contrast matrix."""
    components = variance_components(contrast)
    se = math.sqrt(components.seed_mean_variance / components.n_seeds)
    return Decision(
        components=components,
        alpha=alpha,
        estimate=components.grand_mean,
        se=se,
        critical=t_critical(components.n_seeds, alpha),
    )


def decide(treated: ArrayLike, control: ArrayLike, alpha: float = ALPHA) -> Decision:
    """The pre-registered rule on the two arms: pair, estimate the components, test."""
    return decide_contrast(paired_contrast(treated, control), alpha)
