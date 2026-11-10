"""Dominick Dupuy: the confirmatory decision rule (week 6).

The oracle is the textbook: on the seed means of the paired contrast the rule must be
scipy's one-sample t test, one-sided, statistic and verdict alike. The rest pins the
properties the pre-registration leans on: the shared path cancels, the interval and
the verdict agree, and the clamp on tau^2 never leaks into the test.

Every number is synthetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ttest_1samp

from optspread.analysis import ALPHA, decide, decide_contrast, paired_contrast
from optspread.power import variance_components


def arms(seed: int, delta: float, S: int = 8, N: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    path = rng.normal(0.0, 30.0, size=(S, N))
    treated = delta + rng.normal(0.0, 1.0, size=(S, 1)) + path + rng.normal(0.0, 3.0, size=(S, N))
    control = path + rng.normal(0.0, 3.0, size=(S, N))
    return treated, control


@pytest.mark.parametrize("seed", range(12))
def test_the_rule_is_the_one_sided_t_test_on_seed_means(seed: int) -> None:
    treated, control = arms(seed, delta=1.0)
    d = decide(treated, control)
    oracle = ttest_1samp((treated - control).mean(axis=1), 0.0, alternative="greater")
    assert d.t_stat == pytest.approx(float(oracle.statistic), rel=1e-10)
    assert d.rejected == (float(oracle.pvalue) < ALPHA)
    assert d.df == 7
    assert d.estimate == pytest.approx(float((treated - control).mean()), rel=1e-12)


def test_the_shared_path_cancels_exactly() -> None:
    treated, control = arms(0, delta=1.0)
    shift = np.random.default_rng(99).normal(0.0, 1e3, size=treated.shape)
    a, b = decide(treated, control), decide(treated + shift, control + shift)
    assert b.t_stat == pytest.approx(a.t_stat, rel=1e-9)
    assert b.se == pytest.approx(a.se, rel=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_the_lower_bound_is_positive_exactly_when_the_rule_fires(seed: int) -> None:
    d = decide(*arms(seed, delta=0.8))
    assert (d.lower_bound > 0.0) == d.rejected


def test_the_test_uses_the_unclamped_seed_mean_variance() -> None:
    # Every seed mean equal: MS_between = 0 < MS_within, so the clamp engages.
    rows = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 3.0, 2.0]])
    vc = variance_components(rows)
    assert vc.clamped and vc.var_grand_mean > 0.0
    d = decide_contrast(rows)
    assert d.se == 0.0, "the seed means agree exactly, so the test sees no variance"
    assert d.t_stat == np.inf and d.rejected
    assert decide_contrast(rows - 2.0).t_stat == 0.0
    assert not decide_contrast(rows - 2.0).rejected
    assert decide_contrast(rows - 5.0).t_stat == -np.inf


def test_unpaired_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="one shape"):
        paired_contrast(np.zeros((3, 4)), np.zeros((3, 5)))
    with pytest.raises(ValueError, match="one shape"):
        paired_contrast(np.zeros(4), np.zeros(4))


def test_the_rendering_carries_the_verdict_and_the_interval() -> None:
    text = decide(*arms(1, delta=50.0)).render()
    assert "detected" in text and "not detected" not in text
    assert "lower bound" in text and "on 7 df" in text
