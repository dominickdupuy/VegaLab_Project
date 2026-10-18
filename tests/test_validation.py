"""Dominick Dupuy: the section 8 gate, and the gate's own negative controls (week 6).

A gate that cannot fail is not a gate, so most of this file hands
``validate_pipeline`` rules that are wrong in ways an analysis plausibly is and
requires a FAIL. The right rule must pass on the default case and on the corners the
sweep will visit: no seed term (the scripted arm) and five seeds (published practice).

Every number is synthetic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from optspread.analysis import ALPHA, Decision, decide
from optspread.validation import (
    DEFAULT_CASE,
    DEFAULT_DELTAS,
    POISONED_PIPELINES,
    AnalyticCase,
    Check,
    bootstrap_detection,
    simulate_arms,
    validate_pipeline,
)

R = 4_000  # per effect: MC se 0.0034 at alpha, 0.0079 at power 0.5


def test_the_pipeline_recovers_the_closed_form_on_the_default_case() -> None:
    result = validate_pipeline(replications=R, seed=11)
    assert result.passed, result.render()
    assert [p.delta for p in result.points] == list(DEFAULT_DELTAS)
    assert result.points[0].analytic == ALPHA, "the answer at zero effect is alpha, exactly"
    analytic = [p.analytic for p in result.points]
    assert analytic == sorted(analytic) and analytic[-1] > 0.9
    assert "PASS" in result.render()


@pytest.mark.parametrize(
    "case",
    [
        AnalyticCase(tau2=0.0, sigma2=100.0, n_seeds=5, n_episodes=20, path_sigma2=400.0),
        AnalyticCase(tau2=25.0, sigma2=1.0, n_seeds=5, n_episodes=4),
        AnalyticCase(tau2=1.0, sigma2=50.0, n_seeds=30, n_episodes=10, path_sigma2=10.0),
    ],
    ids=["scripted-no-seed-term", "five-seeds-seed-dominated", "thirty-seeds"],
)
def test_the_pipeline_passes_where_the_sweep_will_go(case: AnalyticCase) -> None:
    se = float(np.sqrt(case.var_grand_mean))
    result = validate_pipeline(case, deltas=(0.0, se, 2.5 * se), replications=R, seed=3)
    assert result.passed, result.render()


@pytest.mark.parametrize("label", list(POISONED_PIPELINES))
def test_every_poisoned_rule_is_refused(label: str) -> None:
    result = validate_pipeline(replications=R, seed=5, pipeline=POISONED_PIPELINES[label])
    assert not result.passed, f"the gate accepted {label}"
    assert max(abs(c.z) for c in result.failures) > 5.0
    assert "FAIL" in result.render()


def test_a_clamped_variance_inside_the_test_is_refused_where_the_clamp_bites() -> None:
    def clamped(treated: np.ndarray, control: np.ndarray, alpha: float) -> Decision:
        d = decide(treated, control, alpha)
        return dataclasses.replace(d, se=d.components.se_grand_mean)

    case = AnalyticCase(tau2=0.0, sigma2=100.0, n_seeds=5, n_episodes=20, path_sigma2=400.0)
    result = validate_pipeline(case, deltas=(0.0, 1.0, 2.5), replications=R, pipeline=clamped)
    assert not result.passed
    # Conservative, not anticonservative: it detects too little at every effect.
    assert all(p.empirical < p.analytic for p in result.points)


def test_the_gate_is_a_function_of_its_seed() -> None:
    a = validate_pipeline(replications=500, seed=7)
    b = validate_pipeline(replications=500, seed=7)
    assert [p.rejections for p in a.points] == [p.rejections for p in b.points]
    assert a.sigma2 == b.sigma2


def test_the_path_term_is_the_only_thing_pairing_removes() -> None:
    # Same draws, path variance 0 against 1e4: the contrast must not move.
    quiet = dataclasses.replace(DEFAULT_CASE, path_sigma2=0.0)
    loud = dataclasses.replace(DEFAULT_CASE, path_sigma2=1e4)
    t0, c0 = simulate_arms(quiet, 1.0, np.random.default_rng(0))
    t1, c1 = simulate_arms(loud, 1.0, np.random.default_rng(0))
    np.testing.assert_allclose(t1 - c1, t0 - c0, rtol=0, atol=1e-9)
    assert np.var(c1) > 100 * np.var(c0)


def test_the_simulated_contrast_has_the_hand_specified_components() -> None:
    rng = np.random.default_rng(1)
    case = AnalyticCase(tau2=4.0, sigma2=9.0, n_seeds=400, n_episodes=10, path_sigma2=50.0)
    treated, control = simulate_arms(case, 0.0, rng)
    contrast = treated - control
    within = float(np.mean(np.var(contrast, axis=1, ddof=1)))
    between = float(np.var(contrast.mean(axis=1), ddof=1))
    # sd of the within estimate 9 sqrt(2/3600) = 0.21; of the seed-mean variance
    # 4.9 sqrt(2/399) = 0.35. Four of each is the tolerance.
    assert within == pytest.approx(9.0, abs=4 * 0.21)
    assert between == pytest.approx(4.0 + 9.0 / 10, abs=4 * 0.35)


def test_check_arithmetic() -> None:
    assert Check("x", 1.0, 1.0, 0.0).within()
    assert not Check("x", 1.1, 1.0, 0.0).within()
    assert Check("x", 1.3, 1.0, 0.1).z == pytest.approx(3.0)
    assert not Check("x", 1.5, 1.0, 0.1).within(k=4.0)


def test_degenerate_cases_are_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        AnalyticCase(tau2=-1.0, sigma2=1.0, n_seeds=5, n_episodes=5)
    with pytest.raises(ValueError, match="at least 2 seeds"):
        AnalyticCase(tau2=1.0, sigma2=1.0, n_seeds=1, n_episodes=5)
    with pytest.raises(ValueError, match="replications"):
        validate_pipeline(replications=1)
    with pytest.raises(ValueError, match="grid"):
        validate_pipeline(deltas=())


# -- section 5.3: the bootstrap on seed means -------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_the_bootstrap_agrees_with_the_analytic_rate_on_normal_seed_means(seed: int) -> None:
    m = np.random.default_rng(seed).normal(0.3, 1.0, size=50)
    check = bootstrap_detection(m, n_boot=10_000, seed=seed)
    # Not Monte Carlo alone: the bootstrap world's variance is (S-1)/S of the sample's,
    # an O(1/S) bias. At S = 50 the two sit within a few points of each other.
    assert check.agrees(0.03), (check.rate, check.analytic)
    assert check.n_seeds == 50 and check.mc_se < 0.006


def test_the_bootstrap_on_identical_seed_means_always_fires() -> None:
    check = bootstrap_detection(np.full(10, 2.0), n_boot=100)
    assert check.rate == 1.0 and check.analytic == 1.0 and check.gap == 0.0
    assert check.se == 0.0


def test_the_bootstrap_refuses_what_it_cannot_resample() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_detection(np.array([1.0]))
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_detection(np.array([1.0, 2.0]), n_boot=0)
