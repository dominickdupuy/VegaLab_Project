"""Dominick Dupuy: variance components and the power arithmetic behind the M2 sample size.

Pre-registration D1 fixes ``Var(grand mean) = tau^2 / S + sigma^2 / (S * N)`` and a
one-sided test at alpha = 0.05 powered at 80%. The oracles here are closed forms: the
estimator is checked on constructed matrices where the answer is exact, then on seeded
synthetic data against its own sampling standard error over a 20-seed sweep; the power
and sample-size functions are checked against the textbook one-sample formula and
against each other (N - 1 misses the target, N meets it).

Every number is synthetic.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm

from optspread.power import (
    budget,
    episodes_for,
    frontier,
    power,
    render_frontier,
    seeds_for,
    var_grand_mean,
    variance_budget,
    variance_components,
)

ALPHA, TARGET = 0.05, 0.80
Z = float(norm.ppf(1.0 - ALPHA) + norm.ppf(TARGET))  # 1.6449 + 0.8416 = 2.4865


def simulate(seed: int, tau2: float, sigma2: float, S: int, N: int) -> npt.NDArray[np.float64]:
    """y_ij = a_i + e_ij, a_i ~ N(0, tau2), e_ij ~ N(0, sigma2), one seeded generator."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, np.sqrt(tau2), size=(S, 1))
    e = rng.normal(0.0, np.sqrt(sigma2), size=(S, N))
    return np.asarray(a + e, dtype=np.float64)


# -- the formula and the estimator, exactly ---------------------------------- #


def test_var_grand_mean_is_the_pre_registered_formula() -> None:
    # 4/4 + 8/(4*2) = 1 + 1: exact in binary, so == rather than approx.
    assert var_grand_mean(tau2=4.0, sigma2=8.0, n_seeds=4, n_episodes=2) == 2.0


def test_zero_within_variance_gives_sigma_zero_and_tau_equal_to_the_seed_mean_variance() -> None:
    means = np.array([1.0, 2.0, 4.0, 7.0])
    outcomes = np.repeat(means[:, None], 3, axis=1)  # every episode in a seed identical
    vc = variance_components(outcomes)
    assert vc.sigma2 == 0.0
    assert vc.ms_within == 0.0
    # sample variance of (1, 2, 4, 7) is 21/3 = 7; float arithmetic on these exact
    # decimals is good to 1e-15, so rel=1e-12 is slack rather than tolerance
    assert vc.tau2 == pytest.approx(7.0, rel=1e-12)
    assert not vc.clamped
    assert vc.grand_mean == pytest.approx(3.5, rel=1e-12)
    assert vc.var_grand_mean == pytest.approx(7.0 / 4, rel=1e-12)


def test_zero_between_variance_clamps_tau_at_zero_and_reports_it() -> None:
    rows = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 3.0, 2.0]])
    vc = variance_components(rows)  # every seed mean is 2: MS_between is exactly 0
    assert vc.ms_between == 0.0
    assert vc.tau2 == 0.0
    assert vc.clamped
    # each row has sample variance 1, so MS_within = (4 rows * 2) / (4 * 2) = 1 exactly
    assert vc.sigma2 == 1.0
    assert vc.var_grand_mean == pytest.approx(1.0 / 12, rel=1e-12)


@pytest.mark.parametrize("seed", range(14))
def test_plug_in_variance_equals_the_seed_mean_variance_over_s_when_unclamped(seed: int) -> None:
    # Algebra: tau2 N + sigma2 = MS_between, so tau2/S + sigma2/(S N) = MS_between/(S N),
    # which is the sample variance of the S seed means divided by S. Holds whenever the
    # clamp is inactive, which tau2 = 100 against sigma2 = 1 at N = 5 guarantees.
    outcomes = simulate(seed, tau2=100.0, sigma2=1.0, S=6, N=5)
    vc = variance_components(outcomes)
    assert not vc.clamped
    seed_means = outcomes.mean(axis=1)
    assert vc.var_grand_mean == pytest.approx(float(np.var(seed_means, ddof=1)) / 6, rel=1e-12)
    assert vc.se_grand_mean == pytest.approx(np.sqrt(vc.var_grand_mean), rel=1e-12)


def test_the_estimator_recovers_planted_components_over_a_seed_sweep() -> None:
    tau2, sigma2, S, N = 4.0, 9.0, 40, 25
    seeds = range(20)
    estimates = [variance_components(simulate(s, tau2, sigma2, S, N)) for s in seeds]
    # Normal data: MS_within ~ sigma2 chi2/(S(N-1)) with Var 2 sigma2^2/(S(N-1)) = 0.169;
    # MS_between ~ (sigma2 + N tau2) chi2/(S-1) with Var 2(sigma2 + N tau2)^2/(S-1) = 609.3.
    # tau2_hat = (MS_between - MS_within)/N, SD = sqrt(609.3 + 0.169)/25 = 0.99; sigma2_hat
    # has SD 0.41. The mean over 20 seeds has SD/sqrt(20); 4 of those is the tolerance,
    # a false alarm of about 6e-5.
    sd_sigma2 = float(np.sqrt(2 * sigma2**2 / (S * (N - 1))))
    sd_tau2 = float(np.sqrt(2 * (sigma2 + N * tau2) ** 2 / (S - 1) + sd_sigma2**2) / N)
    mean_tau2 = float(np.mean([vc.tau2 for vc in estimates]))
    mean_sigma2 = float(np.mean([vc.sigma2 for vc in estimates]))
    assert mean_tau2 == pytest.approx(tau2, abs=4 * sd_tau2 / np.sqrt(len(seeds)))
    assert mean_sigma2 == pytest.approx(sigma2, abs=4 * sd_sigma2 / np.sqrt(len(seeds)))
    # MS_between sits near 109 +- 25 against MS_within near 9: the clamp never engages.
    assert not any(vc.clamped for vc in estimates)
    for vc in estimates:
        assert (vc.n_seeds, vc.n_episodes) == (S, N)
        assert abs(vc.grand_mean) < 4 * vc.se_grand_mean, "the grand mean is unbiased for 0"


# -- power and sample size -------------------------------------------------- #


def test_power_is_alpha_at_zero_effect() -> None:
    assert power(0.0, tau2=1.0, sigma2=1.0, n_seeds=5, n_episodes=10, alpha=ALPHA) == pytest.approx(
        ALPHA, abs=1e-12
    )


def test_power_is_monotone_in_effect_seeds_and_episodes() -> None:
    kw = {"tau2": 1.0, "sigma2": 4.0}
    by_delta = [power(d, n_seeds=8, n_episodes=10, **kw) for d in (0.1, 0.5, 1.0, 2.0)]
    by_seeds = [power(0.5, n_seeds=s, n_episodes=10, **kw) for s in (2, 4, 8, 16)]
    by_episodes = [power(0.5, n_seeds=8, n_episodes=n, **kw) for n in (1, 5, 25, 125)]
    for series in (by_delta, by_seeds, by_episodes):
        assert all(a < b for a, b in pairwise(series)), series
    assert power(50.0, n_seeds=8, n_episodes=10, **kw) == pytest.approx(1.0, abs=1e-12)


def test_variance_budget_is_the_squared_effect_over_the_summed_quantiles() -> None:
    assert variance_budget(0.5, alpha=ALPHA, target_power=TARGET) == pytest.approx(
        (0.5 / Z) ** 2, rel=1e-12
    )


def test_sample_size_reduces_to_the_textbook_formula_when_tau_is_zero() -> None:
    # One seed, no seed effect: N >= ((z_0.95 + z_0.80) sigma / delta)^2 = 24.73, so 25.
    assert episodes_for(0.5, tau2=0.0, sigma2=1.0, n_seeds=1) == 25
    assert int(np.ceil((Z * 1.0 / 0.5) ** 2)) == 25
    assert power(0.5, 0.0, 1.0, n_seeds=1, n_episodes=24) < TARGET
    assert power(0.5, 0.0, 1.0, n_seeds=1, n_episodes=25) >= TARGET


def test_an_infeasible_design_is_reported_not_rounded() -> None:
    # Budget (0.5 / 2.4865)^2 = 0.0404. With tau2 = 1 and 10 seeds the seed term alone is
    # 0.1: no number of episodes can rescue the design, and the answer must say so.
    assert episodes_for(0.5, tau2=1.0, sigma2=1.0, n_seeds=10) is None
    # At 30 seeds the seed term is 0.0333, leaving 0.0071 for sigma2/(S N): N = 5.
    n = episodes_for(0.5, tau2=1.0, sigma2=1.0, n_seeds=30)
    assert n == 5
    assert power(0.5, 1.0, 1.0, n_seeds=30, n_episodes=4) < TARGET
    assert power(0.5, 1.0, 1.0, n_seeds=30, n_episodes=5) >= TARGET
    # Solving for seeds at N = 5 instead: S >= (1 + 1/5) / 0.0404 = 29.7, so 30, and tight.
    s = seeds_for(0.5, tau2=1.0, sigma2=1.0, n_episodes=5)
    assert s == 30
    assert power(0.5, 1.0, 1.0, n_seeds=29, n_episodes=5) < TARGET
    assert power(0.5, 1.0, 1.0, n_seeds=30, n_episodes=5) >= TARGET


def test_budget_arithmetic_is_exact() -> None:
    b = budget(
        n_seeds=10,
        n_episodes=100,
        train_steps_per_seed=1_000_000,
        episode_days=63,
        train_rate=1_000.0,
        eval_rate=1_800.0,
        cores=4,
    )
    assert b.train_steps == 10_000_000
    assert b.eval_steps == 10 * 100 * 63
    assert b.total_steps == 10_000_000 + 63_000
    assert b.train_hours == pytest.approx(10_000_000 / 1_000.0 / 3600.0 / 4, rel=1e-12)
    assert b.eval_hours == pytest.approx(63_000 / 1_800.0 / 3600.0 / 4, rel=1e-12)
    assert b.total_hours == pytest.approx(b.train_hours + b.eval_hours, rel=1e-12)


def test_the_frontier_reports_each_seed_count_with_its_cheapest_episode_count() -> None:
    designs = frontier(
        0.5,
        tau2=1.0,
        sigma2=1.0,
        seeds=(10, 30, 60),
        train_steps_per_seed=100_000,
        episode_days=63,
        train_rate=1_000.0,
        eval_rate=1_800.0,
    )
    assert [d.n_seeds for d in designs] == [10, 30, 60]
    assert not designs[0].feasible and designs[0].n_episodes is None and designs[0].budget is None
    assert designs[1].n_episodes == 5
    assert designs[2].n_episodes == 1  # 1/(60 * (0.0404 - 1/60)) = 0.70, so one episode
    for d in designs[1:]:
        assert d.feasible and d.power is not None and d.budget is not None
        assert d.power >= TARGET
        assert d.budget.total_hours > 0.0
    rendered = render_frontier(designs)
    assert "infeasible" in rendered
    assert all(str(d.n_seeds) in rendered for d in designs)


def test_degenerate_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="seeds"):
        variance_components(np.zeros((1, 5)))
    with pytest.raises(ValueError, match="episodes"):
        variance_components(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="two-dimensional"):
        variance_components(np.zeros(6))
    with pytest.raises(ValueError, match="delta"):
        episodes_for(0.0, tau2=1.0, sigma2=1.0, n_seeds=3)
    with pytest.raises(ValueError, match="negative"):
        power(1.0, tau2=-1.0, sigma2=1.0, n_seeds=3, n_episodes=3)
    with pytest.raises(ValueError, match="alpha"):
        variance_budget(1.0, alpha=0.0, target_power=TARGET)
