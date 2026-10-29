"""Dominick Dupuy: variance components and the power arithmetic behind the M2 sample size.

Pre-registration D1 fixes ``Var(grand mean) = tau^2 / S + sigma^2 / (S * N)`` and a
one-sided test at alpha = 0.05 powered at 80%. The oracles here are closed forms: the
estimator is checked on constructed matrices where the answer is exact, then on seeded
synthetic data against its own sampling standard error over a 20-seed sweep; the power
and sample-size functions are checked against the textbook one-sample formula, against
the noncentral t by quadrature, and against each other (N - 1 misses the target, N
meets it). The planner is held to the exact power of the committed test (#20): the
normal approximation is checked only as the orientation figure it now is.

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
    detectable_effect,
    detectable_effect_t,
    episodes_for,
    frontier,
    power,
    power_t,
    render_frontier,
    seeds_for,
    t_critical,
    t_noncentrality,
    var_grand_mean,
    variance_budget,
    variance_budget_t,
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


def test_the_normal_form_reduces_to_the_textbook_formula_when_tau_is_zero() -> None:
    # One seed, no seed effect: N >= ((z_0.95 + z_0.80) sigma / delta)^2 = 24.73, so 25.
    assert int(np.ceil((Z * 1.0 / 0.5) ** 2)) == 25
    assert power(0.5, 0.0, 1.0, n_seeds=1, n_episodes=24) < TARGET
    assert power(0.5, 0.0, 1.0, n_seeds=1, n_episodes=25) >= TARGET


def test_the_exact_sample_size_at_zero_tau_and_the_normal_form_short_of_it() -> None:
    # Five seeds, no seed effect: 1 / (5 N) <= (0.5 / 3.0398)^2 = 0.02706, N >= 7.39, so 8.
    assert variance_budget_t(0.5, 5) == pytest.approx(0.027056, rel=1e-4)
    assert episodes_for(0.5, tau2=0.0, sigma2=1.0, n_seeds=5) == 8
    assert power_t(0.5, 0.0, 1.0, n_seeds=5, n_episodes=7) < TARGET
    assert power_t(0.5, 0.0, 1.0, n_seeds=5, n_episodes=8) >= TARGET
    # The normal form's 1 / (5 N) <= 0.0404 gives N = 5, at which the test has 66% power:
    # the shortfall of #20, and why nothing is planned on it any more.
    assert int(np.ceil(1.0 / (5 * variance_budget(0.5)))) == 5
    assert power_t(0.5, 0.0, 1.0, n_seeds=5, n_episodes=5) == pytest.approx(0.660, abs=1e-3)


def test_an_infeasible_design_is_reported_not_rounded() -> None:
    # Exact budget at S = 10: (0.5 / 2.6969)^2 = 0.0344. With tau2 = 1 the seed term alone is
    # 0.1: no number of episodes can rescue the design, and the answer must say so.
    assert episodes_for(0.5, tau2=1.0, sigma2=1.0, n_seeds=10) is None
    # At 30 seeds the budget is (0.5 / 2.5466)^2 = 0.03855 and the seed term 0.0333, leaving
    # 0.0052 for sigma2/(S N): N >= 6.39, so 7 (the normal form said 5).
    n = episodes_for(0.5, tau2=1.0, sigma2=1.0, n_seeds=30)
    assert n == 7
    assert power_t(0.5, 1.0, 1.0, n_seeds=30, n_episodes=6) < TARGET
    assert power_t(0.5, 1.0, 1.0, n_seeds=30, n_episodes=7) >= TARGET
    # Solving for seeds at N = 5 instead: 32 (the normal form said 30), and tight.
    s = seeds_for(0.5, tau2=1.0, sigma2=1.0, n_episodes=5)
    assert s == 32
    assert power_t(0.5, 1.0, 1.0, n_seeds=31, n_episodes=5) < TARGET
    assert power_t(0.5, 1.0, 1.0, n_seeds=32, n_episodes=5) >= TARGET
    # No episode variance: one episode, and with none at all the fewest seeds the test runs on.
    assert episodes_for(0.5, tau2=0.0, sigma2=0.0, n_seeds=5) == 1
    assert seeds_for(0.5, tau2=0.0, sigma2=0.0, n_episodes=5) == 2


@pytest.mark.parametrize("k", [3.25, 2.85])
def test_the_last_episode_is_settled_on_the_test_not_on_the_closed_form(
    monkeypatch: pytest.MonkeyPatch, k: float
) -> None:
    # A noncentrality off by enough to round the closed form to 9 (k = 3.25) or 7 (2.85)
    # where the test needs 8: the answer is still the fewest N power_t itself powers.
    import optspread.power as pw

    monkeypatch.setattr(pw, "t_noncentrality", lambda *_: k)
    assert pw.episodes_for(0.5, tau2=0.0, sigma2=1.0, n_seeds=5) == 8


def test_an_effect_no_seed_count_can_reach_is_refused() -> None:
    with pytest.raises(OverflowError, match="no seed count"):
        seeds_for(1e-300, tau2=1.0, sigma2=1.0, n_episodes=10)


@pytest.mark.parametrize(
    ("delta", "tau2", "sigma2"),
    [(0.5, 0.0, 1.0), (0.5, 0.01, 1.0), (824.3, 51_838.0, 105_996_836.0), (1.0, 0.2, 50.0)],
)
def test_the_planner_gives_the_fewest_episodes_and_seeds_the_exact_test_powers(
    delta: float, tau2: float, sigma2: float
) -> None:
    # The last-but-one row is the pilot's PPO contrast of A7 at its delta.
    for S in (2, 3, 5, 8, 20, 50):
        n = episodes_for(delta, tau2, sigma2, S)
        if n is None:  # the seed term alone is past the exact budget
            assert tau2 / S >= variance_budget_t(delta, S)
            continue
        assert power_t(delta, tau2, sigma2, S, n) >= TARGET
        assert n == 1 or power_t(delta, tau2, sigma2, S, n - 1) < TARGET
    for N in (1, 10, 200, 500, 1000):
        s = seeds_for(delta, tau2, sigma2, N)
        assert power_t(delta, tau2, sigma2, s, N) >= TARGET
        assert s == 2 or power_t(delta, tau2, sigma2, s - 1, N) < TARGET


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
    # The cases of the infeasible-design test above, by exact power.
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
    assert designs[1].n_episodes == 7
    assert designs[2].n_episodes == 1  # 1/(60 * (0.03951 - 1/60)) = 0.73, so one episode
    for d in designs[1:]:
        assert d.feasible and d.power is not None and d.budget is not None and d.n_episodes
        assert d.power == power_t(0.5, 1.0, 1.0, d.n_seeds, d.n_episodes) >= TARGET
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
    with pytest.raises(ValueError, match="delta"):
        seeds_for(-1.0, tau2=1.0, sigma2=1.0, n_episodes=3)
    with pytest.raises(ValueError, match="2 seeds"):  # one seed mean has no t test to plan
        episodes_for(0.5, tau2=0.0, sigma2=1.0, n_seeds=1)
    with pytest.raises(ValueError, match="episode"):
        seeds_for(0.5, tau2=1.0, sigma2=1.0, n_episodes=0)
    with pytest.raises(ValueError, match="must exceed alpha"):
        seeds_for(0.5, tau2=1.0, sigma2=1.0, n_episodes=3, target_power=0.01)
    with pytest.raises(ValueError, match="delta"):
        variance_budget_t(0.0, 5)
    with pytest.raises(ValueError, match="delta"):
        variance_budget(-1.0)
    with pytest.raises(ValueError, match="negative"):
        power(1.0, tau2=-1.0, sigma2=1.0, n_seeds=3, n_episodes=3)
    with pytest.raises(ValueError, match="alpha"):
        variance_budget(1.0, alpha=0.0, target_power=TARGET)


# -- the exact power of the confirmatory t test (week 6) --------------------- #


def _power_by_quadrature(delta: float, se: float, S: int, alpha: float) -> float:
    """Independent of scipy's noncentral t: condition on the chi-square in the denominator.

    T = (Z + nc) / sqrt(V / df) with V ~ chi2_df, so P(T > c) = E_V[Phi(nc - c sqrt(V/df))].
    """
    from scipy.integrate import quad
    from scipy.stats import chi2
    from scipy.stats import t as student_t

    df, nc = S - 1, delta / se
    c = float(student_t.ppf(1.0 - alpha, df))
    value, _ = quad(
        lambda v: norm.cdf(nc - c * np.sqrt(v / df)) * chi2.pdf(v, df), 0.0, np.inf, limit=200
    )
    return float(value)


def test_the_critical_value_is_student_t_and_tends_to_the_normal_one() -> None:
    from scipy.stats import t as student_t

    assert t_critical(10, ALPHA) == pytest.approx(float(student_t.ppf(0.95, 9)), rel=1e-12)
    assert t_critical(10, ALPHA) == pytest.approx(1.8331, abs=1e-4)  # the printed table value
    assert t_critical(2, ALPHA) > t_critical(5, ALPHA) > t_critical(50, ALPHA) > norm.ppf(0.95)
    assert t_critical(10**6, ALPHA) == pytest.approx(float(norm.ppf(0.95)), abs=1e-5)
    with pytest.raises(ValueError, match="2 seeds"):
        t_critical(1)
    with pytest.raises(ValueError, match="alpha"):
        t_critical(10, alpha=1.0)


@pytest.mark.parametrize(
    ("delta", "tau2", "sigma2", "S", "N"),
    [
        (0.5, 4.0, 100.0, 10, 25),
        (2.0, 4.0, 100.0, 10, 25),
        (1.0, 0.0, 9.0, 5, 20),
        (3.0, 1.0, 1.0, 3, 2),
    ],
)
def test_exact_power_matches_the_noncentral_t_by_quadrature(
    delta: float, tau2: float, sigma2: float, S: int, N: int
) -> None:
    se = float(np.sqrt(var_grand_mean(tau2, sigma2, S, N)))
    assert power_t(delta, tau2, sigma2, S, N, ALPHA) == pytest.approx(
        _power_by_quadrature(delta, se, S, ALPHA), abs=1e-7
    )


def test_exact_power_is_alpha_at_zero_and_sits_below_the_normal_approximation() -> None:
    assert power_t(0.0, 4.0, 100.0, 10, 25, ALPHA) == ALPHA
    # At S = 10 the t test pays for estimating its variance; the normal form does not.
    for delta in (0.5, 1.0, 2.0, 3.0):
        assert power_t(delta, 4.0, 100.0, 10, 25) < power(delta, 4.0, 100.0, 10, 25)
    # With thousands of seeds the two coincide.
    assert power_t(0.05, 1.0, 1.0, 5000, 10) == pytest.approx(
        power(0.05, 1.0, 1.0, 5000, 10), abs=1e-3
    )
    assert power_t(1.0, 0.0, 0.0, 5, 5) == 1.0
    assert power_t(0.0, 0.0, 0.0, 5, 5) == ALPHA


def test_with_no_variance_a_negative_effect_is_never_detected() -> None:
    # The limit as se falls to 0 (#31): a one-sided test cannot reject on an effect of the
    # wrong sign, so the power is 0, not alpha; the normal form agrees.
    assert power_t(-1.0, 0.0, 0.0, 5, 5) == 0.0
    assert power(-1.0, 0.0, 0.0, 5, 5) == 0.0
    assert power(0.0, 0.0, 0.0, 5, 5) == ALPHA and power(1.0, 0.0, 0.0, 5, 5) == 1.0
    # and continuously: a vanishing variance takes the power of a negative effect to 0
    assert power_t(-1.0, 0.0, 1e-12, 5, 5) < 1e-12


def test_the_detectable_effect_is_the_effect_at_which_power_hits_the_target() -> None:
    mde = detectable_effect(tau2=1.0, sigma2=4.0, n_seeds=8, n_episodes=10)
    assert power(mde, 1.0, 4.0, n_seeds=8, n_episodes=10) == pytest.approx(TARGET, abs=1e-12)
    assert variance_budget(mde) == pytest.approx(var_grand_mean(1.0, 4.0, 8, 10), rel=1e-12)
    # tau2 = 0, one seed: the textbook Z sigma / sqrt(N) = 2.4865 * 1 / 5.
    assert detectable_effect(0.0, 1.0, 1, 25) == pytest.approx(Z / 5.0, rel=1e-12)
    # The floor falls as 1/sqrt(S) at fixed N (project document 4.5): 50 seeds vs 5 is sqrt(10).
    ratio = detectable_effect(1.0, 4.0, 5, 10) / detectable_effect(1.0, 4.0, 50, 10)
    assert ratio == pytest.approx(np.sqrt(10.0), rel=1e-12)


def test_the_exact_t_floor_inverts_power_t_and_sits_above_the_normal_one() -> None:
    for S in (2, 5, 50):
        floor = detectable_effect_t(1.0, 4.0, S, 10)
        assert power_t(floor, 1.0, 4.0, S, 10) == pytest.approx(TARGET, abs=1e-9)
        assert floor > detectable_effect(1.0, 4.0, S, 10)
    # At five seeds the normal form understates the floor by more than a tenth.
    assert detectable_effect_t(1.0, 4.0, 5, 10) > 1.1 * detectable_effect(1.0, 4.0, 5, 10)
    big = detectable_effect_t(1.0, 4.0, 5000, 10) / detectable_effect(1.0, 4.0, 5000, 10)
    assert big == pytest.approx(1.0, abs=1e-3)
    assert detectable_effect_t(0.0, 0.0, 5, 10) == 0.0
    with pytest.raises(ValueError, match="must exceed alpha"):
        detectable_effect_t(1.0, 4.0, 5, 10, alpha=0.05, target_power=0.04)


def test_the_noncentrality_is_the_one_power_t_needs_by_quadrature() -> None:
    # Independent of scipy's nct: at the noncentrality the quadrature power is the target.
    for S in (2, 3, 5, 10, 50):
        k = t_noncentrality(S)
        assert _power_by_quadrature(k, 1.0, S, ALPHA) == pytest.approx(TARGET, abs=1e-7)
    assert t_noncentrality(5) == pytest.approx(3.0398, abs=1e-4)
    assert t_noncentrality(2) > t_noncentrality(3) > t_noncentrality(5) > t_noncentrality(50) > Z
    assert t_noncentrality(10**6) == pytest.approx(Z, abs=1e-5)
    # the budget is (delta / k)^2 and climbs to the normal one as S grows
    assert variance_budget_t(0.5, 5) == pytest.approx((0.5 / t_noncentrality(5)) ** 2, rel=1e-12)
    assert variance_budget_t(0.5, 10**6) == pytest.approx(variance_budget(0.5), rel=1e-5)
    with pytest.raises(ValueError, match="2 seeds"):
        t_noncentrality(1)


def test_the_audit_numbers_for_the_normal_floor() -> None:
    # Week 6 audit (#20). Tau^2 dominating makes the floor a pure function of S.
    def understated(S: int) -> float:
        return 1.0 - detectable_effect(1.0, 0.0, S, 1) / detectable_effect_t(1.0, 0.0, S, 1)

    assert understated(5) == pytest.approx(0.182, abs=1e-3)  # "about 18%"
    assert understated(3) == pytest.approx(0.375, abs=1e-3)  # "about 38%"
    # At the normal floor the test has 66% power at S = 5, not the 80% the form claims.
    normal_floor = detectable_effect(1.0, 0.0, 5, 1)
    assert power_t(normal_floor, 1.0, 0.0, 5, 1) == pytest.approx(0.656, abs=1e-3)
    # The S = 5 to S = 50 floor ratio is 3.8, not sqrt(10) = 3.16.
    exact = detectable_effect_t(1.0, 0.0, 5, 1) / detectable_effect_t(1.0, 0.0, 50, 1)
    assert exact == pytest.approx(3.812, abs=1e-3)
    assert detectable_effect(1.0, 0.0, 5, 1) / detectable_effect(1.0, 0.0, 50, 1) == (
        pytest.approx(np.sqrt(10.0), rel=1e-12)
    )
    # The section 8 case of the README: S = 10, the normal form 0.72 where the test has 0.66.
    assert power(2.0, 4.0, 100.0, 10, 25) == pytest.approx(0.723, abs=1e-3)
    assert power_t(2.0, 4.0, 100.0, 10, 25) == pytest.approx(0.662, abs=1e-3)


def test_component_standard_errors_on_a_constructed_matrix_are_exact() -> None:
    rows = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 3.0, 2.0]])
    vc = variance_components(rows)  # MS_within = 1, MS_between = 0, S = 4, N = 3
    assert vc.se_sigma2 == pytest.approx(np.sqrt(2.0 / 8.0), rel=1e-12)  # 0.5
    assert vc.se_tau2 == pytest.approx(0.5 / 3.0, rel=1e-12)  # sqrt(0 + 0.25) / 3
    assert vc.seed_mean_variance == 0.0
    assert "se 0.5000" in vc.render()


@pytest.mark.parametrize("seed", range(6))
def test_seed_mean_variance_is_the_sample_variance_of_seed_means_with_or_without_the_clamp(
    seed: int,
) -> None:
    # tau2 = 0 engages the clamp about half the time; the identity must not care.
    outcomes = simulate(seed, tau2=0.0, sigma2=1.0, S=5, N=4)
    vc = variance_components(outcomes)
    seed_means = outcomes.mean(axis=1)
    assert vc.seed_mean_variance == pytest.approx(float(np.var(seed_means, ddof=1)), rel=1e-12)
    if vc.clamped:
        assert vc.seed_mean_variance / vc.n_seeds < vc.var_grand_mean


def test_the_component_standard_errors_track_the_sampling_spread() -> None:
    tau2, sigma2, S, N = 4.0, 9.0, 40, 25
    estimates = [variance_components(simulate(s, tau2, sigma2, S, N)) for s in range(200)]
    raw_tau2 = np.array([(vc.ms_between - vc.ms_within) / N for vc in estimates])
    sigma2_hat = np.array([vc.sigma2 for vc in estimates])
    # The plug-in se is itself an estimate; its average should land within 15% of the
    # observed spread of 200 independent estimates (whose own sd is known to ~5%).
    assert np.mean([vc.se_tau2 for vc in estimates]) == pytest.approx(np.std(raw_tau2), rel=0.15)
    assert np.mean([vc.se_sigma2 for vc in estimates]) == pytest.approx(
        np.std(sigma2_hat), rel=0.15
    )
