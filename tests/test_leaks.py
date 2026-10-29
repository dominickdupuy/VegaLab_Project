"""Dominick Dupuy: leak detection suite gates (document 10.4).

Property tests pass across generated paths; the poisoned variant is detected; the
smallest detected planted leak is recorded; and every observation field is both
tested under the probe and shown to have a working positive control. The calendar
invariant has its own controls: it fires on the #29 surface and on a term structure
inverted past what total variance allows, and is quiet on a normal one, on a legal
inversion, on the synthetic world and on the real provider over a fixture store.

The criterion these tests enforce is stated in ``optspread.leaks.PASS_CRITERION``.
"""

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optspread.config import EnvConfig, MarketConfig
from optspread.data import SyntheticSource, conformance_report
from optspread.env import OBS_NAMES, SpreadEnv
from optspread.leaks import (
    CALENDAR_ATOL,
    PASS_CRITERION,
    PoisonedChainProvider,
    calendar_control,
    calendar_violations,
    chain_violations,
    default_env,
    detect_leak,
    field_coverage,
    field_scales,
    leak_floor,
    perturbation_check,
    poison_field,
    sequence_violations,
    total_variance,
)
from optspread.market import SyntheticChainProvider
from optspread.policies import FlatPolicy
from optspread.realdata.panel import K_GRID, TENOR_GRID
from optspread.realdata.provider import HistoricalChainProvider, SurfaceStore
from optspread.rng import SeededStepRNG
from optspread.types import Chain, FloatArray

MARKET = MarketConfig()
AtmVol = Callable[[FloatArray], FloatArray]


def make_provider() -> SyntheticChainProvider:
    return SyntheticChainProvider(MARKET)


def make_env(eps: float = 0.0) -> SpreadEnv:
    """The suite env, from optspread.leaks so the gate report probes the same one."""
    return default_env(eps, MARKET, EnvConfig())


# -- perturbation property: no future stream reaches the present ----------- #


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    other=st.integers(0, 2**31 - 1),
    t=st.integers(0, MARKET.n_days),
)
def test_chain_at_t_is_independent_of_streams_after_t(seed: int, other: int, t: int) -> None:
    past_same, future_differs = perturbation_check(make_provider, seed, other, t)
    assert past_same
    assert future_differs or seed == other


# -- structural invariants -------------------------------------------------- #


def test_clean_sequence_has_no_violations() -> None:
    provider = make_provider()
    chains = [provider.reset(SeededStepRNG(9))]
    chains += [provider.step() for _ in range(provider.n_days)]
    assert sequence_violations(chains) == []


def test_invariants_catch_a_corrupted_chain() -> None:
    c = make_provider().reset(SeededStepRNG(0))
    bad = Chain(c.t, c.spot, c.r, c.strikes[::-1].copy(), c.expiry_days, -c.ivs)
    violations = chain_violations(bad)
    assert any("strikes" in v for v in violations)
    assert any("ivs" in v for v in violations)


# -- calendar arbitrage: total variance must not fall with tenor (#29) ------- #


def test_the_issue_surface_passes_bounds_and_parity_but_not_the_calendar_check() -> None:
    clean = make_provider().reset(SeededStepRNG(0))
    bad = calendar_control(clean)  # 60% at 21 days over 15% at 42
    atm = float(clean.strikes[clean.strike_col(clean.spot)])
    near, far = (int(e) for e in bad.expiry_days)
    assert bad.price("C", atm, far) < bad.price("C", atm, near), "the arbitrage is real"
    violations = chain_violations(bad)
    assert len(violations) == 1, violations  # bounds and parity are blind to it
    assert violations == calendar_violations(bad)
    message = violations[0]
    assert "calendar arbitrage between expiry rows 0 and 1 (21d to 42d)" in message
    # every strike is compared but the top one, whose moneyness at 42 days lies past
    # the longer row's last strike, and the shortfall is 0.36 * 21/252 - 0.0225 * 42/252
    assert f"at {len(clean.strikes) - 1} of {len(clean.strikes) - 1} strikes" in message
    assert "by 0.0262" in message
    assert calendar_violations(clean) == []


def test_the_calendar_check_runs_in_the_conformance_suite() -> None:
    class _Inverted(SyntheticChainProvider):
        def _chain(self, spot: float) -> Chain:
            return calendar_control(super()._chain(spot))

    class _Source(SyntheticSource):
        def provider(self) -> SyntheticChainProvider:
            return _Inverted(self.config)

    assert conformance_report(SyntheticSource(MARKET)).passed
    report = conformance_report(_Source(MARKET))
    assert not report.passed
    assert len(report.violations) == MARKET.n_days + 1  # one per chain on the path
    assert all(v.startswith("calendar arbitrage") for v in report.violations)


def _smile_store(atm_vol: AtmVol, carry: float = 0.04, n_days: int = 40) -> SurfaceStore:
    """A real-looking surface grid: a put skew and a smile that flatten with tenor."""
    k = K_GRID[None, :]
    t = np.asarray(TENOR_GRID, dtype=np.float64)[:, None]
    vols = atm_vol(t) - 0.35 * k * np.sqrt(30 / t) + 0.6 * k * k * np.sqrt(30 / t)
    return SurfaceStore(
        # Consecutive days inside the eval split (#28, pre-registration A8).
        dates=tuple(
            str(np.datetime64("2020-03-02") + np.timedelta64(i, "D")) for i in range(n_days)
        ),
        spot=np.full(n_days, 5000.0),
        carry=np.full(n_days, carry),
        ivs=np.repeat(vols[None], n_days, axis=0),
    )


# ATM vol by calendar days to expiry. The first is the usual contango. The other two
# are inverted, as on a stressed day: the legal one falls slower than 1/sqrt(T), so
# total variance still rises; the arbitrage one, 60% at a week to 20% at two months,
# falls faster, and w at 21 trading days (30 calendar) exceeds w at 42 (61).
def contango(t: FloatArray) -> FloatArray:
    return 0.13 + 0.07 * (1 - np.exp(-t / 150))


def legal_inversion(t: FloatArray) -> FloatArray:
    return 0.18 + 0.40 * np.exp(-t / 20)


def arb_inversion(t: FloatArray) -> FloatArray:
    return 0.15 + 0.60 * np.exp(-t / 25)


LONG_TENORS = (5, 10, 21, 42, 63, 126, 252)


def _historical(store: SurfaceStore, tenors: tuple[int, ...]) -> HistoricalChainProvider:
    return HistoricalChainProvider(MarketConfig(n_days=10, tenor_days=tenors), store)


@pytest.mark.parametrize("carry", [0.04, 0.0, -0.02])
@pytest.mark.parametrize("atm_vol", [contango, legal_inversion])
def test_a_legal_term_structure_passes_through_the_real_provider(
    atm_vol: AtmVol, carry: float
) -> None:
    store = _smile_store(atm_vol, carry)
    grid_w = store.ivs[0] ** 2 * np.asarray(TENOR_GRID, dtype=np.float64)[:, None]
    assert (np.diff(grid_w, axis=0) > 0).all(), "the control must be arbitrage-free on its grid"
    for tenors in ((21, 42), LONG_TENORS):
        provider = _historical(store, tenors)
        chains = [provider.reset(SeededStepRNG(1))]
        chains += [provider.step() for _ in range(provider.n_days)]
        assert sequence_violations(chains) == [], (tenors, carry)


@pytest.mark.parametrize("carry", [0.04, 0.0, -0.02])
def test_an_inverted_real_term_structure_is_caught(carry: float) -> None:
    store = _smile_store(arb_inversion, carry)
    chain = _historical(store, LONG_TENORS).reset(SeededStepRNG(1))
    violations = calendar_violations(chain)
    assert [v.split(" (")[0] for v in violations] == [
        "calendar arbitrage between expiry rows 1 and 2",  # 10d to 21d
        "calendar arbitrage between expiry rows 2 and 3",  # 21d to 42d
    ], violations
    assert chain_violations(chain) == violations, "and nothing else is wrong with it"
    w = total_variance(chain)
    atm = chain.strike_col(chain.spot)
    assert w[2, atm] > w[3, atm], "the 21-day ATM variance exceeds the 42-day"


def _boundary_chain(r: float, curvature: float, far_scale: float = 1.0) -> Chain:
    """Total variance exactly equal across tenors at every forward moneyness: zero
    forward variance, the legal edge. The smile is skewed and bent, so off r = 0 the
    longer row has to be read between its strikes. ``far_scale`` below 1 pushes every
    row after the first under it: one pair of rows in arbitrage."""
    base = make_provider().reset(SeededStepRNG(0))
    tenors = np.asarray([21, 42, 63], dtype=np.int64)
    taus = tenors / base.trading_days_per_year
    k = np.log(base.strikes / base.spot)[None, :] - r * taus[:, None]
    w = 0.006 * (1.0 - 2.5 * k + curvature * k * k)
    w[1:] *= far_scale
    ivs = np.sqrt(w / taus[:, None])
    return Chain(base.t, base.spot, r, base.strikes, base.t + tenors, ivs)


@pytest.mark.parametrize("curvature", [30.0, -10.0], ids=["smile", "frown"])
@pytest.mark.parametrize("r", [0.05, 0.0, -0.03])
def test_flat_total_variance_is_legal_and_one_percent_below_it_is_not(
    r: float, curvature: float
) -> None:
    assert calendar_violations(_boundary_chain(r, curvature)) == []
    caught = calendar_violations(_boundary_chain(r, curvature, far_scale=0.99))
    assert [v.split(" (")[0] for v in caught] == [
        "calendar arbitrage between expiry rows 0 and 1"
    ], caught


def test_the_interpolation_allowance_is_what_keeps_the_boundary_legal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A row that bends down between strikes is under-read by the linear interpolation,
    # so without the allowance the legal edge reads as arbitrage. On a smile flat in k
    # (r = 0 here puts every comparison on a strike) there is nothing to allow for.
    import optspread.leaks as leaks

    monkeypatch.setattr(leaks, "_interpolation_slack", lambda xs, ys, at: np.zeros_like(at))
    assert calendar_violations(_boundary_chain(0.05, -10.0)) != []
    assert calendar_violations(_boundary_chain(0.0, -10.0)) == []


def test_the_rounding_allowance_is_far_below_any_real_violation() -> None:
    bad = calendar_control(make_provider().reset(SeededStepRNG(0)))
    w = total_variance(bad)
    assert w[0, 0] - w[1, 0] > 1e9 * CALENDAR_ATOL
    assert calendar_violations(bad, atol=0.02) != []
    assert calendar_violations(bad, atol=0.03) == []


def test_a_chain_too_narrow_to_bend_is_still_checked() -> None:
    # Two strikes show no slope change to allow for; one expiry has nothing to compare.
    c = make_provider().reset(SeededStepRNG(0))
    narrow = Chain(c.t, c.spot, c.r, c.strikes[15:17], c.expiry_days, c.ivs[:, 15:17])
    assert calendar_violations(narrow) == []
    assert "at 1 of 1 strikes" in calendar_violations(calendar_control(narrow))[0]
    single = Chain(c.t, c.spot, c.r, c.strikes, c.expiry_days[:1], c.ivs[:1])
    assert calendar_violations(single) == []


def test_the_leak_report_runs_the_invariants_and_their_control(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.leak_report import invariants

    assert invariants(range(2))
    out = capsys.readouterr().out
    assert "total variance non-decreasing in tenor" in out
    assert "[PASS] clean world: no structural violation over 128 chains" in out
    assert "[PASS] calendar control" in out
    assert "- calendar arbitrage between expiry rows 0 and 1 (21d to 42d)" in out


# -- poisoned environment: the smoke detector reacts to smoke --------------- #

SEEDS = range(20)


def test_clean_environment_is_not_flagged() -> None:
    report = detect_leak(make_env, SEEDS)
    assert not report.flagged, report
    assert report.n_pairs == len(SEEDS) * MARKET.n_days


def test_poisoned_environment_is_flagged() -> None:
    # exact detector: the chain at t now moves when stream t+1 is resampled
    past_same, _ = perturbation_check(lambda: PoisonedChainProvider(MARKET, 1.0), 1, 2, t=10)
    assert not past_same
    # statistical detector: today's shown return predicts tomorrow's true return
    report = detect_leak(lambda: make_env(1.0), SEEDS)
    assert report.flagged, report
    assert report.worst_feature == "ret_1d"


def test_leak_floor_is_recorded() -> None:
    report = leak_floor(make_env, eps_grid=(1.0, 0.5, 0.25, 0.1, 0.05), seeds=SEEDS)
    assert report.floor is not None and report.floor <= 0.25
    # detection is monotone in the planted size
    flags = np.asarray(report.flagged)
    assert np.all(flags[:-1] >= flags[1:])


# -- every observation field, not just the price fields --------------------- #

COVERAGE_SEEDS = range(8)


def test_the_probe_exercises_all_but_the_constant_fields() -> None:
    # A flat book leaves every inventory, greek and margin slot at zero, so a
    # flat-only probe would report a pass for nine fields it never tested.
    report = detect_leak(make_env, COVERAGE_SEEDS)
    assert report.covers_every_field
    assert set(report.untested) == {"atm_iv"}, report.untested
    reason = next(f.reason for f in report.features if f.name == "atm_iv")
    assert reason, "an untested field must say why"


def test_a_flat_probe_would_have_left_most_fields_untested() -> None:
    # The defect this week's work fixed, kept as a test so it cannot come back.
    report = detect_leak(make_env, COVERAGE_SEEDS, policy=FlatPolicy())
    assert len(report.untested) > 8, report.untested
    assert report.covers_every_field, "even a weak probe must still report every field"


def test_every_field_has_a_working_positive_control() -> None:
    report = field_coverage(make_env, COVERAGE_SEEDS)
    assert report.passed, chr(10) + report.render()
    assert len(report.controls) == len(OBS_NAMES)
    assert report.uncovered == ()
    assert report.misattributed == (), "a leak must be attributed to the field it is in"


@pytest.mark.parametrize("field", ["time_frac", "has_position", "margin_frac", "theta_frac"])
def test_a_leak_planted_in_one_field_is_found_in_that_field(field: str) -> None:
    scales = field_scales(make_env, COVERAGE_SEEDS)
    report = detect_leak(
        lambda: poison_field(make_env(), field, 0.25, scales[field]), COVERAGE_SEEDS
    )
    assert report.flagged, (field, report.p_value)
    assert report.worst_feature == field


def test_an_unplanted_field_leaves_the_world_quiet() -> None:
    report = detect_leak(lambda: poison_field(make_env(), "margin_frac", 0.0), COVERAGE_SEEDS)
    assert not report.flagged, "eps=0 is the clean env and must not fire"


def test_poisoning_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValueError, match="not an observation field"):
        poison_field(make_env(), "not_a_field", 1.0)


def test_the_pass_criterion_is_written_down() -> None:
    # The gap the gate report exposed: this suite had no rule anyone else could
    # check, so it read as a pass every time it ran.
    assert "OBS_NAMES" in PASS_CRITERION
    assert "positive control" in PASS_CRITERION
