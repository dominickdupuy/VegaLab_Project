"""Dominick Dupuy: leak detection suite gates (document 10.4).

Property tests pass across generated paths; the poisoned variant is detected; the
smallest detected planted leak is recorded; and every observation field is both
tested under the probe and shown to have a working positive control.

The criterion these tests enforce is stated in ``optspread.leaks.PASS_CRITERION``.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optspread.config import EnvConfig, MarketConfig
from optspread.env import OBS_NAMES, SpreadEnv
from optspread.leaks import (
    PASS_CRITERION,
    PoisonedChainProvider,
    chain_violations,
    default_env,
    detect_leak,
    field_coverage,
    field_scales,
    leak_floor,
    perturbation_check,
    poison_field,
    sequence_violations,
)
from optspread.market import SyntheticChainProvider
from optspread.policies import FlatPolicy
from optspread.rng import SeededStepRNG
from optspread.types import Chain

MARKET = MarketConfig()


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
