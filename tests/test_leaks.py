"""Dominick Dupuy: leak detection suite gates (document 10.4).

Property tests pass across generated paths; the poisoned variant is detected; the
smallest detected planted leak is recorded.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from optspread.config import CostConfig, EnvConfig, MarketConfig
from optspread.costs import make_cost_model
from optspread.env import SpreadEnv
from optspread.leaks import (
    PoisonedChainProvider,
    chain_violations,
    detect_leak,
    leak_floor,
    perturbation_check,
    sequence_violations,
)
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG
from optspread.types import Chain

MARKET = MarketConfig()


def make_provider() -> SyntheticChainProvider:
    return SyntheticChainProvider(MARKET)


def make_env(eps: float = 0.0) -> SpreadEnv:
    provider = PoisonedChainProvider(MARKET, eps) if eps else make_provider()
    return SpreadEnv(provider, make_cost_model(CostConfig(enabled=False)), EnvConfig())


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
