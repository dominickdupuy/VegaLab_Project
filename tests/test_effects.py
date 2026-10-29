"""Hemdutt Rao: planted effects behind an interface (#32).

The default effect has to be the market as it was, bit for bit, or the confirmatory
sweep would no longer replay. Zero premium has to be the null world under every
effect, or the detection event (each arm against itself at zero) would test the
effect's shape rather than its presence. The two examples are checked for the shape
they claim, for reaching a seller, and against the conformance and leak checks; an
effect that breaks the chain is the positive control.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from optspread.actions import IRON_CONDOR, PUT_CREDIT_SPREAD, SHORT_STRANGLE, build_position
from optspread.config import EnvConfig, MarketConfig
from optspread.costs import ZeroCost
from optspread.data import OptionMetricsStubSource, SyntheticSource, conformance_report
from optspread.effects import (
    CONSTANT_PREMIUM,
    ConstantPremium,
    PlantedChainProvider,
    PlantedEffect,
    SkewPremium,
    TermPremium,
    plant,
)
from optspread.env import SpreadEnv
from optspread.leaks import detect_leak, perturbation_check
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG
from optspread.types import Chain, FloatArray

SHORT = MarketConfig(n_days=15, tenor_days=(8, 15))
EXAMPLES: tuple[PlantedEffect, ...] = (SkewPremium(), TermPremium())


def path(provider: SyntheticChainProvider | PlantedChainProvider, seed: int) -> list[Chain]:
    chains = [provider.reset(SeededStepRNG(seed))]
    return chains + [provider.step() for _ in range(provider.n_days)]


# -- the default effect is the market as it was ------------------------------ #


@pytest.mark.parametrize("premium", [0.0, 0.25, 1.0, 4.0, -3.7])
def test_the_default_effect_is_bit_identical_to_the_constant_premium(premium: float) -> None:
    cfg = MarketConfig(iv_premium_vol_pts=premium, sigma=0.31, tenor_days=(7, 21, 63))
    default = path(SyntheticChainProvider(cfg), 11)
    explicit = path(SyntheticChainProvider(cfg, ConstantPremium()), 11)
    for a, b in zip(default, explicit, strict=True):
        # What the market computed before the interface existed, written out.
        before = np.full(a.ivs.shape, cfg.sigma + cfg.iv_premium_vol_pts / 100.0)
        assert a.ivs.tobytes() == before.tobytes() == b.ivs.tobytes()
        assert a.spot == b.spot


def test_the_sources_plant_the_constant_premium_unless_told_otherwise() -> None:
    assert SyntheticSource(SHORT).effect is CONSTANT_PREMIUM
    assert OptionMetricsStubSource(SHORT).effect is CONSTANT_PREMIUM
    stub = OptionMetricsStubSource(SHORT, effect=TermPremium()).provider()
    assert isinstance(stub, SyntheticChainProvider) and isinstance(stub.effect, TermPremium)
    assert isinstance(SkewPremium(), PlantedEffect)


# -- the example effects ----------------------------------------------------- #


@pytest.mark.parametrize("effect", [CONSTANT_PREMIUM, *EXAMPLES])
def test_zero_premium_is_the_null_world_under_every_effect(effect: PlantedEffect) -> None:
    null = path(SyntheticChainProvider(SHORT), 5)
    for a, b in zip(null, path(SyntheticChainProvider(SHORT, effect), 5), strict=True):
        assert np.array_equal(a.ivs, b.ivs) and a.spot == b.spot
        assert np.all(a.ivs == SHORT.sigma)


def test_the_planted_effect_never_moves_the_path() -> None:
    cfg = SHORT.model_copy(update={"iv_premium_vol_pts": 2.0})
    spots = [c.spot for c in path(SyntheticChainProvider(cfg), 9)]
    for effect in EXAMPLES:
        assert [c.spot for c in path(SyntheticChainProvider(cfg, effect), 9)] == spots


def test_the_skew_premium_is_larger_on_out_of_the_money_puts() -> None:
    cfg = SHORT.model_copy(update={"iv_premium_vol_pts": 2.0})
    chain = SyntheticChainProvider(cfg, SkewPremium(put_slope=0.5)).reset(SeededStepRNG(0))
    premium = (chain.ivs - cfg.sigma) * 100.0
    taus = np.array([chain.tau(int(e)) for e in chain.expiry_days])
    forwards = chain.spot * np.exp(chain.r * taus)
    x = np.log(chain.strikes[None, :] / forwards[:, None]) / (cfg.sigma * np.sqrt(taus)[:, None])
    np.testing.assert_allclose(premium, 2.0 * (1.0 + 0.5 * np.maximum(-x, 0.0)), atol=1e-12)
    for row, xs in zip(premium, x, strict=True):
        assert np.all(np.diff(row[xs < 0]) < 0), "rises into the downside"
        np.testing.assert_allclose(row[xs >= 0], 2.0, atol=1e-12)  # calls: the ATM premium
    calls = SkewPremium(put_slope=0.0, call_slope=1.0).apply(chain, 2.0)
    assert calls[0, -1] > calls[0, chain.strikes.size // 2] > cfg.sigma


def test_the_term_premium_is_larger_at_short_tenors() -> None:
    cfg = SHORT.model_copy(update={"iv_premium_vol_pts": 2.0, "tenor_days": (5, 20, 80)})
    chain = SyntheticChainProvider(cfg, TermPremium(ref_days=20)).reset(SeededStepRNG(0))
    premium = (chain.ivs - cfg.sigma) * 100.0
    assert np.ptp(premium, axis=1).max() < 1e-12, "flat across strikes"
    np.testing.assert_allclose(premium[:, 0], [4.0, 2.0, 1.0], rtol=1e-12)


@pytest.mark.parametrize("effect", EXAMPLES)
def test_the_example_effects_conform_and_do_not_leak(effect: PlantedEffect) -> None:
    cfg = SHORT.model_copy(update={"iv_premium_vol_pts": 3.0})
    assert conformance_report(SyntheticSource(cfg, effect)).passed
    for t in (0, 4, 14):
        past_same, future_differs = perturbation_check(
            partial(SyntheticChainProvider, cfg, effect), 1, 2, t
        )
        assert past_same and future_differs
    env = EnvConfig(episode_days=15)
    assert not detect_leak(
        lambda: SpreadEnv(SyntheticChainProvider(cfg, effect), ZeroCost(), env), seeds=range(12)
    ).flagged


@pytest.mark.parametrize("effect", EXAMPLES)
@pytest.mark.parametrize("action", [IRON_CONDOR, SHORT_STRANGLE, PUT_CREDIT_SPREAD])
def test_the_example_effects_price_a_sold_structure_rich(
    effect: PlantedEffect, action: int
) -> None:
    """What a seller opens on the planted surface is worth more, at fair vol, than it paid."""
    rich = SHORT.model_copy(update={"iv_premium_vol_pts": 2.0})
    planted_chain = SyntheticChainProvider(rich, effect).reset(SeededStepRNG(0))
    fair_chain = SyntheticChainProvider(SHORT).reset(SeededStepRNG(0))
    position = build_position(action, planted_chain, EnvConfig())
    assert position is not None
    assert position.value(fair_chain) - position.entry_value() > 0.0


class _Negative:
    """Drives every vol below zero: a planted effect conformance must refuse."""

    def apply(self, fair: Chain, vol_pts: float) -> FloatArray:
        return fair.ivs - 1.0


def test_an_effect_that_breaks_the_chain_fails_conformance() -> None:
    report = conformance_report(SyntheticSource(SHORT, _Negative()))
    assert not report.passed
    assert any("positive" in v for v in report.violations), report.violations


# -- planting on a provider that plants nothing -------------------------------- #


def test_planting_on_a_null_provider_matches_the_market_at_that_premium() -> None:
    at = SHORT.model_copy(update={"iv_premium_vol_pts": 1.5})
    wrapped = PlantedChainProvider(SyntheticChainProvider(SHORT), CONSTANT_PREMIUM, 1.5)
    assert wrapped.n_days == SHORT.n_days and not wrapped.done
    for a, b in zip(path(wrapped, 4), path(SyntheticChainProvider(at), 4), strict=True):
        assert a.ivs.tobytes() == b.ivs.tobytes() and a.spot == b.spot and a.t == b.t
    assert wrapped.done
    fair = SyntheticChainProvider(SHORT).reset(SeededStepRNG(0))
    assert plant(TermPremium(), fair, 0.0).ivs.tobytes() == fair.ivs.tobytes()
