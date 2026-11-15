"""Synthetic generators: the planted edge is there, the twin is the null world, no look-ahead."""

from __future__ import annotations

import numpy as np
import pytest

from optspread.alpha.generators import (
    GENERATORS,
    ConstVrpParams,
    DriftParams,
    GeneratorParams,
    MrVrpParams,
    NoiseParams,
    SpotMrParams,
    SynthPath,
    generate,
    twin_of,
)
from optspread.config import MarketConfig
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG, SplicedStepRNG

MARKET = MarketConfig()
N = 400


def _paths(params: GeneratorParams, n: int = N, market: MarketConfig = MARKET) -> list[SynthPath]:
    return [generate(params, market, SeededStepRNG(s)) for s in range(n)]


def _spots(params: GeneratorParams, n: int = N) -> np.ndarray:
    return np.stack([p.spot for p in _paths(params, n)])


def test_noise_is_the_existing_null_world() -> None:
    for seed in range(5):
        provider = SyntheticChainProvider(MARKET)
        spots = [provider.reset(SeededStepRNG(seed)).spot]
        while not provider.done:
            spots.append(provider.step().spot)
        path = generate(NoiseParams(), MARKET, SeededStepRNG(seed))
        assert np.array_equal(path.spot, np.array(spots))
        assert np.all(path.iv == MARKET.sigma)
        assert np.all(path.signal == 0.0)
        assert path.spot.shape == path.iv.shape == path.signal.shape == (MARKET.n_days + 1,)


def test_const_vrp_shifts_iv_only() -> None:
    p = generate(ConstVrpParams(premium_pts=3.0), MARKET, SeededStepRNG(7))
    q = generate(NoiseParams(), MARKET, SeededStepRNG(7))
    assert np.array_equal(p.spot, q.spot)
    assert np.allclose(p.iv, MARKET.sigma + 0.03, atol=0, rtol=0)
    assert np.all(p.signal == 3.0)


def test_drift_adds_exactly_its_drift() -> None:
    assert np.array_equal(_spots(DriftParams(bps_per_day=0.0), 5), _spots(NoiseParams(), 5))
    drift = np.diff(np.log(_spots(DriftParams(bps_per_day=5.0), 50)), axis=1)
    noise = np.diff(np.log(_spots(NoiseParams(), 50)), axis=1)
    assert np.allclose(drift - noise, 5e-4, atol=1e-9)
    path = generate(DriftParams(bps_per_day=5.0), MARKET, SeededStepRNG(0))
    assert path.signal[-1] == pytest.approx(5.0 * MARKET.n_days)


def test_spot_mean_reversion_shrinks_terminal_variance() -> None:
    mr = np.log(_spots(SpotMrParams(half_life_days=10.0))[:, -1])
    gbm = np.log(_spots(NoiseParams())[:, -1])
    assert mr.var() < 0.5 * gbm.var()
    slow = _spots(SpotMrParams(half_life_days=1e9), 5)
    assert np.allclose(slow, _spots(NoiseParams(), 5), rtol=1e-6)


def test_mr_vrp_premium_has_its_stated_distribution() -> None:
    flat = generate(MrVrpParams(mean_pts=0.0, vol_pts=0.0), MARKET, SeededStepRNG(3))
    assert np.allclose(flat.iv, MARKET.sigma)
    params = MrVrpParams(mean_pts=1.0, vol_pts=4.0, half_life_days=10.0)
    prem = np.stack([p.signal for p in _paths(params)])
    assert prem.mean() == pytest.approx(1.0, abs=0.5)
    assert prem.std() == pytest.approx(4.0, rel=0.15)
    iv = np.stack([p.iv for p in _paths(params, 20)])
    sig = np.stack([p.signal for p in _paths(params, 20)])
    assert np.allclose(iv, np.maximum(MARKET.sigma + sig / 100.0, 0.01))
    # the spot path is the null world's: the premium is in the price, not the path
    assert np.array_equal(_spots(params, 5), _spots(NoiseParams(), 5))


ALL: tuple[GeneratorParams, ...] = (
    NoiseParams(),
    ConstVrpParams(),
    MrVrpParams(),
    SpotMrParams(),
    DriftParams(),
)


@pytest.mark.parametrize("params", ALL, ids=lambda p: p.kind)
@pytest.mark.parametrize("t", [0, 10, 40])
def test_no_look_ahead(params: GeneratorParams, t: int) -> None:
    base = generate(params, MARKET, SeededStepRNG(1))
    spliced = generate(params, MARKET, SplicedStepRNG(SeededStepRNG(1), SeededStepRNG(2), t))
    for a, b in ((base.spot, spliced.spot), (base.iv, spliced.iv), (base.signal, spliced.signal)):
        assert np.array_equal(a[: t + 1], b[: t + 1])
    assert not np.array_equal(base.spot, spliced.spot)


@pytest.mark.parametrize("params", ALL, ids=lambda p: p.kind)
def test_twin_is_noise_and_every_kind_is_described(params: GeneratorParams) -> None:
    assert twin_of(params) == NoiseParams()
    info = GENERATORS[params.kind]
    assert info.title and info.edge and info.taker
