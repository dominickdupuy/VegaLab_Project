"""The replay provider serves the frozen bytes, and through the env behaves like the generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optspread.actions import SHORT_STRANGLE
from optspread.alpha.dataset import SynthSpec, build_dataset
from optspread.alpha.generators import MrVrpParams, NoiseParams
from optspread.alpha.provider import DatasetChainProvider, dataset_env
from optspread.config import RunConfig
from optspread.harness import build_env, run_episode
from optspread.leaks import sequence_violations
from optspread.policies import AlwaysOnPolicy, DeltaSellerPolicy
from optspread.rng import SeededStepRNG, SplicedStepRNG
from optspread.types import ChainProvider


def test_replays_the_stored_path(tmp_path: Path) -> None:
    ds = build_dataset(
        SynthSpec(generator=MrVrpParams(), n_days=15, n_train=2, n_heldout=4), tmp_path
    )
    p = DatasetChainProvider(ds, "signal", "heldout")
    assert isinstance(p, ChainProvider) and p.n_paths == 4 and p.n_days == 15
    chains = [p.reset(SeededStepRNG(4 + 3))]  # wraps: seed 7 is path 3
    assert p.path_index == 3
    while not p.done:
        chains.append(p.step())
    assert [c.spot for c in chains] == list(ds.spot("signal", "heldout")[3])
    assert [float(c.ivs[0, 0]) for c in chains] == list(ds.iv("signal", "heldout")[3])
    assert np.all(chains[5].ivs == chains[5].ivs[0, 0])
    assert sequence_violations(chains) == []
    with pytest.raises(RuntimeError):
        p.step()


def test_refuses_an_rng_it_cannot_index() -> None:
    rng = SplicedStepRNG(SeededStepRNG(0), SeededStepRNG(1), 3)
    with pytest.raises(TypeError, match="SeededStepRNG"):
        DatasetChainProvider.path_for(rng, 4)


def test_a_noise_dataset_trades_exactly_like_the_live_null_world(tmp_path: Path) -> None:
    spec = SynthSpec(generator=NoiseParams(), n_days=63, n_train=1, n_heldout=3, seed=40)
    ds = build_dataset(spec, tmp_path)
    env = dataset_env(ds, "signal", "heldout")
    live = build_env(RunConfig(market=spec.market()))
    for policy in (AlwaysOnPolicy(SHORT_STRANGLE), DeltaSellerPolicy()):
        for i, path_seed in enumerate(spec.seeds("heldout")):
            replayed = run_episode(env, policy, i)
            original = run_episode(live, policy, path_seed)
            assert replayed.pnl == pytest.approx(original.pnl, abs=1e-6)
            assert replayed.n_trades == original.n_trades
