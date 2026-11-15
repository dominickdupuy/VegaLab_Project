"""The benchmark: an arm on the frozen data, against the oracle, the twin and a noise learner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned

from optspread.actions import FLAT, SHORT_STRANGLE
from optspread.alpha import benchmark as bm
from optspread.alpha.dataset import SynthSpec, build_dataset
from optspread.alpha.generators import ConstVrpParams, MrVrpParams, NoiseParams
from optspread.alpha.provider import DatasetChainProvider
from optspread.config import LearnedArm, RunConfig
from optspread.env import OBS_NAMES
from optspread.types import ChainProvider

SMALL = {"n_days": 30, "n_train": 4, "n_heldout": 16, "seed": 3}


def fake(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    assert config.train is not None and config.policy == config.train.algorithm
    assert isinstance(provider, DatasetChainProvider) and provider.n_days == 30
    return FakeLearned(config.train.seed)


@pytest.fixture(autouse=True)
def small_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "FRESH_N", 16)


def test_paired_unpaired_and_ratio() -> None:
    a = np.array([3.0, 5.0, 7.0, 9.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    value, se = bm.paired(a, b)
    d = a - b
    assert value == pytest.approx(d.mean()) and se == pytest.approx(d.std(ddof=1) / 2)
    value, se = bm.unpaired(a, b)
    assert value == pytest.approx(a.mean() - b.mean())
    assert se == pytest.approx(np.hypot(a.std(ddof=1) / 2, b.std(ddof=1) / 2))
    r, rse = bm.ratio(np.full(50, 2.0), np.full(50, 4.0))
    assert r == pytest.approx(0.5) and rse == pytest.approx(0.0, abs=1e-12)


def test_timing_seller_sells_only_when_rich() -> None:
    seller = bm.TimingSeller(SHORT_STRANGLE, roll_dte=5, iv_floor=0.21)
    obs = np.zeros(len(OBS_NAMES), dtype=np.float32)
    info = {"position": None, "day": 0}
    obs[OBS_NAMES.index("atm_iv")] = 0.20
    assert seller.act(obs, info) == FLAT
    obs[OBS_NAMES.index("atm_iv")] = 0.23
    assert seller.act(obs, info) == SHORT_STRANGLE


def test_menu_adds_timing_sellers_only_for_mr_vrp() -> None:
    plain = bm.menu(SynthSpec(generator=ConstVrpParams()))
    assert "flat" in plain and "roll_strangle" in plain
    assert not any(k.startswith("timing") for k in plain)
    timed = bm.menu(SynthSpec(generator=MrVrpParams()))
    assert {"timing_0", "timing_1", "timing_2"} <= set(timed)


def test_ppo_benchmark_on_a_rich_world(tmp_path: Path) -> None:
    ds = build_dataset(SynthSpec(generator=ConstVrpParams(premium_pts=6.0), **SMALL), tmp_path)
    lines: list[str] = []
    b = bm.run_benchmark(ds, "ppo", 4096, tmp_path, trainer=fake, log=lines.append)
    have = {(r.policy, r.world, r.split) for r in b.rows}
    assert have == {
        ("arm", "signal", "train"),
        ("arm", "signal", "heldout"),
        ("arm", "twin", "heldout"),
        ("noise_control", "twin", "train"),
        ("noise_control", "twin", "heldout"),
        ("oracle", "signal", "train"),
        ("oracle", "signal", "heldout"),
        ("oracle", "twin", "heldout"),
        ("flat", "signal", "heldout"),
    }
    flat = next(r for r in b.rows if r.policy == "flat")
    assert flat.mean == 0.0 and flat.se == 0.0 and flat.n == 16
    assert b.oracle != "flat"  # six vol points of premium is worth selling
    assert len(b.menu) == len(bm.menu(ds.spec))
    assert b.attainable.value is not None and b.attainable.value > 0
    assert b.noise_fit is not None and b.capture_ratio.verdict
    assert any("oracle" in line for line in lines)
    path = bm.bench_path(tmp_path, ds.hash, "ppo", 4096)
    assert path.is_file()
    assert bm.load_benchmark(path) == b
    # a second call reuses the file and never trains
    again = bm.run_benchmark(ds, "ppo", 4096, tmp_path, trainer=_never)
    assert again == b


def _never(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    raise AssertionError("trained again")


def test_scripted_arm_on_noise_has_no_edge_and_no_noise_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Short premium is negatively skewed: on 16 paths a strangle "beats" flat at z 2.8
    # because the rare large loss was not drawn. The production sample is 200.
    monkeypatch.setattr(bm, "FRESH_N", 200)
    ds = build_dataset(SynthSpec(generator=NoiseParams(), **SMALL), tmp_path)
    b = bm.run_benchmark(ds, "delta_seller", None, tmp_path)
    assert b.noise_control is False and b.noise_fit is None
    assert b.oracle == "flat"  # nothing to earn: every seller pays its costs
    assert b.attainable.value == 0.0
    assert b.capture_ratio.value is None and "no attainable edge" in b.capture_ratio.verdict
    # on noise the signal world and the twin are the same paths
    assert b.captured.value == pytest.approx(0.0)


def test_scripted_arm_rejects_a_budget_and_ppo_needs_one(tmp_path: Path) -> None:
    ds = build_dataset(SynthSpec(generator=NoiseParams(), **SMALL), tmp_path)
    with pytest.raises(ValueError, match="budget"):
        bm.run_benchmark(ds, "flat", 100, tmp_path)
    with pytest.raises(ValueError, match="budget"):
        bm.run_benchmark(ds, "ppo", None, tmp_path, trainer=fake)


@pytest.mark.parametrize("arm", ["ppo", "a2c", "dqn"])
def test_each_real_learner_trains_on_the_frozen_paths_only(arm: LearnedArm, tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    ds = build_dataset(SynthSpec(generator=ConstVrpParams(), **SMALL), tmp_path)
    provider = DatasetChainProvider(ds, "signal", "train")
    seen: list[int] = []
    original = provider.reset

    def spy(rng):  # type: ignore[no-untyped-def]
        chain = original(rng)
        seen.append(provider.path_index)
        return chain

    provider.reset = spy  # type: ignore[method-assign]
    policy = bm.learned_trainer(bm._train_config(ds, arm, 2048), provider)
    assert set(seen) == {0, 1, 2, 3}  # cycled every frozen train path, and nothing else
    assert len(bm.evaluate_on(ds, "signal", "heldout", policy)) == 16
