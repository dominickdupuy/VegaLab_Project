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


def test_one_seed_is_the_path_se_alone() -> None:
    a = np.array([3.0, 5.0, 7.0, 9.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    value, se = bm.paired(a, b)
    d = a - b
    assert value == pytest.approx(d.mean()) and se == pytest.approx(d.std(ddof=1) / 2)
    value, se = bm.unpaired(a, b)
    assert value == pytest.approx(a.mean() - b.mean())
    assert se == pytest.approx(np.hypot(a.std(ddof=1) / 2, b.std(ddof=1) / 2))


def test_several_seeds_add_the_seed_spread() -> None:
    # two seeds on the same four paths: path noise of the seed average, plus the seed spread
    m = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    value, se = bm.level(m)
    seed_var = np.var([2.5, 4.5], ddof=1) / 2
    path_var = np.var([2.0, 3.0, 4.0, 5.0], ddof=1) / 4
    assert value == pytest.approx(3.5) and se == pytest.approx(np.sqrt(seed_var + path_var))
    # identical seeds: nothing but the path noise
    assert bm.level(np.array([m[0], m[0]]))[1] == pytest.approx(bm.level(m[0])[1])


def test_ratio_by_the_delta_method() -> None:
    r, se = bm.ratio((2.0, 0.0), (4.0, 0.0))
    assert r == pytest.approx(0.5) and se == pytest.approx(0.0)
    r, se = bm.ratio((2.0, 1.0), (4.0, 2.0))
    assert se == pytest.approx(np.sqrt(1.0 + 0.25 * 4.0) / 4.0)


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
    assert b.n_seeds == 1 and len(b.seeds) == 1 and b.n_train == 4
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
    with pytest.raises(ValueError, match="n_seeds"):
        bm.run_benchmark(ds, "flat", None, tmp_path, n_seeds=2)
    with pytest.raises(ValueError, match="n_seeds"):
        bm.run_benchmark(ds, "ppo", 4096, tmp_path, trainer=fake, n_seeds=bm.MAX_SEEDS + 1)


def test_each_seed_trains_its_own_learner_and_noise_control(tmp_path: Path) -> None:
    ds = build_dataset(SynthSpec(generator=ConstVrpParams(premium_pts=6.0), **SMALL), tmp_path)
    calls: list[tuple[int, str]] = []

    def counting(config: RunConfig, provider: ChainProvider) -> FakeLearned:
        assert config.train is not None and isinstance(provider, DatasetChainProvider)
        calls.append((config.train.seed, provider.world))
        return FakeLearned(config.train.seed)

    lines: list[str] = []
    b = bm.run_benchmark(ds, "ppo", 4096, tmp_path, trainer=counting, log=lines.append, n_seeds=3)
    assert sorted(calls) == [(s, w) for s in range(3) for w in ("signal", "twin")]
    assert len(lines) == bm.stages("ppo", 3)
    assert [r.seed for r in b.seeds] == [0, 1, 2] and b.n_seeds == 3
    # the fake learner's structure depends on the seed, so the seeds disagree
    assert len({round(r.captured, 6) for r in b.seeds}) > 1
    assert b.captured.value == pytest.approx(np.mean([r.captured for r in b.seeds]))
    one = bm.run_benchmark(ds, "ppo", 4096, tmp_path, trainer=fake, n_seeds=1)
    assert bm.bench_path(tmp_path, ds.hash, "ppo", 4096, 3) != bm.bench_path(
        tmp_path, ds.hash, "ppo", 4096, 1
    )
    assert one.n_seeds == 1


def test_the_oracle_is_chosen_once_per_generator(tmp_path: Path) -> None:
    spec = SynthSpec(generator=ConstVrpParams(premium_pts=6.0), **SMALL)
    first: list[str] = []
    bm.run_benchmark(build_dataset(spec, tmp_path), "flat", None, tmp_path, log=first.append)
    bigger = build_dataset(spec.model_copy(update={"n_train": 8}), tmp_path)
    second: list[str] = []
    b = bm.run_benchmark(bigger, "flat", None, tmp_path, log=second.append)
    assert not any("reused" in line for line in first)
    assert any("reused" in line for line in second)
    assert len(second) == bm.stages("flat", 1) and b.oracle != "flat"
    assert bm.oracle_key(spec) == bm.oracle_key(bigger.spec)
    assert bm.oracle_key(spec) != bm.oracle_key(spec.model_copy(update={"seed": 4}))


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
    policy = bm.learned_trainer(bm.train_config(ds, arm, 2048, seed=0), provider)
    assert set(seen) == {0, 1, 2, 3}  # cycled every frozen train path, and nothing else
    assert len(bm.evaluate_on(ds, "signal", "heldout", policy)) == 16


def test_an_unreadable_cached_benchmark_is_recomputed(tmp_path: Path) -> None:
    ds = build_dataset(SynthSpec(generator=ConstVrpParams(premium_pts=6.0), **SMALL), tmp_path)
    first = bm.run_benchmark(ds, "delta_seller", None, tmp_path)
    path = bm.bench_path(tmp_path, ds.hash, "delta_seller", None, 1)
    path.write_text('{"dataset": "cut sh')  # a write cut short, or an older shape
    again = bm.run_benchmark(ds, "delta_seller", None, tmp_path)  # raised ValidationError
    assert again == first and bm.load_benchmark(path) == first
    assert sorted(p.name for p in path.parent.iterdir()) == ["benchmark.json"]
    # whether the oracle shows an edge is the benchmark's verdict, not a nonzero figure
    assert bm.edge_shown(first) == (first.attainable.verdict == bm.EDGE_SHOWN)
    weak = first.model_copy(
        update={"attainable": bm.BenchStat(value=150.0, se=100.0, verdict=bm.NO_EDGE)}
    )
    assert weak.attainable.value and not bm.edge_shown(weak)
