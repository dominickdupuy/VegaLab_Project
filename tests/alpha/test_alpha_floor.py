"""The detection floor: the study's test on per-seed figures, and its curve and floor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned

from optspread.alpha import benchmark as bm
from optspread.alpha.dataset import WORLDS, SynthSpec, build_dataset, generate_arrays, load_dataset
from optspread.alpha.floor import ROUGH, EdgeTest, edge_floor, seed_test
from optspread.alpha.generators import ConstVrpParams, DriftParams, MrVrpParams, SpotMrParams
from optspread.alpha.sweep import (
    AlphaSweepSpec,
    check_edge,
    edge_knob,
    run_sweep,
    with_edge,
    zero_is_null,
)
from optspread.analysis import decide_contrast
from optspread.config import RunConfig
from optspread.m4analysis import noncentrality_for_power
from optspread.types import ChainProvider

HASH = "0123456789abcdef"
SPREAD = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # mean 0, sd 0.79


@pytest.fixture(autouse=True)
def small_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "FRESH_N", 16)


def fake(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    assert config.train is not None
    return FakeLearned(config.train.seed)


def test_the_seed_test_is_the_studys_test() -> None:
    rng = np.random.default_rng(7)
    contrast = rng.normal(0.4, 1.0, size=(5, 40))
    study = decide_contrast(contrast)
    ours = seed_test(1.0, contrast.mean(axis=1), "seeds")
    assert ours.estimate == pytest.approx(study.estimate)
    assert ours.se == pytest.approx(study.se)
    assert ours.t == pytest.approx(study.t_stat)
    assert ours.critical == pytest.approx(study.critical) and ours.df == study.df == 4
    assert ours.detected == study.rejected
    assert ours.lower_bound == pytest.approx(study.lower_bound)
    lo, mid, hi = ours.noncentrality
    assert lo < mid < hi and ours.power_lo < ours.power < ours.power_hi


def test_no_spread_is_an_infinite_t_or_none() -> None:
    sure = seed_test(2.0, [3.0, 3.0, 3.0], "seeds")
    assert sure.t is None and sure.detected and sure.power == pytest.approx(1.0)
    rounding = seed_test(2.0, [2845.504547931717] * 3, "seeds")  # sd of 1e-13, not 0
    assert rounding.se == 0.0 and rounding.t is None and rounding.detected
    huge = seed_test(2.0, [1.0, 1.0 + 1e-9, 1.0 - 1e-9], "seeds")  # t ~ 1e9: capped
    assert huge.t is None and huge.noncentrality[1] > 100
    null = seed_test(0.0, [0.0, 0.0, 0.0], "path blocks")
    assert null.t == 0.0 and not null.detected and null.power == pytest.approx(0.05)
    with pytest.raises(ValueError, match="at least 2"):
        seed_test(1.0, [1.0], "seeds")


def _tests(t_at: dict[float, float]) -> list[EdgeTest]:
    """Five per-seed figures at each size whose t statistic is the one given."""
    sd = float(SPREAD.std(ddof=1))
    return [seed_test(e, SPREAD + t * sd / np.sqrt(5), "seeds") for e, t in t_at.items()]


def test_a_floor_between_two_sizes() -> None:
    tests = _tests({0.0: 0.0, 1.0: 1.0, 2.0: 3.5, 4.0: 8.0})
    assert [t.detected for t in tests] == [False, False, True, True]
    f = edge_floor(tests, "vol points")
    assert f.status == "reached" and f.bracket == (1.0, 2.0)
    assert f.edge is not None and 1.0 < f.edge < 2.0
    assert f.null_detected is False and f.n == 5 and f.unit == "seeds"
    assert "between the swept sizes 1 and 2" in f.reading and f.note == ROUGH
    # the curve is the study's: power at lambda_80 is 0.80 where the floor is
    assert f.curve[0][:2] == (0.0, pytest.approx(0.05))
    lam = noncentrality_for_power(0.8, 5)
    assert lam > 0 and f.max_power is not None and f.max_power > 0.95


def test_below_and_above_the_grid() -> None:
    below = edge_floor(_tests({1.0: 9.0, 2.0: 12.0}), "bps a day")
    assert below.status == "below_grid" and below.edge is not None and below.edge < 1.0
    assert "below the smallest" in below.reading
    above = edge_floor(_tests({0.0: 0.0, 1.0: 0.2, 2.0: 0.5}), "vol points")
    assert above.status == "not_reached" and above.edge is None
    assert above.bracket == (2.0, None) and "above the largest" in above.reading
    none = edge_floor(_tests({0.0: 0.0}))
    assert none.status == "untestable" and none.curve == ()


def test_the_edge_knob_of_each_generator() -> None:
    base = SynthSpec(generator=ConstVrpParams(premium_pts=4.0), n_days=20)
    assert with_edge(base, 1.5).generator == ConstVrpParams(premium_pts=1.5)
    mr = SynthSpec(generator=MrVrpParams(half_life_days=5.0))
    assert with_edge(mr, 2.0).generator == MrVrpParams(vol_pts=2.0, half_life_days=5.0)
    assert edge_knob(DriftParams()) == ("bps_per_day", "bps a day")
    with pytest.raises(ValueError, match="no edge size"):
        edge_knob(SpotMrParams())
    with pytest.raises(ValueError):
        with_edge(base, 99.0)  # past the knob's bound


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"values": (-1, 2), "arm": "flat"}, "zero"),
        ({"values": (0, 0.0001), "arm": "flat"}, None),
        ({"values": (-2, -1), "arm": "flat"}, "zero"),
        ({"values": (1, 2), "total_timesteps": 4096, "n_seeds": 1}, "2 training seeds"),
        ({"values": (1, 2), "n_seeds": 3}, "learned"),
    ],
)
def test_an_edge_sweep_spec(fields: dict[str, object], message: str | None) -> None:
    raw = {"dataset": HASH, "axis": "edge", **fields}
    if message is None:
        assert AlphaSweepSpec.model_validate(raw).values == (0, 0.0001)
        return
    with pytest.raises(ValueError, match=message):
        AlphaSweepSpec.model_validate(raw)


def test_fractional_values_only_on_the_edge_axis() -> None:
    with pytest.raises(ValueError, match="whole numbers"):
        AlphaSweepSpec(dataset=HASH, arm="flat", values=(2, 4.5))
    whole = AlphaSweepSpec(dataset=HASH, arm="flat", values=(2.0, 4.0))
    assert whole.values == (2, 4) and all(isinstance(v, int) for v in whole.values)


@pytest.mark.slow
def test_an_edge_sweep_gives_each_size_a_test_and_the_arm_a_floor(tmp_path: Path) -> None:
    base = build_dataset(
        SynthSpec(generator=ConstVrpParams(premium_pts=4.0), n_days=20, n_train=3, n_heldout=10),
        tmp_path,
    )
    spec = AlphaSweepSpec(
        dataset=base.hash,
        arm="ppo",
        axis="edge",
        values=(0, 2, 6),
        total_timesteps=2048,
        n_seeds=3,
    )
    lines: list[str] = []
    sweep = run_sweep(spec, tmp_path, trainer=fake, log=lines.append)
    assert sweep.edge_knob == "premium_pts" and sweep.edge_unit == "vol points"
    assert len(lines) <= spec.stages and lines[-1].startswith("floor: ")
    sizes = [load_dataset(tmp_path, p.dataset).spec.generator for p in sweep.points]
    assert sizes == [ConstVrpParams(premium_pts=v) for v in (0.0, 2.0, 6.0)]
    held = [load_dataset(tmp_path, p.dataset).spot("signal", "heldout") for p in sweep.points]
    assert all(np.array_equal(h, held[0]) for h in held)  # one set of shocks at every size
    null = sweep.points[0]
    assert null.test is not None and null.test.estimate == 0.0 and not null.test.detected
    assert all(p.test is not None and p.test.unit == "seeds" for p in sweep.points)
    assert all(len(p.figures) == 3 for p in sweep.points)
    assert sweep.floor is not None and sweep.floor.status in {
        "reached",
        "below_grid",
        "not_reached",
    }
    assert sweep.floor.null_detected is False and sweep.floor.n == 3
    # a deterministic arm reads path blocks
    rule = run_sweep(
        AlphaSweepSpec(dataset=base.hash, arm="delta_seller", axis="edge", values=(0, 6)), tmp_path
    )
    assert rule.floor is not None and rule.floor.unit == "path blocks"
    assert rule.floor.n == bm.BLOCKS and len(rule.points[1].figures) == bm.BLOCKS


def test_a_deterministic_result_without_blocks_is_redone(tmp_path: Path) -> None:
    ds = build_dataset(
        SynthSpec(generator=ConstVrpParams(premium_pts=4.0), n_days=20, n_train=3, n_heldout=10),
        tmp_path,
    )
    b = bm.run_benchmark(ds, "delta_seller", None, tmp_path)
    assert len(b.blocks) == bm.BLOCKS
    path = bm.bench_path(tmp_path, ds.hash, "delta_seller", None)
    path.write_text(b.model_copy(update={"blocks": ()}).model_dump_json())
    assert bm.run_benchmark(ds, "delta_seller", None, tmp_path).blocks == b.blocks
    assert bm.captured_blocks(np.zeros(9), np.zeros(9)) == ()  # under 2 paths a block


@pytest.mark.parametrize(
    "generator",
    [ConstVrpParams(premium_pts=4.0), MrVrpParams(vol_pts=4.0), DriftParams(bps_per_day=5.0)],
)
def test_size_zero_is_the_null_where_the_floor_anchors_it(generator: object) -> None:
    base = SynthSpec(generator=generator, n_days=10, n_train=2, n_heldout=2)  # type: ignore[arg-type]
    assert zero_is_null(base.generator)
    arrays = generate_arrays(with_edge(base, 0.0))
    for split in ("train", "heldout"):
        for field in ("spot", "iv", "signal"):
            signal, twin = (arrays[f"{w}_{split}_{field}"] for w in WORLDS)
            assert np.array_equal(signal, twin), (split, field)


def test_an_mr_vrp_base_with_a_mean_premium_is_not_swept() -> None:
    base = SynthSpec(generator=MrVrpParams(mean_pts=3.0, vol_pts=4.0), n_days=10)
    assert not zero_is_null(base.generator)
    # at vol_pts 0 the premium is still 3 points: the signal world is not its twin
    arrays = generate_arrays(with_edge(base, 0.0).model_copy(update={"n_train": 1}))
    assert not np.array_equal(arrays["signal_train_iv"], arrays["twin_train_iv"])
    with pytest.raises(ValueError, match="the null only with mean_pts 0"):
        check_edge(base, (0, 2, 4))
    with pytest.raises(ValueError, match=r"cannot be drawn through \(0, 0\)"):
        edge_floor([seed_test(1.0, SPREAD + 3.0, "seeds")], "vol points", zero_is_null=False)


def test_edge_sizes_are_checked_against_the_knob_range() -> None:
    const = SynthSpec(generator=ConstVrpParams(premium_pts=4.0), n_days=10)
    assert check_edge(const, (0, 10, 20)) == ("premium_pts", "vol points")
    with pytest.raises(ValueError, match=r"edge size 20.5 is out of range for premium_pts"):
        check_edge(const, (0, 20.5))
    with pytest.raises(ValueError, match="bps_per_day"):
        check_edge(SynthSpec(generator=DriftParams(), n_days=10), (0, 51))
    with pytest.raises(ValueError, match="no edge size"):
        check_edge(SynthSpec(generator=SpotMrParams(), n_days=10), (0, 1))
