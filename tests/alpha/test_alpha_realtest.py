"""The real-data test: a benchmark's policies on every real eval window, on a fixture panel.

The panel here is a small flat-surface one written under ``tmp_path`` (never the team's):
business days from June to the end of September 2022, so the eval split's last day
(2022-08-24) falls inside it and A8's holdout days are on disk but must never be played.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fake_learner import FakeLearned

from optspread.alpha import benchmark as bm
from optspread.alpha import plugins as plug
from optspread.alpha import realtest as rt
from optspread.alpha.dataset import SynthDataset, SynthSpec, build_dataset
from optspread.alpha.generators import ConstVrpParams
from optspread.alpha.realjobs import BenchRef, find_benchmark
from optspread.config import RunConfig
from optspread.policies import FlatPolicy
from optspread.realdata import HOLDOUT_START
from optspread.realdata.panel import K_GRID, TENOR_GRID
from optspread.rng import SeededStepRNG
from optspread.types import ChainProvider

SMALL = {"n_days": 10, "n_train": 3, "n_heldout": 8, "seed": 5}
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "alpha_plugin.py"
TIMER, PICKER = f"{EXAMPLE}:iv_timer", f"{EXAMPLE}:iv_picker"
FIRST, LAST = date(2022, 6, 1), date(2022, 9, 30)


def write_panel(root: Path, first: date = FIRST, last: date = LAST) -> Path:
    """The two files a provider reads, a flat 20% surface over a random-walk spot."""
    days = [d.date() for d in pd.bdate_range(first, last)]
    rng = np.random.default_rng(0)
    spot = 4000.0 * np.exp(np.cumsum(0.012 * rng.standard_normal(len(days))))
    grid = pd.DataFrame(
        [
            {"date": d, "tenor_days": t, "k": k, "iv": 0.20}
            for d in days
            for t in TENOR_GRID
            for k in K_GRID
        ]
    )
    daily = pd.DataFrame({"date": days, "spot": spot, "carry30": 0.0})
    (root / "panel").mkdir(parents=True)
    grid.to_parquet(root / "panel" / "surface_grid.parquet", index=False)
    daily.to_parquet(root / "panel" / "spx_daily.parquet", index=False)
    return root


@pytest.fixture(autouse=True)
def small_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "FRESH_N", 16)
    monkeypatch.setenv(plug.PLUGINS_ENV, "")
    monkeypatch.delenv(plug.PLUGINS_FILE_ENV, raising=False)


@pytest.fixture()
def examples(monkeypatch: pytest.MonkeyPatch) -> None:
    """The example plug-ins registered: ``iv_timer`` (a rule), ``iv_picker`` (a learner)."""
    monkeypatch.setenv(plug.PLUGINS_ENV, f"{TIMER},{PICKER}")


@pytest.fixture()
def data(tmp_path: Path) -> Path:
    return write_panel(tmp_path / "data")


@pytest.fixture()
def ds(tmp_path: Path) -> SynthDataset:
    return build_dataset(
        SynthSpec(generator=ConstVrpParams(premium_pts=4), **SMALL), tmp_path / "runs"
    )


def fake(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    assert config.train is not None
    return FakeLearned(config.train.seed)


def other(config: RunConfig, provider: ChainProvider) -> FlatPolicy:
    return FlatPolicy()  # not what the benchmark trained


@pytest.mark.behaviour
def test_a_scripted_arm_plays_every_eval_window_beside_its_synthetic_result(
    ds: SynthDataset, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "delta_seller", None, root)
    lines: list[str] = []
    out = rt.run_real_test(bench, ds, root, data, log=lines.append)
    windows = rt.real_windows(ds.spec.market(), data)
    assert out.n_windows == windows.n > 30 and out.split == "eval"
    assert out.first_date == "2022-06-01" and out.last_date == "2022-08-24"
    assert all(date.fromisoformat(d) < HOLDOUT_START for d in windows.ends), "no holdout day"
    assert [r.policy for r in out.rows] == ["arm", "oracle", "flat"]
    by = {r.policy: r for r in out.rows}
    held = {r.policy: r for r in bench.rows if r.world == "signal" and r.split == "heldout"}
    for policy in ("arm", "oracle", "flat"):
        assert by[policy].synthetic_mean == held[policy].mean
        assert by[policy].synthetic_se == held[policy].se
    assert by["flat"].real_mean == 0.0 and by["flat"].real_se == 0.0
    assert out.edge.value == pytest.approx(by["arm"].real_mean)
    assert out.reproduced is None and out.exploratory and "Exploratory" in out.note
    assert out.block >= ds.spec.n_days, "no block shorter than the overlap"
    assert "overlapping real SPX windows" in out.verdict
    assert len(lines) == rt.stages("delta_seller", 1)
    assert (root / "alpha" / "real" / f"{ds.hash}-delta_seller-0-s1" / "realtest.json").is_file()
    # the same benchmark on the same panel is read back, not played again
    monkeypatch.setattr(rt, "evaluate_real", lambda *a: pytest.fail("recomputed"))
    assert rt.run_real_test(bench, ds, root, data) == out


@pytest.mark.behaviour
@pytest.mark.slow
def test_a_learned_arm_is_rebuilt_per_seed_and_checked_against_its_benchmark(
    ds: SynthDataset, data: Path, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "ppo", 2048, root, trainer=fake, n_seeds=2)
    out = rt.run_real_test(bench, ds, root, data, trainer=fake)
    assert out.reproduced is True and out.n_seeds == 2
    arm = out.rows[0]
    assert len(arm.real_per_seed) == 2 and arm.real_mean == pytest.approx(
        np.mean(arm.real_per_seed)
    )
    assert out.reproduced_mean == pytest.approx(arm.synthetic_mean)
    # policies that are not the benchmark's are scored, and said not to reproduce it
    other_root = tmp_path / "other"
    other_ds = build_dataset(ds.spec, other_root)
    bench2 = bm.run_benchmark(other_ds, "ppo", 2048, other_root, trainer=fake, n_seeds=2)
    assert bench2.rows[1].mean != 0.0
    out2 = rt.run_real_test(bench2, other_ds, other_root, data, trainer=other)
    assert out2.reproduced is False and out2.reproduced_mean == 0.0


@pytest.mark.behaviour
def test_without_a_panel_the_test_says_so_and_runs_nothing(
    ds: SynthDataset, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "flat", None, root)
    empty = tmp_path / "empty"
    assert not rt.real_available(empty)
    assert "not available on this machine" in rt.unavailable_message(empty)
    with pytest.raises(rt.RealDataUnavailableError, match="not available on this machine"):
        rt.run_real_test(bench, ds, root, empty)
    assert not (root / "alpha" / "real").exists()


@pytest.mark.behaviour
def test_refusals(ds: SynthDataset, data: Path, tmp_path: Path) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "flat", None, root)
    with pytest.raises(ValueError, match="registered plug-ins, not mine"):
        rt.run_real_test(bench.model_copy(update={"arm": "mine"}), ds, root, data)
    with pytest.raises(ValueError, match="is not on dataset"):
        rt.run_real_test(bench.model_copy(update={"dataset": "0" * 16}), ds, root, data)


@pytest.mark.invariant
def test_windows_are_served_in_date_order_one_per_seed(ds: SynthDataset, data: Path) -> None:
    windows = rt.real_windows(ds.spec.market(), data)
    assert list(windows.starts) == sorted(windows.starts) and len(set(windows.starts)) == windows.n
    spots = []
    for i in (0, 1, windows.n - 1, windows.n):  # seed n wraps to the first window
        chain = windows.provider.reset(SeededStepRNG(i))
        spots.append(chain.spot)
    assert spots[0] == spots[3] and spots[0] != spots[1]


@pytest.mark.invariant
def test_the_se_is_the_overlap_aware_bootstrap_not_the_independent_one() -> None:
    rng = np.random.default_rng(1)
    shocks = rng.standard_normal(600)
    overlap = 20
    series = np.convolve(shocks, np.ones(overlap), mode="valid")  # windows sharing days
    mean, se, block = rt.overlap_level(series, overlap)
    naive = series.std(ddof=1) / np.sqrt(series.size)
    assert mean == pytest.approx(series.mean()) and block >= overlap
    assert se > 2.5 * naive, "overlapping windows are far from independent"
    # several seeds add the spread of their means, as benchmark.level does
    two = np.vstack([series, series + 10.0])
    _, se2, _ = rt.overlap_level(two, overlap)
    assert se2 > se
    assert rt.overlap_level(np.zeros(50), overlap)[1] == 0.0
    assert rt.usd(-51.2) == "-$51" and rt.usd(1234.0) == "$1,234"


@pytest.mark.behaviour
def test_a_rule_plug_in_is_scored_as_it_stands(
    examples: None, ds: SynthDataset, data: Path, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "iv_timer", None, root)
    assert bench.plugin is not None
    lines: list[str] = []
    out = rt.run_real_test(bench, ds, root, data, log=lines.append)
    assert out.arm == "iv_timer" and out.reproduced is None and out.n_seeds == 1
    assert len(lines) == rt.stages("iv_timer", 1)
    assert "IV timer (plug-in)" in out.verdict
    held = {r.policy: r for r in bench.rows if r.world == "signal" and r.split == "heldout"}
    assert out.rows[0].synthetic_mean == held["arm"].mean
    found = plug.find("iv_timer")
    assert found is not None
    digest = found.digest
    name = f"{ds.hash}-iv_timer.{digest}-0-s1"
    assert (root / "alpha" / "real" / name / "realtest.json").is_file()
    assert out.benchmark == f"alpha/bench/{name}/benchmark.json"
    assert (root / out.benchmark).is_file()


@pytest.mark.behaviour
def test_a_learner_plug_in_is_rebuilt_through_its_own_trainer_and_reproduces(
    examples: None, ds: SynthDataset, data: Path, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "iv_picker", 2048, root, n_seeds=2)
    lines: list[str] = []
    # the built-in trainer is never used for a plug-in: its own registered one is
    out = rt.run_real_test(
        bench,
        ds,
        root,
        data,
        trainer=lambda *a: pytest.fail("not the plug-in's trainer"),
        log=lines.append,
    )
    assert out.n_seeds == 2 and len(out.rows[0].real_per_seed) == 2
    assert out.reproduced is True
    assert out.reproduced_mean == pytest.approx(out.rows[0].synthetic_mean)
    assert len(lines) == rt.stages("iv_picker", 2) == 2 * 2 + 4
    assert "IV picker (plug-in)" in out.verdict


@pytest.mark.behaviour
def test_a_plug_in_gone_or_edited_is_refused_with_the_reason(
    examples: None, ds: SynthDataset, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    bench = bm.run_benchmark(ds, "iv_timer", None, root)
    stored = rt.stored_plugin(bench)
    assert stored is not None
    source, digest = stored
    assert source == TIMER and len(digest) == 8
    edited = bench.model_copy(update={"plugin": f"{source}@00000000"})
    with pytest.raises(rt.PluginChangedError, match=r"has changed since.*00000000 then"):
        rt.run_real_test(edited, ds, root, data)
    ref = BenchRef(dataset=ds.hash, arm="iv_timer")
    assert find_benchmark(root, ref) == bench
    monkeypatch.setenv(plug.PLUGINS_ENV, PICKER)  # iv_timer no longer registered
    # the benchmark is still found by its stored digest, and refused, not tested
    assert find_benchmark(root, ref) == bench
    with pytest.raises(rt.PluginChangedError, match="no longer registered"):
        rt.run_real_test(bench, ds, root, data)
    with pytest.raises(FileNotFoundError, match="run it first"):
        find_benchmark(root, BenchRef(dataset=ds.hash, arm="iv_picker", total_timesteps=2048))
    assert not (root / "alpha" / "real").exists()
