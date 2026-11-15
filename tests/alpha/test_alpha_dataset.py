"""Frozen synthetic datasets: hashed, reused, verified, and exactly what the generator made."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optspread.alpha.dataset import (
    HELDOUT_OFFSET,
    DatasetError,
    SynthSpec,
    build_dataset,
    load_dataset,
    preview,
    spec_hash,
)
from optspread.alpha.generators import ConstVrpParams, DriftParams, NoiseParams, generate
from optspread.rng import SeededStepRNG

SPEC = SynthSpec(
    generator=ConstVrpParams(premium_pts=4.0), n_days=20, n_train=3, n_heldout=5, seed=11
)


def test_hash_is_stable_and_sensitive() -> None:
    assert spec_hash(SPEC) == spec_hash(SynthSpec.model_validate(SPEC.model_dump()))
    assert len(spec_hash(SPEC)) == 16
    others = [
        SPEC.model_copy(update={"seed": 12}),
        SPEC.model_copy(update={"n_train": 4}),
        SPEC.model_copy(update={"generator": ConstVrpParams(premium_pts=4.5)}),
        SPEC.model_copy(update={"generator": DriftParams()}),
    ]
    assert len({spec_hash(SPEC), *(spec_hash(o) for o in others)}) == 5


def test_seeds_are_disjoint_by_split() -> None:
    train, held = SPEC.seeds("train"), SPEC.seeds("heldout")
    assert list(train) == [11, 12, 13]
    assert list(held) == [11 + HELDOUT_OFFSET + i for i in range(5)]


def test_held_out_paths_do_not_depend_on_the_train_size(tmp_path: Path) -> None:
    small = build_dataset(SPEC, tmp_path)
    large = build_dataset(SPEC.model_copy(update={"n_train": 7}), tmp_path)
    assert small.hash != large.hash
    for world in ("signal", "twin"):
        assert np.array_equal(small.spot(world, "heldout"), large.spot(world, "heldout"))
    # the smaller train set is the start of the larger one
    assert np.array_equal(small.spot("signal", "train"), large.spot("signal", "train")[:3])


def test_build_writes_the_generator_output(tmp_path: Path) -> None:
    ds = build_dataset(SPEC, tmp_path)
    assert ds.dir == tmp_path / "alpha" / "datasets" / ds.hash
    assert (ds.dir / "paths.npz").is_file() and (ds.dir / "dataset.json").is_file()
    market = SPEC.market()
    for i, seed in enumerate(SPEC.seeds("heldout")):
        sig = generate(SPEC.generator, market, SeededStepRNG(seed))
        twin = generate(NoiseParams(), market, SeededStepRNG(seed))
        assert np.array_equal(ds.spot("signal", "heldout")[i], sig.spot)
        assert np.array_equal(ds.iv("signal", "heldout")[i], sig.iv)
        assert np.array_equal(ds.signal("signal", "heldout")[i], sig.signal)
        assert np.array_equal(ds.spot("twin", "heldout")[i], twin.spot)
        assert np.array_equal(ds.iv("twin", "heldout")[i], twin.iv)
    assert ds.spot("signal", "train").shape == (3, 21)
    # const_vrp plants nothing in the path: signal and twin share it exactly
    assert np.array_equal(ds.spot("signal", "train"), ds.spot("twin", "train"))
    manifest = json.loads((ds.dir / "dataset.json").read_text())
    assert manifest["hash"] == ds.hash and manifest["spec"]["seed"] == 11


def test_a_second_build_reuses_the_files(tmp_path: Path) -> None:
    ds = build_dataset(SPEC, tmp_path)
    stamp = (ds.dir / "paths.npz").stat().st_mtime_ns
    again = build_dataset(SPEC, tmp_path)
    assert (ds.dir / "paths.npz").stat().st_mtime_ns == stamp
    assert np.array_equal(again.spot("signal", "train"), ds.spot("signal", "train"))
    loaded = load_dataset(tmp_path, ds.hash)
    assert loaded.spec == SPEC


def test_tampering_and_absence_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no dataset"):
        load_dataset(tmp_path, "0" * 16)
    with pytest.raises(DatasetError, match="hash"):
        load_dataset(tmp_path, "../etc")
    ds = build_dataset(SPEC, tmp_path)
    arrays = dict(np.load(ds.dir / "paths.npz"))
    arrays["signal_train_spot"][0, 1] += 1.0
    np.savez(ds.dir / "paths.npz", **arrays)
    with pytest.raises(DatasetError, match="digest"):
        load_dataset(tmp_path, ds.hash)


def test_preview(tmp_path: Path) -> None:
    ds = build_dataset(SPEC, tmp_path)
    view = preview(ds, n=2)
    assert view["days"] == list(range(21))
    assert len(view["signal"]["spot"]) == 2 and len(view["twin"]["iv"]) == 2
    assert view["signal"]["signal"][0][0] == 4.0


def test_datasets_and_benchmarks_are_never_listed_as_runs(tmp_path: Path) -> None:
    from optspread.dashboard.runindex import scan_runs

    build_dataset(SPEC, tmp_path)
    assert scan_runs(tmp_path) == ()
