"""Sweeps: one benchmark per value, the held-out paths shared, nothing recomputed twice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned

from optspread.alpha import benchmark as bm
from optspread.alpha.dataset import SynthSpec, build_dataset, load_dataset
from optspread.alpha.generators import ConstVrpParams
from optspread.alpha.sweep import AlphaSweepSpec, list_sweeps, run_sweep, sweep_path
from optspread.config import RunConfig
from optspread.types import ChainProvider

SPEC = SynthSpec(
    generator=ConstVrpParams(premium_pts=6.0), n_days=30, n_train=4, n_heldout=16, seed=3
)
HASH = "0123456789abcdef"


@pytest.fixture(autouse=True)
def small_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bm, "FRESH_N", 16)


def fake(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    assert config.train is not None
    return FakeLearned(config.train.seed)


def never(config: RunConfig, provider: ChainProvider) -> FakeLearned:
    raise AssertionError("trained again")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"values": (8, 4)}, "increasing"),
        ({"values": (4, 4)}, "increasing"),
        ({"values": (4,)}, "at least 2"),
        ({"axis": "total_timesteps", "arm": "flat", "values": (2048, 4096)}, "learned arm"),
        (
            {"axis": "total_timesteps", "values": (2048, 4096), "total_timesteps": 4096},
            "from values",
        ),
        ({"axis": "total_timesteps", "values": (100, 4096)}, "budgets"),
        ({"values": (0, 4), "total_timesteps": 4096}, "train path counts"),
        ({"values": (2, 4)}, "learned"),
        ({"arm": "flat", "values": (2, 4), "n_seeds": 2}, "deterministic"),
    ],
)
def test_a_sweep_spec_refuses_what_it_cannot_run(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AlphaSweepSpec.model_validate({"dataset": HASH, **fields})


def test_a_sweep_over_train_paths(tmp_path: Path) -> None:
    base = build_dataset(SPEC, tmp_path)
    spec = AlphaSweepSpec(
        dataset=base.hash, arm="ppo", values=(2, 4, 8), total_timesteps=4096, n_seeds=2
    )
    lines: list[str] = []
    sweep = run_sweep(spec, tmp_path, trainer=fake, log=lines.append)
    assert [p.n_train for p in sweep.points] == [2, 4, 8]
    assert sweep.points[1].dataset == base.hash  # n_train 4 is the open dataset itself
    assert len(set(sweep.datasets)) == 3 and base.hash in sweep.datasets
    heldout = [load_dataset(tmp_path, p.dataset).spot("signal", "heldout") for p in sweep.points]
    assert all(np.array_equal(h, heldout[0]) for h in heldout)
    assert len(lines) <= spec.stages
    assert sum("reused" in line for line in lines) == 2  # the oracle, chosen once
    assert sweep_path(tmp_path, spec).is_file()
    assert run_sweep(spec, tmp_path, trainer=never) == sweep
    for p in sweep.points:
        assert [s.key for s in list_sweeps(tmp_path, p.dataset)] == [sweep.key]
    assert list_sweeps(tmp_path, HASH) == []


def test_a_sweep_over_the_training_budget(tmp_path: Path) -> None:
    base = build_dataset(SPEC, tmp_path)
    spec = AlphaSweepSpec(dataset=base.hash, arm="a2c", axis="total_timesteps", values=(2048, 4096))
    sweep = run_sweep(spec, tmp_path, trainer=fake)
    assert [p.total_timesteps for p in sweep.points] == [2048, 4096]
    assert {p.dataset for p in sweep.points} == {base.hash} and sweep.datasets == (base.hash,)
