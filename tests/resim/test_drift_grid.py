"""Hemdutt Rao: the drift sensitivity (A11), on a grid small enough to run here.

The grid is written by the real orchestrator in the study's world (r = 0.04), with the
fixed-delta seller and the fake learner of ``tests/fake_learner.py`` standing in for
PPO, so the cells are the objects a sweep leaves behind. Every number is synthetic.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread.config import STUDY_R, CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.drift import (
    FAIR_R,
    STATEMENT,
    DriftSensitivity,
    Provenance,
    changes,
    drift_grid,
    fair_config,
    reading,
    render,
    render_json,
    resimulate_cell,
)
from optspread.experiment.grid import GridConfig, grid_filename
from optspread.experiment.schedule import run_grid
from optspread.harness import run_episodes
from optspread.manifest import RunManifest
from optspread.policies import Policy
from optspread.runs.rundir import run_id
from optspread.train import POLICY_FILENAME
from scripts.drift_sensitivity import main

BASE = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(r=STUDY_R, n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(
    base=BASE,
    premiums=(0.0, 2.0, 4.0),
    arms=("delta_seller", "ppo"),
    n_seeds=3,
    train=TrainConfig(total_timesteps=64, seed=10),
)
SELLERS = GRID.model_copy(update={"arms": ("delta_seller",), "train": None})
FAST = {"n_boot": 200, "replications": 100}
PROV = Provenance(date(2026, 11, 14), "~/runs", "abc1234567", False, {"numpy": "x"}, ("f00",))


def fake_loader(path: Path, config: RunConfig) -> Policy:
    assert path.name == POLICY_FILENAME and path.read_text().startswith("fake learner")
    assert config.train is not None
    return FakeLearned(config.train.seed)


def other_loader(path: Path, config: RunConfig) -> Policy:
    """Another policy than the one saved: the original world then differs from run.json."""
    assert config.train is not None
    return FakeLearned(config.train.seed + 1)


def snapshot(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


@pytest.fixture(scope="module")
def grid_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("drift") / "sweep"
    run_grid(GRID, root, trainer=fake_trainer)
    return root


@pytest.fixture(scope="module")
def result(grid_root: Path) -> DriftSensitivity:
    before = snapshot(grid_root)
    out = drift_grid(grid_root, load_policy=fake_loader, **FAST)
    assert snapshot(grid_root) == before  # nothing written into the sweep
    return out


@pytest.mark.invariant
def test_the_original_world_reproduces_and_the_fair_world_is_the_config_at_r_zero(
    grid_root: Path, result: DriftSensitivity
) -> None:
    assert len(result.cells) == 3 * 2 * 3
    assert result.n_reproduce == len(result.cells) and result.n_differ == 0
    moved = 0
    for cell, worlds in result.cells:
        assert worlds.saved == worlds.original and worlds.differing == ()
        config = RunManifest.read(grid_root / cell.run_id).config
        assert config.market.r == STUDY_R
        fair = fair_config(config)
        assert worlds.fair_run_id == run_id(fair) != cell.run_id
        policy = (
            fake_loader(grid_root / cell.run_id / POLICY_FILENAME, config) if config.train else None
        )
        assert worlds.fair == tuple(e.pnl for e in run_episodes(fair, policy))
        moved += worlds.fair != worlds.original
    assert moved > 0  # the rate moves the P&L
    # The saved rows and the original world are one set of numbers, so one analysis.
    for arm in GRID.arms:
        saved, original = result.arms["saved"][arm], result.arms["original"][arm]
        assert [r.decision.estimate for r in saved.results] == [
            r.decision.estimate for r in original.results
        ]
        assert saved.floor == original.floor
    assert changes(result, "saved", "original") == {
        "verdicts": [],
        "floors": [],
        "gap": "",
        "rule": [],
    }


@pytest.mark.behaviour
def test_the_report_discloses_first_and_the_json_carries_every_world(
    result: DriftSensitivity,
) -> None:
    text = render(result, PROV)
    head = text.split("\n\n")
    assert head[0].startswith("# Drift sensitivity") and STATEMENT in head[1]
    assert "Post hoc and not pre-registered" in head[1]
    for section in ("## Mean P&L per premium", "## The decisions (A1, A2)", "## Reading"):
        assert section in text
    assert reading(result) in text and "**Gap**" in text
    data = json.loads(render_json(result, PROV))
    assert data["post_hoc"] is True and data["pre_registered"] is False
    assert data["worlds"]["original"]["r"] == STUDY_R and data["worlds"]["fair"]["r"] == FAIR_R
    assert set(data["arms"]) == {"delta_seller", "ppo"}
    assert set(data["arms"]["ppo"]["worlds"]) == {"saved", "original", "fair"}
    assert len(data["arms"]["ppo"]["means"]) == 3 and len(data["cells"]) == 18
    assert data["reproduction"]["reproduce"] == 18 and data["reproduction"]["differing"] == []
    assert set(data["gap"]) == {"saved", "original", "fair"}
    assert set(data["changes"]["original_to_fair"]) == {"verdicts", "floors", "gap", "rule"}


@pytest.mark.behaviour
def test_a_cell_that_does_not_reproduce_is_reported_not_refused(grid_root: Path) -> None:
    cell = next(c for c in sorted(grid_root.iterdir()) if (c / POLICY_FILENAME).is_file())
    worlds = resimulate_cell(cell, load_policy=other_loader)
    assert not worlds.reproduces and not worlds.agrees and worlds.differing
    first = worlds.differing[0]
    assert first.saved_pnl != first.pnl and worlds.max_abs_pnl_diff > 0.0


@pytest.mark.behaviour
def test_a_cell_whose_files_moved_or_whose_policy_is_missing_is_refused(
    grid_root: Path, tmp_path: Path
) -> None:
    import shutil

    cell = next(c for c in sorted(grid_root.iterdir()) if (c / POLICY_FILENAME).is_file())
    copy = tmp_path / cell.name
    shutil.copytree(cell, copy)
    (copy / POLICY_FILENAME).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        resimulate_cell(copy, load_policy=fake_loader)
    (copy / "run.json").write_text("{}")
    with pytest.raises(ValueError, match="no longer match"):
        resimulate_cell(copy, load_policy=fake_loader)


@pytest.mark.behaviour
@pytest.mark.slow
def test_the_command_writes_both_files_and_never_into_the_sweep(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    run_grid(SELLERS, root)
    before = snapshot(root)
    grid = root / grid_filename(SELLERS)
    out = tmp_path / "out" / "DRIFT"
    argv = ["--grid", str(grid), "--out", str(out), "--n-boot", "200", "--replications", "100"]
    assert main(argv, on=date(2026, 11, 14)) == 0
    assert snapshot(root) == before
    md, js = out.with_suffix(".md").read_text(), json.loads(out.with_suffix(".json").read_text())
    assert md.startswith("# Drift sensitivity") and js["kind"] == "drift_sensitivity"
    assert js["gap"]["fair"] is None  # one arm, no gap
    assert main(["--grid", str(tmp_path / "absent.json")]) == 1
    assert main(["--grid", str(grid), "--out", str(root / "DRIFT")]) == 1  # not into the sweep
    with pytest.raises(SystemExit):
        main(["--grid", str(grid), "--workers", "0"])


@pytest.mark.behaviour
@pytest.mark.spawns_processes
@pytest.mark.slow
def test_the_cells_run_on_workers_to_the_same_numbers(tmp_path: Path) -> None:
    root = tmp_path / "sweep"
    run_grid(SELLERS, root)
    serial = drift_grid(root, **FAST)
    parallel = drift_grid(root, workers=2, **FAST)
    assert [w.fair for _, w in serial.cells] == [w.fair for _, w in parallel.cells]
    assert [w.original for _, w in serial.cells] == [w.original for _, w in parallel.cells]


@pytest.mark.invariant
def test_the_fair_config_moves_the_rate_and_nothing_else() -> None:
    study = RunConfig(market=MarketConfig(r=STUDY_R, iv_premium_vol_pts=1.0), seed=7, n_episodes=3)
    fair = fair_config(study)
    assert fair.market.r == FAIR_R == MarketConfig().r == 0.0
    assert fair.model_copy(update={"market": study.market}) == study
    assert fair.market.model_copy(update={"r": STUDY_R}) == study.market
