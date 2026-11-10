"""Ilan Danial: what a run records about what made it, and every reuse check reading it.

The config hash cannot see a component of one's own, a rebuilt panel, the trainer, the
commit or the machine, so the manifest records each and the checks that reuse a run
compare it (the machinery audit's M1, M2, M3 and m1 to m4):

- a run made with components of one's own (``extend.Components``, a planted effect, a
  cost model, a runner) is reused only by the same components, and a default run only
  by the defaults; neither is written over the other;
- a reader that names no trainer still holds a learned cell to the real learner;
- a script's trainer is named with its file, so two scripts' ``__main__.trainer``
  are two trainers;
- a reused cell written at another commit is reused and reported;
- a manifest records the platform it was written on, and a Finder ``.DS_Store`` does
  not make a run dirty;
- the modules that write and read runs say ``encoding="utf-8"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fake_learner import fake_trainer

from optspread.artifacts import read_run, write_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.costs import make_cost_model
from optspread.data import SyntheticSource
from optspread.effects import SkewPremium
from optspread.extend import Components, planted
from optspread.harness import run_episodes, summarize
from optspread.manifest import (
    MANIFEST_FILENAME,
    RunManifest,
    callable_name,
    git_dirty,
    platform_record,
)
from optspread.orchestrate import (
    GridConfig,
    IncompleteGridError,
    execute_run,
    grid_cells,
    grid_record,
    plan_cells,
    run_cells,
    run_grid,
)
from optspread.sweep import (
    REAL_TRAINER,
    SweepConfig,
    components_identity,
    run_id,
    run_sweep,
    valid_run,
)

REPO = Path(__file__).resolve().parent.parent.parent
SHORT = RunConfig(
    seed=100,
    n_episodes=3,
    market=MarketConfig(n_days=15, tenor_days=(8, 15)),
    env=EnvConfig(episode_days=15),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(base=SHORT, premiums=(0.0, 2.0), arms=("delta_seller",), n_seeds=1)
SKEW = Components(source=planted(SkewPremium(put_slope=3.0)))
PPO_GRID = GridConfig(
    base=SHORT,
    premiums=(0.0,),
    arms=("delta_seller", "ppo"),
    n_seeds=1,
    train=TrainConfig(total_timesteps=64, seed=3),
)


# -- M1: components of one's own --------------------------------------------- #


@pytest.mark.invariant
def test_a_component_identity_is_recorded_and_the_defaults_record_none() -> None:
    assert components_identity(run_episodes) is None
    assert components_identity(Components()) is None  # runs exactly what the harness runs
    identity = SKEW.identity
    assert identity is not None and "SkewPremium(put_slope=3.0" in identity
    other = Components(source=planted(SkewPremium(put_slope=2.0))).identity
    assert other != identity, "two effects are two worlds"
    assert Components(costs=make_cost_model, source=SyntheticSource).identity is None


@pytest.mark.invariant
def test_a_cell_of_ones_own_components_is_not_a_default_cell_nor_the_reverse(
    tmp_path: Path,
) -> None:
    cells = grid_cells(GRID)
    run_cells(cells, tmp_path, runner=SKEW)
    cell = tmp_path / cells[1].run_id
    manifest = json.loads((cell / MANIFEST_FILENAME).read_text())
    assert manifest["data"]["components"] == SKEW.identity
    assert manifest["data"]["source"] == "synthetic"
    # The same components reuse it; the defaults do not, and do not overwrite it.
    assert plan_cells(cells, tmp_path, runner=SKEW).to_run == 0
    assert plan_cells(cells, tmp_path).to_run == len(cells)
    assert valid_run(cell, cells[1].config) is None
    with pytest.raises(FileNotFoundError):
        grid_record(GRID, tmp_path)
    skew_mean = read_run(cell).mean_pnl
    with pytest.raises(IncompleteGridError, match="made with Components"):
        run_grid(GRID, tmp_path)
    assert read_run(cell).mean_pnl == skew_mean, "a run of another world is not written over"
    assert grid_record(GRID, tmp_path, runner=SKEW).cell(2.0, "delta_seller", 0).mean_pnl == (
        skew_mean
    )

    # The reverse: a default cell is not reused by, nor overwritten for, the components.
    default = tmp_path / "default"
    record = run_grid(GRID, default)
    assert record.cell(2.0, "delta_seller", 0).mean_pnl != skew_mean
    assert (
        "components"
        not in json.loads((default / cells[1].run_id / MANIFEST_FILENAME).read_text())["data"]
    ), "a default run's manifest is the shape it always was"
    assert plan_cells(cells, default, runner=SKEW).to_run == len(cells)
    with pytest.raises(FileExistsError, match="default components"):
        execute_run(cells[1].config, default / cells[1].run_id, runner=SKEW)


@pytest.mark.invariant
def test_a_runner_of_ones_own_is_named_and_its_cells_are_its_own(tmp_path: Path) -> None:
    calls: list[RunConfig] = []

    def counting(config: RunConfig, policy: object = None) -> tuple:  # type: ignore[type-arg]
        calls.append(config)
        return run_episodes(config, None)

    name = components_identity(counting)
    assert name is not None and "counting" in name
    sweep = SweepConfig(base=SHORT, premiums=(1.0,), arms=("delta_seller",))
    run_sweep(sweep, tmp_path, runner=counting)
    run_sweep(sweep, tmp_path, runner=counting)
    assert len(calls) == 1, "the same runner reuses its cell"
    config = SHORT.model_copy(
        update={
            "policy": "delta_seller",
            "market": SHORT.market.model_copy(update={"iv_premium_vol_pts": 1.0}),
        }
    )
    assert valid_run(tmp_path / run_id(config), config) is None
    with pytest.raises(FileExistsError):
        run_sweep(sweep, tmp_path)  # the default runner does not write over it


# -- m1: run_sweep and another source's run ------------------------------------ #


@pytest.mark.invariant
def test_run_sweep_does_not_write_over_another_sources_run(tmp_path: Path) -> None:
    config = SHORT.model_copy(update={"policy": "delta_seller"})
    run_dir = tmp_path / run_id(config)
    execute_run(config, run_dir, source="optionmetrics_stub")
    sweep = SweepConfig(
        base=SHORT, premiums=(SHORT.market.iv_premium_vol_pts,), arms=("delta_seller",)
    )
    with pytest.raises(FileExistsError, match="optionmetrics_stub"):
        run_sweep(sweep, tmp_path)
    assert RunManifest.read(run_dir).data_record.source == "optionmetrics_stub"


# -- m2: the trainer when a reader names none ------------------------------------ #


@pytest.mark.invariant
def test_a_reader_naming_no_trainer_still_holds_learned_cells_to_the_real_learner(
    tmp_path: Path,
) -> None:
    run_grid(PPO_GRID, tmp_path, trainer=fake_trainer)
    ppo = next(c for c in grid_cells(PPO_GRID) if c.arm == "ppo")
    assert valid_run(tmp_path / ppo.run_id, ppo.config) is None
    assert valid_run(tmp_path / ppo.run_id, ppo.config, trainer=None) is not None  # any trainer
    assert valid_run(tmp_path / ppo.run_id, ppo.config, trainer=callable_name(fake_trainer))
    with pytest.raises(FileNotFoundError, match="ppo"):
        grid_record(PPO_GRID, tmp_path)
    assert grid_record(PPO_GRID, tmp_path, trainer=fake_trainer).cells
    assert plan_cells(grid_cells(PPO_GRID), tmp_path).to_run == 1


@pytest.mark.invariant
def test_two_scripts_trainers_of_one_name_are_two_trainers() -> None:
    found = []
    for path in ("/one/train_a.py", "/two/train_b.py"):
        scope: dict[str, object] = {"__name__": "__main__"}
        exec(compile("def trainer(config):\n    return None\n", path, "exec"), scope)
        found.append(callable_name(scope["trainer"]))
    assert found[0] != found[1]
    assert found[0] == "__main__.trainer@/one/train_a.py"
    assert callable_name(fake_trainer) == "fake_learner.fake_trainer"  # importable: plain
    assert REAL_TRAINER == "optspread.train.train_ppo"


@pytest.mark.invariant
def test_a_scripts_trainer_has_the_same_name_in_a_spawned_worker() -> None:
    # A spawned worker imports the script it was started from as __mp_main__.
    found = []
    for module in ("__main__", "__mp_main__"):
        scope: dict[str, object] = {"__name__": module}
        exec(compile("def trainer(config):\n    return None\n", "/one/train_a.py", "exec"), scope)
        found.append(callable_name(scope["trainer"]))
    assert found == ["__main__.trainer@/one/train_a.py"] * 2


@pytest.mark.invariant
@pytest.mark.spawns_processes
def test_a_scripts_own_trainer_runs_a_grid_on_two_workers_and_is_reused(tmp_path: Path) -> None:
    script = tmp_path / "my_study.py"
    script.write_text(
        """
import sys
from pathlib import Path

from fake_learner import fake_trainer

from optspread.config import EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.orchestrate import GridConfig, grid_cells, plan_cells, run_grid


def trainer(config):  # a researcher's own, at the top of the script
    return fake_trainer(config)


if __name__ == "__main__":
    base = RunConfig(seed=100, n_episodes=2, market=MarketConfig(n_days=10, tenor_days=(5, 10)),
                     env=EnvConfig(episode_days=10))
    grid = GridConfig(base=base, premiums=(0.0,), arms=("delta_seller", "ppo"), n_seeds=2,
                      train=TrainConfig(total_timesteps=64, seed=3))
    run_grid(grid, Path(sys.argv[1]), workers=2, trainer=trainer)
    print(plan_cells(grid_cells(grid), Path(sys.argv[1]), trainer=trainer).to_run)
""",
        encoding="utf-8",
    )
    root = tmp_path / "runs"
    done = subprocess.run(
        [sys.executable, str(script), str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{REPO}{os.pathsep}{REPO / 'tests'}"},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["0"], "a rerun reuses every cell"
    trainers = {
        json.loads(m.read_text(encoding="utf-8")).get("trainer")
        for m in root.glob(f"*/{MANIFEST_FILENAME}")
    }
    assert trainers == {None, f"__main__.trainer@{script.as_posix()}"}


# -- m3: commits ------------------------------------------------------------------ #


@pytest.mark.invariant
def test_a_reused_cell_from_another_commit_is_reused_and_reported(tmp_path: Path) -> None:
    record = run_grid(GRID, tmp_path)
    assert len(record.commits) == 1 and record.commits[0] != "unknown"
    cell = grid_cells(GRID)[0]
    path = tmp_path / cell.run_id / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["git_sha"] = "f" * 40
    path.write_text(json.dumps(manifest))
    plan = plan_cells(grid_cells(GRID), tmp_path)
    assert plan.to_run == 0, "a commit after the freeze computes the same numbers"
    assert plan.other_commits == (("f" * 40, 1),)
    assert "reused cells written at other commits: ffffffff (1)" in plan.render()
    again = grid_record(GRID, tmp_path)
    assert "f" * 40 in again.commits and len(again.commits) == 2
    assert "cells written at commits" in again.render()
    assert "commits" not in json.loads(
        again.model_copy(update={"commits": ()}).model_dump_json()
    ), "a record without commits keeps the old shape"


# -- M3 and m4: the platform and a dirty tree ---------------------------------------- #


@pytest.mark.invariant
def test_a_manifest_records_its_platform_and_an_old_one_reads_without(tmp_path: Path) -> None:
    episodes = run_episodes(SHORT)
    manifest = write_run(tmp_path, SHORT, summarize(episodes), episodes)
    assert manifest.platform == platform_record()
    assert set(platform_record()) == {"os", "arch", "python", "numpy"}
    data = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    del data["platform"]
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(data))
    old = RunManifest.read(tmp_path)
    assert old.platform is None and "platform" not in old.model_dump(mode="json")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.invariant
def test_a_finder_ds_store_does_not_make_runs_dirty(tmp_path: Path) -> None:
    paths = [".DS_Store", "optspread/.DS_Store", "web/.DS_Store"]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if ignored.returncode == 128:
        pytest.skip("not a git checkout")
    assert ignored.stdout.split() == paths, ".DS_Store is not in .gitignore"
    repo = tmp_path / "clone"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_bytes((REPO / ".gitignore").read_bytes())
    _git(repo, "add", ".gitignore")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    (repo / "pkg").mkdir()
    for where in (repo, repo / "pkg"):
        (where / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    assert not git_dirty(repo), "an ignored Finder file is not an uncommitted change"
    (repo / "notes.txt").write_text("untracked and not ignored\n")
    assert git_dirty(repo)


# -- n5: the encoding ---------------------------------------------------------------- #

WRITERS = (
    "manifest",
    "artifacts",
    "sweep",
    "orchestrate",
    "jobs",
    "replay",
    "runs/rundir",
    "experiment/sweep",
)


@pytest.mark.behaviour
def test_the_run_writers_and_readers_name_their_encoding(tmp_path: Path) -> None:
    script = """
import sys, warnings
from pathlib import Path
from optspread.config import RunConfig, MarketConfig, EnvConfig
from optspread.orchestrate import GridConfig, run_grid, read_grid, grid_filename
from optspread.replay import replay_run
from optspread.sweep import SweepConfig, run_sweep, read_sweep
from optspread.jobs import JobSpec, execute_job
root = Path(sys.argv[1])
(root / "jobs" / ".jobs").mkdir(parents=True)
base = RunConfig(n_episodes=2, market=MarketConfig(n_days=10, tenor_days=(5, 10)),
                 env=EnvConfig(episode_days=10))
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always", EncodingWarning)
    grid = GridConfig(base=base, premiums=(0.0,), arms=("delta_seller",))
    record = run_grid(grid, root / "grid")
    read_grid(root / "grid" / grid_filename(grid))
    run_sweep(SweepConfig(base=base, premiums=(0.0,), arms=("flat",)), root / "sweep")
    read_sweep(root / "sweep")
    replay_run(root / "grid" / record.cells[0].run_id)
    spec = JobSpec(n_days=10, tenor_days=(5, 10), n_episodes=2, roll_dte=2)
    execute_job("j1", spec.model_dump_json(), str(root / "jobs"))
mine = [w for w in seen if issubclass(w.category, EncodingWarning)
        and any(f"optspread/{m}.py" in w.filename.replace(chr(92), "/") for m in sys.argv[2:])]
print("\\n".join(f"{w.filename}:{w.lineno}" for w in mine))
sys.exit(1 if mine else 0)
"""
    done = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", script, str(tmp_path), *WRITERS],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
        check=False,
    )
    assert done.returncode == 0, f"locale encoding at:\n{done.stdout}{done.stderr[-2000:]}"
