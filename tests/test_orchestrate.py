"""Ilan Danial: the pilot and sweep grid, parallel and resumable, without torch (week 6).

The grid is premium x arm x seed block. The design properties the pre-registration
depends on are tested as properties of the generated configs, before anything runs:

    paired   within a block, every arm at every premium runs the same path seeds;
    nested   blocks are disjoint, so the pilot formula's independence holds;
    seeded   block i trains with one seed at every premium, and only the ppo arm trains.

Then the machinery: every cell is an ordinary run directory, a learned cell keeps its
policy hashed in the manifest, a second run reads cells back, a failed cell does not
take the others down, and two workers write exactly the bytes one worker writes. The
learner is ``tests/fake_learner.py``; ``tests/test_ppo.py`` runs the real one.

A worker killed outright (#30, ``os._exit`` standing in for the out-of-memory killer)
must not hang the grid: the cells it took are named, every finished cell is kept, the
rest go on to a fresh pool, and a rerun computes the lost cells and nothing else, to
the bytes a run with no death writes. Those tests wait on the pool, not on a clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fake_learner import (
    DOOM_ENV,
    DOOMED_PREMIUM,
    DOOMED_TRAIN_SEED,
    FakeLearned,
    dead_trainer,
    dying_trainer,
    failing_trainer,
    fake_trainer,
    writes_then_dies,
)

from optspread.artifacts import RUN_FILENAME, SchemaVersionError, read_run, write_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.harness import run_episodes, summarize
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256
from optspread.orchestrate import (
    LOST,
    NOT_RUN,
    PILOT_MATRIX,
    PILOT_ORACLE_MATRIX,
    SWEEP_PREMIUMS,
    GridConfig,
    IncompleteGridError,
    Progress,
    block_config,
    cell_seconds,
    default_runs_root,
    execute_run,
    format_seconds,
    grid_cells,
    grid_filename,
    grid_record,
    makespan,
    pilot_grid,
    plan_cells,
    read_grid,
    run_cells,
    run_grid,
    write_pilot_matrices,
)
from optspread.policies import DeltaSellerPolicy
from optspread.sweep import pilot_matrix, run_id
from optspread.train import POLICY_FILENAME, TRAIN_PATH_BASE

SHORT = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
TRAIN = TrainConfig(total_timesteps=4096, seed=10)
GRID = GridConfig(base=SHORT, premiums=(0.0, 2.0), n_seeds=3, train=TRAIN)


# -- the design, before anything runs ----------------------------------------- #


def test_the_default_grid_is_the_pre_registered_axis_and_contrast() -> None:
    grid = GridConfig(train=TRAIN)
    assert grid.premiums == SWEEP_PREMIUMS == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    assert grid.arms == ("delta_seller", "ppo")
    assert grid.base.cost.enabled


def test_within_a_block_every_arm_at_every_premium_runs_the_same_paths() -> None:
    cells = grid_cells(GRID)
    assert len(cells) == 2 * 3 * 2
    for block in range(GRID.n_seeds):
        seeds = {(c.config.seed, c.config.n_episodes) for c in cells if c.block == block}
        assert seeds == {(SHORT.seed + block * SHORT.n_episodes, SHORT.n_episodes)}


def test_blocks_are_disjoint_and_block_i_trains_with_one_seed_at_every_premium() -> None:
    cells = grid_cells(GRID)
    ranges = {
        c.block: set(range(c.config.seed, c.config.seed + c.config.n_episodes)) for c in cells
    }
    for a in ranges:
        for b in ranges:
            assert a == b or not ranges[a] & ranges[b]
    for c in cells:
        if c.arm == "ppo":
            assert c.config.train == TRAIN.model_copy(update={"seed": TRAIN.seed + c.block})
        else:
            assert c.config.train is None
    assert len({c.run_id for c in cells}) == len(cells)


def test_a_cell_changes_only_the_premium_the_arm_the_block_and_the_training() -> None:
    cfg = block_config(GRID, 2.0, "delta_seller", 1)
    assert cfg.market == SHORT.market.model_copy(update={"iv_premium_vol_pts": 2.0})
    assert (cfg.env, cfg.cost, cfg.policy) == (SHORT.env, SHORT.cost, "delta_seller")


def test_a_grid_refuses_an_inconsistent_design() -> None:
    with pytest.raises(ValueError, match="ppo arm needs a TrainConfig"):
        GridConfig(base=SHORT, arms=("delta_seller", "ppo"))
    with pytest.raises(ValueError, match="ppo arm needs a TrainConfig"):
        GridConfig(base=SHORT, arms=("delta_seller",), train=TRAIN)
    with pytest.raises(ValueError, match="premiums"):
        GridConfig(base=SHORT, premiums=(1.0, 1.0), arms=("flat",))
    with pytest.raises(ValueError, match="arms"):
        GridConfig(base=SHORT, arms=("flat", "flat"))
    with pytest.raises(ValueError, match="2\\*\\*15"):
        GridConfig(base=SHORT, n_seeds=2, train=TrainConfig(total_timesteps=1, seed=2**15 - 1))
    with pytest.raises(ValueError, match="training paths"):
        GridConfig(
            base=SHORT.model_copy(update={"seed": TRAIN_PATH_BASE - 5}),
            arms=("flat",),
            n_seeds=2,
        )


def test_the_cost_estimate_rounds_training_up_to_whole_rollouts() -> None:
    ppo = block_config(GRID, 0.0, "ppo", 0)
    scripted = block_config(GRID, 0.0, "delta_seller", 0)
    evaluation = 3 * 21 / 1_800.0
    assert cell_seconds(scripted) == pytest.approx(evaluation)
    assert cell_seconds(ppo) == pytest.approx(evaluation + 4096 / 1_000.0)
    rounded = ppo.model_copy(update={"train": TrainConfig(total_timesteps=2049)})
    assert cell_seconds(rounded) == pytest.approx(evaluation + 4096 / 1_000.0)
    assert format_seconds(75) == "1m15s" and format_seconds(3 * 3600 + 120) == "3h02m"


def test_the_wall_clock_estimate_never_splits_a_cell() -> None:
    assert makespan([10.0] * 8 + [1.0] * 8, workers=16) == 10.0
    assert makespan([10.0] * 10, workers=3) == 40.0  # four rounds, not 33.3
    assert makespan([5.0, 4.0, 3.0, 3.0, 3.0], workers=2) == 10.0  # greedy, as the pool runs
    assert makespan([], workers=4) == 0.0


# -- one run -------------------------------------------------------------------- #


def test_a_scripted_cell_is_exactly_the_run_write_run_writes(tmp_path: Path) -> None:
    cfg = block_config(GRID, 2.0, "delta_seller", 0)
    execute_run(cfg, tmp_path / "a")
    episodes = run_episodes(cfg)
    write_run(tmp_path / "b", cfg, summarize(episodes), episodes)
    assert (tmp_path / "a" / RUN_FILENAME).read_bytes() == (
        tmp_path / "b" / RUN_FILENAME
    ).read_bytes()
    # Nothing trained it, so its manifest has no trainer key at all (#31).
    assert "trainer" not in json.loads((tmp_path / "a" / MANIFEST_FILENAME).read_text())


def test_a_learned_cell_keeps_its_policy_hashed_in_the_manifest(tmp_path: Path) -> None:
    cfg = block_config(GRID, 0.0, "ppo", 1)
    execute_run(cfg, tmp_path, fake_trainer)
    manifest = RunManifest.read(tmp_path)
    assert set(manifest.artifacts) == {RUN_FILENAME, POLICY_FILENAME}
    assert manifest.config.train == cfg.train
    # The stand-in is named, so no gate takes this cell for the real learner's (#31).
    assert manifest.trainer == "fake_learner.fake_trainer"
    expected = run_episodes(cfg, FakeLearned(TRAIN.seed + 1))
    assert [e.pnl for e in read_run(tmp_path).episodes] == [e.pnl for e in expected]


# -- many runs ------------------------------------------------------------------- #


def test_every_cell_is_a_run_directory_listed_in_the_grid_record(tmp_path: Path) -> None:
    record = run_grid(GRID, tmp_path, trainer=fake_trainer)
    assert len(record.cells) == len(grid_cells(GRID))
    for cell in record.cells:
        run = read_run(tmp_path / cell.run_id)
        assert (run.mean_pnl, run.n_episodes) == (cell.mean_pnl, cell.n_episodes)
        assert cell.train_seed == (TRAIN.seed + cell.block if cell.arm == "ppo" else None)
    assert read_grid(tmp_path / grid_filename(GRID)) == record
    assert "grand mean" in record.render() and record.grid_hash in record.render()
    with pytest.raises(KeyError):
        record.cell(9.0, "ppo", 0)


def test_the_grid_record_refuses_an_unknown_schema(tmp_path: Path) -> None:
    record = run_grid(GRID, tmp_path, trainer=fake_trainer)
    bumped = record.model_copy(update={"schema_version": record.schema_version + 1})
    (tmp_path / "bumped.json").write_text(bumped.model_dump_json())
    with pytest.raises(SchemaVersionError):
        read_grid(tmp_path / "bumped.json")


def test_a_second_run_reads_cells_back_and_recomputes_only_what_is_missing(
    tmp_path: Path,
) -> None:
    trained: list[int] = []

    def counting(config: RunConfig) -> FakeLearned:
        trained.append(config.train.seed if config.train else -1)
        return fake_trainer(config)

    lines: list[Progress] = []
    first = run_grid(GRID, tmp_path, trainer=counting, report=lines.append)
    assert len(trained) == 6
    assert all(p.seconds is not None for p in lines)
    assert [p.done for p in lines] == list(range(1, 13))
    assert lines[-1].eta_seconds == pytest.approx(0.0, abs=1e-6)

    victim = first.cell(2.0, "ppo", 1)
    (tmp_path / victim.run_id / RUN_FILENAME).unlink()
    lines.clear()
    second = run_grid(GRID, tmp_path, trainer=counting, report=lines.append)
    assert trained[6:] == [TRAIN.seed + 1]
    assert second == first
    reused = [p for p in lines if p.seconds is None]
    assert len(reused) == 11 and all("reused" in p.render() for p in reused)
    assert plan_cells(grid_cells(GRID), tmp_path, trainer=counting).to_run == 0
    # A reader that names no trainer holds the learned cells to the real learner.
    assert plan_cells(grid_cells(GRID), tmp_path).to_run == 6


def test_a_failed_cell_is_reported_and_the_rest_are_kept(tmp_path: Path) -> None:
    lines: list[Progress] = []
    with pytest.raises(IncompleteGridError) as caught:
        run_grid(GRID, tmp_path, trainer=failing_trainer, report=lines.append)
    failed = {c.run_id for c in grid_cells(GRID) if c.arm == "ppo" and c.block % 2 == 1}
    assert set(caught.value.failures) == failed
    assert "diverged" in str(caught.value) and "rerun to resume" in str(caught.value)
    assert sum("FAILED" in p.render() for p in lines) == len(failed)
    with pytest.raises(FileNotFoundError):
        grid_record(GRID, tmp_path)

    plan = plan_cells(grid_cells(GRID), tmp_path, workers=4, trainer=failing_trainer)
    assert (plan.reused, plan.to_run) == (12 - len(failed), len(failed))
    assert "to run" in plan.render() and "4 worker(s)" in plan.render()
    run_grid(GRID, tmp_path, trainer=fake_trainer)


def test_two_workers_write_exactly_the_bytes_one_worker_writes(tmp_path: Path) -> None:
    one = run_grid(GRID, tmp_path / "one", trainer=fake_trainer)
    two = run_grid(GRID, tmp_path / "two", workers=2, trainer=fake_trainer)
    assert one == two
    for cell in one.cells:
        for name in (RUN_FILENAME, POLICY_FILENAME):
            a, b = tmp_path / "one" / cell.run_id / name, tmp_path / "two" / cell.run_id / name
            assert a.exists() == b.exists()
            if a.exists():
                assert a.read_bytes() == b.read_bytes()


def test_a_dead_worker_loses_only_the_cells_in_flight_and_the_grid_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DOOM_ENV, "1")  # the spawned workers inherit it
    doomed = next(
        c
        for c in grid_cells(GRID)
        if c.config.train is not None
        and (c.config.train.seed, c.premium) == (DOOMED_TRAIN_SEED, DOOMED_PREMIUM)
    )
    lines: list[Progress] = []
    with pytest.raises(IncompleteGridError) as caught:
        run_grid(GRID, tmp_path / "two", workers=2, trainer=dying_trainer, report=lines.append)
    lost = set(caught.value.lost)
    # The doomed cell, and at most the one cell in flight beside it when the pool died.
    assert doomed.run_id in lost and len(lost) <= 2
    assert set(caught.value.failures) == lost  # nothing else failed, nothing was left unrun
    assert all(caught.value.failures[rid] == LOST for rid in lost)
    assert "lost with a dead worker" in str(caught.value) and "rerun to resume" in str(caught.value)
    # One line per cell: the broken pool's survivors and a fresh pool's cells all report.
    assert len(lines) == len(grid_cells(GRID)) and lines[-1].done == len(lines)
    assert sum("FAILED" in p.render() and "lost" in p.render() for p in lines) == len(lost)

    plan = plan_cells(grid_cells(GRID), tmp_path / "two", trainer=dying_trainer)
    assert (plan.reused, plan.to_run) == (12 - len(lost), len(lost))
    lines.clear()
    monkeypatch.delenv(DOOM_ENV)  # the same trainer, so its finished cells are reused
    resumed = run_grid(
        GRID, tmp_path / "two", workers=2, trainer=dying_trainer, report=lines.append
    )
    assert {p.cell.run_id for p in lines if p.seconds is not None} == lost

    reference = run_grid(GRID, tmp_path / "one", trainer=fake_trainer)
    assert resumed == reference
    for cell in reference.cells:
        for name in (RUN_FILENAME, POLICY_FILENAME):
            a, b = tmp_path / "one" / cell.run_id / name, tmp_path / "two" / cell.run_id / name
            assert a.exists() == b.exists()
            if a.exists():
                assert a.read_bytes() == b.read_bytes()


def test_a_pool_that_dies_before_finishing_a_cell_is_not_fed_again(tmp_path: Path) -> None:
    lines: list[Progress] = []
    with pytest.raises(IncompleteGridError) as caught:
        run_grid(GRID, tmp_path, workers=2, trainer=dead_trainer, report=lines.append)
    failures = caught.value.failures
    # The first round was lost; nothing else was started on a machine that cannot fit it.
    assert len(caught.value.lost) == 2 and set(failures) == {c.run_id for c in grid_cells(GRID)}
    assert sorted(failures.values()).count(NOT_RUN) == 10
    assert len(lines) == 12 and all(p.error for p in lines)
    assert plan_cells(grid_cells(GRID), tmp_path, trainer=fake_trainer).to_run == 12
    run_grid(GRID, tmp_path, workers=2, trainer=fake_trainer)
    assert plan_cells(grid_cells(GRID), tmp_path, trainer=fake_trainer).to_run == 0


def test_a_cell_written_in_full_before_its_worker_died_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = GridConfig(base=SHORT, premiums=(0.0, 2.0), arms=("delta_seller",), n_seeds=2)
    doomed = block_config(grid, 2.0, "delta_seller", 1)
    monkeypatch.setenv("FAKE_LEARNER_ROOT", str(tmp_path))
    lines: list[Progress] = []
    lost: set[str] = set()
    try:
        run_grid(grid, tmp_path, workers=2, runner=writes_then_dies, report=lines.append)
    except IncompleteGridError as exc:  # only if a cell was in flight beside it
        lost = set(exc.lost)
        assert set(exc.failures) == lost and len(lost) == 1
    # Its result never came back, but it is on disk and valid: read back, not lost.
    assert run_id(doomed) not in lost
    assert any(p.cell.run_id == run_id(doomed) and not p.error for p in lines)
    assert plan_cells(grid_cells(grid), tmp_path, runner=writes_then_dies).to_run == len(lost)


def test_a_task_that_cannot_reach_a_worker_fails_its_cell(tmp_path: Path) -> None:
    def local(config: RunConfig) -> FakeLearned:  # a closure does not pickle
        return fake_trainer(config)

    with pytest.raises(IncompleteGridError) as caught:
        run_cells(grid_cells(GRID), tmp_path, workers=2, trainer=local)
    assert len(caught.value.failures) == 12 and not caught.value.lost
    assert all("local object" in why for why in caught.value.failures.values())


def test_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        run_cells(grid_cells(GRID), tmp_path, workers=0)


# -- the pilot ------------------------------------------------------------------- #


def test_the_pilot_writes_the_contrast_the_budget_reads_and_its_control(tmp_path: Path) -> None:
    S, N = 3, 4
    grid = pilot_grid(SHORT, n_seeds=S, n_episodes=N, premium=1.0, train=TRAIN)
    # The detection event is each arm against itself at zero (#26), so both worlds run.
    assert grid.premiums == (0.0, 1.0) and set(grid.arms) == {"ppo", "delta_seller"}
    record = run_grid(grid, tmp_path, trainer=fake_trainer)
    paths = write_pilot_matrices(record, tmp_path)
    assert paths == {"ppo": tmp_path / PILOT_MATRIX, "delta_seller": tmp_path / PILOT_ORACLE_MATRIX}

    learner = np.load(tmp_path / PILOT_MATRIX)
    control = np.load(tmp_path / PILOT_ORACLE_MATRIX)
    assert learner.shape == control.shape == (S, N)

    # Each matrix is the arm at 1 vol point minus the same arm at 0, block by block on
    # the nested design sweep.pilot_matrix describes, value for value.
    def world(premium: float) -> RunConfig:
        market = SHORT.market.model_copy(update={"iv_premium_vol_pts": premium})
        return SHORT.model_copy(update={"market": market})

    def seller(_: int) -> DeltaSellerPolicy:
        return DeltaSellerPolicy(roll_dte=SHORT.env.roll_dte)

    def learned(i: int) -> FakeLearned:
        return FakeLearned(TRAIN.seed + i)

    for matrix, make in ((control, seller), (learner, learned)):
        at_p, at_0 = (pilot_matrix(world(p), S, N, make) for p in (1.0, 0.0))
        assert np.array_equal(matrix, at_p - at_0)
    raw = np.load(tmp_path / "matrix_delta_seller_p1.npy")
    assert np.array_equal(raw, pilot_matrix(world(1.0), S, N, seller))
    assert control.any()  # the premium moves the seller, so the control is not trivially zero

    # The matrices carry their own manifest, so the M2 report can tell if one changed.
    manifest = RunManifest.read(tmp_path)
    raw = [f"matrix_{a}_p{p:g}.npy" for a in ("ppo", "delta_seller") for p in (0.0, 1.0)]
    assert manifest.artifacts == {
        name: file_sha256(tmp_path / name) for name in (PILOT_MATRIX, PILOT_ORACLE_MATRIX, *raw)
    }
    assert manifest.config.policy == "ppo" and manifest.config.market.iv_premium_vol_pts == 1.0


def test_the_pilot_refuses_to_contrast_zero_with_itself() -> None:
    with pytest.raises(ValueError, match="must differ from 0"):
        pilot_grid(SHORT, n_seeds=3, n_episodes=4, premium=0.0, train=TRAIN)


def test_a_grid_contrast_is_zero_at_zero_and_needs_the_null_world(tmp_path: Path) -> None:
    grid = pilot_grid(SHORT, n_seeds=2, n_episodes=3, premium=0.5, train=TRAIN)
    record = run_grid(grid, tmp_path, trainer=fake_trainer)
    assert not record.contrast(tmp_path, 0.0, "ppo").any()  # the same policy on the same paths
    without = record.model_copy(update={"grid": grid.model_copy(update={"premiums": (0.5,)})})
    with pytest.raises(KeyError, match="no 0 premium"):
        without.contrast(tmp_path, 0.5, "ppo")


# -- the commands ------------------------------------------------------------------ #


def test_the_pilot_command_runs_resumes_and_writes_the_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import pilot

    args = ["--root", str(tmp_path), "--seeds", "2", "--episodes", "2", "--timesteps", "64"]
    assert pilot.main([*args, "--dry-run"], trainer=fake_trainer) == 0
    assert "8 cells: 0 on disk, 8 to run" in capsys.readouterr().out
    assert pilot.main(args, trainer=fake_trainer) == 0
    out = capsys.readouterr().out
    assert "tau^2" in out and "pilot_budget.py" in out
    assert np.load(tmp_path / PILOT_MATRIX).shape == (2, 2)
    assert pilot.main(args, trainer=fake_trainer) == 0
    assert capsys.readouterr().out.count("reused") == 8
    # Another trainer's cells are not reused (#31): it retrains the PPO cells, and fails.
    assert pilot.main(args, trainer=failing_trainer) == 1
    assert capsys.readouterr().out.count("reused") == 4  # the scripted cells only
    fresh = ["--root", str(tmp_path / "f"), "--seeds", "2", "--episodes", "1", "--timesteps", "1"]
    assert pilot.main(fresh, trainer=failing_trainer) == 1


def test_without_a_root_the_pilot_and_sweep_land_where_the_reports_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import m2_gate_report, pilot, sweep

    data = tmp_path / "shared"
    assert default_runs_root({}) == Path("runs")
    assert default_runs_root({"VEGALAB_DATA": str(data)}) == data / "runs"
    for env in ({}, {"VEGALAB_DATA": str(data)}):
        runs = default_runs_root(env)
        assert m2_gate_report.default_roots(env) == (runs, runs / "pilot")

    monkeypatch.setenv("VEGALAB_DATA", str(data))
    args = ["--seeds", "2", "--episodes", "2", "--timesteps", "64"]
    assert pilot.main(args, trainer=fake_trainer) == 0
    assert (data / "runs" / "pilot" / PILOT_MATRIX).exists()
    common = ["--premiums", "0", "--arms", "delta_seller", "--seeds", "2", "--episodes", "1"]
    assert sweep.main(common) == 0
    assert len(list((data / "runs").glob("grid-*.json"))) == 1
    capsys.readouterr()
    for command in (pilot, sweep):
        with pytest.raises(SystemExit):
            command.main(["--help"])
        assert "$VEGALAB_DATA/runs" in capsys.readouterr().out


def test_the_sweep_command_needs_a_budget_only_when_it_trains(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import sweep

    common = ["--root", str(tmp_path), "--premiums", "0", "1", "--seeds", "2", "--episodes", "1"]
    with pytest.raises(SystemExit):
        sweep.main(common)
    capsys.readouterr()
    assert sweep.main([*common, "--arms", "delta_seller", "--dry-run"]) == 0
    assert "4 cells" in capsys.readouterr().out
    assert sweep.main([*common, "--timesteps", "64"], trainer=fake_trainer) == 0
    out = capsys.readouterr().out
    assert out.count("  eta ") == 8 and "grid-" in out
    retrain = [*common, "--timesteps", "64", "--train-seed", "5"]
    assert sweep.main(retrain, trainer=failing_trainer) == 1  # seeds 5 and 6: 5 fails


def test_workers_are_spawned_with_one_blas_thread_set_before_they_import_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A spawned worker imports numpy before any initializer runs, so the variables must be
    # in the environment it is started with; this process's own come back afterwards.
    import concurrent.futures
    import os

    from optspread import orchestrate

    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    seen: list[dict[str, str | None]] = []

    class Recording(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, workers: int, **_: object) -> None:
            super().__init__(workers)

        def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            seen.append({n: os.environ.get(n) for n in orchestrate.SINGLE_THREAD_ENV})
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(orchestrate, "ProcessPoolExecutor", Recording)
    grid = GridConfig(
        base=RunConfig(
            n_episodes=2,
            market=MarketConfig(n_days=10, tenor_days=(5, 10)),
            env=EnvConfig(episode_days=10),
        ),
        premiums=(0.0, 1.0),
        arms=("delta_seller",),
    )
    run_grid(grid, tmp_path, workers=2)
    assert seen and all(set(s.values()) == {"1"} for s in seen)
    assert os.environ["OMP_NUM_THREADS"] == "7" and "MKL_NUM_THREADS" not in os.environ
