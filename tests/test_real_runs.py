"""Ilan Danial: runs on the real SPX panel from the command line and the API (GH-21, #28).

A real-data run is an ordinary run directory whose manifest records the source and the
split. The source and the split are not in the config hash, so a real-data run has the
run id of its synthetic twin: it lands under its own root by default, every resume
check refuses a run on other data, and nothing overwrites one. A learned arm on the real
source trains on the ``train`` split and is evaluated on ``eval`` (the dates of
pre-registration A8, not a flag); the dates the provider served are recorded here and
checked to be disjoint. A cell trained by another trainer is never reused (#31).

Everything runs on a small panel generated into a temporary folder, across the split
boundary (2018-12-31 / 2019-01-01): the shared data folder is never read.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned, fake_real_trainer, fake_trainer

from optspread.artifacts import RUN_FILENAME, read_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.data import OptionMetricsSource
from optspread.extend import Components
from optspread.manifest import (
    MANIFEST_FILENAME,
    DataRecord,
    RunManifest,
    callable_name,
    file_sha256,
)
from optspread.orchestrate import (
    GridConfig,
    IncompleteGridError,
    execute_run,
    foreign_run,
    grid_cells,
    plan_cells,
    run_grid,
    run_single,
    source_root,
)
from optspread.realdata import SPLITS, SplitRanges
from optspread.replay import replay_matches, replay_run
from optspread.sweep import (
    REAL_TRAINER,
    SB3_ARCHIVE_MEMBER,
    SweepConfig,
    recorded_trainer,
    run_id,
    run_sweep,
    valid_run,
)
from optspread.train import POLICY_FILENAME, train_on, train_ppo

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

SMALL = RunConfig(
    policy="delta_seller",
    seed=8000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
LEARNED = SMALL.model_copy(
    update={"policy": "ppo", "train": TrainConfig(total_timesteps=64, seed=2)}
)
# The first and last panel dates: a training side of about 150 trading days, an
# evaluation side of about 100, either side of the split boundary.
FIRST, LAST = "2018-06-01", "2019-05-31"
TRAIN_END, EVAL_START, EVAL_END = (str(d) for d in (SPLITS["train"][1], *SPLITS["eval"]))


def _panel(root: Path) -> Path:
    """The two panel files the real source reads, every business day ``FIRST..LAST``: a
    surface flat in strike and tenor whose level moves day to day, on a random-walk spot."""
    from optspread.realdata.panel import K_GRID, TENOR_GRID

    rng = np.random.default_rng(28)
    days = [d.date() for d in pd.bdate_range(FIRST, LAST)]
    level = 0.15 + 0.08 * rng.random(len(days))
    grid = pd.DataFrame(
        [
            (d, t, float(k), v)
            for d, v in zip(days, level, strict=True)
            for t in TENOR_GRID
            for k in K_GRID
        ],
        columns=["date", "tenor_days", "k", "iv"],
    )
    spot = 2800.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(days))))
    daily = pd.DataFrame({"date": days, "spot": spot, "carry30": 0.02})
    (root / "panel").mkdir(parents=True)
    grid.to_parquet(root / "panel" / "surface_grid.parquet", index=False)
    daily.to_parquet(root / "panel" / "spx_daily.parquet", index=False)
    return root


@pytest.fixture(scope="module")
def panel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _panel(tmp_path_factory.mktemp("data"))


@pytest.fixture()
def data(panel: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``VEGALAB_DATA`` names the generated panel, as it names the shared folder."""
    monkeypatch.setenv("VEGALAB_DATA", str(panel))
    return panel


@dataclass
class Served:
    """Every date a real provider served, by the split its store was cut to."""

    dates: dict[str, set[str]] = field(default_factory=lambda: {"train": set(), "eval": set()})


@pytest.fixture()
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[Served]:
    import optspread.realdata.provider as provider

    record = Served()
    cut: dict[int, str] = {}  # id(store) -> the split it was loaded for
    load, chain = provider.load_store, provider.HistoricalChainProvider._chain

    def load_spy(panel_dir: Path, split: str, ranges: SplitRanges | None = None):  # type: ignore[no-untyped-def]
        store = load(panel_dir, split, ranges)  # type: ignore[arg-type]
        cut[id(store)] = split
        return store

    def spy(self: provider.HistoricalChainProvider):  # type: ignore[no-untyped-def]
        record.dates[cut[id(self.store)]].add(self.store.dates[self._start + self._day][:10])
        return chain(self)

    monkeypatch.setattr(provider, "load_store", load_spy)
    monkeypatch.setattr(provider.HistoricalChainProvider, "_chain", spy)
    yield record


def _manifest(run_dir: Path) -> dict[str, object]:
    out: dict[str, object] = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    return out


# -- the single-run command -------------------------------------------------------- #


def test_a_scripted_real_run_from_the_command_records_its_source_and_replays(
    data: Path, served: Served, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import replay, run

    args = ["--policy", "delta_seller", "--episodes", "3", "--seed", "5"]
    assert run.main([*args, "--source", "optionmetrics"]) == 0
    out = capsys.readouterr().out
    config = RunConfig(seed=5, n_episodes=3, policy="delta_seller")
    run_dir = data / "runs" / "optionmetrics" / "single" / run_id(config)
    assert "written" in out and str(run_dir) in out and "evaluates on 2019-01-01" in out
    recorded = _manifest(run_dir)["data"]
    assert recorded == {
        "source": "optionmetrics",
        "panel": {
            f"panel/{n}": file_sha256(data / "panel" / n)
            for n in ("surface_grid.parquet", "spx_daily.parquet")
        },
        "split": "eval",
    }
    assert served.dates["eval"] and not served.dates["train"]
    assert min(served.dates["eval"]) >= "2019-01-01"

    assert replay.main([str(run_dir)]) == 0
    assert "IDENTICAL" in capsys.readouterr().out
    assert run.main([*args, "--source", "optionmetrics"]) == 0  # read back, not recomputed
    assert "reused" in capsys.readouterr().out

    # The synthetic run of the same config has the same run id: it is refused the real run,
    # lands in its own folder by default, and never overwrites the real one.
    assert valid_run(run_dir, config) is None
    assert valid_run(run_dir, config, "optionmetrics") is not None
    before = (run_dir / RUN_FILENAME).read_bytes()
    assert run.main([*args, "--root", str(run_dir.parent)]) == 1
    assert "already holds" in capsys.readouterr().err
    assert (run_dir / RUN_FILENAME).read_bytes() == before
    assert run.main(args) == 0
    synthetic = data / "runs" / "single" / run_id(config)
    assert RunManifest.read(synthetic).data_record.source == "synthetic"
    assert read_run(synthetic).episodes != read_run(run_dir).episodes


def test_the_dates_are_a8s_not_a_flag_and_a_record_of_other_dates_is_another_run(
    data: Path, served: Served, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import run

    args = ["--policy", "flat", "--episodes", "2", "--source", "optionmetrics"]
    for flag in ("--train-end", "--eval-start", "--eval-end"):
        with pytest.raises(SystemExit):  # A8 fixed the dates; nothing past them is reachable
            run.main([*args, flag, "2024-01-31"])
        assert "unrecognized arguments" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run.main(["--policy", "ppo", "--episodes", "1"])  # a learned arm needs a budget
    capsys.readouterr()
    assert run.main(args) == 0
    assert f"evaluates on {EVAL_START}..{EVAL_END}" in capsys.readouterr().out
    config = RunConfig(n_episodes=2, policy="flat")
    run_dir = data / "runs" / "optionmetrics" / "single" / run_id(config)
    assert RunManifest.read(run_dir).data_record.split == "eval"
    assert served.dates["eval"] and max(served.dates["eval"]) <= EVAL_END
    assert valid_run(run_dir, config, "optionmetrics") is not None

    # A record of dates of its own (a surface of your own, docs/EXTENDING.md) is another
    # run: never reused for, nor replayed on, the team's panel.
    manifest = _manifest(run_dir)
    own = {"train": ["2018-06-01", "2018-12-31"], "eval": ["2019-01-01", "2019-04-30"]}
    manifest["data"]["ranges"] = own  # type: ignore[index]
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    record = RunManifest.read(run_dir).data_record
    assert record.ranges is not None and str(record.ranges["eval"][1]) == "2019-04-30"
    assert valid_run(run_dir, config, "optionmetrics") is None
    matches, reason = replay_matches(run_dir)
    assert not matches and "A8" in reason


def test_the_real_source_without_a_panel_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import run

    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path))
    with pytest.raises(SystemExit):
        run.main(["--source", "optionmetrics", "--root", str(tmp_path / "runs")])
    assert "no SPX panel" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_the_makefile_runs_one_run_with_its_args() -> None:
    import shutil
    import subprocess

    if shutil.which("make") is None:
        pytest.skip("needs make")
    repo = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        ["make", "-n", "-f", str(repo / "Makefile"), "run", "ARGS=--source optionmetrics", "PY=py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout
    assert out.strip() == "py -m scripts.run --source optionmetrics"


# -- the learned arm trains on training dates and is scored on evaluation dates ---- #


def test_a_learned_real_cell_trains_and_evaluates_on_disjoint_dates(
    data: Path, served: Served, tmp_path: Path
) -> None:
    grid = GridConfig(
        base=SMALL.model_copy(update={"seed": 9000, "n_episodes": 2}),
        premiums=(0.0,),
        arms=("delta_seller", "ppo"),
        n_seeds=2,
        train=TrainConfig(total_timesteps=64, seed=4),
    )
    root = tmp_path / "grid"
    record = run_grid(grid, root, trainer=fake_real_trainer, source="optionmetrics")
    train, evaluate = served.dates["train"], served.dates["eval"]
    assert train and evaluate and not train & evaluate
    assert max(train) <= TRAIN_END
    assert min(evaluate) >= EVAL_START and max(evaluate) <= EVAL_END
    for cell in record.cells:
        manifest = RunManifest.read(root / cell.run_id)
        assert manifest.data_record.source == "optionmetrics"
        assert manifest.data_record.split == "eval"
        if cell.arm == "ppo":
            assert manifest.trainer == callable_name(fake_real_trainer)

    ppo = next(c for c in record.cells if c.arm == "ppo")
    served.dates["train"].clear()
    served.dates["eval"].clear()
    outcome = replay_run(root / ppo.run_id, trainer=fake_real_trainer)
    assert outcome.matches, outcome.reason
    assert max(served.dates["train"]) <= TRAIN_END  # retrained where it trained
    assert min(served.dates["eval"]) >= EVAL_START

    # Resumed with the same trainer and source: every cell read back.
    assert (
        plan_cells(grid_cells(grid), root, trainer=fake_real_trainer, source="optionmetrics").to_run
        == 0
    )
    # The synthetic grid of the same design shares the run ids and reuses none of them.
    assert plan_cells(grid_cells(grid), root, trainer=fake_real_trainer).to_run == 4
    with pytest.raises(IncompleteGridError) as caught:
        run_grid(grid, root, trainer=fake_trainer)
    assert all("already holds" in why for why in caught.value.failures.values())
    assert valid_run(root / ppo.run_id, LEARNED) is None  # and nothing was overwritten
    assert RunManifest.read(root / ppo.run_id).data_record.source == "optionmetrics"


def test_a_trainer_that_takes_no_source_is_refused_on_real_data(data: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="takes no source"):
        train_on(fake_trainer, LEARNED, "optionmetrics")
    assert isinstance(train_on(fake_trainer, LEARNED), FakeLearned)  # the generator: as ever
    with pytest.raises(ValueError, match="takes no source"):
        execute_run(LEARNED, tmp_path / "run", fake_trainer, source="optionmetrics")
    assert not (tmp_path / "run" / RUN_FILENAME).exists()


def test_the_learner_builds_its_training_environment_on_the_training_range(
    data: Path, served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("stable_baselines3")
    import optspread.train as train

    built: list[tuple[str, str]] = []
    build = train.build_env

    def spy(config: RunConfig, source: str = "synthetic", **kw: object):  # type: ignore[no-untyped-def]
        built.append((source, str(kw.get("split"))))
        return build(config, source, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(train, "build_env", spy)
    config = LEARNED.model_copy(update={"n_episodes": 1})
    train_ppo(
        config.model_copy(update={"train": TrainConfig(total_timesteps=64)}), source="optionmetrics"
    )
    assert built == [("optionmetrics", "train")]
    assert served.dates["train"] and not served.dates["eval"]
    assert max(served.dates["train"]) <= TRAIN_END


# -- a different trainer's cell is not reused (#31) ------------------------------- #


def test_a_cell_is_reused_only_for_the_trainer_that_made_it(tmp_path: Path) -> None:
    run_dir = tmp_path / run_id(LEARNED)
    execute_run(LEARNED, run_dir, fake_trainer)
    fake = callable_name(fake_trainer)
    assert valid_run(run_dir, LEARNED, trainer=fake) is not None
    assert valid_run(run_dir, LEARNED, trainer=REAL_TRAINER) is None
    assert valid_run(run_dir, LEARNED) is None  # no trainer named: the real learner's rule
    assert valid_run(run_dir, LEARNED, trainer=None) is not None  # any trainer, asked for
    assert valid_run(run_dir, SMALL, trainer=REAL_TRAINER) is None  # another config anyway

    # A manifest from before the trainer was recorded: the real learner's only when its
    # policy is a Stable-Baselines3 archive, as the M2 gate reads it.
    manifest = _manifest(run_dir)
    del manifest["trainer"]
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    old = RunManifest.read(run_dir)
    assert recorded_trainer(run_dir, old) is None
    assert valid_run(run_dir, LEARNED, trainer=REAL_TRAINER) is None
    assert valid_run(run_dir, LEARNED, trainer=fake) is not None
    with zipfile.ZipFile(run_dir / POLICY_FILENAME, "w") as archive:
        archive.writestr(SB3_ARCHIVE_MEMBER, "2.3.2")
    manifest["artifacts"][POLICY_FILENAME] = file_sha256(run_dir / POLICY_FILENAME)  # type: ignore[index]
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert recorded_trainer(run_dir, RunManifest.read(run_dir)) == REAL_TRAINER
    assert valid_run(run_dir, LEARNED, trainer=REAL_TRAINER) is not None
    assert valid_run(run_dir, LEARNED, trainer=fake) is None
    (run_dir / POLICY_FILENAME).write_bytes(b"not a zip")
    manifest["artifacts"][POLICY_FILENAME] = file_sha256(run_dir / POLICY_FILENAME)  # type: ignore[index]
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert recorded_trainer(run_dir, RunManifest.read(run_dir)) is None
    (run_dir / POLICY_FILENAME).unlink()
    assert recorded_trainer(run_dir, RunManifest.read(run_dir)) is None


def test_run_single_reads_back_its_own_and_recomputes_another_trainers(tmp_path: Path) -> None:
    run_dir, _, reused = run_single(LEARNED, tmp_path, trainer=fake_trainer)
    assert not reused and run_dir == tmp_path / run_id(LEARNED)
    assert run_single(LEARNED, tmp_path, trainer=fake_trainer)[2]
    assert not run_single(LEARNED, tmp_path, trainer=fake_real_trainer)[2]
    assert RunManifest.read(run_dir).trainer == callable_name(fake_real_trainer)


# -- old manifests, other orchestrators, the job runner -------------------------- #


def test_a_real_manifest_from_before_the_split_was_recorded_reads_as_the_default(
    data: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    execute_run(SMALL, run_dir, source="optionmetrics")
    manifest = _manifest(run_dir)
    del manifest["data"]["split"]  # type: ignore[attr-defined]
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    record = RunManifest.read(run_dir).data_record
    assert record.split is None and record.effective_split == "eval"
    assert "split" not in record.model_dump() and "ranges" not in record.model_dump()
    assert valid_run(run_dir, SMALL, "optionmetrics") is not None
    matches, reason = replay_matches(run_dir)
    assert matches, reason
    assert DataRecord().effective_split is None  # the generator has no dates


def test_the_scripted_sweep_runs_on_the_source_it_is_given(data: Path, tmp_path: Path) -> None:
    sweep = SweepConfig(
        base=SMALL.model_copy(update={"n_episodes": 2}), premiums=(0.0,), arms=("flat",)
    )
    record = run_sweep(sweep, tmp_path, source="optionmetrics")
    cell = tmp_path / record.cells[0].run_id
    assert RunManifest.read(cell).data_record.source == "optionmetrics"
    assert run_sweep(sweep, tmp_path, source="optionmetrics") == record
    with pytest.raises(ValueError, match="brings its own data"):
        run_sweep(sweep, tmp_path, runner=lambda c, p: (), source="optionmetrics")
    with pytest.raises(ValueError, match="brings its own data"):
        run_grid(
            GridConfig(base=SMALL, premiums=(0.0,), arms=("flat",)),
            tmp_path,
            runner=lambda c, p: (),
            source="optionmetrics",
        )


def test_the_pilot_and_the_sweep_take_the_source_and_land_in_its_folder(
    data: Path, served: Served, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import pilot, sweep

    args = ["--seeds", "1", "--episodes", "2", "--timesteps", "64", "--source", "optionmetrics"]
    assert pilot.main([*args, "--dry-run"], trainer=fake_real_trainer) == 0
    out = capsys.readouterr()
    assert (
        "4 cells: 0 on disk, 4 to run" in out.out
        and f"trains on 2010-01-01..{TRAIN_END}" in out.out
    )
    assert "not applied on the real source" in out.err
    assert pilot.main(args, trainer=fake_real_trainer) == 0
    root = data / "runs" / "optionmetrics" / "pilot"
    matrices = RunManifest.read(root)
    assert matrices.data_record.source == "optionmetrics"
    assert not served.dates["train"] & served.dates["eval"]

    common = ["--premiums", "0", "--arms", "delta_seller", "--seeds", "1", "--episodes", "2"]
    assert sweep.main([*common, "--source", "optionmetrics"]) == 0
    assert "optionmetrics: trains on" in capsys.readouterr().out
    assert len(list((data / "runs" / "optionmetrics").glob("grid-*.json"))) == 1
    assert not list((data / "runs").glob("grid-*.json"))  # nothing in the synthetic root


def test_a_directory_holding_another_sources_run_is_not_written(data: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / run_id(SMALL)
    execute_run(SMALL, run_dir, source="optionmetrics")
    assert foreign_run(run_dir, SMALL, "optionmetrics") == ""
    assert "ran on optionmetrics, not synthetic" in foreign_run(run_dir, SMALL, "synthetic")
    other = SMALL.model_copy(update={"seed": 1})
    assert foreign_run(run_dir, other, "synthetic") == ""  # not this config's directory
    assert foreign_run(tmp_path / "empty", SMALL, "synthetic") == ""
    with pytest.raises(FileExistsError):
        execute_run(SMALL, run_dir)
    assert source_root(tmp_path, "synthetic") == tmp_path
    assert source_root(tmp_path, "optionmetrics") == tmp_path / "optionmetrics"
    manifest = _manifest(run_dir)
    manifest["data"]["split"] = "train"  # type: ignore[index]
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert "ran under the split train, not eval" in foreign_run(run_dir, SMALL, "optionmetrics")


def test_a_job_on_the_real_source_lands_in_its_folder_and_reports_it(
    data: Path, tmp_path: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor, wait

    from optspread.jobs import LAUNCH_DIR, JobManager, JobSpec, available_sources, execute_job
    from optspread.webapi import job_options, job_out

    spec = JobSpec(n_days=21, tenor_days=(7, 14), n_episodes=2, seed=3, source="optionmetrics")
    (tmp_path / ".jobs").mkdir()  # the manager makes it
    rel = execute_job("j1", spec.model_dump_json(), str(tmp_path))
    assert rel.startswith(f"optionmetrics/{LAUNCH_DIR}/")
    assert RunManifest.read(tmp_path / rel).data_record.source == "optionmetrics"
    assert execute_job("j2", spec.model_dump_json(), str(tmp_path)) == rel
    assert json.loads((tmp_path / ".jobs" / "j2.json").read_text())["phase"] == "reused"
    synthetic = execute_job(
        "j3", spec.model_copy(update={"source": "synthetic"}).model_dump_json(), str(tmp_path)
    )
    assert synthetic.startswith(f"{LAUNCH_DIR}/") and synthetic.split("/")[-1] == rel.split("/")[-1]

    assert available_sources() == ("synthetic", "optionmetrics")
    options = job_options()
    knobs = {k.name: k for k in options.knobs}
    assert knobs["source"].kind == "choice" and knobs["source"].choices == [
        "synthetic",
        "optionmetrics",
    ]
    assert (
        options.sources == ["synthetic", "optionmetrics"]
        and options.defaults["source"] == "synthetic"
    )

    manager = JobManager(tmp_path / "m", executor=ThreadPoolExecutor(1))
    try:
        job = manager.submit(spec)
        wait([manager.jobs[job].future], timeout=600)
        status = manager.status(job)
        assert status.state == "done" and status.source == "optionmetrics"
        assert job_out(status).source == "optionmetrics"
    finally:
        manager.shutdown()


def test_a_job_on_a_source_whose_data_is_absent_is_refused_at_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from optspread.jobs import JobManager, JobSpec, available_sources

    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path / "nothing"))
    assert available_sources() == ("synthetic",)
    manager = JobManager(tmp_path, executor=ThreadPoolExecutor(1))
    try:
        with pytest.raises(ValueError, match="optionmetrics data is not available"):
            manager.submit(JobSpec(source="optionmetrics"))
        assert not manager.jobs
    finally:
        manager.shutdown()


# -- the panel a cell read, and components over the real chains ---------------------- #


def test_a_real_cell_from_a_panel_since_rebuilt_is_not_reused(
    panel: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    data = Path(shutil.copytree(panel, tmp_path / "data"))
    monkeypatch.setenv("VEGALAB_DATA", str(data))
    grid = GridConfig(base=SMALL, premiums=(0.0,), arms=("delta_seller",))
    root = tmp_path / "runs"
    run_grid(grid, root, source="optionmetrics")
    [cell] = grid_cells(grid)
    assert valid_run(root / cell.run_id, cell.config, "optionmetrics") is not None
    assert plan_cells([cell], root, source="optionmetrics").to_run == 0
    # The panel is rebuilt: same dates, another surface.
    grid_file = data / "panel" / "surface_grid.parquet"
    frame = pd.read_parquet(grid_file)
    frame["iv"] = frame["iv"] * 1.01
    frame.to_parquet(grid_file, index=False)
    assert valid_run(root / cell.run_id, cell.config, "optionmetrics") is None
    assert plan_cells([cell], root, source="optionmetrics").to_run == 1
    matches, reason = replay_matches(root / cell.run_id)
    assert not matches and "is not the one the run read" in reason
    # The resume recomputes it on the new build rather than refusing to write over it.
    run_grid(grid, root, source="optionmetrics")
    assert valid_run(root / cell.run_id, cell.config, "optionmetrics") is not None
    # A panel that is not there at all validates nothing.
    shutil.rmtree(data / "panel")
    assert valid_run(root / cell.run_id, cell.config, "optionmetrics") is None


def _surface(market: MarketConfig, *, root: Path) -> OptionMetricsSource:
    return OptionMetricsSource(market, root=root, split="eval")


def test_components_over_the_real_chains_record_the_real_source_not_synthetic(
    panel: Path, tmp_path: Path
) -> None:
    world = Components(source=partial(_surface, root=panel))
    grid = GridConfig(base=SMALL, premiums=(0.0,), arms=("delta_seller",))
    root = tmp_path / "own"
    run_grid(grid, root, runner=world)
    [cell] = grid_cells(grid)
    data = RunManifest.read(root / cell.run_id).data_record
    assert data.source == "optionmetrics" and data.split == "eval" and data.panel
    assert data.components == world.identity and "_surface" in (world.identity or "")
    assert plan_cells([cell], root, runner=world).to_run == 0
    assert plan_cells([cell], root).to_run == 1
    assert replay_run(root / cell.run_id, runner=world).matches
