"""Ilan Danial: the reproducibility gate, any completed run reproduces bit for bit (week 6).

Project document 10.7. The check has to be able to fail, so most of this file is
about the ways it says no: a results file edited after its manifest was written, a
replay that lands on a different number, a run directory that is not one, and a
learner that is not installed. Each refusal names what failed first.

A learned run replays by retraining from its recorded config. Here the learner is
``tests/fake_learner.py``; ``tests/test_ppo.py`` replays a real PPO run.

A run replays on the data source its manifest records (GH-21): a manifest from
before the source was recorded replays on the generator, as it was made, and a run
on the real panel replays only against the panel it read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread.artifacts import RUN_FILENAME, RunRecord, dump_run, write_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.data import DataUnavailableError
from optspread.experiment.schedule import execute_run
from optspread.harness import run_episodes, summarize
from optspread.manifest import MANIFEST_FILENAME, RunManifest, callable_name, file_sha256
from optspread.replay import find_runs, replay_matches, replay_run
from optspread.runs.rundir import changed_artifacts, valid_run
from optspread.train import POLICY_FILENAME

SCRIPTED = RunConfig(
    policy="delta_seller",
    seed=8000,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21), iv_premium_vol_pts=1.0),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
LEARNED = SCRIPTED.model_copy(
    update={"policy": "ppo", "train": TrainConfig(total_timesteps=64, seed=2)}
)


def _rewrite(run_dir: Path, record: RunRecord, *, update_digest: bool) -> None:
    (run_dir / RUN_FILENAME).write_text(dump_run(record))
    if update_digest:
        manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
        manifest["artifacts"][RUN_FILENAME] = file_sha256(run_dir / RUN_FILENAME)
        (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))


@pytest.mark.invariant
def test_a_scripted_run_reproduces_bit_for_bit(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    before = (tmp_path / RUN_FILENAME).read_bytes()
    matches, reason = replay_matches(tmp_path)
    assert matches, reason
    assert "reproduced" in reason
    assert (tmp_path / RUN_FILENAME).read_bytes() == before, "replay never writes the run"
    assert replay_run(tmp_path).render().startswith("IDENTICAL")


@pytest.mark.invariant
def test_a_learned_run_reproduces_by_retraining_from_its_config(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    assert replay_matches(tmp_path, fake_trainer)[0]


@pytest.mark.invariant
def test_a_learner_that_lands_elsewhere_is_caught_at_the_first_episode(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    # A manifest from before the trainer was recorded names none, so the replay retrains.
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    del manifest["trainer"]
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    outcome = replay_run(tmp_path, lambda _: FakeLearned(seed=1))
    assert not outcome.matches
    assert outcome.reason.startswith("episode seed 8000: saved pnl")
    assert outcome.render().startswith("DIFFERS")


@pytest.mark.invariant
def test_a_run_is_retrained_only_by_the_trainer_its_manifest_names(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    outcome = replay_run(tmp_path)  # the real learner was asked for; a stand-in made it
    assert not outcome.matches
    assert outcome.reason.startswith("trained by fake_learner.fake_trainer, not optspread.train")
    assert replay_run(tmp_path, fake_trainer).matches


def _scaled(run_dir: Path, factor: float, *, trades: int = 0) -> None:
    """Every P&L of the run moved by ``factor``, relatively, as another libm would move it."""
    record = RunRecord.model_validate_json((run_dir / RUN_FILENAME).read_text())
    episodes = tuple(
        e.model_copy(update={"pnl": e.pnl * factor, "n_trades": e.n_trades + trades})
        for e in record.episodes
    )
    moved = record.model_copy(update={"episodes": episodes, "mean_pnl": record.mean_pnl * factor})
    _rewrite(run_dir, moved, update_digest=True)


@pytest.mark.invariant
def test_a_tolerance_replay_says_identical_within_or_differs(tmp_path: Path) -> None:
    from scripts import replay as script

    execute_run(SCRIPTED, tmp_path)
    exact = replay_run(tmp_path, rtol=1e-9)
    assert exact.verdict == "IDENTICAL" and exact.matches and exact.reproduced
    _scaled(tmp_path, 1 + 3e-12)
    assert replay_run(tmp_path).verdict == "DIFFERS", "bytes are still the default"
    within = replay_run(tmp_path, rtol=1e-9)
    assert within.verdict == "WITHIN" and within.reproduced and not within.matches
    assert within.max_rel_diff is not None and 1e-12 < within.max_rel_diff < 1e-11
    assert within.render().startswith("WITHIN rtol 1e-09 (max rel diff 3e-12)")
    assert "every number of run.json within 1e-09" in within.reason
    tight = replay_run(tmp_path, rtol=1e-13)
    assert tight.verdict == "DIFFERS" and "beyond 1e-13 at mean_pnl" in tight.reason
    assert script.main([str(tmp_path)]) == 1
    assert script.main([str(tmp_path), "--rtol", "1e-9"]) == 0
    # A count is exact whatever the tolerance.
    execute_run(SCRIPTED, tmp_path / "counts")
    _scaled(tmp_path / "counts", 1.0, trades=1)
    counted = replay_run(tmp_path / "counts", rtol=1.0)
    assert counted.verdict == "DIFFERS" and "n_trades" in counted.reason


@pytest.mark.behaviour
def test_the_platform_a_run_was_recorded_on_is_named_when_it_differs(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    assert "recorded on" not in replay_run(tmp_path).reason
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    manifest["platform"] = {"os": "Windows", "arch": "AMD64", "python": "3.12.0", "numpy": "2.5.3"}
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    reason = replay_run(tmp_path).reason
    assert "recorded on Windows AMD64 (python 3.12.0, numpy 2.5.3), replayed on " in reason
    del manifest["platform"]
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    _scaled(tmp_path, 1 + 1e-12)
    assert "platform not recorded" in replay_run(tmp_path).reason


@pytest.mark.invariant
def test_a_results_file_edited_after_its_manifest_is_a_failure_of_the_run(
    tmp_path: Path,
) -> None:
    execute_run(SCRIPTED, tmp_path)
    record = RunRecord.model_validate_json((tmp_path / RUN_FILENAME).read_text())
    _rewrite(tmp_path, record.model_copy(update={"mean_pnl": 1.0}), update_digest=False)
    matches, reason = replay_matches(tmp_path)
    assert not matches and "digest" in reason


@pytest.mark.invariant
def test_a_replay_that_lands_on_a_different_number_says_which(tmp_path: Path) -> None:
    # The digest is consistent, so this is what a nondeterministic run would look like.
    execute_run(SCRIPTED, tmp_path)
    record = RunRecord.model_validate_json((tmp_path / RUN_FILENAME).read_text())
    episodes = list(record.episodes)
    episodes[2] = episodes[2].model_copy(update={"pnl": episodes[2].pnl + 1e-9})
    _rewrite(tmp_path, record.model_copy(update={"episodes": tuple(episodes)}), update_digest=True)
    matches, reason = replay_matches(tmp_path)
    assert not matches
    assert reason.startswith(f"episode seed {SCRIPTED.seed + 2}: saved pnl")

    _rewrite(tmp_path, record.model_copy(update={"t_stat": 0.0}), update_digest=True)
    assert replay_matches(tmp_path)[1].startswith("t_stat: saved 0.0")
    _rewrite(
        tmp_path, record.model_copy(update={"episodes": record.episodes[:2]}), update_digest=True
    )
    assert replay_matches(tmp_path)[1].startswith("2 episodes saved, 4 replayed")


@pytest.mark.behaviour
def test_provenance_that_moved_is_reported_beside_the_verdict(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    manifest["versions"]["numpy"] = "0.0.1"
    manifest["git_sha"] = "0" * 40
    manifest["git_dirty"] = True
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    matches, reason = replay_matches(tmp_path)
    assert matches, reason
    assert "recorded at git 00000000" in reason
    assert "numpy 0.0.1 ->" in reason and "dirty tree" in reason


@pytest.mark.invariant
def test_a_manifest_from_before_gymnasium_and_pandas_were_recorded_replays(
    tmp_path: Path,
) -> None:
    execute_run(SCRIPTED, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert "gymnasium" in manifest["versions"]
    for name in ("gymnasium", "pandas", "pyarrow"):  # what a manifest from before #31 lacks
        manifest["versions"].pop(name, None)
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    matches, reason = replay_matches(tmp_path)
    assert matches, reason
    assert "gymnasium" not in reason and "pandas" not in reason  # unrecorded is not "moved"


@pytest.mark.invariant
def test_what_is_not_a_reproducible_run_says_so(tmp_path: Path) -> None:
    assert replay_matches(tmp_path)[1].startswith("not a readable run")

    execute_run(SCRIPTED, tmp_path / "a")
    manifest = json.loads((tmp_path / "a" / MANIFEST_FILENAME).read_text())
    manifest["run_id"] = "0000000000000000-s0"
    (tmp_path / "a" / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert "does not belong" in replay_matches(tmp_path / "a")[1]


@pytest.mark.behaviour
def test_a_learned_run_without_the_learner_names_the_extra(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    del manifest["trainer"]  # as from before it was recorded: retrained by whatever is given
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    def absent(config: RunConfig) -> FakeLearned:
        raise ImportError("No module named 'stable_baselines3'")

    matches, reason = replay_matches(tmp_path, absent)
    assert not matches and "install the train extra" in reason


@pytest.mark.behaviour
def test_the_replay_command_takes_runs_or_folders_of_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import replay

    for k in range(3):
        execute_run(SCRIPTED.model_copy(update={"seed": 8000 + 10 * k}), tmp_path / f"r{k}")
    assert find_runs(tmp_path / "r0") == [tmp_path / "r0"]
    assert find_runs(tmp_path) == [tmp_path / f"r{k}" for k in range(3)]
    assert replay.main([str(tmp_path), "--max", "2"]) == 0
    assert "2/2 runs reproduce bit for bit" in capsys.readouterr().out
    assert replay.main([str(tmp_path / "r1")]) == 0

    (tmp_path / "r2" / RUN_FILENAME).write_text("{}")
    assert replay.main([str(tmp_path)]) == 1
    assert "2/3" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert replay.main([str(tmp_path / "empty")]) == 1


@pytest.mark.behaviour
def test_the_replay_command_refuses_to_replay_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import replay

    execute_run(SCRIPTED, tmp_path / "r0")
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit) as exit_:
            replay.main([str(tmp_path), "--max", bad])
        assert exit_.value.code == 2
        assert "--max must be at least 1" in capsys.readouterr().err
    assert replay.main([str(tmp_path / "absent")]) == 1
    assert "no such run directory" in capsys.readouterr().err
    assert replay.main([str(tmp_path / "r0" / RUN_FILENAME)]) == 1  # a file, not a run


@pytest.mark.invariant
def test_a_swapped_policy_fails_the_replay_and_the_resume(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    fake = callable_name(fake_trainer)
    assert valid_run(tmp_path, LEARNED, trainer=fake) is not None
    (tmp_path / POLICY_FILENAME).write_text("a different policy\n")
    matches, reason = replay_matches(tmp_path, fake_trainer)
    assert not matches and reason == f"{POLICY_FILENAME} no longer match the manifest's digests"
    assert valid_run(tmp_path, LEARNED, trainer=fake) is None  # the resume recomputes it
    (tmp_path / POLICY_FILENAME).unlink()
    assert not replay_matches(tmp_path, fake_trainer)[0]


@pytest.mark.invariant
def test_a_manifest_that_does_not_record_the_results_is_not_replayed(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    manifest["artifacts"] = {}
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert changed_artifacts(tmp_path, RunManifest.read(tmp_path)) == (RUN_FILENAME,)
    assert not replay_matches(tmp_path)[0]
    assert valid_run(tmp_path, SCRIPTED) is None


# -- the data source a run read (GH-21) --------------------------------------------- #


@pytest.mark.invariant
def test_a_run_from_before_the_data_record_replays_on_the_generator(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["data"] == {"source": "synthetic", "panel": {}}
    del manifest["data"]  # the manifest every run written before GH-21 has
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert valid_run(tmp_path, SCRIPTED) is not None, "a sweep still resumes from it"
    matches, reason = replay_matches(tmp_path)
    assert matches, reason


def _flat_panel(root: Path, n_days: int = 40) -> Path:
    """The two panel files the real source reads: a surface flat in strike and tenor,
    its level moving day to day, on a random-walk spot."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from optspread.realdata.panel import K_GRID, TENOR_GRID

    rng = np.random.default_rng(21)
    days = [
        d.date() for d in pd.bdate_range("2020-03-02", periods=n_days)
    ]  # inside the eval range (#28)
    level = 0.18 + 0.04 * rng.random(n_days)
    grid = pd.DataFrame(
        [
            (d, t, float(k), v)
            for d, v in zip(days, level, strict=True)
            for t in TENOR_GRID
            for k in K_GRID
        ],
        columns=["date", "tenor_days", "k", "iv"],
    )
    spot = 5000.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n_days)))
    daily = pd.DataFrame({"date": days, "spot": spot, "carry30": 0.02})
    (root / "panel").mkdir(parents=True)
    grid.to_parquet(root / "panel" / "surface_grid.parquet", index=False)
    daily.to_parquet(root / "panel" / "spx_daily.parquet", index=False)
    return root


@pytest.mark.invariant
def test_a_run_on_the_real_panel_records_it_and_replays_only_against_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SCRIPTED.model_copy(update={"n_episodes": 3})
    with pytest.raises(DataUnavailableError):  # no panel: refused, never the generator
        run_episodes(config, source="optionmetrics")
    data = _flat_panel(tmp_path / "data")
    monkeypatch.setenv("VEGALAB_DATA", str(data))
    run_dir = tmp_path / "run"
    episodes = run_episodes(config, source="optionmetrics")
    assert episodes != run_episodes(config), "the episodes ran on the panel"
    write_run(run_dir, config, summarize(episodes), episodes, source="optionmetrics")

    recorded = RunManifest.read(run_dir).data
    assert recorded is not None and recorded.source == "optionmetrics"
    assert recorded.panel == {
        f"panel/{name}": file_sha256(data / "panel" / name)
        for name in ("surface_grid.parquet", "spx_daily.parquet")
    }
    matches, reason = replay_matches(run_dir)
    assert matches, reason
    # Same run id as the synthetic run of this config: a synthetic resume must not take it.
    assert valid_run(run_dir, config) is None
    assert valid_run(run_dir, config, "optionmetrics") is not None

    (data / "panel" / "spx_daily.parquet").write_bytes(b"rebuilt since")
    matches, reason = replay_matches(run_dir)
    assert not matches and reason.startswith("the optionmetrics panel is not the one the run read")
    assert recorded.panel_id in reason
    (data / "panel" / "spx_daily.parquet").unlink()
    matches, reason = replay_matches(run_dir)
    assert not matches and reason.startswith("cannot replay on optionmetrics: no SPX panel")
