"""Ilan Danial: the reproducibility gate, any completed run reproduces bit for bit (week 6).

Project document 10.7. The check has to be able to fail, so most of this file is
about the ways it says no: a results file edited after its manifest was written, a
replay that lands on a different number, a run directory that is not one, and a
learner that is not installed. Each refusal names what failed first.

A learned run replays by retraining from its recorded config. Here the learner is
``tests/fake_learner.py``; ``tests/test_ppo.py`` replays a real PPO run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread.artifacts import RUN_FILENAME, RunRecord, dump_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256
from optspread.orchestrate import execute_run
from optspread.replay import find_runs, replay_matches, replay_run
from optspread.sweep import changed_artifacts, valid_run
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


def test_a_scripted_run_reproduces_bit_for_bit(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    before = (tmp_path / RUN_FILENAME).read_bytes()
    matches, reason = replay_matches(tmp_path)
    assert matches, reason
    assert "reproduced" in reason
    assert (tmp_path / RUN_FILENAME).read_bytes() == before, "replay never writes the run"
    assert replay_run(tmp_path).render().startswith("IDENTICAL")


def test_a_learned_run_reproduces_by_retraining_from_its_config(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    assert replay_matches(tmp_path, fake_trainer)[0]


def test_a_learner_that_lands_elsewhere_is_caught_at_the_first_episode(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    outcome = replay_run(tmp_path, lambda _: FakeLearned(seed=1))
    assert not outcome.matches
    assert outcome.reason.startswith("episode seed 8000: saved pnl")
    assert outcome.render().startswith("DIFFERS")


def test_a_results_file_edited_after_its_manifest_is_a_failure_of_the_run(
    tmp_path: Path,
) -> None:
    execute_run(SCRIPTED, tmp_path)
    record = RunRecord.model_validate_json((tmp_path / RUN_FILENAME).read_text())
    _rewrite(tmp_path, record.model_copy(update={"mean_pnl": 1.0}), update_digest=False)
    matches, reason = replay_matches(tmp_path)
    assert not matches and "digest" in reason


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


def test_what_is_not_a_reproducible_run_says_so(tmp_path: Path) -> None:
    assert replay_matches(tmp_path)[1].startswith("not a readable run")

    execute_run(SCRIPTED, tmp_path / "a")
    manifest = json.loads((tmp_path / "a" / MANIFEST_FILENAME).read_text())
    manifest["run_id"] = "0000000000000000-s0"
    (tmp_path / "a" / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert "does not belong" in replay_matches(tmp_path / "a")[1]


def test_a_learned_run_without_the_learner_names_the_extra(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)

    def absent(config: RunConfig) -> FakeLearned:
        raise ImportError("No module named 'stable_baselines3'")

    matches, reason = replay_matches(tmp_path, absent)
    assert not matches and "install the train extra" in reason


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


def test_a_swapped_policy_fails_the_replay_and_the_resume(tmp_path: Path) -> None:
    execute_run(LEARNED, tmp_path, fake_trainer)
    assert valid_run(tmp_path, LEARNED) is not None
    (tmp_path / POLICY_FILENAME).write_text("a different policy\n")
    matches, reason = replay_matches(tmp_path, fake_trainer)
    assert not matches and reason == f"{POLICY_FILENAME} no longer match the manifest's digests"
    assert valid_run(tmp_path, LEARNED) is None  # the resume recomputes it
    (tmp_path / POLICY_FILENAME).unlink()
    assert not replay_matches(tmp_path, fake_trainer)[0]


def test_a_manifest_that_does_not_record_the_results_is_not_replayed(tmp_path: Path) -> None:
    execute_run(SCRIPTED, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    manifest["artifacts"] = {}
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert changed_artifacts(tmp_path, RunManifest.read(tmp_path)) == (RUN_FILENAME,)
    assert not replay_matches(tmp_path)[0]
    assert valid_run(tmp_path, SCRIPTED) is None
