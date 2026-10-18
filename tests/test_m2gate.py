"""Dominick Dupuy: the M2 gate items (week 6).

Every item is tested both ways: with its deliverable absent it must read NOT CLEARED
and say what failed first, and with evidence that meets its criterion it must pass.
The deliverables written in parallel (training, replay, dashboard) are stood in for
by small modules registered under the names the gate imports, so these tests do not
change when the real modules land.
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optspread import m2gate
from optspread.config import RunConfig
from optspread.data import ConformanceReport
from optspread.gatereport import GateItem
from optspread.manifest import RunManifest
from optspread.validation import POISONED_PIPELINES


def blocked(item: GateItem, contains: str) -> None:
    assert not item.passed and item.verdict == "FAIL"
    assert contains in item.failed_first, item.failed_first
    assert item.has_criterion


def passes(item: GateItem) -> None:
    assert item.passed and item.verdict == "PASS", item.failed_first
    assert item.evidence


def write_run(root: Path, name: str, *, learner: bool = False) -> Path:
    run_dir = root / name
    manifest = RunManifest.create(RunConfig(policy="always_on", seed=len(name)))
    if learner:
        versions = {**manifest.versions, "torch": "2.4.0", "stable_baselines3": "2.3.2"}
        manifest = manifest.model_copy(update={"versions": versions})
    (run_dir / "run.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text("{}")
    manifest.add_artifact(run_dir, run_dir / "run.json")
    manifest.write(run_dir)
    return run_dir


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    def install(name: str, **attrs: Any) -> None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    yield install


class FakePPO:
    def __init__(self, policy: str, env: object, learning_rate: float = 3e-4, n_steps: int = 2048):
        pass


# -- import discipline ------------------------------------------------------- #


def _record(values: object) -> Callable[[], types.SimpleNamespace]:
    """A stand-in for train.frozen_hyperparameters: a callable returning ``.values``."""
    return lambda: types.SimpleNamespace(values=values)


def test_load_names_the_import_that_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "needs_dep.py").write_text("import not_a_real_dependency_xyz\n")
    (tmp_path / "broken_mod.py").write_text("raise RuntimeError('half written')\n")
    (tmp_path / "plain_mod.py").write_text("x = 1\n")
    assert m2gate._load("optspread.not_here", "f")[1] == "optspread.not_here does not exist yet"
    assert m2gate._load("not_here_either", "f")[1] == "not_here_either is not installed"
    assert "cannot import not_a_real_dependency_xyz" in m2gate._load("needs_dep", "f")[1]
    assert "half written" in m2gate._load("broken_mod", "f")[1]
    assert m2gate._load("plain_mod", "y")[1] == "plain_mod has no y"
    assert m2gate._load("plain_mod", "x") == (1, "")


def test_run_manifests_counts_the_valid_and_the_refused(tmp_path: Path) -> None:
    assert m2gate.run_manifests(tmp_path / "absent") == ([], 0)
    write_run(tmp_path, "a")
    write_run(tmp_path, "nested/b")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("{not json")
    runs, refused = m2gate.run_manifests(tmp_path)
    assert [p.name for p, _ in runs] == ["a", "b"] and refused == 1


# -- Ilan Danial ------------------------------------------------------------- #


def test_ppo_harness_absent_then_present(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "optspread.train_not_written")
    blocked(m2gate.ppo_harness_item(tmp_path), "does not exist yet")

    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "fake_train")
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record({}))
    blocked(m2gate.ppo_harness_item(tmp_path), "not a non-empty mapping")

    pinned = {"learning_rate": 3e-4, "n_steps": 252}
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record(pinned))
    fake_module("stable_baselines3", PPO=FakePPO)
    blocked(m2gate.ppo_harness_item(tmp_path), "no training run")

    write_run(tmp_path, "scripted")
    blocked(m2gate.ppo_harness_item(tmp_path), "no training run")
    write_run(tmp_path, "trained", learner=True)
    item = m2gate.ppo_harness_item(tmp_path)
    passes(item)
    assert "1 off the library default (n_steps)" in item.evidence
    assert "1 training run(s)" in item.evidence

    fake_module(
        "fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record({"learning_rat": 1e-3})
    )
    blocked(m2gate.ppo_harness_item(tmp_path), "not PPO arguments: learning_rat")


def test_ppo_harness_without_the_library(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "fake_train")
    fake_module(
        "fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record({"n_steps": 252})
    )
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)  # import raises
    item = m2gate.ppo_harness_item(tmp_path)
    assert not item.passed and "stable_baselines3" in item.failed_first


def test_replay_absent_empty_mismatched_raising_and_matching(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m2gate, "REPLAY_MODULE", "optspread.replay_not_written")
    blocked(m2gate.replay_item(tmp_path), "does not exist yet")

    monkeypatch.setattr(m2gate, "REPLAY_MODULE", "fake_replay")
    fake_module("fake_replay", replay_matches=lambda run_dir: (True, "identical"))
    blocked(m2gate.replay_item(tmp_path), "no valid run manifest")

    write_run(tmp_path, "one")
    passes(m2gate.replay_item(tmp_path))

    fake_module("fake_replay", replay_matches=lambda run_dir: (False, "run.json digest moved"))
    blocked(m2gate.replay_item(tmp_path), "run.json digest moved")
    # A bare False (no reason) still names what failed.
    fake_module("fake_replay", replay_matches=lambda run_dir: False)
    blocked(m2gate.replay_item(tmp_path), "digest differs")

    def boom(run_dir: Path) -> bool:
        raise OSError("disk full")

    fake_module("fake_replay", replay_matches=boom)
    blocked(m2gate.replay_item(tmp_path), "disk full")


def test_dashboard_absent_then_live(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, docs = tmp_path / "runs", tmp_path / "docs"
    docs.mkdir()
    (docs / "M1_GATE_REPORT.json").write_text("{}")
    monkeypatch.setattr(m2gate, "DASHBOARD_MODULE", "optspread.dashboard_not_written")
    blocked(m2gate.dashboard_item(runs, docs), "does not exist yet")

    monkeypatch.setattr(m2gate, "DASHBOARD_MODULE", "fake_dash")
    every_run = lambda root: [p for p, _ in m2gate.run_manifests(root)[0]]  # noqa: E731
    every_board = lambda d: sorted(d.glob("*_GATE_REPORT.json"))  # noqa: E731
    fake_module("fake_dash", list_runs=every_run, list_gate_reports=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "nothing real to browse")

    write_run(runs, "r1")
    write_run(runs, "r2")
    passes(m2gate.dashboard_item(runs, docs))

    fake_module("fake_dash", list_runs=lambda root: ["only one"], list_gate_reports=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "lists 1 runs, 2 have valid manifests")

    fake_module("fake_dash", list_runs=every_run, list_gate_reports=lambda d: [])
    blocked(m2gate.dashboard_item(runs, docs), "reads 0 reports, 1 sidecars")

    def boom(root: Path) -> list[Path]:
        raise KeyError("manifest_v2")

    fake_module("fake_dash", list_runs=boom, list_gate_reports=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "manifest_v2")


# -- Hemdutt Rao ------------------------------------------------------------- #


def suite(real: ConformanceReport) -> Callable[[], tuple[ConformanceReport, ...]]:
    synthetic = ConformanceReport("synthetic", True, 64, ())
    return lambda: (synthetic, real)


def test_the_adapter_clears_only_on_a_real_conforming_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m2gate, "conformance_suite", suite(ConformanceReport("optionmetrics", False, 0, ()))
    )
    blocked(m2gate.adapter_item(), "SKIPPED")
    bad = ConformanceReport("optionmetrics", True, 10, ("reset must return the chain at t=0",))
    monkeypatch.setattr(m2gate, "conformance_suite", suite(bad))
    blocked(m2gate.adapter_item(), "reset must return")
    monkeypatch.setattr(
        m2gate, "conformance_suite", suite(ConformanceReport("optionmetrics", True, 64, ()))
    )
    item = m2gate.adapter_item()
    passes(item)
    assert "optionmetrics PASS (64 chains)" in item.evidence


# -- Dominick Dupuy ---------------------------------------------------------- #


def write_pilot(root: Path, matrix: np.ndarray, *, record: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "matrix.npy", matrix)
    manifest = RunManifest.create(RunConfig(policy="delta_seller", seed=0))
    if record:
        manifest.add_artifact(root, root / "matrix.npy")
    manifest.write(root)


def pilot_matrix(seed: int = 0, S: int = 8, N: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 300.0, size=(S, 1)) + rng.normal(0.0, 400.0, size=(S, N))


def test_the_pilot_item_refuses_everything_short_of_a_recorded_matrix(tmp_path: Path) -> None:
    blocked(m2gate.pilot_item(tmp_path / "pilot"), "pilot not run")

    bare = tmp_path / "bare"
    bare.mkdir()
    np.save(bare / "matrix.npy", pilot_matrix())
    blocked(m2gate.pilot_item(bare), "no manifest.json")

    (bare / "manifest.json").write_text('{"run_id": ""}')
    blocked(m2gate.pilot_item(bare), "manifest refused")

    write_pilot(tmp_path / "unrecorded", pilot_matrix(), record=False)
    blocked(m2gate.pilot_item(tmp_path / "unrecorded"), "does not record matrix.npy")

    write_pilot(tmp_path / "changed", pilot_matrix())
    np.save(tmp_path / "changed" / "matrix.npy", pilot_matrix(seed=1))
    blocked(m2gate.pilot_item(tmp_path / "changed"), "changed under its manifest")

    write_pilot(tmp_path / "thin", pilot_matrix(S=1))
    blocked(m2gate.pilot_item(tmp_path / "thin"), "the matrix is 1x40")

    nan = pilot_matrix()
    nan[0, 0] = np.nan
    write_pilot(tmp_path / "nan", nan)
    blocked(m2gate.pilot_item(tmp_path / "nan"), "every entry finite")


def test_the_pilot_item_reports_components_and_the_floor(tmp_path: Path) -> None:
    write_pilot(tmp_path, pilot_matrix())
    item = m2gate.pilot_item(tmp_path)
    passes(item)
    assert "8 x 40 pilot (delta_seller" in item.evidence
    assert "tau^2 = " in item.evidence and "(se " in item.evidence
    assert "at S = 5" in item.evidence and "at S = 50" in item.evidence

    planned = m2gate.pilot_item(tmp_path, delta=200.0)
    passes(planned)
    assert "for delta = 200 the cheapest design is" in planned.evidence
    hopeless = m2gate.pilot_item(tmp_path, delta=1e-3)
    passes(hopeless)  # the pilot ran; that no design is feasible is a finding, not a failure
    assert "no design up to 64 seeds is feasible" in hopeless.evidence


def test_the_analysis_item_passes_live_and_names_its_numbers() -> None:
    item = m2gate.analysis_item(replications=4_000)
    passes(item)
    assert "3 of 3 wrong rules refused" in item.evidence
    assert "normal approximation reads" in item.evidence
    assert "noncentral-t" in item.criterion


def test_the_analysis_item_fails_when_the_gate_would_accept_a_wrong_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from optspread.validation import _default_pipeline

    monkeypatch.setattr(m2gate, "POISONED_PIPELINES", {"not actually wrong": _default_pipeline})
    blocked(m2gate.analysis_item(replications=500), "accepted a wrong rule: not actually wrong")


def test_the_analysis_item_fails_when_the_pipeline_misses_the_closed_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = m2gate.validate_pipeline
    wrong = POISONED_PIPELINES["normal critical values"]

    def with_wrong_rule(*args: Any) -> Any:
        return real(*args[:4], wrong)

    monkeypatch.setattr(m2gate, "validate_pipeline", with_wrong_rule)
    blocked(m2gate.analysis_item(replications=2_000), "rejection rate at delta")


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("x\n")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "commit", "-q", "-m", "scaffold")
    return tmp_path


def test_the_pre_registration_item_walks_from_untracked_to_tagged(repo: Path) -> None:
    prereg = repo / "docs" / "PRE_REGISTRATION.md"
    blocked(m2gate.prereg_item(repo), "is not tracked")

    prereg.write_text("# Pre-registration\n\nD1.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "prereg")
    blocked(m2gate.prereg_item(repo), "committed, not tagged")

    git(repo, "tag", "v0.2.0")  # a release tag is not the convention
    blocked(m2gate.prereg_item(repo), "committed, not tagged")

    git(repo, "tag", "-a", "prereg-m2", "-m", "M2 pre-registration")
    item = m2gate.prereg_item(repo)
    passes(item)
    assert item.evidence.startswith("prereg-m2 at ") and item.evidence.endswith("unchanged")

    prereg.write_text("# Pre-registration\n\nD1.\n\n## Amendment 2026-11-03\n")
    blocked(m2gate.prereg_item(repo), "uncommitted changes")
    git(repo, "commit", "-q", "-am", "amend")
    item = m2gate.prereg_item(repo)
    passes(item)
    assert "appends 25 characters of amendments" in item.evidence

    prereg.write_text("# Pre-registration\n\nD1, revised.\n")
    git(repo, "commit", "-q", "-am", "edit in place")
    blocked(m2gate.prereg_item(repo), "edited in place after prereg-m2")


def test_the_pre_registration_tag_must_hold_the_file_and_sit_on_this_history(
    repo: Path,
) -> None:
    git(repo, "tag", "prereg-early")  # tags the scaffold, before the file existed
    prereg = repo / "docs" / "PRE_REGISTRATION.md"
    prereg.write_text("# Pre-registration\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "prereg")
    blocked(m2gate.prereg_item(repo), "points at a commit without")

    git(repo, "tag", "-d", "prereg-early")
    git(repo, "checkout", "-q", "-b", "side")
    prereg.write_text("# Pre-registration\n\nside branch\n")
    git(repo, "commit", "-q", "-am", "side")
    git(repo, "tag", "prereg-side")
    git(repo, "checkout", "-q", "-")
    blocked(m2gate.prereg_item(repo), "is not on the history of HEAD")
