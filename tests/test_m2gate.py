"""Dominick Dupuy: the M2 gate items (week 6).

Every item is tested both ways: with its deliverable absent it must read NOT CLEARED
and say what failed first, and with evidence that meets its criterion it must pass.
The deliverables written in parallel (training, replay, dashboard) are stood in for
by small modules registered under the names the gate imports, so these tests do not
change when the real modules land.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optspread import m2gate
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.data import ConformanceReport
from optspread.gatereport import GateItem, build_report
from optspread.manifest import RunManifest
from optspread.orchestrate import pilot_grid, run_grid, write_pilot_matrices
from optspread.power import detectable_effect_t, variance_components
from optspread.validation import POISONED_PIPELINES


def blocked(item: GateItem, contains: str) -> None:
    assert not item.passed and item.verdict == "FAIL"
    assert contains in item.failed_first, item.failed_first
    assert item.has_criterion


def passes(item: GateItem) -> None:
    assert item.passed and item.verdict == "PASS", item.failed_first
    assert item.evidence


TRAIN = TrainConfig(total_timesteps=2048)


def write_run(root: Path, name: str, *, learner: bool = False, ppo: bool = False) -> Path:
    """A run directory. ``learner`` records torch and SB3 versions, as any manifest written
    with the train extra installed does; ``ppo`` makes it a PPO run."""
    run_dir = root / name
    config = (
        RunConfig(policy="ppo", seed=len(name), train=TRAIN)
        if ppo
        else RunConfig(policy="always_on", seed=len(name))
    )
    manifest = RunManifest.create(config)
    if learner:
        versions = {**manifest.versions, "torch": "2.4.0", "stable_baselines3": "2.3.2"}
        manifest = manifest.model_copy(update={"versions": versions})
    (run_dir / "run.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text("{}")
    manifest.add_artifact(run_dir, run_dir / "run.json")
    manifest.write(run_dir)
    return run_dir


def write_matrix_manifest(root: Path) -> Path:
    """What ``write_pilot_matrices`` leaves: a PPO-keyed manifest over matrices, no run.json."""
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "matrix.npy", np.zeros((2, 2)))
    manifest = RunManifest.create(RunConfig(policy="ppo", seed=1, train=TRAIN))
    manifest.add_artifact(root, root / "matrix.npy")
    manifest.write(root)
    return root


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    def install(name: str, **attrs: Any) -> None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    yield install


# -- import discipline ------------------------------------------------------- #


def _record(
    values: object, matches: bool | None = True, differences: tuple[str, ...] = ()
) -> Callable[[], types.SimpleNamespace]:
    """A stand-in for train.frozen_hyperparameters: a callable returning the record."""
    return lambda: types.SimpleNamespace(
        values=values,
        hash="0123abcd",
        library_version=None if matches is None else "2.3.2",
        matches_library=matches,
        differences=differences,
    )


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


def test_a_pilot_matrix_manifest_is_not_a_run(tmp_path: Path) -> None:
    write_matrix_manifest(tmp_path / "pilot")
    write_run(tmp_path, "r")
    assert len(m2gate.run_manifests(tmp_path)[0]) == 2
    assert [d.name for d, _ in m2gate.results_runs(tmp_path)] == ["r"]


# -- Ilan Danial ------------------------------------------------------------- #

# Policy settings beside PPO arguments, as the real record has: the record's own
# comparison with the library is what the item reads, not PPO's signature.
PINNED = {"learning_rate": 3e-4, "n_steps": 2048, "net_arch_pi": [64, 64], "ortho_init": True}


def test_ppo_harness_absent_then_present(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "optspread.train_not_written")
    blocked(m2gate.ppo_harness_item(tmp_path), "does not exist yet")

    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "fake_train")
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record({}))
    blocked(m2gate.ppo_harness_item(tmp_path), "not a non-empty mapping")

    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record(PINNED))
    blocked(m2gate.ppo_harness_item(tmp_path), "no PPO run")

    # Torch and SB3 are in every manifest written with the train extra installed.
    write_run(tmp_path, "scripted", learner=True)
    write_matrix_manifest(tmp_path / "pilot")
    blocked(m2gate.ppo_harness_item(tmp_path), "no PPO run")
    write_run(tmp_path, "trained", learner=True, ppo=True)
    item = m2gate.ppo_harness_item(tmp_path)
    passes(item)
    assert "4 pinned values, hash 0123abcd" in item.evidence
    assert "equal to the stable-baselines3 2.3.2 defaults" in item.evidence
    assert "1 PPO run(s)" in item.evidence

    moved = _record(PINNED, matches=False, differences=("n_steps", "ortho_init"))
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=moved)
    blocked(m2gate.ppo_harness_item(tmp_path), "2.3.2 defaults: n_steps, ortho_init")


def test_ppo_harness_without_the_library(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run(tmp_path, "trained", ppo=True)
    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "fake_train")
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record(PINNED, None))
    item = m2gate.ppo_harness_item(tmp_path)
    blocked(item, "stable-baselines3 is not installed")
    assert "4 pinned values, hash 0123abcd" in item.evidence


def test_ppo_harness_reads_the_real_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from optspread import train

    write_run(tmp_path, "trained", ppo=True)
    real = train.frozen_hyperparameters()
    assert {"activation_fn", "net_arch_pi", "ortho_init"} <= set(real.values)
    as_installed = dataclasses.replace(real, library_version="2.3.2", matches_library=True)
    monkeypatch.setattr(train, "frozen_hyperparameters", lambda: as_installed)
    item = m2gate.ppo_harness_item(tmp_path)
    passes(item)
    assert f"{len(real.values)} pinned values, hash {real.hash}" in item.evidence


def test_replay_absent_empty_mismatched_raising_and_matching(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m2gate, "REPLAY_MODULE", "optspread.replay_not_written")
    blocked(m2gate.replay_item(tmp_path), "does not exist yet")

    monkeypatch.setattr(m2gate, "REPLAY_MODULE", "fake_replay")
    fake_module("fake_replay", replay_matches=lambda run_dir: (True, "identical"))
    blocked(m2gate.replay_item(tmp_path), "no valid run manifest")
    write_matrix_manifest(tmp_path / "a_pilot")  # sorts first, and is not a run
    blocked(m2gate.replay_item(tmp_path), "no valid run manifest")

    write_run(tmp_path, "one")
    item = m2gate.replay_item(tmp_path)
    passes(item)
    assert "of 1 runs (0 PPO, 1 scripted)" in item.evidence
    assert "no PPO run under" in item.evidence

    fake_module("fake_replay", replay_matches=lambda run_dir: (False, "run.json digest moved"))
    blocked(m2gate.replay_item(tmp_path), "run.json digest moved")
    # A bare False (no reason) still names what failed.
    fake_module("fake_replay", replay_matches=lambda run_dir: False)
    blocked(m2gate.replay_item(tmp_path), "digest differs")

    def boom(run_dir: Path) -> bool:
        raise OSError("disk full")

    fake_module("fake_replay", replay_matches=boom)
    blocked(m2gate.replay_item(tmp_path), "disk full")


def test_replay_covers_a_ppo_run_and_a_scripted_run(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    replayed: list[str] = []

    def replay(run_dir: Path) -> tuple[bool, str]:
        replayed.append(run_dir.name)
        return (run_dir / "run.json").exists(), "no run.json to replay"

    monkeypatch.setattr(m2gate, "REPLAY_MODULE", "fake_replay")
    fake_module("fake_replay", replay_matches=replay)
    write_matrix_manifest(tmp_path / "a_pilot")
    write_run(tmp_path, "b_scripted", learner=True)
    write_run(tmp_path, "c_scripted")
    write_run(tmp_path, "d_trained", ppo=True)
    item = m2gate.replay_item(tmp_path)
    passes(item)
    assert replayed == ["d_trained", "b_scripted"]
    assert "(PPO, 1 artifacts) and " in item.evidence and "(scripted, 1 artifacts)" in item.evidence
    assert "of 3 runs (1 PPO, 2 scripted)" in item.evidence
    assert "no PPO run" not in item.evidence and "no scripted run" not in item.evidence


def test_dashboard_absent_then_live(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, docs = tmp_path / "runs", tmp_path / "docs"
    docs.mkdir()
    (docs / "M1_GATE_REPORT.json").write_text("{}")
    monkeypatch.setattr(m2gate, "DASHBOARD_MODULE", "optspread.dashboard_not_written")
    blocked(m2gate.dashboard_item(runs, docs), "does not exist yet")

    monkeypatch.setattr(m2gate, "DASHBOARD_MODULE", "fake_dash")

    def every_run(root: Path) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(path=p) for p, _ in m2gate.run_manifests(root)[0]]

    def every_board(d: Path) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(source=p) for p in sorted(d.glob("*_GATE_REPORT.json"))]

    fake_module("fake_dash", scan_runs=every_run, load_gate_boards=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "nothing real to browse")

    write_run(runs, "r1")
    write_run(runs, "r2")
    passes(m2gate.dashboard_item(runs, docs))

    def first_run(root: Path) -> list[types.SimpleNamespace]:
        return every_run(root)[:1]

    fake_module("fake_dash", scan_runs=first_run, load_gate_boards=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "does not list 1 valid run(s)")

    fake_module("fake_dash", scan_runs=every_run, load_gate_boards=lambda d: [])
    blocked(m2gate.dashboard_item(runs, docs), "does not read M1_GATE_REPORT.json")

    def boom(root: Path) -> list[Path]:
        raise KeyError("manifest_v2")

    fake_module("fake_dash", scan_runs=boom, load_gate_boards=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "manifest_v2")


def test_dashboard_reads_the_real_data_layer(tmp_path: Path) -> None:
    runs, docs = tmp_path / "runs", tmp_path / "docs"
    write_run(runs, "r1")
    write_run(runs, "nested/r2", ppo=True)
    write_matrix_manifest(runs / "pilot")
    (runs / "bad").mkdir()
    (runs / "bad" / "manifest.json").write_text("{not json")  # listed, but not a valid run
    build_report("M2", [GateItem("x", "y", "z", "e", passed=True)]).write(
        docs / "M2_GATE_REPORT.md"
    )
    item = m2gate.dashboard_item(runs, docs)
    passes(item)
    # the pilot's matrix manifest is not a run: the browser skips it and so does the count
    assert "lists 2 of 2 valid manifests" in item.evidence and "(3 rows in all)" in item.evidence
    assert "reads 1 of 1 sidecars" in item.evidence


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


def pilot_config(N: int = 40, policy: str = "ppo") -> RunConfig:
    if policy == "ppo":
        return RunConfig(policy="ppo", seed=0, n_episodes=N, train=TRAIN)
    return RunConfig(policy="delta_seller", seed=0, n_episodes=N)


def write_pilot(
    root: Path,
    matrix: np.ndarray,
    oracle: np.ndarray | None = None,
    *,
    config: RunConfig | None = None,
    record: tuple[str, ...] = ("matrix.npy", "matrix_oracle.npy"),
) -> None:
    """The two matrices and their manifest, as ``write_pilot_matrices`` leaves them."""
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "matrix.npy", matrix)
    np.save(root / "matrix_oracle.npy", np.zeros_like(matrix) if oracle is None else oracle)
    manifest = RunManifest.create(config or pilot_config(N=matrix.shape[-1]))
    for name in record:
        manifest.add_artifact(root, root / name)
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

    write_pilot(tmp_path / "unrecorded", pilot_matrix(), record=())
    blocked(m2gate.pilot_item(tmp_path / "unrecorded"), "does not record matrix.npy")

    write_pilot(tmp_path / "no_oracle", pilot_matrix(), record=("matrix.npy",))
    blocked(m2gate.pilot_item(tmp_path / "no_oracle"), "does not record matrix_oracle.npy")

    write_pilot(tmp_path / "changed", pilot_matrix())
    np.save(tmp_path / "changed" / "matrix.npy", pilot_matrix(seed=1))
    blocked(m2gate.pilot_item(tmp_path / "changed"), "matrix.npy changed under its manifest")

    write_pilot(tmp_path / "thin", pilot_matrix(S=1))
    blocked(m2gate.pilot_item(tmp_path / "thin"), "the matrix is 1x40")

    nan = pilot_matrix()
    nan[0, 0] = np.nan
    write_pilot(tmp_path / "nan", nan)
    blocked(m2gate.pilot_item(tmp_path / "nan"), "every entry finite")


def test_the_pilot_item_refuses_a_throwaway_or_mislabelled_matrix(tmp_path: Path) -> None:
    # What `make train-smoke` writes: 2 seeds x 3 episodes.
    write_pilot(tmp_path / "smoke", pilot_matrix(S=2, N=3))
    blocked(m2gate.pilot_item(tmp_path / "smoke"), "the matrix is 2x3, not S x N with S >= 5")
    write_pilot(tmp_path / "short", pilot_matrix(S=8, N=19))
    blocked(m2gate.pilot_item(tmp_path / "short"), "N >= 20")

    write_pilot(tmp_path / "scripted", pilot_matrix(), config=pilot_config(policy="delta_seller"))
    blocked(m2gate.pilot_item(tmp_path / "scripted"), "the delta_seller arm, not PPO")

    write_pilot(tmp_path / "unpaired", pilot_matrix(), pilot_matrix(S=7))
    blocked(m2gate.pilot_item(tmp_path / "unpaired"), "is 8x40 and matrix_oracle.npy 7x40")

    write_pilot(tmp_path / "relabelled", pilot_matrix(), config=pilot_config(N=100))
    blocked(
        m2gate.pilot_item(tmp_path / "relabelled"), "N = 40 episodes, the manifest's config 100"
    )


def test_the_pilot_item_reports_the_contrast_and_both_arms(tmp_path: Path) -> None:
    ppo, oracle = pilot_matrix(), pilot_matrix(seed=1) * 0.1
    write_pilot(tmp_path, ppo, oracle)
    item = m2gate.pilot_item(tmp_path)
    passes(item)
    assert "8 x 40 pilot (run " in item.evidence and "2,048 training steps" in item.evidence
    for label in ("contrast ppo - delta_seller tau^2 = ", "; ppo tau^2 = ", "delta_seller tau^2"):
        assert label in item.evidence
    # The floor is the exact t one, on the contrast rather than on the PPO arm alone.
    vc = variance_components(ppo - oracle)
    exact = f"{detectable_effect_t(vc.tau2, vc.sigma2, 5, 40):,.4g} at S = 5"
    assert exact in item.evidence and "at S = 50" in item.evidence
    alone = variance_components(ppo)
    assert f"{detectable_effect_t(alone.tau2, alone.sigma2, 5, 40):,.4g} at S" not in item.evidence

    planned = m2gate.pilot_item(tmp_path, delta=200.0)
    passes(planned)
    assert "for delta = 200 the cheapest design is" in planned.evidence
    hopeless = m2gate.pilot_item(tmp_path, delta=1e-3)
    passes(hopeless)  # the pilot ran; that no design is feasible is a finding, not a failure
    assert "no design up to 64 seeds is feasible" in hopeless.evidence


def test_the_pilot_item_on_a_real_pilot_checks_the_grid(tmp_path: Path) -> None:
    from fake_learner import fake_trainer

    base = RunConfig(
        seed=7000,
        market=MarketConfig(n_days=21, tenor_days=(10, 21)),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
    )
    grid = pilot_grid(base, n_seeds=5, n_episodes=20, premium=0.0, train=TRAIN)
    write_pilot_matrices(run_grid(grid, tmp_path, trainer=fake_trainer), tmp_path)
    item = m2gate.pilot_item(tmp_path)
    passes(item)
    assert "5 x 20 pilot" in item.evidence and "grid of 5 seeds" in item.evidence

    # Matrices re-recorded with a sixth seed no longer match the grid that produced them.
    manifest = RunManifest.read(tmp_path)
    for name in ("matrix.npy", "matrix_oracle.npy"):
        np.save(tmp_path / name, np.vstack([np.load(tmp_path / name), np.ones((1, 20))]))
        manifest.add_artifact(tmp_path, tmp_path / name)
    manifest.write(tmp_path)
    blocked(m2gate.pilot_item(tmp_path), "S = 6 seeds, the grid record 5")


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


# -- the report command ------------------------------------------------------ #


def test_the_report_defaults_its_roots_to_the_shared_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import m2_gate_report

    assert m2_gate_report.default_roots({}) == (Path("runs"), Path("runs/pilot"))
    data = tmp_path / "shared"
    runs, pilot = m2_gate_report.default_roots({"VEGALAB_DATA": str(data)})
    assert (runs, pilot) == (data / "runs", data / "runs" / "pilot")

    def stub(*args: Any) -> GateItem:
        return GateItem("stub", "x", "c", "e", passed=True)

    monkeypatch.setattr(m2_gate_report, "analysis_item", stub)
    monkeypatch.setattr(m2_gate_report, "adapter_item", stub)
    monkeypatch.setenv("VEGALAB_DATA", str(data))
    out = tmp_path / "docs" / "M2_GATE_REPORT.md"
    assert m2_gate_report.main(["--out", str(out), "--repo", str(tmp_path)]) == 1
    text = out.read_text()
    # the roots are the shared folder's, named as $VEGALAB_DATA in the committed file
    assert "Runs root: `$VEGALAB_DATA/runs`; pilot root: `$VEGALAB_DATA/runs/pilot`" in text
    assert "no $VEGALAB_DATA/runs/pilot/matrix.npy" in text
    assert str(data) not in text


def test_a_written_report_names_roots_portably(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.m2_gate_report import portable

    monkeypatch.setenv("VEGALAB_DATA", "/shared/vega data")
    text = f"/shared/vega data/runs and {Path.home()}/elsewhere"
    assert portable(text) == "$VEGALAB_DATA/runs and ~/elsewhere"


def test_a_run_whose_files_moved_under_its_manifest_is_not_evidence(tmp_path: Path) -> None:
    good = write_run(tmp_path, "good")
    moved = write_run(tmp_path, "moved")
    (moved / "run.json").write_text((moved / "run.json").read_text() + " ")
    kept = [d.resolve() for d, _ in m2gate.results_runs(tmp_path)]
    assert good.resolve() in kept and moved.resolve() not in kept
