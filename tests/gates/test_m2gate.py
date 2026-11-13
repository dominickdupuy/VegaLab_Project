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
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optspread import m2gate
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.data import ConformanceReport
from optspread.experiment.pilot import pilot_grid, write_pilot_matrices
from optspread.experiment.schedule import run_grid
from optspread.gatereport import GateItem, build_report
from optspread.manifest import RunManifest
from optspread.power import detectable_effect_t, variance_components
from optspread.validation import POISONED_PIPELINES, PREREGISTERED_EVENT, REJECTED_EVENTS


def blocked(item: GateItem, contains: str) -> None:
    assert not item.passed and item.verdict == "FAIL"
    assert contains in item.failed_first, item.failed_first
    assert item.has_criterion


def passes(item: GateItem) -> None:
    assert item.passed and item.verdict == "PASS", item.failed_first
    assert item.evidence


TRAIN = TrainConfig(total_timesteps=2048)


def write_run(
    root: Path,
    name: str,
    *,
    learner: bool = False,
    ppo: bool = False,
    trainer: str | None = m2gate.REAL_TRAINER,
) -> Path:
    """A run directory. ``learner`` records torch and SB3 versions, as any manifest written
    with the train extra installed does; ``ppo`` makes it a PPO run, trained by ``trainer``
    (the real learner unless a test says otherwise)."""
    run_dir = root / name
    config = (
        RunConfig(policy="ppo", seed=len(name), train=TRAIN)
        if ppo
        else RunConfig(policy="always_on", seed=len(name))
    )
    manifest = RunManifest.create(config, trainer=trainer if ppo else None)
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


@pytest.mark.internal
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


@pytest.mark.behaviour
def test_run_manifests_counts_the_valid_and_the_refused(tmp_path: Path) -> None:
    assert m2gate.run_manifests(tmp_path / "absent") == ([], 0)
    write_run(tmp_path, "a")
    write_run(tmp_path, "nested/b")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("{not json")
    runs, refused = m2gate.run_manifests(tmp_path)
    assert [p.name for p, _ in runs] == ["a", "b"] and refused == 1


@pytest.mark.behaviour
def test_a_pilot_matrix_manifest_is_not_a_run(tmp_path: Path) -> None:
    write_matrix_manifest(tmp_path / "pilot")
    write_run(tmp_path, "r")
    assert len(m2gate.run_manifests(tmp_path)[0]) == 2
    assert [d.name for d, _ in m2gate.results_runs(tmp_path)] == ["r"]


# -- Ilan Danial ------------------------------------------------------------- #

# Policy settings beside PPO arguments, as the real record has: the record's own
# comparison with the library is what the item reads, not PPO's signature.
PINNED = {"learning_rate": 3e-4, "n_steps": 2048, "net_arch_pi": [64, 64], "ortho_init": True}


@pytest.mark.behaviour
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


@pytest.mark.behaviour
def test_ppo_harness_without_the_library(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_run(tmp_path, "trained", ppo=True)
    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "fake_train")
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record(PINNED, None))
    item = m2gate.ppo_harness_item(tmp_path)
    blocked(item, "stable-baselines3 is not installed")
    assert "4 pinned values, hash 0123abcd" in item.evidence


@pytest.mark.behaviour
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


def sb3_archive(path: Path) -> None:
    """A zip laid out as Stable-Baselines3 saves one; the gate reads only its member names."""
    with zipfile.ZipFile(path, "w") as archive:
        for member in ("data", "policy.pth", m2gate.SB3_ARCHIVE_MEMBER):
            archive.writestr(member, "")


def with_policy(run_dir: Path, write: Callable[[Path], None]) -> None:
    """Save a policy.zip beside the run and hash it into its manifest, as execute_run does."""
    write(run_dir / "policy.zip")
    manifest = RunManifest.read(run_dir)
    manifest.add_artifact(run_dir, run_dir / "policy.zip")
    manifest.write(run_dir)


@pytest.mark.invariant
def test_ppo_harness_refuses_a_stand_in_learners_run(
    tmp_path: Path, fake_module: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m2gate, "TRAIN_MODULE", "fake_train")
    monkeypatch.setattr(m2gate, "REAL_TRAINER", "fake_train.train_ppo")
    fake_module("fake_train", train_ppo=lambda: None, frozen_hyperparameters=_record(PINNED))
    write_run(tmp_path, "fake", ppo=True, trainer="fake_learner.fake_trainer")
    item = m2gate.ppo_harness_item(tmp_path)
    blocked(item, "was trained by fake_learner.fake_trainer, not fake_train.train_ppo")
    assert "0 PPO run(s) by fake_train.train_ppo" in item.evidence and "1 more" in item.evidence

    # Before the trainer was recorded: a policy.zip that is not an SB3 archive, or none.
    runs = tmp_path / "before"
    old = write_run(runs, "old", ppo=True, trainer=None)
    blocked(m2gate.ppo_harness_item(runs), "records no trainer and no policy.zip")
    with_policy(old, lambda p: p.write_text("fake learner, training seed 0\n"))
    blocked(m2gate.ppo_harness_item(runs), "policy.zip is not a Stable-Baselines3 archive")
    with_policy(old, sb3_archive)
    passes(m2gate.ppo_harness_item(runs))
    # One real run among stand-ins is enough; the others are counted, not hidden.
    item = m2gate.ppo_harness_item(tmp_path)
    passes(item)
    assert "1 PPO run(s) by fake_train.train_ppo" in item.evidence and "1 more" in item.evidence

    # The real learner names itself the way the gate expects.
    from optspread.manifest import callable_name
    from optspread.train import train_ppo

    assert callable_name(train_ppo) == "optspread.train.train_ppo"


@pytest.mark.invariant
@pytest.mark.slow
def test_a_pilot_from_before_the_trainer_was_recorded_is_judged_by_its_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pilot(tmp_path / "gridless", pilot_matrix(), trainer=None)
    blocked(m2gate.pilot_item(tmp_path / "gridless"), "does not record which trainer")

    from fake_learner import fake_trainer

    base = RunConfig(
        seed=7000,
        market=MarketConfig(n_days=21, tenor_days=(10, 21)),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
    )
    grid = pilot_grid(base, n_seeds=5, n_episodes=20, premium=1.0, train=TRAIN)
    root = tmp_path / "old"
    record = run_grid(grid, root, trainer=fake_trainer)
    write_pilot_matrices(record, root)
    matrices = RunManifest.read(root)
    matrices.trainer = None  # what write_pilot_matrices wrote before #31
    matrices.write(root)
    blocked(m2gate.pilot_item(root), "was trained by fake_learner.fake_trainer")
    # Cells from before #31 too, each saved by the real learner: the pilot counts.
    for cell in (c for c in record.cells if c.arm == "ppo"):
        cell_manifest = RunManifest.read(root / cell.run_id)
        cell_manifest.trainer = None
        cell_manifest.write(root / cell.run_id)
        with_policy(root / cell.run_id, sb3_archive)
    passes(m2gate.pilot_item(root))
    gone = next(c for c in record.cells if c.arm == "ppo")
    (root / gone.run_id / "manifest.json").unlink()
    blocked(m2gate.pilot_item(root), f"PPO cell {gone.run_id} has no valid manifest")


@pytest.mark.invariant
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
    assert "run.json regenerated byte for byte" in item.evidence
    # The criterion says what is checked: run.json reproduced, policy.zip only for tampering.
    assert "regenerates its run.json byte for byte" in item.criterion
    assert "policy.zip) must still match its recorded digest" in item.criterion
    assert "reproduces every artifact digest" not in item.criterion
    assert "no PPO run under" in item.evidence

    fake_module("fake_replay", replay_matches=lambda run_dir: (False, "run.json digest moved"))
    blocked(m2gate.replay_item(tmp_path), "run.json digest moved")
    # A bare False (no reason) still names what failed.
    fake_module("fake_replay", replay_matches=lambda run_dir: False)
    blocked(m2gate.replay_item(tmp_path), "regenerated run.json differs")

    def boom(run_dir: Path) -> bool:
        raise OSError("disk full")

    fake_module("fake_replay", replay_matches=boom)
    blocked(m2gate.replay_item(tmp_path), "disk full")


@pytest.mark.behaviour
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


@pytest.mark.invariant
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

    def board(p: Path, stated: str = "PASS", derived: str = "PASS") -> types.SimpleNamespace:
        return types.SimpleNamespace(
            source=p, milestone=p.name.split("_")[0], stated_verdict=stated, verdict=derived
        )

    def every_board(d: Path) -> list[types.SimpleNamespace]:
        return [board(p) for p in sorted(d.glob("*_GATE_REPORT.json"))]

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

    # Read, but the file says PASS over an item that does not pass (#31).
    def overstated(d: Path) -> list[types.SimpleNamespace]:
        return [board(p, "PASS", "NOT CLEARED") for p in sorted(d.glob("*_GATE_REPORT.json"))]

    fake_module("fake_dash", scan_runs=every_run, load_gate_boards=overstated)
    item = m2gate.dashboard_item(runs, docs)
    blocked(item, "M1 states PASS where its items support NOT CLEARED")
    assert "0 of 1 boards stating the verdict their items support" in item.evidence

    def boom(root: Path) -> list[Path]:
        raise KeyError("manifest_v2")

    fake_module("fake_dash", scan_runs=boom, load_gate_boards=every_board)
    blocked(m2gate.dashboard_item(runs, docs), "manifest_v2")


@pytest.mark.behaviour
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


@pytest.mark.invariant
def test_dashboard_fails_on_a_real_board_whose_verdict_line_was_edited(tmp_path: Path) -> None:
    runs, docs = tmp_path / "runs", tmp_path / "docs"
    write_run(runs, "r1")
    failing = GateItem("x", "y", "z", "e", passed=False, failed_first="not yet")
    build_report("M2", [failing]).write(docs / "M2_GATE_REPORT.md")
    honest = m2gate.dashboard_item(runs, docs)
    assert honest.passed  # NOT CLEARED, honestly stated: the board reads it right
    sidecar = docs / "M2_GATE_REPORT.json"
    sidecar.write_text(
        sidecar.read_text().replace('"verdict": "NOT CLEARED"', '"verdict": "PASS"', 1)
    )
    blocked(m2gate.dashboard_item(runs, docs), "M2 states PASS where its items support NOT CLEARED")


# -- Hemdutt Rao ------------------------------------------------------------- #


def suite(real: ConformanceReport) -> Callable[[], tuple[ConformanceReport, ...]]:
    synthetic = ConformanceReport("synthetic", True, 64, ())
    return lambda: (synthetic, real)


@pytest.mark.behaviour
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
    market = MarketConfig(iv_premium_vol_pts=1.0)  # the pilot premium the contrast is at
    if policy == "ppo":
        return RunConfig(policy="ppo", market=market, seed=0, n_episodes=N, train=TRAIN)
    return RunConfig(policy="delta_seller", market=market, seed=0, n_episodes=N)


def write_pilot(
    root: Path,
    matrix: np.ndarray,
    oracle: np.ndarray | None = None,
    *,
    config: RunConfig | None = None,
    record: tuple[str, ...] = ("matrix.npy", "matrix_oracle.npy"),
    trainer: str | None = m2gate.REAL_TRAINER,
) -> None:
    """The two matrices and their manifest, as ``write_pilot_matrices`` leaves them."""
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "matrix.npy", matrix)
    np.save(root / "matrix_oracle.npy", np.zeros_like(matrix) if oracle is None else oracle)
    manifest = RunManifest.create(config or pilot_config(N=matrix.shape[-1]), trainer=trainer)
    for name in record:
        manifest.add_artifact(root, root / name)
    manifest.write(root)


def pilot_matrix(seed: int = 0, S: int = 8, N: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 300.0, size=(S, 1)) + rng.normal(0.0, 400.0, size=(S, N))


@pytest.mark.behaviour
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


@pytest.mark.invariant
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


@pytest.mark.invariant
def test_the_pilot_item_reports_the_contrast_and_the_control(tmp_path: Path) -> None:
    ppo, oracle = pilot_matrix(), pilot_matrix(seed=1) * 0.1
    write_pilot(tmp_path, ppo, oracle)
    item = m2gate.pilot_item(tmp_path)
    passes(item)
    assert "8 x 40 pilot (run " in item.evidence and "2,048 training steps" in item.evidence
    for label in ("contrast ppo at 1 - ppo at 0 tau^2 = ", "; delta_seller at 1 - at 0 tau^2"):
        assert label in item.evidence
    # The matrices are already the event's contrasts (#26): the floor is the exact t one on
    # the PPO matrix as written, not on PPO minus the seller.
    vc = variance_components(ppo)
    exact = f"{detectable_effect_t(vc.tau2, vc.sigma2, 5, 40):,.4g} at S = 5"
    assert exact in item.evidence and "at S = 50" in item.evidence
    old = variance_components(ppo - oracle)
    assert f"{detectable_effect_t(old.tau2, old.sigma2, 5, 40):,.4g} at S" not in item.evidence

    planned = m2gate.pilot_item(tmp_path, delta=200.0)
    passes(planned)
    assert "for delta = 200 the cheapest design is" in planned.evidence
    assert "(exact power " in planned.evidence and "normal" not in planned.evidence  # #20
    hopeless = m2gate.pilot_item(tmp_path, delta=1e-3)
    passes(hopeless)  # the pilot ran; that no design is feasible is a finding, not a failure
    assert "no design up to 64 seeds is feasible" in hopeless.evidence


@pytest.mark.invariant
@pytest.mark.slow
def test_the_pilot_item_on_a_real_pilot_checks_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_learner import fake_trainer

    base = RunConfig(
        seed=7000,
        market=MarketConfig(n_days=21, tenor_days=(10, 21)),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
    )
    grid = pilot_grid(base, n_seeds=5, n_episodes=20, premium=1.0, train=TRAIN)
    write_pilot_matrices(run_grid(grid, tmp_path, trainer=fake_trainer), tmp_path)
    # A pilot of the stand-in learner is not a pilot of PPO (#31), however it is laid out.
    assert RunManifest.read(tmp_path).trainer == "fake_learner.fake_trainer"
    blocked(m2gate.pilot_item(tmp_path), "trained by fake_learner.fake_trainer, not optspread")
    # The rest of the item, with the stand-in taken for the real learner.
    monkeypatch.setattr(m2gate, "REAL_TRAINER", "fake_learner.fake_trainer")
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


@pytest.mark.invariant
@pytest.mark.slow
def test_the_analysis_item_passes_live_and_names_its_numbers() -> None:
    item = m2gate.analysis_item(replications=4_000)
    passes(item)
    assert "3 of 3 wrong rules refused" in item.evidence
    assert "normal approximation reads" in item.evidence
    assert "noncentral-t" in item.criterion
    assert "null world: each arm against itself at zero premium fires at 0.0" in item.evidence
    assert "the learner against the fixed-delta seller at one premium at 0.99" in item.evidence
    assert "cost drag" in item.criterion


@pytest.mark.invariant
@pytest.mark.slow
def test_the_analysis_item_fails_when_the_event_fires_in_the_null_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #26: the contrast PPO minus the seller detects a flat learner beating a seller that
    # pays its costs. Pre-registering that event must not clear the gate.
    rejected = next(iter(REJECTED_EVENTS.values()))
    monkeypatch.setattr(m2gate, "PREREGISTERED_EVENT", ("against the seller", rejected))
    blocked(m2gate.analysis_item(replications=2_000), "fires at 0.99")


@pytest.mark.invariant
@pytest.mark.slow
def test_the_analysis_item_fails_when_the_null_world_accepts_a_replaced_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m2gate, "REJECTED_EVENTS", {"not actually wrong": PREREGISTERED_EVENT[1]})
    blocked(m2gate.analysis_item(replications=2_000), "did not refuse a replaced event")


@pytest.mark.invariant
def test_the_analysis_item_fails_when_the_gate_would_accept_a_wrong_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from optspread.validation import _default_pipeline

    monkeypatch.setattr(m2gate, "POISONED_PIPELINES", {"not actually wrong": _default_pipeline})
    blocked(m2gate.analysis_item(replications=500), "accepted a wrong rule: not actually wrong")


@pytest.mark.invariant
@pytest.mark.slow
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


@pytest.mark.invariant
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


@pytest.mark.invariant
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


@pytest.mark.behaviour
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


@pytest.mark.behaviour
def test_a_written_report_names_roots_portably(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.m2_gate_report import portable

    monkeypatch.setenv("VEGALAB_DATA", "/shared/vega data")
    text = f"/shared/vega data/runs and {Path.home()}/elsewhere"
    assert portable(text) == "$VEGALAB_DATA/runs and ~/elsewhere"


@pytest.mark.behaviour
def test_a_run_whose_files_moved_under_its_manifest_is_not_evidence(tmp_path: Path) -> None:
    good = write_run(tmp_path, "good")
    moved = write_run(tmp_path, "moved")
    (moved / "run.json").write_text((moved / "run.json").read_text() + " ")
    kept = [d.resolve() for d, _ in m2gate.results_runs(tmp_path)]
    assert good.resolve() in kept and moved.resolve() not in kept
