"""Dominick Dupuy: the partner reproduction (#10): the expected file, the comparison, the
steps, the record and the one command that writes it.

The comparison runs for real: a seller cell of the tiny grid of ``tests/m3_fixtures.py``
regenerated through the sweep's own path and compared with what the grid wrote, and one
fixed-delta seller cell of the confirmatory sweep regenerated and compared with the
committed ``docs/SWEEP_EXPECTED.json`` within the stated tolerance. The bootstrap never
creates a venv or installs anything here: its subprocess calls are recorded instead. The
steps' commands are held to what ``make -n`` says the Makefile runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fake_learner import fake_trainer
from m3_fixtures import TINY, World, build_world, git

from optspread import reproduction as rp
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.orchestrate import grid_cells, grid_filename, read_grid

REPO = Path(__file__).resolve().parent.parent.parent
COMMITTED = REPO / "docs" / rp.EXPECTED_FILE


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return build_world(tmp_path_factory.mktemp("reproduction"))


# -- the expected file ---------------------------------------------------------------- #


@pytest.mark.invariant
def test_the_reference_cells_are_a_seller_per_premium_and_the_replayed_ppo_cell() -> None:
    refs = rp.reference_cells(CONFIRMATORY_GRID)
    sellers = [c for c in refs if c.arm == rp.SELLER]
    assert [c.premium for c in sellers] == sorted(CONFIRMATORY_GRID.premiums)
    assert [c.block for c in sellers] == [0, 1, 2, 3, 4, 0]
    assert [c.run_id for c in refs if c.arm == rp.PPO] == ["e608f959c816fc46-s102000"]


@pytest.mark.behaviour
def test_spread_takes_both_ends_and_the_middle_first() -> None:
    assert rp.spread([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]) == [0.0, 4.0, 1.0, 0.25, 0.5, 2.0]
    assert rp.spread([0.0, 1.0]) == [0.0, 1.0]


@pytest.mark.invariant
def test_the_committed_expected_file_is_the_whole_confirmatory_grid_and_small() -> None:
    expected = rp.read_expected(COMMITTED)
    assert COMMITTED.stat().st_size < 400_000
    assert expected.grid_hash == "7d9d53d340006f02" and expected.tolerance == rp.tolerances()
    places = {(c.premium, c.arm, c.block): c.run_id for c in grid_cells(CONFIRMATORY_GRID)}
    assert {(c.premium, c.arm, c.block): c.run_id for c in expected.cells} == places
    refs = {c.run_id for c in rp.reference_cells(CONFIRMATORY_GRID)}
    assert {c.run_id for c in expected.cells if c.episodes is not None} == refs
    assert all(c.n_episodes == 500 and len(c.run_json_sha256) == 64 for c in expected.cells)
    assert [c.premium for c in expected.sellers(3)] == [0.0, 4.0, 1.0]
    assert len(expected.sellers(99)) == 30 and expected.ppo_reference() is not None


@pytest.mark.invariant
@pytest.mark.slow
def test_one_confirmatory_seller_cell_regenerates_within_the_tolerance(tmp_path: Path) -> None:
    """The freeze, checked on whatever machine runs the suite: about 40 seconds."""
    expected = rp.read_expected(COMMITTED)
    cell = expected.sellers(1)[0]
    [result] = rp.check_cells(CONFIRMATORY_GRID, [cell], rp.SELLER_REL_TOL, tmp_path)
    assert result.within_tolerance, result.first_difference
    assert result.episodes_compared == 500 and result.max_rel_diff < 1e-11


@pytest.mark.invariant
def test_build_expected_round_trips_and_refuses_an_invalid_cell(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / rp.EXPECTED_FILE
    path.write_text(world.expected_text)
    assert rp.read_expected(path) == world.expected
    rows = [ln for ln in world.expected_text.splitlines() if ln.startswith('  {"run_id"')]
    assert len(world.expected.cells) == 18 and len(rows) == 18  # one line a cell
    head = json.loads(world.expected_text)
    assert head["sweep_git_shas"] == [world.at_tag]
    path.write_text(world.expected_text.replace('"schema_version": 1', '"schema_version": 9'))
    with pytest.raises(ValueError, match="schema v9"):
        rp.read_expected(path)
    root = Path(shutil.copytree(world.root, tmp_path / "runs"))
    record = read_grid(root / grid_filename(TINY))
    with pytest.raises(ValueError, match="is not valid under"):
        rp.build_expected(record, root)  # the real trainer is asked for, the fake one trained
    monkeypatch.setattr(rp, "REAL_TRAINER", f"{fake_trainer.__module__}.fake_trainer")
    shutil.rmtree(root / grid_cells(TINY)[0].run_id)
    with pytest.raises(ValueError, match="is not valid under"):
        rp.build_expected(record, root)
    fewer = record.model_copy(update={"cells": record.cells[:-1]})
    with pytest.raises(ValueError, match="exactly its grid's cells"):
        rp.build_expected(fewer, root)


@pytest.mark.behaviour
def test_the_expected_script_writes_the_file_and_refuses_a_missing_grid(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import sweep_expected

    out = tmp_path / "out.json"
    assert sweep_expected.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    monkeypatch.setattr(sweep_expected, "CONFIRMATORY_GRID", TINY)
    assert sweep_expected.main(["--root", str(world.root), "--out", str(out)]) == 1  # trainer
    monkeypatch.setattr(rp, "REAL_TRAINER", f"{fake_trainer.__module__}.fake_trainer")
    assert sweep_expected.main(["--root", str(world.root), "--out", str(out)]) == 0
    assert out.read_text() == world.expected_text


# -- comparing a regenerated cell ---------------------------------------------------- #


def saved(world: World, index: int = 0) -> tuple[rp.ExpectedCell, bytes]:
    cell = world.expected.sellers(3)[index]
    return cell, (world.root / cell.run_id / "run.json").read_bytes()


def edited(text: bytes, edit: Any) -> bytes:
    data = json.loads(text)
    edit(data)
    return json.dumps(data, indent=2).encode()


@pytest.mark.invariant
def test_the_saved_bytes_compare_identical_and_within(world: World) -> None:
    cell, text = saved(world)
    result = rp.compare_run(cell, text, rp.SELLER_REL_TOL)
    assert result.byte_identical and result.within_tolerance and result.max_rel_diff == 0.0
    assert result.episodes_compared == 3 and result.values_compared == 5 + 3 * 2


@pytest.mark.invariant
def test_last_digit_noise_differs_in_bytes_and_passes_the_tolerance(world: World) -> None:
    cell, text = saved(world)

    def nudge(d: dict[str, Any]) -> None:
        d["mean_pnl"] *= 1 + 3e-13
        d["episodes"][1]["pnl"] *= 1 + 8e-13

    result = rp.compare_run(cell, edited(text, nudge), rp.SELLER_REL_TOL)
    assert not result.byte_identical and result.within_tolerance
    assert 1e-13 < result.max_rel_diff < 1e-12


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda d: d["episodes"][2].update(pnl=d["episodes"][2]["pnl"] * (1 + 1e-6)), "pnl"),
        (lambda d: d["episodes"][0].update(n_trades=99), "n_trades: expected"),
        (lambda d: d["episodes"][0].update(steps=1), "steps: expected"),
        (lambda d: d["episodes"].pop(), "episode seeds"),
        (lambda d: d.update(run_id="other-s1"), "run_id: expected"),
        (lambda d: d.update(se_pnl="wide"), "se_pnl: expected"),
        (lambda d: d.update(mean_cost=d["mean_cost"] + 1.0), "mean_cost"),
    ],
)
def test_a_difference_the_model_could_make_is_outside_the_tolerance(
    world: World, edit: Any, why: str
) -> None:
    cell, text = saved(world)
    result = rp.compare_run(cell, edited(text, edit), rp.SELLER_REL_TOL)
    assert not result.within_tolerance and why in result.first_difference


@pytest.mark.invariant
def test_a_regeneration_that_is_not_a_run_is_refused(world: World) -> None:
    cell, _ = saved(world)
    assert "not JSON" in rp.compare_run(cell, b"raised", 1e-9).first_difference
    assert "not a JSON object" in rp.compare_run(cell, b"[1]", 1e-9).first_difference
    assert rp.close(0.0, 5e-7, 1e-9) and not rp.close(0.0, 5e-6, 1e-9)
    assert rp.relative_difference(0.0, 0.0) == 0.0


@pytest.mark.invariant
def test_a_tiny_seller_cell_regenerates_byte_identically_here(world: World, tmp_path: Path) -> None:
    lines: list[str] = []
    cells = world.expected.sellers(1)
    found = rp.check_cells(TINY, cells, rp.SELLER_REL_TOL, tmp_path, report=lines.append)
    assert [c.within_tolerance and c.byte_identical for c in found] == [True]
    assert (
        lines[0].startswith("  [1/1] PASS delta_seller at 0 block 0")
        and "byte-identical" in (lines[0])
    )
    ppo = world.expected.ppo_reference()
    assert ppo is not None
    [result] = rp.check_cells(
        TINY, [ppo], rp.SELLER_REL_TOL, tmp_path, trainer=fake_trainer, structure=_structure
    )
    assert result.within_tolerance and result.byte_identical
    assert result.check == "statistical" and result.z == 0.0 and result.k_se == rp.PPO_K_SE
    assert "checked statistically (PPO training is not bit-reproducible" in rp.describe(result)


def _structure(run_dir: Path) -> tuple[str, list[str]]:
    return "20 episodes: 9 opens, 0 refused, residual share 1.7%", []


def _ppo_run(world: World, tmp_path: Path, **update: Any) -> tuple[rp.ExpectedCell, Path]:
    """The tiny grid's reference PPO cell regenerated, its run.json then moved by ``update``."""
    from optspread.artifacts import RUN_FILENAME, dump_run, read_run

    ppo = world.expected.ppo_reference()
    assert ppo is not None
    rp.check_cells(
        TINY, [ppo], rp.SELLER_REL_TOL, tmp_path, trainer=fake_trainer, structure=_structure
    )
    run_dir = tmp_path / ppo.run_id
    if update:
        (run_dir / RUN_FILENAME).write_text(dump_run(read_run(run_dir).model_copy(update=update)))
    return ppo, run_dir


@pytest.mark.invariant
def test_a_retrained_ppo_cell_is_compared_statistically_not_by_its_numbers(
    world: World, tmp_path: Path
) -> None:
    # Another platform trains another policy: other trades and another mean. Within 3 se
    # of the reference it passes; a policy that did not learn, or does not trade, fails.
    ppo, run_dir = _ppo_run(world, tmp_path / "a")
    scale = (ppo.se_pnl**2 * 2) ** 0.5
    moved = {"mean_pnl": ppo.mean_pnl + 2.0 * scale, "mean_trades": ppo.mean_trades + 1.5}
    _, run_dir = _ppo_run(world, tmp_path / "b", **moved)
    found = rp.compare_ppo(ppo, run_dir, _structure)
    assert found.within_tolerance and not found.byte_identical
    assert found.z is not None and abs(found.z - 2.0) < 1e-9
    _, run_dir = _ppo_run(world, tmp_path / "c", mean_pnl=ppo.mean_pnl - 3.5 * scale)
    found = rp.compare_ppo(ppo, run_dir, _structure)
    assert not found.within_tolerance and "beyond 3 standard errors" in found.first_difference
    _, run_dir = _ppo_run(world, tmp_path / "d", mean_trades=0.0)
    assert "does not trade" in rp.compare_ppo(ppo, run_dir, _structure).first_difference

    def refused(run_dir: Path) -> tuple[str, list[str]]:
        return "20 episodes", ["2 of 11 opens refused"]

    _, run_dir = _ppo_run(world, tmp_path / "e")
    assert rp.compare_ppo(ppo, run_dir, refused).first_difference == "2 of 11 opens refused"
    step = rp.cells_step(rp.PPO_STEP, [rp.compare_ppo(ppo, run_dir, _structure)], 1.0)
    assert step.passed and "PPO checked statistically" in step.detail
    assert "not bit-reproducible across platforms" in step.detail


@pytest.mark.behaviour
@pytest.mark.needs_torch
def test_the_ppo_structure_is_read_off_the_policy_the_cell_saved(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
    from optspread.orchestrate import execute_run

    config = RunConfig(
        policy="ppo",
        seed=9000,
        n_episodes=3,
        market=MarketConfig(n_days=21, tenor_days=(10, 21), iv_premium_vol_pts=4.0),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
        train=TrainConfig(total_timesteps=2048, seed=0),
    )
    execute_run(config, tmp_path)
    summary, problems = rp.ppo_structure(tmp_path)
    assert (
        summary.startswith("3 episodes:") and "refused" in summary and "residual share" in summary
    )
    assert all("refused" in p or "residual" in p for p in problems)


@pytest.mark.behaviour
def test_a_cell_that_cannot_be_regenerated_is_a_finding(world: World, tmp_path: Path) -> None:
    cell = world.expected.sellers(1)[0]
    other = TINY.model_copy(update={"premiums": (1.0,), "arms": ("delta_seller",), "train": None})
    [result] = rp.check_cells(other, [cell], rp.SELLER_REL_TOL, tmp_path)
    assert not result.within_tolerance and "not JSON" in result.first_difference
    assert "FAIL" in rp.describe(result) and "first difference" in rp.describe(result)


# -- the steps ------------------------------------------------------------------------ #


def _make_n(target: str) -> list[list[str]]:
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
    out = subprocess.run(
        ["make", "-n", "-f", str(REPO / "Makefile"), target, "PY=python"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.split() for line in out.stdout.splitlines() if line.strip()]


@pytest.mark.behaviour
@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_the_steps_run_what_the_makefile_runs() -> None:
    targets = {
        "ruff": "lint",
        "mypy --strict": "typecheck",
        "pytest": "test",
        "wave0 gate": "gate",
        "cost drag": "drag",
        "leak suite": "leaks",
    }
    assert set(targets) == set(rp.STEP_COMMANDS)
    for step, target in targets.items():
        assert [list(c) for c in rp.STEP_COMMANDS[step]] == _make_n(target), step
    assert _make_n("check") == [c for s in ("lint", "typecheck", "test") for c in _make_n(s)]


@pytest.mark.behaviour
def test_a_step_runs_each_command_under_the_interpreter_and_stops_at_a_failure(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(command: Any, cwd: Path, env: Any) -> tuple[int, str]:
        calls.append(list(command))
        assert cwd == tmp_path and str(tmp_path) in env["PYTHONPATH"]
        return (1 if "check" in command else 0), "line\n" * 100

    step = rp.run_step("ruff", tmp_path, "/venv/python", runner)
    assert not step.passed and step.returncode == 1 and len(calls) == 1
    assert calls[0][0] == "/venv/python" and step.commands[0][0] == rp.PYTHON
    assert step.detail == "python -m ruff check optspread tests scripts exited 1"
    assert len(step.output_tail.splitlines()) == rp.OUTPUT_LINES
    calls.clear()
    assert rp.run_step("mypy --strict", tmp_path, "py", lambda c, d, e: (0, "ok")).passed


@pytest.mark.behaviour
def test_run_process_reports_what_it_could_not_start(tmp_path: Path) -> None:
    code, text = rp.run_process([sys.executable, "-c", "print('hi')"], tmp_path, os.environ)
    assert (code, text.strip()) == (0, "hi")
    code, text = rp.run_process(["/no/such/binary"], tmp_path, os.environ)
    assert code == 127 and "cannot run" in text


@pytest.mark.behaviour
def test_the_child_environment_tests_this_clone() -> None:
    clone = Path("/clone")  # \clone on Windows: the path as this OS writes it
    env = rp.child_env(clone, {"MAKEFLAGS": "-j", "PYTHONPATH": "/elsewhere"})
    assert "MAKEFLAGS" not in env
    assert env["PYTHONPATH"] == os.pathsep.join([str(clone), "/elsewhere"])
    assert rp.child_env(clone, {})["PYTHONPATH"] == str(clone)


@pytest.mark.behaviour
def test_the_poisoned_demo_step_is_the_dashboards_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    step = rp.poisoned_demo_step()
    assert step.passed and "poisoned flagged True" in step.detail
    from optspread import dashboard_data

    def broken(**_: Any) -> Any:
        raise RuntimeError("no detector")

    monkeypatch.setattr(dashboard_data, "run_leak_demo", broken)
    step = rp.poisoned_demo_step()
    assert not step.passed and "no detector" in step.detail


@pytest.mark.behaviour
def test_run_steps_runs_every_step_whatever_the_others_do(world: World, tmp_path: Path) -> None:
    def runner(command: Any, cwd: Path, env: Any) -> tuple[int, str]:
        return (2 if "mypy" in command else 0), "out"

    lines: list[str] = []
    names = ["ruff", "mypy --strict", "sweep cells", rp.PPO_STEP]
    steps, cells = rp.run_steps(
        names,
        repo=tmp_path,
        python="py",
        grid=TINY,
        sellers=world.expected.sellers(1),
        ppo=world.expected.ppo_reference(),
        report=lines.append,
        runner=runner,
        trainer=fake_trainer,
        structure=_structure,
    )
    assert [s.passed for s in steps] == [True, False, True, True]
    assert [c.arm for c in cells] == ["delta_seller", "ppo"]
    assert any("| out" in ln for ln in lines) and lines[0] == "[1/4] ruff ..."
    with pytest.raises(ValueError, match="no step named"):
        rp.run_steps(
            ["nope"], repo=tmp_path, python="py", grid=TINY, sellers=[], ppo=None, report=print
        )
    empty = rp.cells_step("sweep cells", [], 0.0)
    assert not empty.passed and empty.detail == "no cell regenerated"


# -- the record ----------------------------------------------------------------------- #


def record(**override: Any) -> rp.PartnerRecord:
    fields: dict[str, Any] = {
        "generated": "2026-11-13T15:00:00+00:00",
        "commit": "a" * 40,
        "dirty": False,
        "tag": rp.CUTOFF_TAG,
        "tag_commit": "b" * 40,
        "after_tag": True,
        "platform": rp.platform_facts(),
        "versions": rp.versions_now(),
        "expected_file": "docs/SWEEP_EXPECTED.json",
        "expected_sha256": "c" * 64,
        "tolerance": rp.tolerances(),
        "steps": [rp.StepResult(name=n, passed=True, seconds=1.0) for n in rp.STEPS],
        "cells": [],
        "seconds": 60.0,
    }
    return rp.PartnerRecord(**{**fields, **override})


@pytest.mark.invariant
def test_the_record_passes_only_clean_complete_and_passed() -> None:
    assert record().passed
    assert not record(dirty=True).passed and not record(tag_commit="").passed
    assert not record(after_tag=False).passed, "a commit before the tag is not the freeze"
    assert record(steps=record().steps[:-1]).missing_steps == ["sweep cells"]
    failed = [*record().steps, rp.StepResult(name=rp.PPO_STEP, passed=False, seconds=1.0)]
    assert (
        not record(steps=failed).passed and record(steps=failed).failed_steps[0].name == "ppo cell"
    )


@pytest.mark.behaviour
def test_the_summary_is_one_screen_saying_what_to_send_back(world: World, tmp_path: Path) -> None:
    cells = rp.check_cells(TINY, world.expected.sellers(1), rp.SELLER_REL_TOL, tmp_path)
    text = rp.render_summary(record(cells=cells, steps=record().steps[:-1]), "docs/R.json")
    assert text.startswith("# M3 partner reproduction: FAIL")
    assert "| sweep cells | NOT RUN | -- | |" in text and "(not named; pass --team)" in text
    assert "Send back docs/R.json and the .md beside it" in text and "PASS delta_seller" in text
    assert len(text.splitlines()) < 40


@pytest.mark.behaviour
def test_git_facts_reads_the_commit_the_tag_and_changes_but_the_record(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "a")
    git(tmp_path, "tag", "t1")
    facts = rp.git_facts(tmp_path, "t1")
    assert not facts.dirty and facts.commit == facts.tag_commit and len(facts.commit) == 40
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "R.json").write_text("{}")
    assert not rp.git_facts(tmp_path, "t1", ["docs/R.json"]).dirty
    assert rp.git_facts(tmp_path, "t1").dirty and rp.git_facts(tmp_path, "t2").tag_commit == ""
    assert facts.after_tag and not rp.git_facts(tmp_path, "t2").after_tag


@pytest.mark.invariant
def test_git_facts_says_whether_head_is_at_or_after_the_tag(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.txt").write_text(f"{name}\n")
        git(tmp_path, "add", ".")
        git(tmp_path, "commit", "-q", "-m", name)
        if name == "b":
            git(tmp_path, "tag", "t1")
    assert rp.git_facts(tmp_path, "t1").after_tag  # one commit after the tag
    git(tmp_path, "checkout", "-q", "HEAD~2")
    early = rp.git_facts(tmp_path, "t1")
    assert early.tag_commit and not early.after_tag, "the record must not pass before the tag"


@pytest.mark.behaviour
def test_the_expected_files_digest_is_the_same_from_an_lf_and_a_crlf_checkout() -> None:
    text = COMMITTED.read_bytes()
    assert b"\r\n" not in text
    assert rp.text_sha256(text.replace(b"\n", b"\r\n")) == rp.text_sha256(text)
    assert rp.text_sha256(text) == rp.sha256_bytes(text)


# -- the one command ------------------------------------------------------------------ #


class Calls:
    """Stands in for subprocess.run: records each command and its environment."""

    def __init__(self, fail_at: int | None = None, code: int = 0) -> None:
        self.commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.fail_at = fail_at
        self.code = code

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> Any:
        self.commands.append(command)
        self.envs.append(env)
        failed = self.fail_at is not None and len(self.commands) - 1 == self.fail_at
        return subprocess.CompletedProcess(command, 3 if failed else self.code)


@pytest.mark.behaviour
def test_the_bootstrap_makes_the_venv_installs_and_runs_itself_inside_it(tmp_path: Path) -> None:
    from scripts import reproduce

    calls = Calls()
    venv = tmp_path / ".venv"
    code = reproduce.bootstrap(
        ["--cells", "2"],
        run=calls,
        python="/usr/bin/python3",
        platform="darwin",
        version=(3, 12),
        venv=venv,
        env={"PIP_USER": "1", "HOME": "/h"},
    )
    vpy = str(venv / "bin" / "python")
    assert code == 0
    assert calls.commands[0] == ["/usr/bin/python3", "-m", "venv", str(venv)]
    assert calls.commands[1] == [vpy, "-m", "pip", "install", "-e", ".[dev,train]"]
    assert calls.commands[2][0] == vpy and calls.commands[2][-3:] == [
        "--cells",
        "2",
        "--skip-install",
    ]
    assert calls.commands[2][1].endswith("reproduce.py")
    assert all(env["PIP_USER"] == "0" and env["HOME"] == "/h" for env in calls.envs)


@pytest.mark.behaviour
def test_the_bootstrap_on_windows_and_linux_and_with_a_venv_already_there(tmp_path: Path) -> None:
    from scripts import reproduce

    win = reproduce.bootstrap_commands("py", tmp_path, exists=False, platform="win32")
    assert win[-1][0] == str(tmp_path / "Scripts" / "python.exe")
    linux = reproduce.bootstrap_commands("python3", tmp_path, exists=True, platform="linux")
    vpy = str(tmp_path / "bin" / "python")
    assert linux == [
        [vpy, "-m", "pip", "install", *reproduce.CPU_TORCH],
        [vpy, "-m", "pip", "install", "-e", ".[dev,train]"],
    ]
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").write_text("")
    calls = Calls()
    assert (
        reproduce.bootstrap([], run=calls, platform="darwin", version=(3, 13), venv=tmp_path) == 0
    )
    assert len(calls.commands) == 2  # install and re-run, no new venv


@pytest.mark.behaviour
def test_the_bootstrap_stops_at_a_failure_and_refuses_an_old_python(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import reproduce

    calls = Calls(fail_at=1)
    assert (
        reproduce.bootstrap([], run=calls, platform="darwin", version=(3, 12), venv=tmp_path) == 3
    )
    assert len(calls.commands) == 2 and "failed (exit 3)" in capsys.readouterr().err
    calls = Calls()
    assert reproduce.bootstrap([], run=calls, version=(3, 9), venv=tmp_path) == 2
    assert calls.commands == [] and "3.12 or newer is needed" in capsys.readouterr().err
    calls = Calls(code=1)  # every command exits 1: the first one's status is ours
    assert (
        reproduce.bootstrap([], run=calls, platform="darwin", version=(3, 12), venv=tmp_path) == 1
    )
    assert len(calls.commands) == 1


@pytest.mark.behaviour
def test_main_bootstraps_unless_told_to_run_here(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import reproduce

    seen: list[Any] = []
    monkeypatch.setattr(reproduce, "bootstrap", lambda argv: seen.append(("boot", argv)) or 5)
    monkeypatch.setattr(reproduce, "reproduce", lambda args: seen.append(("run", args.cells)) or 6)
    assert reproduce.main(["--cells", "4"]) == 5
    assert reproduce.main(["--skip-install", "--cells", "4"]) == 6
    assert seen == [("boot", ["--cells", "4"]), ("run", 4)]
    with pytest.raises(SystemExit):
        reproduce.main(["--cells", "0"])


@pytest.mark.behaviour
def test_the_run_writes_the_record_and_its_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import reproduce

    expected = rp.read_expected(COMMITTED)
    asked: dict[str, Any] = {}

    def run_steps(names: Any, **kw: Any) -> Any:
        asked.update(names=list(names), **kw)
        steps = [rp.StepResult(name=n, passed=True, seconds=1.0) for n in names]
        return steps, []

    monkeypatch.setattr(rp, "run_steps", run_steps)
    monkeypatch.setattr(
        rp, "git_facts", lambda repo, tag, ignore=(): rp.GitFacts("a" * 40, False, "b" * 40, True)
    )
    out = tmp_path / "R.json"
    args = reproduce.parse(["--skip-install", "--cells", "2", "--with-ppo", "--out", str(out)])
    assert reproduce.reproduce(args) == 0
    assert asked["names"] == [*rp.STEPS, rp.PPO_STEP]
    assert [c.premium for c in asked["sellers"]] == [0.0, 4.0]
    assert asked["ppo"] == expected.ppo_reference() and asked["python"] == sys.executable
    written = rp.PartnerRecord.model_validate_json(out.read_text())
    assert written.passed and written.expected_sha256 == rp.sha256_bytes(COMMITTED.read_bytes())
    assert out.with_suffix(".md").read_text().startswith("# M3 partner reproduction: PASS")
    assert "M3 partner reproduction: PASS" in capsys.readouterr().out
    monkeypatch.setattr(
        rp, "git_facts", lambda repo, tag, ignore=(): rp.GitFacts("a" * 40, True, "")
    )
    assert reproduce.reproduce(args) == 1
    shown = capsys.readouterr().out
    assert "uncommitted changes" in shown and "git fetch --tags" in shown
    monkeypatch.setattr(
        rp, "git_facts", lambda repo, tag, ignore=(): rp.GitFacts("a" * 40, False, "b" * 40)
    )
    assert reproduce.reproduce(args) == 1, "a commit before the tag does not pass"
    assert "is not at or after cutoff-2026-11-01" in capsys.readouterr().out
