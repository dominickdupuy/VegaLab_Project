"""What a partner team reproduces from a clean clone, and the record they send back (#10).

Lane: Dominick Dupuy (statistical spine).

The M3 row of the schedule ends with "partner team clones the repository and
independently runs the gate commands". A clone has the code and nothing else: no shared
folder, no sweep. So the sweep comes with the repository in miniature,
``docs/SWEEP_EXPECTED.json`` (``build_expected``, written by ``scripts/sweep_expected.py``
from the confirmatory sweep's run directories): for every cell of the A7 grid its run id,
place in the grid, the sha256 of its ``run.json`` and its headline numbers, and for a
handful of reference cells every episode's P&L, cost, trade count and length. A few
hundred kilobytes, and no market data: the cells are on the generator.

``scripts/reproduce.py`` is the one command. It runs the gate commands (the parts of
``make check``, ``make gate``, ``make drag``, ``make leaks``, and the poisoned-environment
demo), then regenerates sweep cells from scratch, through ``orchestrate.execute_run``, the
path the sweep wrote them by, and compares each with the expected file twice:

- **byte for byte**: the sha256 of the regenerated ``run.json`` against the sweep's. This
  is the replay gate's promise, and it is made on the platform and library versions that
  produced the run. The sweep ran on Windows; on macOS every one of the 30 fixed-delta
  seller cells regenerates with each P&L and cost within 1.70e-11 of the Windows value,
  relatively (the largest in 085c6b5f8479e932-s102000), and their bytes differ, because the
  platforms' libm round ``exp``, ``log`` and ``erf`` differently in the last bit, and that
  last bit reaches the printed digits.
- **within a stated tolerance**: every number of the run (the headline fields, and every
  episode of a reference cell) within ``rel_tol`` of the sweep's, relatively, or
  ``ABS_TOL`` dollars absolutely for a value near zero; every count and identifier exactly.

``SELLER_REL_TOL = 1e-9`` is some 60 times the largest cross-platform difference measured
over the whole seller arm, and far below anything the model can do to a number: a cent on
a $10,000 episode is 1e-6, a changed seed or strike moves an episode by whole percent. A
scripted cell outside it has been computed by a different simulation, not a different libm.

A PPO cell (``--with-ppo``, one million training steps, about 20 minutes) is checked
**statistically**, because PPO training is not bit-reproducible across platforms, not even
at the sweep's exact library versions: a last-bit difference in one observation flips a
greedy action, and a million SGD steps compound it into another policy. The reference cell
(e608f959c816fc46-s102000, mean P&L 4,958.07 on Windows) retrained on macOS gave 4,661.39
and, at the sweep's pinned versions, 4,246.41, with 471 and 427 of its 500 trade counts
different; the Windows policy itself, scored on macOS, matches its run.json to 1.8e-12, so
it is the training that diverges, not the evaluation. The regenerated cell is therefore
compared as a draw of the same training recipe (``compare_ppo``):

- its mean P&L against the reference's, as ``z = (mean - reference mean) / sqrt(se**2 +
  reference se**2)``, within ``PPO_K_SE = 3`` standard errors. Both cells are scored on the
  same 500 evaluation paths, which correlates them positively, so the independent-samples
  denominator is the larger one and the test errs towards passing a cell trained the same
  way (|z| > 3 is a 0.27% event for two draws of the same expected P&L). The two macOS
  retrains above sit at z = -0.41 and -0.91. A policy that did not learn (a flat book
  earns 0) sits at z = -6.3 and fails, as does a broken environment;
- its structure: it trades (``mean_trades > 0``), no open is refused by the margin rule on
  its first ``STRUCTURE_EPISODES`` episodes (``actionmix``; the reference's policy refuses
  none), and the P&L attribution's residual share on them stays under
  ``optspread.m4gate.RESIDUAL_SHARE_TOL`` (the reference's: 1.7%), so the greeks still
  explain what it earns.

The PPO step is optional: a reproduction without it passes on the other steps alone, and a
reproduction with it must pass it too. ``optspread.replay``'s tolerance mode does not make
a retrained PPO cell comparable across platforms either: it is the same policy-level
divergence, which no tolerance on the numbers absorbs.

The record (``PartnerRecord``, ``docs/M3_PARTNER_REPRODUCTION.json`` by default) is what
the M3 gate's partner item reads: the commit and the tag reproduced, the platform, the
Python and library versions, each step with its commands, result and timing, and each
regenerated cell's comparison. Its schema is fixed here, so the script that writes it and
the item that reads it cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from optspread.artifacts import RUN_FILENAME, read_run
from optspread.config import ArmName
from optspread.manifest import RunManifest, file_sha256, library_versions
from optspread.orchestrate import Cell, GridConfig, GridRecord, execute_run, grid_cells
from optspread.sweep import REAL_TRAINER, valid_run
from optspread.train import Trainer, train_ppo

SCHEMA_VERSION = 1
EXPECTED_FILE = "SWEEP_EXPECTED.json"  # under docs/
RECORD_FILE = "M3_PARTNER_REPRODUCTION.json"  # under docs/, with a .md beside it
# The tag a reproduction is of: the cutoff, after which nothing that determines a run's
# numbers changes (project document 12.1), and so the commit the sweep's numbers belong to.
CUTOFF_TAG = "cutoff-2026-11-01"

SELLER_REL_TOL = 1e-9
ABS_TOL = 1e-6  # dollars: the floor under a relative tolerance for a value near zero
PPO_K_SE = 3.0  # a retrained PPO cell's mean within this many standard errors (docstring)
STRUCTURE_EPISODES = 20  # episodes of a retrained PPO cell re-simulated for its structure

SELLER: ArmName = "delta_seller"
PPO: ArmName = "ppo"
SUMMARY_FIELDS = ("mean_pnl", "se_pnl", "t_stat", "mean_cost", "mean_trades")
DEFAULT_CELLS = 3

# The steps of a reproduction, in order. The first six are the gate commands as the
# Makefile runs them (``make check`` is ruff, mypy and pytest); ``STEP_COMMANDS`` holds
# them and tests/gates/test_m3gate.py holds them to ``make -n``.
STEPS = (
    "ruff",
    "mypy --strict",
    "pytest",
    "wave0 gate",
    "cost drag",
    "leak suite",
    "poisoned demo",
    "sweep cells",
)
PPO_STEP = "ppo cell"  # only with --with-ppo; when recorded it must pass too
PYTHON = "python"  # how a recorded command names the interpreter it ran under
_SOURCES = ("optspread", "tests", "scripts")
STEP_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "ruff": (
        (PYTHON, "-m", "ruff", "check", *_SOURCES),
        (PYTHON, "-m", "ruff", "format", "--check", *_SOURCES),
    ),
    "mypy --strict": ((PYTHON, "-m", "mypy", "--strict", "optspread", "scripts"),),
    "pytest": (
        (
            PYTHON,
            "-m",
            "pytest",
            "tests/",
            "--cov=optspread",
            "--cov-report=term-missing",
            "--cov-fail-under=85",
        ),
    ),
    "wave0 gate": ((PYTHON, "-m", "scripts.wave0_gate"),),
    "cost drag": ((PYTHON, "-m", "scripts.cost_drag"),),
    "leak suite": ((PYTHON, "-m", "scripts.leak_report"),),
}
OUTPUT_LINES = 60  # the tail of a step's output kept in the record: every table fits


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(data: bytes) -> str:
    """The sha256 of a committed text file with its line endings as git stores them (LF).

    A Windows clone with ``core.autocrlf`` checks a text file out with CRLF; the bytes git
    holds for it, which the gate reads back with ``git show``, are LF. Both sides hash this,
    so they hash the same bytes (``.gitattributes`` also keeps the committed JSON LF)."""
    return sha256_bytes(data.replace(b"\r\n", b"\n"))


# -- the expected results ------------------------------------------------------ #


class ExpectedEpisodes(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: list[int]
    pnl: list[float]
    cost: list[float]
    n_trades: list[int]
    steps: list[int]


class ExpectedCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    arm: ArmName
    premium: float
    block: int
    seed: int
    train_seed: int | None
    n_episodes: int
    run_json_sha256: str
    mean_pnl: float
    se_pnl: float
    t_stat: float
    mean_cost: float
    mean_trades: float
    episodes: ExpectedEpisodes | None = None  # a reference cell's every episode

    @property
    def label(self) -> str:
        return f"{self.arm} at {self.premium:g} block {self.block}"


class SweepExpected(BaseModel):
    """``docs/SWEEP_EXPECTED.json``: the confirmatory sweep, small enough to commit."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    grid_hash: str
    grid_file: str
    sweep_git_shas: list[str]
    sweep_versions: dict[str, str]
    tolerance: dict[str, float]
    cells: tuple[ExpectedCell, ...]

    def sellers(self, n: int) -> list[ExpectedCell]:
        """The first ``n`` seller cells to regenerate: the reference cells, spread over the
        premiums (``spread``), then every other seller cell in grid order."""
        refs = {c.premium: c for c in self.cells if c.arm == SELLER and c.episodes is not None}
        first = [refs[p] for p in spread(sorted(refs))]
        rest = [c for c in self.cells if c.arm == SELLER and c not in first]
        return [*first, *rest][:n]

    def ppo_reference(self) -> ExpectedCell | None:
        return next((c for c in self.cells if c.arm == PPO and c.episodes is not None), None)


def spread(premiums: Sequence[float]) -> list[float]:
    """The lowest, the highest and the middle first, then the rest in order: three cells
    cover the curve's two ends and its middle."""
    if len(premiums) <= 2:
        return list(premiums)
    head = [premiums[0], premiums[-1], premiums[len(premiums) // 2]]
    return [*head, *(p for p in premiums if p not in head)]


def reference_cells(grid: GridConfig) -> list[Cell]:
    """The cells whose every episode is kept: one seller cell per premium, block ``i``
    for the ``i``-th premium (wrapping), and the PPO cell at the largest premium in the
    last block (in the A7 grid e608f959c816fc46-s102000, which replayed bit for bit on
    the machine that trained it)."""
    premiums = sorted(grid.premiums)
    cells = {(c.premium, c.arm, c.block): c for c in grid_cells(grid)}
    refs = [
        cells[(p, SELLER, i % grid.n_seeds)]
        for i, p in enumerate(premiums)
        if (p, SELLER, i % grid.n_seeds) in cells
    ]
    if (premiums[-1], PPO, grid.n_seeds - 1) in cells:
        refs.append(cells[(premiums[-1], PPO, grid.n_seeds - 1)])
    return refs


def build_expected(record: GridRecord, root: Path) -> SweepExpected:
    """The expected file from a finished grid under ``root``; raises on any invalid cell."""
    grid = record.grid
    listed = {(c.premium, c.arm, c.block): c.run_id for c in record.cells}
    if listed != {(c.premium, c.arm, c.block): c.run_id for c in grid_cells(grid)}:
        raise ValueError("the record does not list exactly its grid's cells")
    refs = {c.run_id for c in reference_cells(grid)}
    cells: list[ExpectedCell] = []
    shas: set[str] = set()
    versions: dict[str, str] = {}
    for cell in grid_cells(grid):
        run_dir = root / cell.run_id
        trainer = REAL_TRAINER if cell.config.train is not None else None
        if valid_run(run_dir, cell.config, trainer=trainer) is None:
            raise ValueError(f"cell {cell.label.strip()} ({cell.run_id}) is not valid under {root}")
        run = read_run(run_dir)
        manifest = RunManifest.read(run_dir)
        shas.add(manifest.git_sha)
        versions = versions or dict(manifest.versions)
        episodes = None
        if cell.run_id in refs:
            episodes = ExpectedEpisodes(
                seed=[e.seed for e in run.episodes],
                pnl=[e.pnl for e in run.episodes],
                cost=[e.cost for e in run.episodes],
                n_trades=[e.n_trades for e in run.episodes],
                steps=[e.steps for e in run.episodes],
            )
        cells.append(
            ExpectedCell(
                run_id=cell.run_id,
                arm=cell.arm,
                premium=cell.premium,
                block=cell.block,
                seed=cell.config.seed,
                train_seed=None if cell.config.train is None else cell.config.train.seed,
                n_episodes=run.n_episodes,
                run_json_sha256=file_sha256(run_dir / RUN_FILENAME),
                mean_pnl=run.mean_pnl,
                se_pnl=run.se_pnl,
                t_stat=run.t_stat,
                mean_cost=run.mean_cost,
                mean_trades=run.mean_trades,
                episodes=episodes,
            )
        )
    return SweepExpected(
        grid_hash=record.grid_hash,
        grid_file=f"grid-{record.grid_hash}.json",
        sweep_git_shas=sorted(shas),
        sweep_versions=versions,
        tolerance=tolerances(),
        cells=tuple(cells),
    )


def tolerances() -> dict[str, float]:
    return {"seller_rel": SELLER_REL_TOL, "abs": ABS_TOL, "ppo_k_se": PPO_K_SE}


def render_expected(expected: SweepExpected) -> str:
    """One cell per line, its episodes on the same line: small, and diffable by cell."""
    head = expected.model_dump(mode="json", exclude={"cells"})
    lines = [json.dumps(head, indent=1)[:-2] + ",", ' "cells": [']
    rows = [json.dumps(c.model_dump(mode="json")) for c in expected.cells]
    lines += [f"  {row}{',' if i < len(rows) - 1 else ''}" for i, row in enumerate(rows)]
    lines += [" ]", "}"]
    return "\n".join(lines) + "\n"


def read_expected(path: Path) -> SweepExpected:
    expected = SweepExpected.model_validate_json(path.read_text(encoding="utf-8"))
    if expected.schema_version != SCHEMA_VERSION:
        raise ValueError(f"{path.name} is schema v{expected.schema_version}, not v{SCHEMA_VERSION}")
    return expected


# -- comparing one regenerated cell ---------------------------------------------- #


class CellComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    arm: ArmName
    premium: float
    block: int
    rel_tol: float
    byte_identical: bool
    within_tolerance: bool
    max_rel_diff: float
    values_compared: int
    episodes_compared: int
    first_difference: str = ""
    seconds: float = 0.0
    # "numeric" for a seller, every number within ``rel_tol``; "statistical" for a PPO
    # cell (``compare_ppo``): ``z`` within ``k_se``, and its ``structure`` checked.
    check: Literal["numeric", "statistical"] = "numeric"
    z: float | None = None
    k_se: float | None = None
    structure: str = ""


def relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def close(a: float, b: float, rel_tol: float) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), ABS_TOL)


def compare_run(expected: ExpectedCell, text: bytes, rel_tol: float) -> CellComparison:
    """The regenerated ``run.json`` (its bytes) against the expected cell."""
    worst, compared, episodes = 0.0, 0, 0
    problems: list[str] = []

    def number(name: str, want: float, got: Any) -> None:
        nonlocal worst, compared
        compared += 1
        if not isinstance(got, int | float) or isinstance(got, bool):
            problems.append(f"{name}: expected {want!r}, regenerated {got!r}")
            return
        if math.isnan(want) and math.isnan(got):
            return
        worst = max(worst, relative_difference(want, float(got)))
        if not close(want, float(got), rel_tol):
            problems.append(f"{name}: expected {want!r}, regenerated {got!r}")

    def exact(name: str, want: Any, got: Any) -> None:
        if want != got:
            problems.append(f"{name}: expected {want!r}, regenerated {got!r}")

    try:
        data = json.loads(text)
    except ValueError as exc:
        data = {}
        problems.append(f"the regenerated run.json is not JSON: {exc}")
    if isinstance(data, dict):
        exact("run_id", expected.run_id, data.get("run_id"))
        exact("config_hash", expected.run_id.split("-s")[0], data.get("config_hash"))
        exact("policy", expected.arm, data.get("policy"))
        exact("seed", expected.seed, data.get("seed"))
        exact("n_episodes", expected.n_episodes, data.get("n_episodes"))
        for name in SUMMARY_FIELDS:
            number(name, float(getattr(expected, name)), data.get(name))
        rows = data.get("episodes") or []
        seeds = [r.get("seed") for r in rows if isinstance(r, dict)]
        exact(
            "episode seeds", list(range(expected.seed, expected.seed + expected.n_episodes)), seeds
        )
        ref = expected.episodes
        if ref is not None and len(rows) == len(ref.pnl):
            for i, row in enumerate(rows):
                at = f"episode seed {ref.seed[i]}"
                exact(f"{at} n_trades", ref.n_trades[i], row.get("n_trades"))
                exact(f"{at} steps", ref.steps[i], row.get("steps"))
                number(f"{at} pnl", ref.pnl[i], row.get("pnl"))
                number(f"{at} cost", ref.cost[i], row.get("cost"))
            episodes = len(rows)
    else:
        problems.append("the regenerated run.json is not a JSON object")
    return CellComparison(
        run_id=expected.run_id,
        arm=expected.arm,
        premium=expected.premium,
        block=expected.block,
        rel_tol=rel_tol,
        byte_identical=sha256_bytes(text) == expected.run_json_sha256,
        within_tolerance=not problems,
        max_rel_diff=worst,
        values_compared=compared,
        episodes_compared=episodes,
        first_difference=problems[0] if problems else "",
    )


Structure = Callable[[Path], tuple[str, list[str]]]


def ppo_structure(run_dir: Path) -> tuple[str, list[str]]:
    """``(summary, problems)`` of a regenerated PPO run's structure: its first
    ``STRUCTURE_EPISODES`` episodes re-simulated with the policy it saved, for the opens
    the margin rule refused (``actionmix``) and the attribution's residual share."""
    from optspread.actionmix import mix_run
    from optspread.attribution import attribute_run
    from optspread.m4gate import RESIDUAL_SHARE_TOL

    config = RunManifest.read(run_dir).config
    seeds = list(range(config.seed, config.seed + min(STRUCTURE_EPISODES, config.n_episodes)))
    mix = mix_run(run_dir, seeds).summary
    share = attribute_run(run_dir, seeds).residual_share
    problems = []
    if mix.n_refused:
        problems.append(f"{mix.n_refused} of {mix.n_opens + mix.n_refused} opens refused")
    if not share < RESIDUAL_SHARE_TOL:
        problems.append(f"residual share {share:.1%}, not under {RESIDUAL_SHARE_TOL:.0%}")
    summary = (
        f"{len(seeds)} episodes: {mix.n_opens} opens, {mix.n_refused} refused, residual "
        f"share {share:.1%}"
    )
    return summary, problems


def compare_ppo(
    expected: ExpectedCell, run_dir: Path, structure: Structure = ppo_structure
) -> CellComparison:
    """A regenerated PPO cell against the reference, statistically (the module docstring):
    identifiers and episode seeds exact, the mean within ``PPO_K_SE`` standard errors,
    and its structure (it trades, no refused open, the residual share)."""
    path = run_dir / RUN_FILENAME
    text = path.read_bytes() if path.is_file() else b""
    problems: list[str] = []
    z: float | None = None
    rel = math.inf
    summary = ""
    try:
        run = read_run(run_dir)
    except (OSError, ValueError) as exc:
        problems.append(f"the regenerated run.json is refused: {exc}")
    else:
        for name, want, got in (
            ("run_id", expected.run_id, run.run_id),
            ("policy", expected.arm, run.policy),
            ("seed", expected.seed, run.seed),
            ("n_episodes", expected.n_episodes, run.n_episodes),
        ):
            if want != got:
                problems.append(f"{name}: expected {want!r}, regenerated {got!r}")
        seeds = [e.seed for e in run.episodes]
        if seeds != list(range(expected.seed, expected.seed + expected.n_episodes)):
            problems.append("the episode seeds are not the reference's")
        scale = math.hypot(run.se_pnl, expected.se_pnl)
        z = (run.mean_pnl - expected.mean_pnl) / scale if scale > 0 else math.inf
        rel = relative_difference(run.mean_pnl, expected.mean_pnl)
        if not abs(z) <= PPO_K_SE:
            problems.append(
                f"mean P&L {run.mean_pnl:,.2f} against the reference's {expected.mean_pnl:,.2f} "
                f"is z = {z:.2f}, beyond {PPO_K_SE:g} standard errors"
            )
        if not run.mean_trades > 0:
            problems.append("it does not trade (mean trades 0)")
        if not problems:
            try:
                summary, found = structure(run_dir)
            except Exception as exc:  # a structure that cannot be read is a finding
                summary, found = "", [f"its structure could not be re-simulated: {exc!r}"]
            problems += found
    return CellComparison(
        run_id=expected.run_id,
        arm=expected.arm,
        premium=expected.premium,
        block=expected.block,
        rel_tol=0.0,
        byte_identical=bool(text) and sha256_bytes(text) == expected.run_json_sha256,
        within_tolerance=not problems,
        max_rel_diff=rel,
        values_compared=1 if z is not None else 0,
        episodes_compared=expected.n_episodes if z is not None else 0,
        first_difference=problems[0] if problems else "",
        check="statistical",
        z=z,
        k_se=PPO_K_SE,
        structure=summary,
    )


def grid_cell(grid: GridConfig, expected: ExpectedCell) -> Cell:
    """The cell of ``grid`` at the expected cell's place, by premium, arm and block.

    Not by run id: the run id is what the regenerated config hashes to on this machine,
    and that is one of the things compared."""
    for cell in grid_cells(grid):
        if (cell.premium, cell.arm, cell.block) == (expected.premium, expected.arm, expected.block):
            return cell
    raise KeyError(f"the grid has no cell {expected.label}")


def regenerate(cell: Cell, workdir: Path, trainer: Trainer = train_ppo) -> bytes:
    """Run ``cell`` from scratch under ``workdir`` the way the sweep wrote it; its run.json."""
    run_dir = workdir / cell.run_id
    execute_run(cell.config, run_dir, trainer)
    return (run_dir / RUN_FILENAME).read_bytes()


def check_cells(
    grid: GridConfig,
    cells: Sequence[ExpectedCell],
    rel_tol: float,
    workdir: Path,
    *,
    trainer: Trainer = train_ppo,
    report: Callable[[str], None] | None = None,
    structure: Structure = ppo_structure,
) -> list[CellComparison]:
    """Regenerate each cell under ``workdir`` and compare it: a seller within ``rel_tol``
    (``compare_run``), a PPO cell statistically (``compare_ppo``)."""
    out: list[CellComparison] = []
    for i, expected in enumerate(cells, 1):
        start = time.perf_counter()
        raised = ""
        try:
            text = regenerate(grid_cell(grid, expected), workdir, trainer)
        except Exception as exc:  # a cell that cannot be regenerated is a finding
            raised = f"regeneration raised {exc!r}"
            text = raised.encode()
        if expected.arm == PPO:
            found = compare_ppo(expected, workdir / expected.run_id, structure)
            if raised:
                found = found.model_copy(
                    update={"within_tolerance": False, "first_difference": raised}
                )
        else:
            found = compare_run(expected, text, rel_tol)
        result = found.model_copy(update={"seconds": time.perf_counter() - start})
        out.append(result)
        if report is not None:
            report(f"  [{i}/{len(cells)}] {describe(result)}")
    return out


def describe(c: CellComparison) -> str:
    verdict = "PASS" if c.within_tolerance else "FAIL"
    bytes_ = "byte-identical" if c.byte_identical else "bytes differ"
    if c.check == "statistical":
        z = f"z = {c.z:.2f}" if c.z is not None else "no z"
        text = (
            f"{verdict} {c.arm} at {c.premium:g} block {c.block} ({c.run_id}): checked "
            f"statistically (PPO training is not bit-reproducible across platforms): mean "
            f"P&L {z} (within {c.k_se:g} se), {c.structure or 'structure not checked'}; "
            f"{bytes_}, {c.seconds:.0f}s"
        )
        return text + (f"; first difference {c.first_difference}" if c.first_difference else "")
    text = (
        f"{verdict} {c.arm} at {c.premium:g} block {c.block} ({c.run_id}): {bytes_}, "
        f"max relative difference {c.max_rel_diff:.2g} over {c.values_compared} values "
        f"(tolerance {c.rel_tol:g}), {c.seconds:.0f}s"
    )
    return text + (f"; first difference {c.first_difference}" if c.first_difference else "")


# -- the steps and the record ------------------------------------------------------ #


class StepResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    commands: list[list[str]] = Field(default_factory=list)
    passed: bool
    returncode: int | None = None
    seconds: float
    detail: str = ""
    output_tail: str = ""


Runner = Callable[[Sequence[str], Path, Mapping[str, str]], tuple[int, str]]


def run_process(command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> tuple[int, str]:
    """``(exit status, stdout and stderr together)``; a command that cannot start is 127."""
    try:
        done = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError as exc:
        return 127, f"cannot run {command[0]}: {exc}"
    return done.returncode, done.stdout


def child_env(repo: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The steps' environment: this clone first on the path, so its code is what is tested,
    and no make variables inherited from an outer make."""
    env = {k: v for k, v in (os.environ if base is None else base).items()}
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo), env.get("PYTHONPATH", "")) if p)
    return env


def run_step(name: str, repo: Path, python: str, runner: Runner = run_process) -> StepResult:
    """One of the Makefile's steps, each command under ``python``; stops at the first failure."""
    commands = [list(c) for c in STEP_COMMANDS[name]]
    env = child_env(repo)
    start = time.perf_counter()
    output: list[str] = []
    code = 0
    for command in commands:
        code, text = runner([python if a == PYTHON else a for a in command], repo, env)
        output.append(text)
        if code != 0:
            break
    tail = "\n".join("".join(output).rstrip().splitlines()[-OUTPUT_LINES:])
    return StepResult(
        name=name,
        commands=commands,
        passed=code == 0,
        returncode=code,
        seconds=time.perf_counter() - start,
        detail="" if code == 0 else f"{' '.join(commands[len(output) - 1])} exited {code}",
        output_tail=tail,
    )


def poisoned_demo_step() -> StepResult:
    """The dashboard's leak demo, run in-process: the poisoned env caught twice, the clean
    one quiet (``dashboard_data.run_leak_demo``, the function the demo endpoint serves)."""
    from optspread.dashboard_data import run_leak_demo

    start = time.perf_counter()
    try:
        demo = run_leak_demo()
    except Exception as exc:
        return StepResult(
            name="poisoned demo",
            passed=False,
            seconds=time.perf_counter() - start,
            detail=f"run_leak_demo raised {exc!r}",
        )
    detail = (
        f"eps = {demo.eps:g} over {len(demo.seeds)} seeds: poisoned flagged "
        f"{demo.poisoned.flagged} "
        f"(worst {demo.poisoned.worst_feature}, p = {demo.poisoned.p_value:.2g}), its past moved "
        f"{not demo.poisoned_past_unchanged}; clean flagged {demo.clean.flagged}, its past moved "
        f"{not demo.clean_past_unchanged}"
    )
    return StepResult(
        name="poisoned demo",
        commands=[["optspread.dashboard_data.run_leak_demo()"]],
        passed=demo.caught,
        seconds=time.perf_counter() - start,
        detail=detail,
    )


def cells_step(name: str, comparisons: Sequence[CellComparison], seconds: float) -> StepResult:
    failed = [c for c in comparisons if not c.within_tolerance]
    identical = sum(c.byte_identical for c in comparisons)
    detail = (
        f"{len(comparisons)} cell(s) regenerated: {len(comparisons) - len(failed)} within "
        f"tolerance, {identical} byte-identical"
    )
    if any(c.check == "statistical" for c in comparisons):
        detail += (
            "; PPO checked statistically, since its training is not bit-reproducible "
            f"across platforms (mean within {PPO_K_SE:g} se, trades, no refused open, "
            "residual share)"
        )
    if not comparisons:
        detail = "no cell regenerated"
    elif failed:
        detail += f"; {failed[0].run_id}: {failed[0].first_difference}"
    return StepResult(
        name=name,
        commands=[["optspread.reproduction.check_cells()"]],
        passed=bool(comparisons) and not failed,
        seconds=seconds,
        detail=detail,
    )


def run_steps(
    names: Sequence[str],
    *,
    repo: Path,
    python: str,
    grid: GridConfig,
    sellers: Sequence[ExpectedCell],
    ppo: ExpectedCell | None,
    report: Callable[[str], None] = print,
    runner: Runner = run_process,
    trainer: Trainer = train_ppo,
    structure: Structure = ppo_structure,
) -> tuple[list[StepResult], list[CellComparison]]:
    """Run ``names`` in order, each whatever happens to the others; the steps and the cells."""
    steps: list[StepResult] = []
    compared: list[CellComparison] = []
    for i, name in enumerate(names, 1):
        report(f"[{i}/{len(names)}] {name} ...")
        began = time.perf_counter()
        if name in STEP_COMMANDS:
            step = run_step(name, repo, python, runner)
        elif name == "poisoned demo":
            step = poisoned_demo_step()
        elif name in ("sweep cells", PPO_STEP):
            cells = list(sellers) if name == "sweep cells" else [c for c in (ppo,) if c is not None]
            tol = SELLER_REL_TOL
            with tempfile.TemporaryDirectory(prefix="optspread-reproduce-") as work:
                found = check_cells(
                    grid,
                    cells,
                    tol,
                    Path(work),
                    trainer=trainer,
                    report=report,
                    structure=structure,
                )
            compared += found
            step = cells_step(name, found, time.perf_counter() - began)
        else:
            raise ValueError(f"no step named {name!r}")
        steps.append(step)
        report(f"      {'PASS' if step.passed else 'FAIL'} in {step.seconds:.0f}s {step.detail}")
        if not step.passed and step.output_tail:
            report("\n".join("      | " + ln for ln in step.output_tail.splitlines()[-15:]))
    return steps, compared


class PartnerRecord(BaseModel):
    """``docs/M3_PARTNER_REPRODUCTION.json``: what the partner ran and what they got."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    kind: Literal["m3-partner-reproduction"] = "m3-partner-reproduction"
    team: str = ""
    generated: str
    commit: str
    dirty: bool
    tag: str
    tag_commit: str
    # HEAD is the tag's commit or a descendant of it; False in a record from before
    # it was recorded, which then cannot pass (the gate checks it again in any case).
    after_tag: bool = False
    platform: dict[str, str]
    versions: dict[str, str]
    expected_file: str
    expected_sha256: str
    tolerance: dict[str, float]
    steps: list[StepResult]
    cells: list[CellComparison]
    seconds: float

    @property
    def missing_steps(self) -> list[str]:
        names = {s.name for s in self.steps}
        return [s for s in STEPS if s not in names]

    @property
    def failed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if not s.passed]

    @property
    def passed(self) -> bool:
        """Every step present and passed, from a clean checkout of a commit at or after
        the tag. The PPO step is optional; recorded, it must pass too."""
        return (
            not self.dirty
            and bool(self.tag_commit)
            and self.after_tag
            and not self.missing_steps
            and not self.failed_steps
        )


def platform_facts() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
    }


def versions_now() -> dict[str, str]:
    return library_versions()


def render_summary(record: PartnerRecord, record_path: str) -> str:
    """The one screen the script prints, and the Markdown beside the record."""
    lines = [
        f"# M3 partner reproduction: {'PASS' if record.passed else 'FAIL'}",
        "",
        f"Team: {record.team or '(not named; pass --team)'}",
        f"Commit: {record.commit}{' (dirty)' if record.dirty else ''}; tag {record.tag} at "
        f"{record.tag_commit[:12] or '(not found in this clone)'}",
        f"Platform: {record.platform.get('platform')}, Python {record.platform.get('python')}",
        "Versions: " + ", ".join(f"{k} {v}" for k, v in sorted(record.versions.items())),
        f"Generated: {record.generated}, {record.seconds / 60:.1f} minutes in all",
        "",
        "| Step | Result | Seconds | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for s in record.steps:
        detail = (s.detail or "").replace("|", "/")
        lines.append(
            f"| {s.name} | {'PASS' if s.passed else 'FAIL'} | {s.seconds:.0f} | {detail} |"
        )
    for name in record.missing_steps:
        lines.append(f"| {name} | NOT RUN | -- | |")
    if record.cells:
        lines += ["", "Regenerated sweep cells:", ""]
        lines += [f"- {describe(c)}" for c in record.cells]
    lines += [
        "",
        f"Send back {record_path} and the .md beside it (a pull request adding both under "
        "docs/, or by email to the Vega Lab team). If a step failed, send them anyway: the "
        "failure is the finding.",
    ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class GitFacts:
    commit: str
    dirty: bool
    tag_commit: str
    after_tag: bool = False  # HEAD is the tag's commit or a descendant of it


def git_facts(repo: Path, tag: str, ignore: Sequence[str] = ()) -> GitFacts:
    """HEAD, whether the tree has changes other than ``ignore``, and the tag's commit."""

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8", check=False
        )
        return done.stdout.strip() if done.returncode == 0 else ""

    status = [
        ln[3:]
        for ln in git("status", "--porcelain", "--untracked-files=all").splitlines()
        if len(ln) > 3
    ]
    commit = git("rev-parse", "HEAD")
    changed = [p for p in status if p not in ignore]
    tag_commit = git("rev-parse", f"{tag}^{{commit}}")
    after = bool(commit and tag_commit) and (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", tag_commit, commit],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    return GitFacts(commit or "unknown", bool(changed) or not commit, tag_commit, after)
