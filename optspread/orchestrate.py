"""The pilot and the sweep as one grid: premium x arm x seed block, parallel and resumable.

Lane: Ilan Danial (machinery).

``optspread.sweep`` runs scripted arms on one shared seed range. A learned arm
adds the seed axis, because two training seeds are two different policies
(project document 4.1). A ``GridConfig`` is the design both the pilot and the full
sweep use:

    for every planted premium p, seed block i and arm a, one run on
        path seeds  base.seed + i*N  ..  base.seed + (i+1)*N - 1
        trained with seed  train.seed + i  (the ppo arm only)

So every arm at every premium sees the same paths in block i (paired, as D1 and
the pre-registration require), the blocks are disjoint (nested, as the formula in
``optspread.power`` requires), and block i's policy is the same training seed at
every premium. The pilot is this grid at one premium; ``GridRecord.matrix`` reads
its S x N outcomes back for ``scripts/pilot_budget.py``.

Every cell is an ordinary run directory under ``root``, named by its run id and
written by ``write_run``, so a cell is the same object whichever command produced
it. ``run_cells`` reads back every cell that ``sweep.valid_run`` accepts and
computes the rest, most expensive first, across ``workers`` processes. A killed
run loses only the cells in flight; the same command resumes it. One process per
root: two orchestrators writing the same cell at once are not guarded against.

Workers are spawned rather than forked, so a worker never inherits a parent's
torch or BLAS thread pools, and the start method is the same on Linux and macOS.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from optspread.artifacts import SchemaVersionError, read_run, write_run
from optspread.config import ArmName, CostConfig, RunConfig, TrainConfig, config_hash
from optspread.harness import run_episodes, summarize
from optspread.manifest import RunManifest
from optspread.sweep import run_id, valid_run
from optspread.train import POLICY_FILENAME, Trainer, check_disjoint, train_ppo

SCHEMA_VERSION = 1

# Decision D2's axis in vol points, the 4.3 levels plus the null world.
SWEEP_PREMIUMS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
# D1's primary contrast: the fixed-delta seller (the oracle) against the trained policy.
PRIMARY_ARMS: tuple[ArmName, ...] = ("delta_seller", "ppo")

# M1-measured rates, steps/s/core (README, Throughput). Only their ratio enters the ETA,
# which is rescaled by the measured wall clock as cells finish; the absolute values
# give the estimate printed before the first cell is done.
TRAIN_RATE = 1_000.0
EVAL_RATE = 1_800.0


class GridConfig(BaseModel):
    """premium x arm x seed block. ``base.seed`` is the first path seed and
    ``base.n_episodes`` the episodes per block; ``train`` is the template the ppo arm
    trains with, block ``i`` at ``train.seed + i``."""

    model_config = ConfigDict(frozen=True)

    base: RunConfig = Field(default_factory=lambda: RunConfig(cost=CostConfig(enabled=True)))
    premiums: tuple[float, ...] = SWEEP_PREMIUMS
    arms: tuple[ArmName, ...] = PRIMARY_ARMS
    n_seeds: int = Field(default=1, ge=1)
    train: TrainConfig | None = None

    @field_validator("premiums")
    @classmethod
    def _premiums_distinct(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("premiums must list at least one premium, each once")
        return value

    @field_validator("arms")
    @classmethod
    def _arms_distinct(cls, value: tuple[ArmName, ...]) -> tuple[ArmName, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("arms must list at least one arm, each once")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> GridConfig:
        if ("ppo" in self.arms) != (self.train is not None):
            raise ValueError("a grid with the ppo arm needs a TrainConfig, and only then")
        if self.train is not None and self.train.seed + self.n_seeds > 2**15:
            raise ValueError("training seeds must stay below 2**15")
        check_disjoint(
            self.base.model_copy(update={"n_episodes": self.n_seeds * self.base.n_episodes})
        )
        return self


@dataclass(frozen=True, slots=True)
class Cell:
    premium: float
    arm: ArmName
    block: int
    config: RunConfig

    @property
    def run_id(self) -> str:
        return run_id(self.config)

    @property
    def label(self) -> str:
        return f"p={self.premium:<5.2f} {self.arm:<13}block {self.block:>3}"


def block_config(grid: GridConfig, premium: float, arm: ArmName, block: int) -> RunConfig:
    """The base config with the premium, the arm, block ``block``'s seeds and training set."""
    base, n = grid.base, grid.base.n_episodes
    train = None
    if arm == "ppo" and grid.train is not None:
        train = grid.train.model_copy(update={"seed": grid.train.seed + block})
    return RunConfig(
        market=base.market.model_copy(update={"iv_premium_vol_pts": premium}),
        cost=base.cost,
        env=base.env,
        policy=arm,
        seed=base.seed + block * n,
        n_episodes=n,
        train=train,
    )


def grid_cells(grid: GridConfig) -> tuple[Cell, ...]:
    return tuple(
        Cell(premium, arm, block, block_config(grid, premium, arm, block))
        for premium in grid.premiums
        for block in range(grid.n_seeds)
        for arm in grid.arms
    )


def cell_seconds(config: RunConfig) -> float:
    """Core-seconds at the M1 rates. Training rounds up to whole rollouts, as SB3 does."""
    seconds = config.n_episodes * config.env.episode_days / EVAL_RATE
    if config.train is not None:
        n_steps = config.train.hyperparameters.n_steps
        steps = math.ceil(config.train.total_timesteps / n_steps) * n_steps
        seconds += steps / TRAIN_RATE
    return seconds


def format_seconds(seconds: float) -> str:
    s = round(seconds)
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s"


# -- one run ----------------------------------------------------------------- #


def execute_run(config: RunConfig, run_dir: Path, trainer: Trainer = train_ppo) -> None:
    """Train if the arm is learned, evaluate, and write the run directory.

    The trained policy is saved beside ``run.json`` and hashed into the manifest, so
    the expensive half of a PPO run is kept; the replay gate retrains from the config
    rather than trusting it.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    extra: list[Path] = []
    if config.train is None:
        episodes = run_episodes(config)
    else:
        check_disjoint(config)
        policy = trainer(config)
        policy.save(run_dir / POLICY_FILENAME)
        extra.append(run_dir / POLICY_FILENAME)
        episodes = run_episodes(config, policy)
    write_run(run_dir, config, summarize(episodes), episodes, extra=extra)


def _single_thread() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def _execute(task: tuple[RunConfig, Path, Trainer]) -> tuple[str, float, str]:
    config, run_dir, trainer = task
    start = time.perf_counter()
    try:
        execute_run(config, run_dir, trainer)
    except Exception as exc:  # reported per cell; the other cells keep running
        return run_id(config), time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return run_id(config), time.perf_counter() - start, ""


# -- many runs ---------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Progress:
    """One line per cell: reused from disk, computed, or failed."""

    done: int
    total: int
    cell: Cell
    seconds: float | None  # None when the cell was read back
    eta_seconds: float | None
    mean_pnl: float | None
    error: str = ""

    def render(self) -> str:
        width = len(str(self.total))
        head = f"[{self.done:>{width}}/{self.total}] {self.cell.label}  {self.cell.run_id}"
        if self.error:
            return f"{head}  FAILED after {format_seconds(self.seconds or 0.0)}: {self.error}"
        pnl = f"mean {self.mean_pnl:>10,.0f}" if self.mean_pnl is not None else ""
        if self.seconds is None:
            return f"{head}  {pnl}  reused"
        eta = f"  eta {format_seconds(self.eta_seconds)}" if self.eta_seconds is not None else ""
        return f"{head}  {pnl}  {self.seconds:>7.1f}s{eta}"


Reporter = Callable[[Progress], None]


class IncompleteGridError(RuntimeError):
    """Some cells failed. The rest are on disk; the same command resumes."""

    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        lines = "\n".join(f"  {k}: {v}" for k, v in sorted(failures.items()))
        super().__init__(f"{len(failures)} cell(s) failed; rerun to resume:\n{lines}")


@dataclass(frozen=True, slots=True)
class Plan:
    """What a run of ``run_cells`` will do before it does it."""

    total: int
    reused: int
    to_run: int
    core_seconds: float
    wall_seconds: float
    workers: int

    def render(self) -> str:
        return (
            f"{self.total} cells: {self.reused} on disk, {self.to_run} to run, "
            f"~{self.core_seconds / 3600:,.2f} core-hours at the M1 rates "
            f"({TRAIN_RATE:,.0f} steps/s training, {EVAL_RATE:,.0f} evaluation), "
            f"~{format_seconds(self.wall_seconds)} wall on {self.workers} worker(s)"
        )


def makespan(costs: Sequence[float], workers: int) -> float:
    """Wall clock of ``costs`` run longest first, each to the next free worker.

    The schedule ``run_cells`` follows, so the estimate respects that one cell
    cannot be split: eight PPO cells on sixteen workers take one cell's time."""
    loads = [0.0] * max(1, workers)
    for cost in sorted(costs, reverse=True):
        loads[loads.index(min(loads))] += cost
    return max(loads)


def plan_cells(cells: Sequence[Cell], root: Path, workers: int = 1) -> Plan:
    pending = [c for c in cells if valid_run(root / c.run_id, c.config) is None]
    costs = [cell_seconds(c.config) for c in pending]
    return Plan(
        total=len(cells),
        reused=len(cells) - len(pending),
        to_run=len(pending),
        core_seconds=sum(costs),
        wall_seconds=makespan(costs, workers),
        workers=workers,
    )


def run_cells(
    cells: Sequence[Cell],
    root: Path,
    *,
    workers: int = 1,
    trainer: Trainer = train_ppo,
    report: Reporter | None = None,
) -> None:
    """Compute every cell not valid on disk under ``root``; raise if any failed."""
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    root.mkdir(parents=True, exist_ok=True)
    total, done = len(cells), 0
    pending: list[Cell] = []
    for cell in cells:
        record = valid_run(root / cell.run_id, cell.config)
        if record is None:
            pending.append(cell)
            continue
        done += 1
        if report is not None:
            report(Progress(done, total, cell, None, None, record.mean_pnl))

    # Longest first, so the slow ppo cells do not end up as the tail of a parallel run.
    pending.sort(key=lambda c: cell_seconds(c.config), reverse=True)
    by_id = {c.run_id: c for c in pending}
    tasks = [(c.config, root / c.run_id, trainer) for c in pending]
    # The ETA is the plan's schedule over the cells not yet finished, rescaled by how
    # long finished cells took against their estimate. Cells in flight count in full,
    # so it errs long by at most one round.
    unfinished = [cell_seconds(c.config) for c in pending]
    predicted, measured = 0.0, 0.0
    failures: dict[str, str] = {}

    pool = None
    if workers > 1 and len(tasks) > 1:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(min(workers, len(tasks)), initializer=_single_thread)
    try:
        stream = map(_execute, tasks) if pool is None else pool.imap_unordered(_execute, tasks)
        for rid, seconds, error in stream:
            cell = by_id[rid]
            done += 1
            cost = cell_seconds(cell.config)
            unfinished.remove(cost)
            predicted += cost
            measured += seconds
            eta = makespan(unfinished, workers) * measured / predicted
            mean = None
            if error:
                failures[rid] = error
            else:
                mean = read_run(root / rid).mean_pnl
            if report is not None:
                report(Progress(done, total, cell, seconds, eta, mean, error))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if failures:
        raise IncompleteGridError(failures)


# -- the grid record ---------------------------------------------------------- #


class GridCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    premium: float
    arm: ArmName
    block: int
    run_id: str
    config_hash: str
    seed: int
    n_episodes: int
    train_seed: int | None
    mean_pnl: float
    se_pnl: float
    mean_cost: float
    mean_trades: float


class GridRecord(BaseModel):
    """Which run is which cell, and each cell's headline. The runs hold the episodes."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    grid_hash: str
    grid: GridConfig
    cells: tuple[GridCell, ...]

    def cell(self, premium: float, arm: ArmName, block: int) -> GridCell:
        for c in self.cells:
            if (c.premium, c.arm, c.block) == (premium, arm, block):
                return c
        raise KeyError(f"no cell for premium {premium}, arm {arm!r}, block {block}")

    def matrix(self, root: Path, premium: float, arm: ArmName) -> NDArray[np.float64]:
        """``(S, N)`` episode P&L at one premium for one arm: row i is block i, in seed order.

        The input ``optspread.power.variance_components`` and ``scripts/pilot_budget.py
        --matrix`` take."""
        rows = []
        for block in range(self.grid.n_seeds):
            record = read_run(root / self.cell(premium, arm, block).run_id)
            rows.append([e.pnl for e in record.episodes])
        return np.asarray(rows, dtype=np.float64)

    def render(self) -> str:
        """Per premium and arm: the grand mean over blocks and the se of the block means."""
        lines = [
            f"{'premium':>8}  {'arm':<14}{'S':>4}{'N':>6}{'grand mean':>13}{'se':>10}{'trades':>8}"
        ]
        for premium in self.grid.premiums:
            for arm in self.grid.arms:
                cells = [self.cell(premium, arm, b) for b in range(self.grid.n_seeds)]
                means = np.array([c.mean_pnl for c in cells])
                se = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else 0.0
                trades = float(np.mean([c.mean_trades for c in cells]))
                lines.append(
                    f"{premium:>8.2f}  {arm:<14}{len(cells):>4}{cells[0].n_episodes:>6}"
                    f"{float(means.mean()):>13,.0f}{se:>10,.0f}{trades:>8.1f}"
                )
        lines.append(f"grid {self.grid_hash}")
        return "\n".join(lines)


def grid_record(grid: GridConfig, root: Path) -> GridRecord:
    """Read every cell back from ``root``. Raises if any is missing or invalid."""
    cells = []
    for c in grid_cells(grid):
        record = valid_run(root / c.run_id, c.config)
        if record is None:
            raise FileNotFoundError(f"cell {c.label} ({c.run_id}) is not valid under {root}")
        cells.append(
            GridCell(
                premium=c.premium,
                arm=c.arm,
                block=c.block,
                run_id=c.run_id,
                config_hash=record.config_hash,
                seed=c.config.seed,
                n_episodes=record.n_episodes,
                train_seed=None if c.config.train is None else c.config.train.seed,
                mean_pnl=record.mean_pnl,
                se_pnl=record.se_pnl,
                mean_cost=record.mean_cost,
                mean_trades=record.mean_trades,
            )
        )
    return GridRecord(grid_hash=config_hash(grid), grid=grid, cells=tuple(cells))


def grid_filename(grid: GridConfig) -> str:
    return f"grid-{config_hash(grid)}.json"


def run_grid(
    grid: GridConfig,
    root: Path,
    *,
    workers: int = 1,
    trainer: Trainer = train_ppo,
    report: Reporter | None = None,
) -> GridRecord:
    """Run the grid under ``root`` and write ``grid-<hash>.json`` beside its cells."""
    run_cells(grid_cells(grid), root, workers=workers, trainer=trainer, report=report)
    record = grid_record(grid, root)
    (root / grid_filename(grid)).write_text(record.model_dump_json(indent=2))
    return record


def read_grid(path: Path) -> GridRecord:
    record = GridRecord.model_validate_json(path.read_text())
    if record.schema_version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path.name} is schema v{record.schema_version}, this build reads v{SCHEMA_VERSION}"
        )
    return record


# -- the pilot ------------------------------------------------------------------ #

PILOT_MATRIX = "matrix.npy"  # the learner: what scripts/pilot_budget.py --matrix reads
PILOT_ORACLE_MATRIX = "matrix_oracle.npy"  # the negative control, tau^2 = 0 by construction


def pilot_grid(
    base: RunConfig,
    n_seeds: int,
    n_episodes: int,
    premium: float,
    train: TrainConfig,
) -> GridConfig:
    """The variance-components pilot: one premium, the learner and its scripted control."""
    return GridConfig(
        base=base.model_copy(update={"n_episodes": n_episodes}),
        premiums=(premium,),
        arms=("ppo", "delta_seller"),
        n_seeds=n_seeds,
        train=train,
    )


def write_pilot_matrices(record: GridRecord, root: Path) -> dict[ArmName, Path]:
    """Save each arm's ``(S, N)`` matrix at the pilot's premium beside its cells.

    The matrices get a manifest of their own at ``root``, keyed by the learned arm's
    first block config and recording every matrix's digest, so the M2 report can
    refuse a matrix that changed after it was written.
    """
    premium = record.grid.premiums[0]
    names: dict[ArmName, str] = {"ppo": PILOT_MATRIX, "delta_seller": PILOT_ORACLE_MATRIX}
    paths: dict[ArmName, Path] = {}
    for arm in record.grid.arms:
        path = root / names.get(arm, f"matrix_{arm}.npy")
        np.save(path, record.matrix(root, premium, arm))
        paths[arm] = path
    lead: ArmName = "ppo" if "ppo" in record.grid.arms else record.grid.arms[0]
    manifest = RunManifest.create(block_config(record.grid, premium, lead, 0))
    for path in paths.values():
        manifest.add_artifact(root, path)
    manifest.write(root)
    return paths
