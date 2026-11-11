"""The pilot and the sweep as one grid: premium x arm x seed block, and its record.

Lane: Ilan Danial (machinery).

``optspread.experiment.sweep`` runs scripted arms on one shared seed range. A learned arm
adds the seed axis, because two training seeds are two different policies
(project document 4.1). A ``GridConfig`` is the design both the pilot and the full
sweep use:

    for every planted premium p, seed block i and arm a, one run on
        path seeds  base.seed + i*N  ..  base.seed + (i+1)*N - 1
        trained with seed  train.seed + i  (the ppo arm only)

So every arm at every premium sees the same paths in block i (paired, as D1 and
the pre-registration require), the blocks are disjoint (nested, as the formula in
``optspread.power`` requires), and block i's policy is the same training seed at
every premium. That last is what the detection event (#26) needs: each arm at
premium p against itself at zero, block by block, which ``GridRecord.contrast``
reads back. The pilot is this grid at zero and one premium; its S x N contrast
matrices are what ``scripts/pilot_budget.py`` reads.

This module is the design and its record: the cells and their configs, when a cell on
disk is reused (``reusable``), and the grid read back from disk.
``optspread.experiment.schedule`` computes the cells and ``optspread.experiment.pilot`` is
the pilot's grid. Moved from ``optspread.orchestrate``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from optspread.analysis import NULL_PREMIUM, paired_contrast
from optspread.artifacts import RunRecord, read_run

# Used here before read_grid's check moved to sweep.check_schema; still importable from here.
from optspread.artifacts import SchemaVersionError as SchemaVersionError
from optspread.config import ArmName, CostConfig, RunConfig, TrainConfig, config_hash
from optspread.data import SourceName
from optspread.experiment.sweep import block_seed, check_schema, distinct, premium_market
from optspread.harness import run_episodes
from optspread.manifest import RunManifest, callable_name
from optspread.runs.rundir import Runner, run_id, valid_run
from optspread.train import Trainer, check_disjoint, train_ppo

SCHEMA_VERSION = 1

# Decision D2's axis in vol points, the 4.3 levels plus the null world.
SWEEP_PREMIUMS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
# The arms of the primary analysis: the fixed-delta seller (the oracle) and the trained
# policy. Each is tested against itself at zero premium (#26), so each has its own power
# curve and the oracle-to-PPO gap is the horizontal distance between them.
PRIMARY_ARMS: tuple[ArmName, ...] = ("delta_seller", "ppo")


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
        message = "premiums must list at least one premium, each once"
        return distinct(value, message, message)

    @field_validator("arms")
    @classmethod
    def _arms_distinct(cls, value: tuple[ArmName, ...]) -> tuple[ArmName, ...]:
        message = "arms must list at least one arm, each once"
        return distinct(value, message, message)

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
        market=premium_market(base, premium),
        cost=base.cost,
        env=base.env,
        policy=arm,
        seed=block_seed(base.seed, block, n),
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


def reusable(
    cell: Cell,
    root: Path,
    *,
    trainer: Trainer | None = train_ppo,
    source: SourceName = "synthetic",
    runner: Runner = run_episodes,
) -> RunRecord | None:
    """The cell's run on disk when it can be reused: ``sweep.valid_run`` on ``source`` with
    ``runner``'s components, and for a learned cell trained by ``trainer`` (the real
    learner unless told otherwise; None accepts any)."""
    name = callable_name(trainer) if trainer is not None else None
    return valid_run(root / cell.run_id, cell.config, source, trainer=name, runner=runner)


def _commits(root: Path, cells: Sequence[Cell]) -> list[str]:
    """The commit each of ``cells``' manifests under ``root`` was written at."""
    return [RunManifest.read(root / c.run_id).git_sha for c in cells]


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
    # The commits the cells were written at, sorted; empty (and no key) in a record from
    # before they were listed. More than one is reported, not refused (``valid_run``).
    commits: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_absent_commits(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        out: dict[str, Any] = handler(self)
        if not self.commits:
            out.pop("commits", None)
        return out

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

    def contrast(self, root: Path, premium: float, arm: ArmName) -> NDArray[np.float64]:
        """The detection event's ``(S, N)`` matrix: ``arm`` at ``premium`` minus ``arm`` at
        zero, block by block on the same paths (#26). Identically zero at zero itself."""
        if NULL_PREMIUM not in self.grid.premiums:
            raise KeyError(f"the grid has no {NULL_PREMIUM:g} premium to contrast against")
        return paired_contrast(
            self.matrix(root, premium, arm), self.matrix(root, NULL_PREMIUM, arm)
        )

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
        if len(self.commits) > 1:
            lines.append("cells written at commits " + ", ".join(sha[:8] for sha in self.commits))
        return "\n".join(lines)


def grid_record(
    grid: GridConfig,
    root: Path,
    *,
    trainer: Trainer | None = train_ppo,
    source: SourceName = "synthetic",
    runner: Runner = run_episodes,
) -> GridRecord:
    """Read every cell back from ``root``. Raises if any is missing or invalid: not on
    ``source``, made with components other than ``runner``'s, or a learned cell trained by
    another than ``trainer`` (the real learner unless told otherwise; None accepts any).
    The record lists the commits the cells were written at."""
    cells = []
    for c in grid_cells(grid):
        record = reusable(c, root, trainer=trainer, source=source, runner=runner)
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
    commits = tuple(sorted(set(_commits(root, grid_cells(grid)))))
    return GridRecord(grid_hash=config_hash(grid), grid=grid, cells=tuple(cells), commits=commits)


def grid_filename(grid: GridConfig) -> str:
    return f"grid-{config_hash(grid)}.json"


def read_grid(path: Path) -> GridRecord:
    record = GridRecord.model_validate_json(path.read_text(encoding="utf-8"))
    check_schema(path.name, record.schema_version, SCHEMA_VERSION)
    return record
