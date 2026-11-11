"""API payloads for sweeps and grids: the list, and one sweep's or grid's detail.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi``, which re-exports it.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from optspread.analysis import NULL_PREMIUM, decide_premium, paired_contrast
from optspread.api._base import Out, _f, _fl, _from_attrs, _inside
from optspread.artifacts import SchemaVersionError
from optspread.config import ArmName
from optspread.dashboard.jobs.spec import JOBS_DIR

# an orchestrated grid's record, as the run index finds it (re-exported for optspread.webapi)
from optspread.dashboard.runindex import GRID_PATTERN as GRID_PATTERN
from optspread.experiment.grid import GridRecord, read_grid
from optspread.experiment.sweep import SWEEP_FILENAME, SweepRecord, read_sweep
from optspread.manifest import MANIFEST_FILENAME, RunManifest

# -- sweeps ----------------------------------------------------------------------- #


# "sweep": a scripted sweep.json; "grid": an orchestrated grid-*.json. The two record
# kinds the summaries and the details are built for; nothing else sets it.
SweepKind = Literal["sweep", "grid"]


class SweepSummaryOut(Out):
    key: str
    sweep_hash: str
    premiums: list[float]
    arms: list[str]
    n_cells: int
    seed: int
    n_episodes: int
    kind: SweepKind = "sweep"
    n_seeds: int = 1  # seed blocks; a scripted sweep is one block on shared seeds
    n_present: int = 0  # cells whose run directory is under the root


class OraclePointOut(Out):
    # A scripted sweep's cell figures as its sweep.json records them: null where the
    # file holds a value that is not finite.
    premium: float
    mean_pnl: float | None
    se_pnl: float | None
    lift: float | None  # against the first premium for the same arm: a paired difference
    run_id: str


class GridCellOut(Out):
    premium: float
    arm: str
    block: int
    run_id: str
    key: str  # the cell's run key under the runs root, for /runs/?key=
    train_seed: int | None
    mean_pnl: float | None
    se_pnl: float | None
    mean_cost: float | None
    mean_trades: float | None
    present: bool


class ArmMeanOut(Out):
    """One arm at one premium over the seed blocks: the grand mean and the se of block means."""

    premium: float
    mean: float | None
    se: float | None
    n_blocks: int
    block_means: list[float | None]


class ContrastOut(Out):
    """The detection event read off the grid: the arm at p minus itself at 0, block by block.

    The pre-registered rule (``analysis.decide_premium``, A1 and A2) on the cells' episodes.
    """

    premium: float
    estimate: float | None
    se: float | None
    t: float | None
    df: int
    critical: float | None
    lower_bound: float | None
    detected: bool
    block_means: list[float | None]
    reason: str = ""  # why there is no decision, when there is none


class GridDetailOut(Out):
    grid_hash: str
    n_seeds: int
    n_episodes: int
    premiums: list[float]
    arms: list[str]
    total_timesteps: int | None
    first_path_seed: int
    costs: bool
    cells: list[GridCellOut]
    means: dict[str, list[ArmMeanOut]]
    contrasts: dict[str, list[ContrastOut]]
    missing: list[str]  # cells listed in the record whose run is not under the root
    git_shas: list[str]


class SweepOut(Out):
    summary: SweepSummaryOut
    oracle: dict[str, list[OraclePointOut]]
    source: str
    kind: SweepKind = "sweep"
    grid: GridDetailOut | None = None


class SweepListOut(Out):
    root: str
    sweeps: list[SweepSummaryOut]


def _sweep_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob(SWEEP_FILENAME)) if root.is_dir() else []


def _grid_files(root: Path) -> list[Path]:
    """Every orchestrated grid record under the root, the job queue's bookkeeping aside."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob(GRID_PATTERN) if JOBS_DIR not in p.parts and p.is_file())


def _sweep_summary(key: str, rec: SweepRecord, n_present: int = 0) -> SweepSummaryOut:
    """A scripted sweep's summary. The list passes ``n_present``; the detail leaves it at
    its default, as each did before they shared this."""
    return SweepSummaryOut(
        key=key,
        sweep_hash=rec.sweep_hash,
        premiums=list(rec.premiums),
        arms=list(rec.arms),
        n_cells=len(rec.cells),
        seed=rec.seed,
        n_episodes=rec.n_episodes,
        n_present=n_present,
    )


def _grid_summary(path: Path, root: Path, record: GridRecord) -> SweepSummaryOut:
    grid = record.grid
    present = sum((path.parent / c.run_id / MANIFEST_FILENAME).is_file() for c in record.cells)
    return SweepSummaryOut(
        key=path.relative_to(root).as_posix(),
        sweep_hash=record.grid_hash,
        premiums=sorted(grid.premiums),
        arms=list(grid.arms),
        n_cells=len(record.cells),
        seed=grid.base.seed,
        n_episodes=grid.base.n_episodes,
        kind="grid",
        n_seeds=grid.n_seeds,
        n_present=present,
    )


def sweeps(root: Path) -> SweepListOut:
    out: list[SweepSummaryOut] = []
    for d in _sweep_dirs(root):
        try:
            rec = read_sweep(d)
        except (OSError, ValueError, SchemaVersionError):
            continue
        out.append(_sweep_summary(d.relative_to(root).as_posix(), rec, n_present=len(rec.cells)))
    for path in _grid_files(root):
        try:
            record = read_grid(path)
        except (OSError, ValueError, SchemaVersionError):
            continue
        out.append(_grid_summary(path, root, record))
    return SweepListOut(root=str(root), sweeps=out)


def _arm_mean(premium: float, cells: Sequence[GridCellOut]) -> ArmMeanOut:
    means = np.asarray([c.mean_pnl for c in cells if c.mean_pnl is not None], dtype=np.float64)
    n = len(means)
    return ArmMeanOut(
        premium=premium,
        mean=_f(float(means.mean())) if n else None,
        se=_f(float(means.std(ddof=1) / math.sqrt(n))) if n > 1 else None,
        n_blocks=n,
        block_means=[c.mean_pnl for c in cells],
    )


def _contrast(root: Path, record: GridRecord, premium: float, arm: ArmName) -> ContrastOut:
    try:
        treated = record.matrix(root, premium, arm)
        null = record.matrix(root, NULL_PREMIUM, arm)
        decision = decide_premium(treated, null)
        block = paired_contrast(treated, null).mean(axis=1)
    except (OSError, ValueError, KeyError, SchemaVersionError) as exc:
        return ContrastOut(
            premium=premium,
            estimate=None,
            se=None,
            t=None,
            df=record.grid.n_seeds - 1,
            critical=None,
            lower_bound=None,
            detected=False,
            block_means=[],
            reason=str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
    return ContrastOut(
        premium=premium,
        estimate=_f(decision.estimate),
        se=_f(decision.se),
        t=_f(decision.t_stat),
        df=decision.df,
        critical=_f(decision.critical),
        lower_bound=_f(decision.lower_bound),
        detected=decision.rejected,
        block_means=_fl(block),
    )


def grid_detail(path: Path, root: Path) -> SweepOut:
    """An orchestrated grid: per-arm means by premium with the se of block means, every
    cell, and the pre-registered contrast at each premium above zero."""
    record = read_grid(path)
    grid = record.grid
    base_dir = path.parent
    cells: list[GridCellOut] = []
    missing: list[str] = []
    shas: set[str] = set()
    for c in sorted(record.cells, key=lambda c: (c.arm, c.premium, c.block)):
        run_dir = base_dir / c.run_id
        present = (run_dir / MANIFEST_FILENAME).is_file()
        if present:
            with contextlib.suppress(OSError, ValueError):
                shas.add(RunManifest.read(run_dir).git_sha)
        else:
            missing.append(f"{c.arm} at {c.premium:g}, block {c.block}: {c.run_id}")
        cells.append(
            GridCellOut(
                premium=c.premium,
                arm=c.arm,
                block=c.block,
                run_id=c.run_id,
                key=run_dir.relative_to(root).as_posix(),
                train_seed=c.train_seed,
                mean_pnl=_f(c.mean_pnl),
                se_pnl=_f(c.se_pnl),
                mean_cost=_f(c.mean_cost),
                mean_trades=_f(c.mean_trades),
                present=present,
            )
        )
    premiums = sorted(grid.premiums)
    means = {
        arm: [_arm_mean(p, [c for c in cells if c.arm == arm and c.premium == p]) for p in premiums]
        for arm in grid.arms
    }
    contrasts: dict[str, list[ContrastOut]] = {}
    if NULL_PREMIUM in grid.premiums and grid.n_seeds >= 2:
        contrasts = {
            arm: [_contrast(base_dir, record, p, arm) for p in premiums if p > NULL_PREMIUM]
            for arm in grid.arms
        }
    oracle = {
        arm: [
            OraclePointOut(
                premium=m.premium,
                mean_pnl=m.mean or 0.0,
                se_pnl=m.se or 0.0,
                lift=(m.mean or 0.0) - (pts[0].mean or 0.0),
                run_id=record.grid_hash,
            )
            for m in pts
            if m.mean is not None
        ]
        for arm, pts in means.items()
    }
    return SweepOut(
        summary=_grid_summary(path, root, record),
        oracle=oracle,
        source=str(path),
        kind="grid",
        grid=GridDetailOut(
            grid_hash=record.grid_hash,
            n_seeds=grid.n_seeds,
            n_episodes=grid.base.n_episodes,
            premiums=premiums,
            arms=list(grid.arms),
            total_timesteps=None if grid.train is None else grid.train.total_timesteps,
            first_path_seed=grid.base.seed,
            costs=grid.base.cost.enabled,
            cells=cells,
            means=means,
            contrasts=contrasts,
            missing=missing,
            git_shas=sorted(shas),
        ),
    )


def sweep_detail(root: Path, key: str) -> SweepOut:
    target = _inside(root, key)
    if target is not None and target.is_file() and target.match(GRID_PATTERN):
        try:
            return grid_detail(target, root)
        except (OSError, ValueError, SchemaVersionError) as exc:
            raise KeyError(key) from exc
    if target is None or not (target / SWEEP_FILENAME).is_file():
        raise KeyError(key)
    rec = read_sweep(target)
    oracle: dict[str, list[OraclePointOut]] = {}
    for arm in rec.arms:
        base = rec.cell(rec.premiums[0], arm).mean_pnl
        oracle[arm] = [
            _from_attrs(OraclePointOut, c, lift=c.mean_pnl - base)
            for c in (rec.cell(p, arm) for p in rec.premiums)
        ]
    return SweepOut(
        summary=_sweep_summary(key, rec),
        oracle=oracle,
        source=str(target / SWEEP_FILENAME),
    )
