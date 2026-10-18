"""Sweep orchestration: a grid of RunConfigs on paired seeds, written as run directories.

Lane: Ilan Danial (machinery).

The M2 power curve has one axis, the planted premium in vol points (decision D2),
and one primary contrast per point, the fixed-delta seller against the trained
policy on the same seeds (D1). A sweep here is that grid: for every premium and
every arm one ``RunConfig``, differing from the base only in
``market.iv_premium_vol_pts`` and ``policy``. Every arm at every premium runs the
same episode seeds, so the arms are paired by construction and are never compared
across different draws of paths.

Each cell is written through ``write_run`` as an ordinary run directory named by
its run id, so a cell is reproducible from its config hash and seed alone and the
dashboard reads it with the same reader as any other run. A sweep resumes: a cell
whose ``run.json`` and manifest are present, self-consistent and keyed by the
expected hash is read back rather than recomputed; anything else is recomputed.

``pilot_matrix`` is the other half of the variance-components pilot: S policy
instances by N episodes, each seed block on its own disjoint path seeds, which is
the nested design the pre-registered formula describes. With a scripted policy the
between-seed component is zero by construction, the pilot's negative control; with
a learner, block i is the policy trained with seed i.

Every number produced here is synthetic until a real ``ChainSource`` exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from optspread.artifacts import RUN_FILENAME, RunRecord, SchemaVersionError, read_run, write_run
from optspread.config import PolicyName, RunConfig, config_hash
from optspread.harness import EpisodeResult, run_episodes, summarize
from optspread.manifest import MANIFEST_FILENAME, RunManifest, file_sha256
from optspread.policies import Policy

SCHEMA_VERSION = 1
SWEEP_FILENAME = "sweep.json"

# The one path from a config to episodes. Injected so a caller can count or wrap it;
# the default is the harness runner every gate uses.
Runner = Callable[[RunConfig, Policy | None], tuple[EpisodeResult, ...]]


class SweepConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base: RunConfig = Field(default_factory=RunConfig)
    premiums: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)  # D2 axis, in vol points
    arms: tuple[PolicyName, ...] = ("flat", "always_on", "delta_seller")

    @field_validator("premiums")
    @classmethod
    def _premiums_distinct(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("premiums must list at least one planted premium")
        if len(set(value)) != len(value):
            raise ValueError("premiums must be distinct")
        return value

    @field_validator("arms")
    @classmethod
    def _arms_distinct(cls, value: tuple[PolicyName, ...]) -> tuple[PolicyName, ...]:
        if not value:
            raise ValueError("arms must list at least one policy")
        if len(set(value)) != len(value):
            raise ValueError("arms must be distinct")
        return value


def cell_config(sweep: SweepConfig, premium: float, arm: PolicyName) -> RunConfig:
    """The base config with only the planted premium and the policy changed."""
    market = sweep.base.market.model_copy(update={"iv_premium_vol_pts": premium})
    return sweep.base.model_copy(update={"market": market, "policy": arm})


def run_id(config: RunConfig) -> str:
    """``<config hash>-s<seed>``: the id ``RunRecord`` and ``RunManifest`` give the same run."""
    return f"{config_hash(config)}-s{config.seed}"


class CellRecord(BaseModel):
    """One cell of the grid, as read back from its run directory."""

    model_config = ConfigDict(frozen=True)

    premium: float
    policy: PolicyName
    run_id: str
    config_hash: str
    n_episodes: int
    mean_pnl: float
    se_pnl: float
    mean_cost: float
    mean_trades: float

    @classmethod
    def of(cls, premium: float, policy: PolicyName, record: RunRecord) -> CellRecord:
        return cls(
            premium=premium,
            policy=policy,
            run_id=record.run_id,
            config_hash=record.config_hash,
            n_episodes=record.n_episodes,
            mean_pnl=record.mean_pnl,
            se_pnl=record.se_pnl,
            mean_cost=record.mean_cost,
            mean_trades=record.mean_trades,
        )


class SweepRecord(BaseModel):
    """The sweep's own record: which cells, and the headline of each."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    sweep_hash: str
    seed: int
    n_episodes: int
    premiums: tuple[float, ...]
    arms: tuple[PolicyName, ...]
    cells: tuple[CellRecord, ...]

    def cell(self, premium: float, arm: PolicyName) -> CellRecord:
        for cell in self.cells:
            if cell.premium == premium and cell.policy == arm:
                return cell
        raise KeyError(f"no cell for premium {premium} and arm {arm!r}")

    def series(self, arm: PolicyName) -> tuple[float, ...]:
        """Mean P&L of ``arm`` along the premium axis, in grid order."""
        return tuple(self.cell(premium, arm).mean_pnl for premium in self.premiums)

    def render(self) -> str:
        """One row per cell. ``lift`` is the mean against the first premium for the same arm:
        the arms share seeds, so it is a paired difference, not a difference of noisy means."""
        lines = [
            f"{'premium':>8}  {'arm':<14}{'n':>5}{'mean P&L':>12}{'se':>10}"
            f"{'lift':>10}{'trades':>8}"
        ]
        first = self.premiums[0]
        for cell in self.cells:
            lift = cell.mean_pnl - self.cell(first, cell.policy).mean_pnl
            lines.append(
                f"{cell.premium:>8.2f}  {cell.policy:<14}{cell.n_episodes:>5}"
                f"{cell.mean_pnl:>12,.0f}{cell.se_pnl:>10,.0f}{lift:>10,.0f}{cell.mean_trades:>8.1f}"
            )
        lines.append(
            f"sweep {self.sweep_hash}: seeds {self.seed}..{self.seed + self.n_episodes - 1}"
        )
        return "\n".join(lines)


def valid_run(cell_dir: Path, config: RunConfig) -> RunRecord | None:
    """The run already on disk for ``config``, or None when it has to be (re)computed.

    The resume rule every orchestrator here shares (``run_sweep`` and
    ``optspread.orchestrate``): present, readable, keyed by the expected hash and
    run id, and the results file still the one its manifest hashed.
    """
    if not (cell_dir / RUN_FILENAME).exists() or not (cell_dir / MANIFEST_FILENAME).exists():
        return None
    try:
        record = read_run(cell_dir)
        manifest = RunManifest.read(cell_dir)
    except (OSError, ValueError):  # unreadable, wrong schema, incomplete, or malformed JSON
        return None
    expected = config_hash(config)
    if record.config_hash != expected or manifest.config_hash != expected:
        return None
    if record.run_id != run_id(config) or manifest.run_id != run_id(config):
        return None
    if manifest.artifacts.get(RUN_FILENAME) != file_sha256(cell_dir / RUN_FILENAME):
        return None  # the results file changed under its manifest
    return record


def run_sweep(sweep: SweepConfig, root: Path, runner: Runner = run_episodes) -> SweepRecord:
    """Run every cell not already valid on disk and write ``sweep.json`` under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    cells: list[CellRecord] = []
    for premium in sweep.premiums:
        for arm in sweep.arms:
            config = cell_config(sweep, premium, arm)
            cell_dir = root / run_id(config)
            record = valid_run(cell_dir, config)
            if record is None:
                episodes = runner(config, None)
                write_run(cell_dir, config, summarize(episodes), episodes)
                record = read_run(cell_dir)
            cells.append(CellRecord.of(premium, arm, record))
    result = SweepRecord(
        sweep_hash=config_hash(sweep),
        seed=sweep.base.seed,
        n_episodes=sweep.base.n_episodes,
        premiums=sweep.premiums,
        arms=sweep.arms,
        cells=tuple(cells),
    )
    (root / SWEEP_FILENAME).write_text(result.model_dump_json(indent=2))
    return result


def read_sweep(root: Path) -> SweepRecord:
    record = SweepRecord.model_validate_json((root / SWEEP_FILENAME).read_text())
    if record.schema_version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"sweep.json is schema v{record.schema_version}, this build reads v{SCHEMA_VERSION}"
        )
    return record


# -- the pilot matrix ------------------------------------------------------- #


def pilot_matrix(
    base: RunConfig,
    n_seeds: int,
    n_episodes: int,
    make_policy: Callable[[int], Policy],
    runner: Runner = run_episodes,
) -> NDArray[np.float64]:
    """``(S, N)`` P&L matrix for the variance-components pilot.

    Block ``i`` is ``make_policy(i)`` (for a learner, the policy trained with seed
    ``i``) evaluated on path seeds ``base.seed + i * N`` through ``+ N - 1``: disjoint
    across blocks, so the within-block outcomes are independent of every other block,
    as the nested formula in ``optspread.power`` requires.
    """
    if n_seeds < 1 or n_episodes < 1:
        raise ValueError(
            f"the pilot needs at least one seed and one episode, got {n_seeds} x {n_episodes}"
        )
    rows: list[list[float]] = []
    for i in range(n_seeds):
        config = base.model_copy(
            update={"seed": base.seed + i * n_episodes, "n_episodes": n_episodes}
        )
        rows.append([episode.pnl for episode in runner(config, make_policy(i))])
    return np.asarray(rows, dtype=np.float64)


def pilot_seeds(base: RunConfig, n_seeds: int, n_episodes: int) -> Sequence[Sequence[int]]:
    """The path seeds each block of ``pilot_matrix`` runs on, for reports and audits."""
    return [
        range(base.seed + i * n_episodes, base.seed + (i + 1) * n_episodes) for i in range(n_seeds)
    ]
