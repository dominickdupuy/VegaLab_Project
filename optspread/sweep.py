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

A sweep runs on the generator unless told otherwise (``source``), and every cell's
manifest records the source it ran on, and on the real source the split and the panel.
A cell made with components of one's own (a runner other than the harness) records them
(``components_identity``), and every reuse check here compares the whole record with the
one the run asked for would write (``expected_data``): the source, the split, the
components, and on the real panel its digests, so a cell read from a panel since rebuilt
is recomputed rather than mixed into a grid with cells of the new build.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from optspread.artifacts import RUN_FILENAME, RunRecord, SchemaVersionError, read_run, write_run
from optspread.config import PolicyName, RunConfig, config_hash
from optspread.data import ChainSource, DataUnavailableError, SourceName, make_source
from optspread.extend import describe
from optspread.harness import EpisodeResult, run_episodes, summarize
from optspread.manifest import (
    MANIFEST_FILENAME,
    DataRecord,
    RunManifest,
    callable_name,
    file_sha256,
)
from optspread.policies import Policy
from optspread.train import POLICY_FILENAME, train_ppo

SCHEMA_VERSION = 1
SWEEP_FILENAME = "sweep.json"

# What the real learner records as a learned run's trainer (as ``optspread.m2gate``
# names it), and a member of every archive Stable-Baselines3 saves, which no stand-in
# writes: how a manifest from before the trainer was recorded is told apart (#31).
REAL_TRAINER = callable_name(train_ppo)
SB3_ARCHIVE_MEMBER = "_stable_baselines3_version"

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


def changed_artifacts(run_dir: Path, manifest: RunManifest) -> tuple[str, ...]:
    """Recorded artifacts that are missing or no longer match their digest, in order.

    ``run.json`` counts as changed when the manifest does not record it at all: a
    manifest that hashes the policy but not the results vouches for nothing shown.
    """
    recorded = dict(manifest.artifacts)
    recorded.setdefault(RUN_FILENAME, "")
    return tuple(
        rel
        for rel, digest in sorted(recorded.items())
        if not (run_dir / rel).is_file() or file_sha256(run_dir / rel) != digest
    )


def recorded_trainer(cell_dir: Path, manifest: RunManifest) -> str | None:
    """The trainer a learned run's manifest names, or for one written before the trainer
    was recorded, ``REAL_TRAINER`` when its hashed policy is a Stable-Baselines3 archive
    (the rule of ``optspread.m2gate.not_real_learner``); None when nothing tells."""
    if manifest.trainer is not None:
        return manifest.trainer
    policy = cell_dir / POLICY_FILENAME
    recorded = manifest.artifacts.get(POLICY_FILENAME)
    if recorded is None or not policy.is_file() or recorded != file_sha256(policy):
        return None
    try:
        with zipfile.ZipFile(policy) as archive:
            return REAL_TRAINER if SB3_ARCHIVE_MEMBER in archive.namelist() else None
    except (OSError, zipfile.BadZipFile):
        return None


_NO_IDENTITY = object()


def components_identity(runner: Runner) -> str | None:
    """What a run made by ``runner`` records as its components (``DataRecord.components``).

    None for the harness, ``run_episodes`` itself or bound to a source, and for the
    default ``Components()``; a ``Components``' ``identity`` otherwise; and for any other
    runner its name (``extend.describe``), since nothing says it runs the default world.
    """
    if runner is run_episodes:
        return None
    if (
        isinstance(runner, partial)
        and runner.func is run_episodes
        and not runner.args
        and set(runner.keywords) <= {"source"}
    ):
        return None
    identity: Any = getattr(runner, "identity", _NO_IDENTITY)
    if identity is None or isinstance(identity, str):
        return identity
    return describe(runner)


def expected_data(
    runner: Runner, config: RunConfig, source: SourceName = "synthetic"
) -> DataRecord:
    """The data record a run of ``config`` by ``runner`` on ``source`` writes, as it stands
    now: on the real panel with the panel's digests (raises ``DataUnavailableError``
    when it is absent). A runner of one's own records its components, and a
    ``Components`` the source it builds (its panel too, when that is the real one)."""
    components = components_identity(runner)
    if components is None:
        return DataRecord.of(make_source(source, config.market, split="eval"))
    factory = getattr(runner, "source", None)
    if callable(factory):
        chains = factory(config.market)
        if isinstance(chains, ChainSource):
            return DataRecord.of(chains, components)
    return DataRecord(components=components)


def _components(name: str | None) -> str:
    return name if name is not None else "the default components"


def source_mismatch(manifest: RunManifest, source: SourceName) -> str:
    """Why ``manifest`` is not a default run on ``source``'s ``eval`` split; "" when it is.

    The source and the split are not in the config hash, so the same run id on other data,
    or on other dates of the same data, is a different run: the harness evaluates on
    ``eval`` under A8's ``SPLITS``, so a record of another split or of ranges of its own
    is not its run. Nor is a run made with components of one's own. A manifest from before
    the source was recorded ran on the generator, and a real-data one from before the split
    was recorded ran on ``eval``. The panel is not compared here (``data_mismatch``).
    """
    data = manifest.data_record
    if data.components is not None:
        return f"{manifest.run_id} was made with {data.components}, not the default components"
    if data.source != source:
        return f"{manifest.run_id} ran on {data.source}, not {source}"
    want = "eval" if source == "optionmetrics" else None
    if data.effective_split != want:
        return f"{manifest.run_id} ran under the split {data.effective_split}, not {want}"
    if data.ranges is not None:
        return f"{manifest.run_id} ran on dates of its own ({data.ranges}), not A8's splits"
    return ""


def data_mismatch(manifest: RunManifest, want: DataRecord, *, panel: bool = True) -> str:
    """Why ``manifest``'s data is not ``want`` (``expected_data``); "" when it is.

    The components, the source, the split and the dates, and with ``panel`` the panel's
    digests: a real-data run read from a panel since rebuilt is a run of other data, as
    the replay says too. A record from before the split was recorded ran on ``eval``.
    """
    data = manifest.data_record
    if data.components != want.components:
        return (
            f"{manifest.run_id} was made with {_components(data.components)}, "
            f"not {_components(want.components)}"
        )
    if data.source != want.source:
        return f"{manifest.run_id} ran on {data.source}, not {want.source}"
    if data.effective_split != want.effective_split:
        return (
            f"{manifest.run_id} ran under the split {data.effective_split}, "
            f"not {want.effective_split}"
        )
    if data.ranges != want.ranges:
        return f"{manifest.run_id} ran on dates {data.ranges}, not {want.ranges}"
    if panel and data.panel != want.panel:
        return (
            f"{manifest.run_id} read the {data.source} panel {data.panel_id or '(none)'}, "
            f"the panel now is {want.panel_id or '(none)'}"
        )
    return ""


def foreign_run(
    run_dir: Path, config: RunConfig, source: SourceName, runner: Runner = run_episodes
) -> str:
    """Why ``run_dir`` must not be written for ``config`` on ``source``; "" when it may.

    It may not when it already holds this config's run on another source or split, or
    made with other components: the run id is the same, and writing over it would
    destroy a finished run of other data or another world. A run of the same data on a
    panel since rebuilt may be written over: it is the cell the resume recomputes.
    """
    try:
        manifest = RunManifest.read(run_dir)
    except (OSError, ValueError):
        return ""
    if manifest.run_id != run_id(config):
        return ""
    components = components_identity(runner)
    if components is None:
        why = source_mismatch(manifest, source)
    else:
        try:
            why = data_mismatch(manifest, expected_data(runner, config, source), panel=False)
        except DataUnavailableError:
            why = ""
    if not why:
        return ""
    return (
        f"{run_dir} already holds {why}; runs on another source, or made with other "
        "components, go under their own root (the default is <runs root>/<source>), so it "
        "is not overwritten"
    )


def valid_run(
    cell_dir: Path,
    config: RunConfig,
    source: SourceName = "synthetic",
    *,
    trainer: str | None = REAL_TRAINER,
    runner: Runner = run_episodes,
) -> RunRecord | None:
    """The run already on disk for ``config``, or None when it has to be (re)computed.

    The resume rule every orchestrator here shares (``run_sweep`` and
    ``optspread.orchestrate``): present, readable, keyed by the expected hash and
    run id, its data the record a run by ``runner`` on ``source`` writes now
    (``expected_data``, ``data_mismatch``: the source, on the real source its ``eval``
    split and the panel's digests, and the components, so a run made with components of
    one's own is reused only by the same components), and every file its manifest hashed
    (the results, and the policy of a learned run) still the one it hashed. A learned
    run must also have been trained by ``trainer`` (a ``callable_name``; the real
    learner unless told otherwise, None to accept any): a stand-in's cell carries the
    same run id as the real learner's and is never reused for it (#31). A learned run
    that records no trainer and whose policy is not a Stable-Baselines3 archive is
    reused by no one asking for ``REAL_TRAINER``. The commit a run was written at is not
    compared: the freeze keeps the numbers of every commit after the cutoff the same,
    and the planner reports the commits it reuses (``orchestrate.Plan``).
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
    try:
        want = expected_data(runner, config, source)
    except DataUnavailableError:  # the panel it would be checked against is not here
        return None
    if data_mismatch(manifest, want):
        return None
    if changed_artifacts(cell_dir, manifest):
        return None  # a recorded file changed under its manifest
    if trainer is not None and config.train is not None:
        by = recorded_trainer(cell_dir, manifest)
        if by != trainer and (by is not None or trainer == REAL_TRAINER):
            return None
    return record


def run_sweep(
    sweep: SweepConfig,
    root: Path,
    runner: Runner = run_episodes,
    *,
    source: SourceName = "synthetic",
) -> SweepRecord:
    """Run every cell not already valid on disk and write ``sweep.json`` under ``root``.

    On another ``source`` the default runner runs there, on its ``eval`` split; a runner
    of your own brings its own data and is refused with one. A cell directory holding
    this config's run on other data, or made with other components, is refused rather
    than written over (``foreign_run``).
    """
    if source != "synthetic":
        if runner is not run_episodes:
            raise ValueError("a runner of your own brings its own data; source is for the default")
        runner = partial(run_episodes, source=source)
    root.mkdir(parents=True, exist_ok=True)
    cells: list[CellRecord] = []
    for premium in sweep.premiums:
        for arm in sweep.arms:
            config = cell_config(sweep, premium, arm)
            cell_dir = root / run_id(config)
            record = valid_run(cell_dir, config, source, runner=runner)
            if record is None:
                if why := foreign_run(cell_dir, config, source, runner):
                    raise FileExistsError(why)
                data = expected_data(runner, config, source)
                episodes = runner(config, None)
                write_run(cell_dir, config, summarize(episodes), episodes, source=source, data=data)
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
    (root / SWEEP_FILENAME).write_text(
        result.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    return result


def read_sweep(root: Path) -> SweepRecord:
    record = SweepRecord.model_validate_json((root / SWEEP_FILENAME).read_text(encoding="utf-8"))
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
