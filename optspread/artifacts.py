"""The run artifact schema: the fixed output format every consumer reads.

Lane: Ilan Danial (machinery).

A run's outputs are an interface, not an implementation detail. The market layer
and the statistical code may change freely inside as long as they keep writing
this shape, which is what lets the dashboard and the experiments move in
parallel instead of in sequence.

Two files per run directory:

    manifest.json   provenance (``optspread.manifest``): what produced the run
    run.json        results (``RunRecord``): what the run produced

``SCHEMA_VERSION`` is written into every record and checked on read. Old runs
stay readable and are clearly labelled; a reader that meets a version it does
not know says so instead of silently misreading fields.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from optspread.config import RunConfig, config_hash
from optspread.harness import EpisodeResult, EvalSummary
from optspread.manifest import RunManifest

SCHEMA_VERSION = 1
RUN_FILENAME = "run.json"


class EpisodeRow(BaseModel):
    """One episode. The grain of every downstream table: one row per seed."""

    model_config = ConfigDict(frozen=True)

    seed: int
    pnl: float
    cost: float
    n_trades: int
    steps: int

    @classmethod
    def of(cls, result: EpisodeResult) -> EpisodeRow:
        return cls(
            seed=result.seed,
            pnl=result.pnl,
            cost=result.cost,
            n_trades=result.n_trades,
            steps=result.steps,
        )


class RunRecord(BaseModel):
    """The results half of a run directory. Keyed by ``run_id``, same as the manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    run_id: str
    config_hash: str
    policy: str
    seed: int
    n_episodes: int
    mean_pnl: float
    se_pnl: float
    t_stat: float
    mean_cost: float
    mean_trades: float
    episodes: tuple[EpisodeRow, ...] = Field(default_factory=tuple)

    @classmethod
    def of(
        cls, config: RunConfig, summary: EvalSummary, episodes: tuple[EpisodeResult, ...]
    ) -> RunRecord:
        h = config_hash(config)
        return cls(
            run_id=f"{h}-s{config.seed}",
            config_hash=h,
            policy=config.policy,
            seed=config.seed,
            n_episodes=summary.n,
            mean_pnl=summary.mean_pnl,
            se_pnl=summary.se_pnl,
            t_stat=summary.t_stat,
            mean_cost=summary.mean_cost,
            mean_trades=summary.mean_trades,
            episodes=tuple(EpisodeRow.of(e) for e in episodes),
        )


class SchemaVersionError(ValueError):
    """A record written by a schema version this build cannot read."""


def write_run(
    run_dir: Path,
    config: RunConfig,
    summary: EvalSummary,
    episodes: tuple[EpisodeResult, ...],
) -> RunManifest:
    """Write ``run.json`` then ``manifest.json`` with the results file hashed into it.

    Order matters: the manifest records a digest of the results, so the results
    must exist first. The returned manifest is the one on disk.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    record = RunRecord.of(config, summary, episodes)
    target = run_dir / RUN_FILENAME
    target.write_text(record.model_dump_json(indent=2))

    manifest = RunManifest.create(config)
    manifest.add_artifact(run_dir, target)
    manifest.results = {
        "mean_pnl": summary.mean_pnl,
        "se_pnl": summary.se_pnl,
        "t_stat": summary.t_stat,
        "mean_cost": summary.mean_cost,
        "mean_trades": summary.mean_trades,
    }
    manifest.write(run_dir)
    return manifest


def read_run(run_dir: Path) -> RunRecord:
    record = RunRecord.model_validate_json((run_dir / RUN_FILENAME).read_text())
    if record.schema_version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"run.json is schema v{record.schema_version}, this build reads v{SCHEMA_VERSION}"
        )
    return record
