"""Ilan Danial: the run artifact schema is a contract, so it is tested as one.

A run directory is readable, round-trips, carries its schema version, and hashes
its own results file into the manifest.
"""

from pathlib import Path

import pytest

from optspread.artifacts import (
    RUN_FILENAME,
    SCHEMA_VERSION,
    RunRecord,
    SchemaVersionError,
    read_run,
    write_run,
)
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, config_hash
from optspread.harness import run_episodes, summarize
from optspread.manifest import MANIFEST_FILENAME, RunManifest

SMALL = RunConfig(
    policy="always_on",
    seed=7,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)


@pytest.fixture
def written(tmp_path: Path) -> Path:
    episodes = run_episodes(SMALL)
    write_run(tmp_path, SMALL, summarize(episodes), episodes)
    return tmp_path


def test_run_directory_has_both_halves(written: Path) -> None:
    assert (written / RUN_FILENAME).exists()
    assert (written / MANIFEST_FILENAME).exists()


def test_record_round_trips_every_episode(written: Path) -> None:
    record = read_run(written)
    assert record.schema_version == SCHEMA_VERSION
    assert record.n_episodes == SMALL.n_episodes
    assert [row.seed for row in record.episodes] == [7, 8, 9]
    assert record.config_hash == config_hash(SMALL)
    assert record.run_id == f"{config_hash(SMALL)}-s7"


def test_summary_fields_agree_with_the_episode_rows(written: Path) -> None:
    record = read_run(written)
    assert record.mean_pnl == pytest.approx(
        sum(row.pnl for row in record.episodes) / len(record.episodes)
    )
    assert record.mean_trades == pytest.approx(
        sum(row.n_trades for row in record.episodes) / len(record.episodes)
    )


def test_manifest_hashes_the_results_file(written: Path) -> None:
    manifest = RunManifest.read(written)
    assert RUN_FILENAME in manifest.artifacts
    assert manifest.results["mean_pnl"] == pytest.approx(read_run(written).mean_pnl)
    assert manifest.config_hash == config_hash(SMALL)


def test_an_unknown_schema_version_is_refused_not_misread(written: Path) -> None:
    record = read_run(written)
    bumped = record.model_copy(update={"schema_version": SCHEMA_VERSION + 1})
    (written / RUN_FILENAME).write_text(bumped.model_dump_json())
    with pytest.raises(SchemaVersionError):
        read_run(written)


def test_the_record_is_frozen() -> None:
    record = RunRecord.of(SMALL, summarize(run_episodes(SMALL)), run_episodes(SMALL))
    with pytest.raises(Exception, match=r"frozen"):
        record.mean_pnl = 1.0  # type: ignore[misc]
