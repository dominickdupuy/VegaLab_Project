"""Ilan Danial: config hashing, manifest round-trip, and the manifest gate.

The gate is the point of the second half: a run missing anything a replay needs
never reaches the results folder, which is a rule the tooling enforces rather
than one the team remembers.
"""

import json
from pathlib import Path

import pytest

from optspread.config import CostConfig, MarketConfig, RunConfig, config_hash
from optspread.manifest import (
    MANIFEST_FILENAME,
    REQUIRED_FIELDS,
    REQUIRED_VERSIONS,
    SCHEMA_VERSION,
    IncompleteManifestError,
    RunManifest,
    file_sha256,
)


def test_hash_is_stable_and_content_addressed() -> None:
    a = RunConfig()
    b = RunConfig(market=MarketConfig(), cost=CostConfig())  # same content, built differently
    c = RunConfig(seed=1)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 16


def test_manifest_round_trip(tmp_path: Path) -> None:
    cfg = RunConfig(policy="always_on", seed=7)
    m = RunManifest.create(cfg)
    artifact = tmp_path / "returns.npy"
    artifact.write_bytes(b"\x00\x01")
    m.add_artifact(tmp_path, artifact)
    m.results["mean_pnl"] = -12.5
    path = m.write(tmp_path)

    back = RunManifest.read(tmp_path)
    assert path.exists()
    assert back == m
    assert back.run_id == f"{config_hash(cfg)}-s7"
    assert back.artifacts["returns.npy"] == file_sha256(artifact)
    assert back.config.policy == "always_on"


# -- the manifest gate: a run nobody can reproduce cannot be written -------- #


def _complete(tmp_path: Path) -> RunManifest:
    m = RunManifest.create(RunConfig(policy="random", seed=3))
    if m.git_sha == "unknown":  # pragma: no cover - only outside a git checkout
        m.git_sha = "0" * 40
    return m


def test_a_complete_manifest_validates_and_writes(tmp_path: Path) -> None:
    m = _complete(tmp_path)
    m.validate_complete()
    assert m.write(tmp_path).exists()
    assert m.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_field_blocks_the_write(tmp_path: Path, field: str) -> None:
    m = _complete(tmp_path)
    setattr(m, field, "" if field != "versions" else {})
    with pytest.raises(IncompleteManifestError, match=field):
        m.write(tmp_path)
    assert not (tmp_path / MANIFEST_FILENAME).exists(), "nothing may be written on a failure"


@pytest.mark.parametrize("library", REQUIRED_VERSIONS)
def test_a_missing_library_version_blocks_the_write(tmp_path: Path, library: str) -> None:
    m = _complete(tmp_path)
    m.versions = {k: v for k, v in m.versions.items() if k != library}
    with pytest.raises(IncompleteManifestError, match=library):
        m.write(tmp_path)


def test_a_hash_that_drifted_from_its_config_is_caught(tmp_path: Path) -> None:
    # The quiet failure: two runs that look alike but differ in one setting. If the
    # hash and the config disagree, the record identifies neither.
    m = _complete(tmp_path)
    m.config_hash = config_hash(RunConfig(seed=999))
    with pytest.raises(IncompleteManifestError, match="drifted apart"):
        m.write(tmp_path)


def test_unidentified_code_blocks_the_write(tmp_path: Path) -> None:
    m = _complete(tmp_path)
    m.git_sha = "unknown"
    with pytest.raises(IncompleteManifestError, match="git_sha"):
        m.write(tmp_path)


def test_reading_an_incomplete_manifest_raises_rather_than_returning_it(tmp_path: Path) -> None:
    m = _complete(tmp_path)
    m.write(tmp_path)
    broken = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    broken["git_sha"] = "unknown"
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(broken))
    with pytest.raises(IncompleteManifestError):
        RunManifest.read(tmp_path)
