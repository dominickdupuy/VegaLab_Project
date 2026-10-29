"""Ilan Danial: config hashing, manifest round-trip, and the manifest gate.

The gate is the point of the second half: a run missing anything a replay needs
never reaches the results folder, which is a rule the tooling enforces rather
than one the team remembers.
"""

import functools
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import pytest

from optspread.config import CostConfig, MarketConfig, RunConfig, config_hash
from optspread.data import DataUnavailableError, OptionMetricsSource, make_source
from optspread.manifest import (
    MANIFEST_FILENAME,
    OPTIONAL_VERSIONS,
    REQUIRED_FIELDS,
    REQUIRED_VERSIONS,
    SCHEMA_VERSION,
    DataRecord,
    IncompleteManifestError,
    RunManifest,
    callable_name,
    file_sha256,
    git_dirty,
    git_sha,
    library_versions,
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


def test_the_optional_versions_are_recorded_when_installed() -> None:
    # The learner, gymnasium and the panel's reader can each move a number; each is
    # recorded where it is installed and is not required (#31).
    versions = library_versions()
    assert set(REQUIRED_VERSIONS) <= versions.keys()
    assert {"gymnasium", "pandas", "pyarrow", "torch", "stable_baselines3"} == set(
        OPTIONAL_VERSIONS
    )
    for name in OPTIONAL_VERSIONS:
        installed = importlib.util.find_spec(name) is not None
        assert (name in versions) == installed, name
    assert versions["gymnasium"] == importlib.metadata.version("gymnasium")  # a core dependency


def test_a_manifest_without_the_optional_versions_still_validates(tmp_path: Path) -> None:
    # What every manifest written before #31 records: no gymnasium, no pandas.
    m = _complete(tmp_path)
    m.versions = {k: v for k, v in m.versions.items() if k in REQUIRED_VERSIONS}
    m.validate_complete()
    assert m.write(tmp_path).exists()


# -- the data source: recorded beside the config, never in it (GH-21) ------------ #


def test_the_trainer_is_recorded_only_when_a_policy_was_trained(tmp_path: Path) -> None:
    m = _complete(tmp_path)
    assert m.trainer is None
    m.write(tmp_path)
    assert "trainer" not in json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    m.trainer = callable_name(_complete)
    assert m.trainer == "test_config_manifest._complete"
    m.write(tmp_path)
    assert RunManifest.read(tmp_path).trainer == m.trainer
    assert callable_name(functools.partial(_complete)) == "functools.partial"


def test_a_new_manifest_records_its_data_source(tmp_path: Path) -> None:
    m = _complete(tmp_path)
    assert m.data == DataRecord() and m.data_record.source == "synthetic"
    m.write(tmp_path)
    written = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert written["data"] == {"source": "synthetic", "panel": {}}
    assert "source" not in written["config"], "the source is provenance, not config"
    assert RunManifest.read(tmp_path).data == DataRecord()


def test_a_manifest_from_before_the_data_record_still_reads_and_keeps_its_keys(
    tmp_path: Path,
) -> None:
    m = _complete(tmp_path)
    m.write(tmp_path)
    old = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    del old["data"]  # what every manifest written before GH-21 looks like
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(old))
    back = RunManifest.read(tmp_path)
    assert back.data is None and back.data_record == DataRecord()
    back.write(tmp_path)  # read and written again: no key appears that was not there
    assert json.loads((tmp_path / MANIFEST_FILENAME).read_text()).keys() == old.keys()


def test_the_real_source_records_the_panel_files_it_reads(tmp_path: Path) -> None:
    source = OptionMetricsSource(root=tmp_path)
    with pytest.raises(DataUnavailableError, match="no SPX panel"):
        DataRecord.of(source)
    (tmp_path / "panel").mkdir()
    for name in ("surface_grid.parquet", "spx_daily.parquet"):
        (tmp_path / "panel" / name).write_bytes(name.encode())
    record = DataRecord.of(source)
    assert record.source == "optionmetrics"
    assert record.panel == {  # keyed as docs/DATA_MANIFEST.json keys them
        f"panel/{name}": file_sha256(tmp_path / "panel" / name)
        for name in ("surface_grid.parquet", "spx_daily.parquet")
    }
    assert len(record.panel_id) == 16 and record.panel_id == DataRecord.of(source).panel_id
    (tmp_path / "panel" / "spx_daily.parquet").write_bytes(b"rebuilt")
    assert DataRecord.of(source).panel_id != record.panel_id


def test_the_generator_and_the_stub_read_no_panel() -> None:
    for name in ("synthetic", "optionmetrics_stub"):
        record = DataRecord.of(make_source(name))
        assert record.source == name and record.panel == {} and record.panel_id == ""


def test_the_commit_is_the_code_checkouts_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # GH-23: git used to be asked about the cwd, so a run started (or a worker left)
    # outside the checkout recorded "unknown" and its manifest was refused.
    here = git_sha()
    if here == "unknown":  # pragma: no cover - only outside a git checkout
        pytest.skip("not a git checkout")
    monkeypatch.chdir(tmp_path)  # not a repository
    assert git_sha() == here and RunManifest.create(RunConfig()).git_sha == here
    assert git_sha(tmp_path) == "unknown" and git_dirty(tmp_path) is True
