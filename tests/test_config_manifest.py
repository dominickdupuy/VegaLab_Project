"""Ilan Danial: config hashing and manifest round-trip."""

from pathlib import Path

from optspread.config import CostConfig, MarketConfig, RunConfig, config_hash
from optspread.manifest import RunManifest, file_sha256


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
