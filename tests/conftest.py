"""Shared fixtures for the dashboard tests: tiny run roots, gate docs and an SPX panel.

The run root is written by the real sweep, so the browser is tested on the same
files a sweep leaves behind. The panel builders live in ``dashboard_fixtures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dashboard_fixtures import write_panel

from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.sweep import SweepConfig, run_sweep

TINY = RunConfig(
    seed=7,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(7, 14)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
TINY_SWEEP = SweepConfig(base=TINY, premiums=(0.0, 2.0), arms=("flat", "delta_seller"))


@pytest.fixture
def fake_a2c_config() -> RunConfig:
    from optspread.config import EnvConfig, MarketConfig, TrainConfig

    return RunConfig(
        market=MarketConfig(n_days=5, tenor_days=(3,), n_strikes_each_side=2),
        env=EnvConfig(episode_days=5),
        policy="a2c",
        n_episodes=2,
        train=TrainConfig(algorithm="a2c", total_timesteps=50),
    )


@pytest.fixture
def fake_dqn_config() -> RunConfig:
    from optspread.config import EnvConfig, MarketConfig, TrainConfig

    return RunConfig(
        market=MarketConfig(n_days=5, tenor_days=(3,), n_strikes_each_side=2),
        env=EnvConfig(episode_days=5),
        policy="dqn",
        n_episodes=2,
        train=TrainConfig(algorithm="dqn", total_timesteps=50),
    )


@pytest.fixture(scope="session")
def tiny_sweep() -> SweepConfig:
    return TINY_SWEEP


@pytest.fixture(scope="session")
def run_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A runs root with one real sweep (four cells) plus a broken and a tampered run."""
    root = tmp_path_factory.mktemp("runs")
    run_sweep(TINY_SWEEP, root / "sweeps" / "tiny")
    # A manifest with no git SHA: refused by the gate, so listed as INVALID.
    broken = root / "adhoc" / "broken-run"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(json.dumps({"run_id": "broken-run", "config": {}}))
    # A real run whose results file was edited after its manifest was written.
    cells = sorted((root / "sweeps" / "tiny").glob("*-s7"))
    tampered = root / "adhoc" / cells[0].name
    tampered.mkdir(parents=True)
    for name in ("manifest.json", "run.json"):
        (tampered / name).write_bytes((cells[0] / name).read_bytes())
    record = json.loads((tampered / "run.json").read_text())
    record["mean_pnl"] = 1e9
    (tampered / "run.json").write_text(json.dumps(record))
    return root


def gate_json(milestone: str, items: list[dict[str, str]], verdict: str) -> str:
    return json.dumps(
        {"milestone": milestone, "report_date": "2026-10-26", "verdict": verdict, "items": items}
    )


@pytest.fixture()
def gate_docs(tmp_path: Path) -> Path:
    """M1 as markdown only (the repository's own file) and M2 as a JSON sidecar."""
    docs = tmp_path / "docs"
    docs.mkdir()
    repo_docs = Path(__file__).resolve().parent.parent / "docs"
    (docs / "M1_GATE_REPORT.md").write_text((repo_docs / "M1_GATE_REPORT.md").read_text())
    items = [
        {
            "name": "Analysis pipeline validated",
            "owner": "Dominick Dupuy",
            "verdict": "PASS",
            "criterion": "recovers the closed-form detection probability within 2 se",
            "evidence": "200 synthetic replicates; run 0123456789abcdef-s3",
            "failed_first": "",
        },
        {
            "name": "Replay",
            "owner": "Ilan Danial",
            "verdict": "NOT CLEARED",
            "criterion": "a replay reproduces every artifact digest",
            "evidence": "",
            "failed_first": "no replay has been run",
        },
    ]
    (docs / "M2_GATE_REPORT.json").write_text(gate_json("M2", items, "NOT CLEARED"))
    (docs / "M2_GATE_REPORT.md").write_text("# M2 gate report\n")  # the JSON wins
    return docs


@pytest.fixture(scope="session")
def panel_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_panel(tmp_path_factory.mktemp("vegalab"))


@pytest.fixture(autouse=True)
def _no_shared_data_folder(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No test reads the team's shared data folder: VEGALAB_DATA points at an empty dir.

    A test that wants data builds it under its own tmp_path and sets VEGALAB_DATA
    itself; the default (the synced OneDrive folder) is never reached from the suite.
    """
    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path_factory.mktemp("no_shared_data")))
