"""The M3 tests' world: a git history with a cutoff tag, and a tiny sweep written on it.

Not collected (no ``test_`` prefix). The sweep is the real orchestrator's, run with the
fake learner of ``tests/fake_learner.py``; its manifests name a commit of a throwaway
repository (``optspread.manifest.git_sha`` is patched while it runs), so the items'
history checks read that repository and never this one's tags. Every number is synthetic.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from fake_learner import fake_trainer

from optspread import manifest as manifest_module
from optspread import reproduction as rp
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.manifest import callable_name
from optspread.orchestrate import GridConfig, GridRecord, grid_filename, read_grid, run_grid

SHORT = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
TINY = GridConfig(
    base=SHORT, premiums=(0.0, 2.0, 4.0), n_seeds=3, train=TrainConfig(total_timesteps=64, seed=10)
)
TAG = "cutoff-test"
FAKE_TRAINER = callable_name(fake_trainer)


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@dataclass(frozen=True)
class World:
    repo: Path  # C0 -> C1 (tagged TAG; the sweep's commit) -> C2 (adds the expected file)
    before: str  # C0
    at_tag: str  # C1
    after: str  # C2
    root: Path  # the runs root: the grid record and its cells
    record: GridRecord
    expected: rp.SweepExpected
    expected_text: str


def build_world(base: Path) -> World:
    repo = base / "repo"
    (repo / "docs").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "README.md").write_text("scaffold\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "scaffold")
    before = git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("frozen\n")
    git(repo, "commit", "-q", "-am", "freeze")
    git(repo, "tag", "-a", TAG, "-m", "the cutoff")
    at_tag = git(repo, "rev-parse", "HEAD")
    root = base / "runs"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifest_module, "git_sha", lambda cwd=None: at_tag)
        mp.setattr(manifest_module, "git_dirty", lambda cwd=None: False)
        run_grid(TINY, root, trainer=fake_trainer)
        mp.setattr(rp, "REAL_TRAINER", FAKE_TRAINER)
        record = read_grid(root / grid_filename(TINY))
        expected = rp.build_expected(record, root)
    text = rp.render_expected(expected)
    (repo / "docs" / rp.EXPECTED_FILE).write_text(text, newline="\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "expected results")
    after = git(repo, "rev-parse", "HEAD")
    return World(repo, before, at_tag, after, root, record, expected, text)
