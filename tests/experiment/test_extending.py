"""Hemdutt Rao: bring your own data, agent, planted effect and cost model (#32).

docs/EXTENDING.md is run here block by block, so a guide that drifts from the code
fails the suite. The rest pins what the guide promises: the default components are
the harness exactly; the drag check takes a cost model of one's own; and a source,
cost model, effect and learner of one's own run through the paired grid, in
parallel, and replay only through the components they were made with.
"""

from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fake_learner import fake_trainer

from optspread.artifacts import RUN_FILENAME
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.costs import CostModel, QuotedSpreadCost, make_cost_model
from optspread.effects import SkewPremium, TermPremium
from optspread.extend import Components, planted
from optspread.harness import run_episodes
from optspread.orchestrate import GridConfig, run_grid
from optspread.replay import replay_run
from optspread.scripted import cost_drag_check
from optspread.types import Chain, OptionLeg

REPO = Path(__file__).resolve().parent.parent.parent
GUIDE = REPO / "docs" / "EXTENDING.md"
SHORT = MarketConfig(n_days=15, tenor_days=(8, 15))


# -- the guide runs -------------------------------------------------------- #


def guide_blocks() -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", GUIDE.read_text(), flags=re.DOTALL | re.MULTILINE)


@pytest.mark.slow
def test_every_block_in_the_guide_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("pyarrow")
    blocks = guide_blocks()
    assert len(blocks) >= 6, "the guide lost its examples"
    # The guide's relative roots land in the test's tmp dir; its manifests still name
    # this checkout's commit, as they would run from the repository.
    git_dir = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    if not git_dir:
        pytest.skip("not a git checkout, so a run manifest cannot name its commit")
    monkeypatch.setenv("GIT_DIR", git_dir)
    monkeypatch.setenv("GIT_WORK_TREE", str(REPO))
    monkeypatch.chdir(tmp_path)
    module = types.ModuleType("extending_guide")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{GUIDE.name}[block {i}]", "exec"), module.__dict__)
        except Exception as exc:  # name the block, so a broken example is found at once
            raise AssertionError(f"docs/EXTENDING.md block {i} failed: {exc!r}") from exc
    out = capsys.readouterr().out
    assert "detected" in out and "grid " in out
    assert (tmp_path / "runs" / "extend" / "skew-wing").is_dir()


def test_the_readme_links_the_guide() -> None:
    readme = (REPO / "README.md").read_text()
    assert "docs/EXTENDING.md" in readme


# -- the defaults through the runner ----------------------------------------- #


def test_the_default_components_run_exactly_what_the_harness_runs() -> None:
    cfg = RunConfig(seed=3, n_episodes=3, market=SHORT, env=EnvConfig(episode_days=15))
    config = cfg.model_copy(update={"policy": "delta_seller"})
    assert Components()(config) == run_episodes(config)


# -- a cost model of one's own ---------------------------------------------- #


class _DoubleSpread:
    """Twice the quoted cost: deterministic in the chain and the leg, so it must pass."""

    def __init__(self, config: CostConfig) -> None:
        self.quoted = QuotedSpreadCost(config)

    def leg_cost(self, chain: Chain, leg: OptionLeg) -> float:
        return 2.0 * self.quoted.leg_cost(chain, leg)


def test_the_drag_check_runs_on_a_cost_model_of_ones_own() -> None:
    own = cost_drag_check(n_episodes=3, sellers=["hold_condor", "roll_condor"])
    doubled = cost_drag_check(
        n_episodes=3, sellers=["hold_condor", "roll_condor"], costs=_DoubleSpread(CostConfig())
    )
    assert own.passed and doubled.passed, doubled.render()
    for a, b in zip(own.rows, doubled.rows, strict=True):
        assert a.mean_pnl_free == b.mean_pnl_free  # the free arm never sees the model
        assert b.measured_drag == pytest.approx(2.0 * a.measured_drag)


def _double(config: CostConfig) -> CostModel:
    return _DoubleSpread(config) if config.enabled else make_cost_model(config)


# -- all four through the grid ------------------------------------------------- #

GRID = GridConfig(
    base=RunConfig(
        seed=900,
        n_episodes=3,
        market=MarketConfig(n_days=21, tenor_days=(10, 21)),
        env=EnvConfig(episode_days=21),
        cost=CostConfig(enabled=True),
    ),
    premiums=(0.0, 2.0),
    arms=("delta_seller", "ppo"),
    n_seeds=2,
    train=TrainConfig(total_timesteps=64, seed=3),
)
# Module-level and built from importable pieces, so it pickles into spawned workers.
SKEW_WORLD = Components(source=planted(SkewPremium()))


def test_components_of_ones_own_run_through_the_grid_and_replay_only_through_them(
    tmp_path: Path,
) -> None:
    world = Components(source=planted(TermPremium()), costs=_double)
    record = run_grid(GRID, tmp_path, trainer=fake_trainer, runner=world)
    default = run_grid(GRID, tmp_path / "default", trainer=fake_trainer)
    # The same cells and run ids (the config hash cannot see the components), other numbers.
    assert [c.run_id for c in record.cells] == [c.run_id for c in default.cells]
    assert record.cell(2.0, "ppo", 0).mean_pnl != default.cell(2.0, "ppo", 0).mean_pnl
    assert np.all(record.contrast(tmp_path, 0.0, "delta_seller") == 0.0)
    for arm in GRID.arms:
        cell = tmp_path / record.cell(2.0, arm, 1).run_id
        assert replay_run(cell, trainer=fake_trainer, runner=world).matches
        assert not replay_run(cell, trainer=fake_trainer).matches


@pytest.mark.spawns_processes
@pytest.mark.slow
def test_components_pickle_into_parallel_workers(tmp_path: Path) -> None:
    one = run_grid(GRID, tmp_path / "one", trainer=fake_trainer, runner=SKEW_WORLD)
    two = run_grid(GRID, tmp_path / "two", workers=2, trainer=fake_trainer, runner=SKEW_WORLD)
    assert one == two
    for cell in one.cells:
        a = (tmp_path / "one" / cell.run_id / RUN_FILENAME).read_bytes()
        assert a == (tmp_path / "two" / cell.run_id / RUN_FILENAME).read_bytes()
