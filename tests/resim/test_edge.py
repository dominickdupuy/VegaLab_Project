"""Dominick Dupuy: the edge attribution, on a grid small enough to run here.

The grid is written by the real orchestrator (the fixed-delta seller, the flat arm and
the fake learner of ``tests/fake_learner.py``), so the cells are the same objects a
sweep leaves behind. The identity is tested on every edge, the pairing where its
answer is known (an arm that never trades, and a component the book never books, have
an edge of exactly zero), and the se against a hand computation from the attribution's
own episodes. Every number is synthetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread.actions import ACTION_NAMES
from optspread.analysis import ALPHA
from optspread.attribution import COMPONENTS, Terms, attribute_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.edge import (
    FIELDS,
    STRUCTURES,
    CellParts,
    GridEdge,
    arm_edge,
    default_root,
    edge_grid,
    edge_of,
    output_stem,
    read_edge,
    render_markdown,
    write_edge,
)
from optspread.orchestrate import GridConfig, grid_filename, read_grid, run_grid
from optspread.policies import Policy
from optspread.power import t_critical
from optspread.train import POLICY_FILENAME
from scripts.edge_attribution import main

BASE = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(
    base=BASE,
    premiums=(0.0, 2.0, 4.0),
    arms=("delta_seller", "flat", "ppo"),
    n_seeds=3,
    train=TrainConfig(total_timesteps=64, seed=10),
)


def fake_loader(path: Path, config: RunConfig) -> Policy:
    assert path.name == POLICY_FILENAME and path.read_text().startswith("fake learner")
    assert config.train is not None
    return FakeLearned(config.train.seed)


def snapshot(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


@pytest.fixture(scope="module")
def grid_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("edge") / "sweep"
    run_grid(GRID, root, trainer=fake_trainer)
    return root


@pytest.fixture(scope="module")
def result(grid_root: Path) -> GridEdge:
    before = snapshot(grid_root)
    out = edge_grid(grid_root, load_policy=fake_loader)
    assert snapshot(grid_root) == before  # nothing written into the sweep
    return out


def _scale(*values: float) -> float:
    return 1.0 + math.fsum(abs(v) for v in values)


# -- the identity -------------------------------------------------------------- #


def test_the_components_edges_sum_to_the_pnl_edge(result: GridEdge) -> None:
    assert len(result.edges) == 2 * 3  # premiums 2 and 4, three arms
    assert result.reproduction == "reproduces" and result.reproduces and result.agrees
    for e in result.edges:
        estimates = [e.components[c].estimate for c in COMPONENTS]
        total = math.fsum(estimates)
        assert abs(total - e.pnl.estimate) <= 1e-9 * _scale(*estimates)
        assert abs(e.identity_gap) <= 1e-9 * _scale(*estimates)
        # Block by block too: the contrast is linear, so the block means add up.
        for i, m in enumerate(e.pnl.block_means):
            parts = [e.components[c].block_means[i] for c in COMPONENTS]
            assert abs(math.fsum(parts) - m) <= 1e-9 * _scale(*parts)
        # The saved P&L is the re-simulated one: the cells reproduce bit for bit here.
        assert e.pnl_saved == e.pnl


def test_the_structures_split_each_component_and_add_back_to_it(result: GridEdge) -> None:
    for e in result.edges:
        pnls = [e.by_structure[s]["pnl"].estimate for s in STRUCTURES]
        assert abs(math.fsum(pnls) - e.pnl.estimate) <= 1e-9 * _scale(*pnls)
        assert abs(e.structure_gap) <= 1e-9 * _scale(*pnls)
        for c in COMPONENTS:
            parts = [e.by_structure[s][c].estimate for s in STRUCTURES]
            assert abs(math.fsum(parts) - e.components[c].estimate) <= 1e-9 * _scale(*parts)
    for cell in result.cells:
        for f in FIELDS:
            parts = [getattr(cell.by_structure[s], f) for s in STRUCTURES]
            assert abs(math.fsum(parts) - getattr(cell.mean, f)) <= 1e-9 * _scale(*parts)
        explained = math.fsum(getattr(cell.mean, c) for c in COMPONENTS)
        assert abs(explained - cell.mean.pnl) <= 1e-9 * _scale(cell.mean.pnl, cell.gross)


def test_each_structure_gets_its_own_steps_and_a_close_is_charged_to_what_it_closed(
    result: GridEdge,
) -> None:
    for cell in result.cells:
        flat = cell.by_structure["FLAT"]
        # A step about nothing moves nothing: every cost is a structure's.
        assert flat.cost == 0.0 and flat.market == 0.0 and abs(flat.pnl) < 1e-9
        if cell.arm == "delta_seller":
            # It only ever holds iron condors, so the condor carries all of its cost.
            for s in ("SHORT_STRANGLE", "PUT_CREDIT_SPREAD"):
                assert cell.by_structure[s] == Terms()
            condor = cell.by_structure["IRON_CONDOR"].cost
            assert condor == pytest.approx(cell.mean.cost, rel=1e-12) and condor < 0.0
        if cell.arm == "flat":
            assert cell.held_share["FLAT"] == 1.0 and cell.mean.cost == 0.0
    # The fake learner of block i holds an iron condor on even training seeds (10, 12)
    # and a put spread on odd ones (11): both appear, and nothing else.
    ppo = [c for c in result.cells if c.arm == "ppo"]
    assert {b for c in ppo for b, share in c.held_share.items() if share > 0} == {
        "FLAT",
        "IRON_CONDOR",
        "PUT_CREDIT_SPREAD",
    }


# -- the pairing --------------------------------------------------------------- #


def test_a_component_equal_at_p_and_at_zero_has_an_edge_of_exactly_zero(
    result: GridEdge,
) -> None:
    zero = {"estimate": 0.0, "se": 0.0, "t_stat": 0.0, "lower_bound": 0.0, "upper_bound": 0.0}
    for e in result.edges:
        # The book never books carry, at any premium.
        assert e.components["carry"].model_dump(exclude={"rejected", "block_means"}) == zero
        assert e.components["carry"].block_means == (0.0,) * GRID.n_seeds
        if e.arm == "flat":  # never trades: every component equal in both worlds
            for c in (*COMPONENTS, "pnl"):
                edge = e.pnl if c == "pnl" else e.components[c]
                assert edge.estimate == 0.0 and edge.se == 0.0 and not edge.rejected
    # And on any matrix: what differs in one cell moves the edge; what does not, does not.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 6))
    assert edge_of(x, x).estimate == 0.0 and edge_of(x, x).se == 0.0
    assert edge_of(x + 2.5, x).estimate == pytest.approx(2.5) and edge_of(x + 2.5, x).se < 1e-12


def _parts(seeds: tuple[int, ...], n: int = 3) -> CellParts:
    z = np.zeros((n, len(FIELDS)))
    return CellParts(
        run_id="r",
        source="synthetic",
        seeds=seeds,
        totals=z,
        by_structure=np.zeros((n, len(STRUCTURES), len(FIELDS))),
        saved_pnl=np.zeros(n),
        held_share={},
        reproduces=True,
        agrees=True,
        max_abs_pnl_diff=0.0,
        gross=0.0,
        abs_residual=0.0,
        residual_share=0.0,
        max_abs_step_residual=0.0,
    )


def test_cells_that_are_not_paired_are_refused(grid_root: Path) -> None:
    record = read_grid(grid_root / grid_filename(GRID))
    c1 = [record.cell(2.0, "ppo", b) for b in range(2)]
    c0 = [record.cell(0.0, "ppo", b) for b in range(2)]
    good = [(c, _parts((c.seed, c.seed + 1, c.seed + 2))) for c in c1]
    null = [(c, _parts((c.seed, c.seed + 1, c.seed + 2))) for c in c0]
    assert arm_edge("ppo", 2.0, good, null).pnl.estimate == 0.0
    shifted = [(c, _parts((c.seed + 1, c.seed + 2, c.seed + 3))) for c in c0]
    with pytest.raises(ValueError, match="same paths and training seed"):
        arm_edge("ppo", 2.0, good, shifted)
    retrained = [(c.model_copy(update={"train_seed": 99}), p) for c, p in null]
    with pytest.raises(ValueError, match="same paths and training seed"):
        arm_edge("ppo", 2.0, good, retrained)
    with pytest.raises(ValueError, match="blocks against"):
        arm_edge("ppo", 2.0, good, null[:1])


# -- the test against a hand computation -------------------------------------------- #


def test_the_se_and_interval_are_a2s_on_the_block_means_by_hand(
    grid_root: Path, result: GridEdge
) -> None:
    record = read_grid(grid_root / grid_filename(GRID))
    S = GRID.n_seeds
    crit = t_critical(S, ALPHA)
    assert result.critical == crit
    for arm, component in (("delta_seller", "theta"), ("delta_seller", "gamma"), ("ppo", "cost")):
        means = []
        for b in range(S):
            at_p = attribute_run(
                grid_root / record.cell(4.0, arm, b).run_id, load_policy=fake_loader
            )
            at_0 = attribute_run(
                grid_root / record.cell(0.0, arm, b).run_id, load_policy=fake_loader
            )
            diffs = [
                getattr(e1.totals, component) - getattr(e0.totals, component)
                for e1, e0 in zip(at_p.episodes, at_0.episodes, strict=True)
            ]
            means.append(sum(diffs) / len(diffs))
        grand = sum(means) / S
        se = math.sqrt(sum((m - grand) ** 2 for m in means) / (S - 1) / S)
        edge = result.edge(arm, 4.0).components[component]
        assert edge.block_means == pytest.approx(means, rel=1e-12, abs=1e-9)
        assert edge.estimate == pytest.approx(grand, rel=1e-12, abs=1e-9)
        assert edge.se == pytest.approx(se, rel=1e-9, abs=1e-9)
        assert edge.lower_bound == pytest.approx(grand - crit * se, rel=1e-9, abs=1e-9)
        assert edge.upper_bound == pytest.approx(grand + crit * se, rel=1e-9, abs=1e-9)
        assert edge.rejected == (edge.lower_bound > 0.0)


# -- the grid, the files and the command --------------------------------------------- #


@pytest.mark.spawns_processes
@pytest.mark.slow
def test_the_grid_keeps_arms_reads_a_prefix_and_runs_on_workers(grid_root: Path) -> None:
    serial = edge_grid(grid_root, arms=["delta_seller"], first=2)
    assert serial.arms == ("delta_seller",) and serial.n_episodes == 2 and serial.first == 2
    assert {c.n_episodes for c in serial.cells} == {2} and len(serial.cells) == 3 * 3
    parallel = edge_grid(grid_root / grid_filename(GRID), arms=["delta_seller"], first=2, workers=2)
    assert parallel.model_copy(update={"grid_file": serial.grid_file}) == serial
    with pytest.raises(ValueError, match="not in this grid"):
        edge_grid(grid_root, arms=["random"])
    with pytest.raises(ValueError, match="at least 2"):
        edge_grid(grid_root, arms=["flat"], first=1)


def test_the_json_round_trips_and_is_never_written_into_a_sweep(
    tmp_path: Path, grid_root: Path, result: GridEdge
) -> None:
    json_path, md_path = write_edge(result, tmp_path / "out")
    assert json_path.name == f"edge-{result.grid_hash}.json" == f"{output_stem(result)}.json"
    assert read_edge(json_path) == result
    raw = json.loads(json_path.read_text())
    assert raw["schema_version"] == 1 and raw["components"] == list(COMPONENTS)
    assert raw["structures"] == list(ACTION_NAMES)
    assert set(raw["edges"][0]["by_structure"]["IRON_CONDOR"]) == {*COMPONENTS, "pnl"}
    assert md_path.read_text() == render_markdown(result)
    assert "## Edge by component" in md_path.read_text()
    with pytest.raises(ValueError, match="holds a sweep"):
        write_edge(result, grid_root)
    with pytest.raises(ValueError, match="run directory"):
        write_edge(result, grid_root / result.cells[0].run_id)
    path = tmp_path / "v99.json"
    path.write_text(result.model_copy(update={"schema_version": 99}).model_dump_json())
    with pytest.raises(ValueError, match="schema v99"):
        read_edge(path)


def test_the_default_root_is_under_the_attribution_folder(tmp_path: Path) -> None:
    assert default_root({}).parts[-2:] == ("attribution", "edge")
    assert default_root({"OPTSPREAD_ATTRIBUTION": str(tmp_path)}) == tmp_path / "edge"


def test_the_command_writes_the_edge_and_says_no(
    tmp_path: Path, grid_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "edge"
    argv = ["--grid", str(grid_root), "--arms", "delta_seller", "flat", "--first", "2"]
    assert main([*argv, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "p=2     delta_seller" in printed and "carried by" in printed
    stem = f"edge-{read_grid(grid_root / grid_filename(GRID)).grid_hash}-first2"
    assert read_edge(out / f"{stem}.json").first == 2 and (out / f"{stem}.md").is_file()
    assert main([*argv, "--out", str(grid_root)]) == 1
    assert "holds a sweep" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--grid", str(grid_root), "--first", "1"])
    with pytest.raises(SystemExit):
        main(["--grid", str(grid_root), "--workers", "0"])
