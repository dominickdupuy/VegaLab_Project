"""Ilan Danial: the Attribution page reads the edge and the action mix as they were written.

A tiny grid (the fixed-delta seller and the fake learner of ``tests/fake_learner.py``,
two premiums, two blocks) is run by the real orchestrator, its edge written by
``optspread.edge`` and its action mix by ``optspread.actionmix``, both outside the runs
root. The payloads are then read back through the builders and the server, so the
page is tested on the schemas those modules write. Every number is synthetic.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread import webviews as v
from optspread.actionmix import mix_grid, output_name, write_mix
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.dashboard_data import REPO_ROOT
from optspread.edge import COMPONENTS, edge_grid, write_edge
from optspread.orchestrate import GridConfig, grid_filename, run_grid
from optspread.policies import Policy
from optspread.train import POLICY_FILENAME

BASE = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(
    base=BASE,
    premiums=(0.0, 2.0),
    arms=("delta_seller", "ppo"),
    n_seeds=2,
    train=TrainConfig(total_timesteps=64, seed=10),
)


def fake_loader(path: Path, config: RunConfig) -> Policy:
    assert path.name == POLICY_FILENAME and config.train is not None
    return FakeLearned(config.train.seed)


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """(runs root, attribution root, action-mix root), every file written by the real code."""
    base = tmp_path_factory.mktemp("webedge")
    runs, attr, mix = base / "runs", base / "attribution", base / "actionmix"
    run_grid(GRID, runs, trainer=fake_trainer)
    write_edge(edge_grid(runs, load_policy=fake_loader), attr / v.EDGE_DIR)
    write_edge(edge_grid(runs, first=2, load_policy=fake_loader), attr / v.EDGE_DIR)
    grid, cells = mix_grid(runs, load_policy=fake_loader)
    write_mix(grid, mix / output_name(grid))
    for cell in cells:
        write_mix(cell, mix / output_name(cell))
    return runs, attr, mix


GH = grid_filename(GRID).removeprefix("grid-").removesuffix(".json")


@pytest.mark.behaviour
def test_the_list_names_every_edge_file_and_each_grids_commands(
    roots: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    runs, attr, _ = roots
    listed = v.edge_list(attr, runs)
    assert {(f.grid_hash, f.first) for f in listed.files} == {(GH, None), (GH, 2)}
    full = next(f for f in listed.files if f.first is None)
    assert full.arms == ["delta_seller", "ppo"] and full.premiums == [0.0, 2.0]
    assert full.n_seeds == 2 and full.reproduction == "reproduces" and not full.error
    (grid,) = listed.grids
    assert grid.grid_hash == GH and grid.has_edge and grid.key == grid_filename(GRID)
    assert grid.edge_command == f'make edge ARGS="--grid <runs root>/{grid.key} --workers 2"'
    assert grid.actionmix_command.startswith('make actionmix ARGS="--grid <runs root>/')
    # an empty root: no files, and the command the page shows says what writes them
    empty = v.edge_list(tmp_path / "none", tmp_path / "noruns")
    assert empty.files == [] and empty.grids == [] and "make edge" in empty.command


@pytest.mark.invariant
def test_the_detail_is_the_edge_as_written_with_each_cells_run(
    roots: tuple[Path, Path, Path],
) -> None:
    runs, attr, _ = roots
    out = v.edge_detail(attr, runs, GH)
    assert out.name == f"edge-{GH}.json" and out.grid_key == grid_filename(GRID)
    edge = out.edge
    assert edge["components"] == list(COMPONENTS) and edge["first"] is None
    assert len(edge["edges"]) == 2  # premium 2, both arms
    for e in edge["edges"]:
        total = sum(e["components"][c]["estimate"] for c in COMPONENTS)
        assert abs(total - e["pnl"]["estimate"]) <= 1e-6 * (1 + abs(total))
        assert set(e["by_structure"]) == set(edge["structures"])
        assert e["pnl"]["lower_bound"] <= e["pnl"]["estimate"] <= e["pnl"]["upper_bound"]
    for cell in edge["cells"]:
        key = out.run_keys[cell["run_id"]]
        assert key is not None and (runs / key / "manifest.json").is_file()
        assert cell["reproduction"] == "reproduces"
    # the partial read on request, and by default only when no full read exists
    assert v.edge_detail(attr, runs, GH, first=2).edge["first"] == 2
    assert v.edge_detail(attr, runs, f"grid-{GH}.json").name == out.name


@pytest.mark.behaviour
def test_a_partial_read_alone_is_served_and_a_missing_or_bad_grid_is_refused(
    roots: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    runs, attr, _ = roots
    only = tmp_path / "attr"
    (only / v.EDGE_DIR).mkdir(parents=True)
    shutil.copy(attr / v.EDGE_DIR / f"edge-{GH}-first2.json", only / v.EDGE_DIR)
    assert v.edge_detail(only, runs, GH).edge["first"] == 2
    # cells whose runs are not under this root are still read, with no run to link
    elsewhere = v.edge_detail(only, tmp_path / "noruns", GH)
    assert elsewhere.grid_key is None and set(elsewhere.run_keys.values()) == {None}
    with pytest.raises(KeyError):
        v.edge_detail(only, runs, "abc123")
    with pytest.raises(KeyError):
        v.edge_detail(only, runs, GH, first=7)
    for bad in ("../etc", "ABC", ""):
        with pytest.raises(ValueError):
            v.edge_detail(only, runs, bad)
    with pytest.raises(ValueError):
        v.edge_detail(only, runs, GH, first=1)
    (only / v.EDGE_DIR / f"edge-{GH}.json").write_text("{not json")
    listed = v.edge_list(only, runs)
    assert any(f.error for f in listed.files)
    with pytest.raises(ValueError):
        v.edge_detail(only, runs, GH)


# -- through the server ---------------------------------------------------------- #


@pytest.fixture()
def client(roots: tuple[Path, Path, Path], tmp_path: Path, gate_docs: Path) -> Iterator[object]:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from optspread.jobs import JobManager
    from optspread.server import Settings, create_app

    runs, attr, mix = roots
    settings = Settings(
        runs,
        gate_docs,
        tmp_path / "nodata",
        tmp_path / "noweb",
        REPO_ROOT / "README.md",
        attribution=attr,
        actionmix=mix,
    )
    manager = JobManager(tmp_path / "jobs", executor=ThreadPoolExecutor(1))
    with TestClient(create_app(settings, manager), base_url="http://127.0.0.1") as c:
        yield c


@pytest.mark.invariant
def test_the_routes_serve_the_edge_files_and_one_grids_edge(
    client: object, roots: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    assert isinstance(client, TestClient)
    runs, _, _ = roots
    listed = client.get("/api/edge").json()
    assert {f["name"] for f in listed["files"]} == {f"edge-{GH}.json", f"edge-{GH}-first2.json"}
    detail = client.get("/api/edge/detail", params={"grid": GH})
    assert detail.status_code == 200
    body = detail.json()
    assert body["edge"]["grid_hash"] == GH and len(body["run_keys"]) == 2 * 2 * 2
    assert (
        client.get("/api/edge/detail", params={"grid": GH, "first": 2}).json()["edge"]["first"] == 2
    )
    missing = client.get("/api/edge/detail", params={"grid": "abc123"})
    assert missing.status_code == 404 and "make edge" in missing.json()["detail"]
    assert client.get("/api/edge/detail", params={"grid": "../x"}).status_code == 422
    assert client.get("/api/edge/detail", params={"grid": GH, "first": 1}).status_code == 422
    assert str(tmp_path) not in missing.text and str(runs) not in missing.text


@pytest.mark.invariant
def test_the_routes_serve_a_grids_action_mix_and_a_runs(
    client: object, roots: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    assert isinstance(client, TestClient)
    runs, _, _ = roots
    body = client.get("/api/edge/detail", params={"grid": GH}).json()
    # the grid's action mix, pooled per arm and premium, and one cell's own
    grid = client.get("/api/runs/actionmix", params={"grid": GH}).json()
    assert grid["kind"] == "grid" and len(grid["grid"]["pooled"]) == 2 * 2
    run_id = body["edge"]["cells"][0]["run_id"]
    run = client.get("/api/runs/actionmix", params={"run": body["run_keys"][run_id]}).json()
    assert run["kind"] == "run" and run["run"]["run_id"] == run_id
    summary = run["run"]["summary"]
    assert summary["n_episodes"] == BASE.n_episodes and len(summary["by_day"]) > 0
    assert abs(sum(summary["held_share"].values()) - 1.0) < 1e-9
    absent = client.get("/api/runs/actionmix", params={"grid": "abc123"})
    assert absent.status_code == 404 and "make actionmix" in absent.json()["detail"]
    assert str(tmp_path) not in absent.text and str(runs) not in absent.text


@pytest.mark.behaviour
def test_a_differing_cell_counts_its_episodes_from_its_runs_attribution(
    roots: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """The grid page says "1 of N episodes differs", not a failed cell, when it can."""
    import json
    import shutil

    runs, attr, _ = roots
    mine = tmp_path / "attribution"
    shutil.copytree(attr, mine)
    path = mine / v.EDGE_DIR / f"edge-{GH}.json"
    edge = json.loads(path.read_text())
    cell = next(c for c in edge["cells"] if c["arm"] == "delta_seller")
    cell["reproduction"], cell["reproduces"], cell["agrees"] = "differs", False, False
    path.write_text(json.dumps(edge))
    assert v.edge_detail(mine, runs, GH).episode_counts == {}  # no run attribution cached yet
    target = mine / f"{cell['run_id']}.json"
    v.compute_attribution(runs / cell["run_id"], target)
    att = json.loads(target.read_text())
    att["episodes"][0]["agrees"] = False
    target.write_text(json.dumps(att))
    counts = v.edge_detail(mine, runs, GH).episode_counts
    assert counts == {cell["run_id"]: {"n_episodes": len(att["episodes"]), "n_differ": 1}}
