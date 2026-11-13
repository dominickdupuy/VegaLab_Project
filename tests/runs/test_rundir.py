"""Ilan Danial: two refusals of the resume rule that the orchestrator tests never reach.

``optspread.runs.rundir.valid_run`` reads a run directory back instead of recomputing it
only when everything about it checks out. The grid and sweep tests delete a results file
or change one under its manifest; here, a results file or manifest that cannot be read
(a write cut off half way, say) and a run whose recorded id is not its config's, with
every digest still matching, are each recomputed: never read back, and never a crash of
the resume. Every number is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optspread.artifacts import RUN_FILENAME, read_run
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.experiment.grid import GridConfig, grid_cells
from optspread.experiment.schedule import Progress, plan_cells, run_grid
from optspread.manifest import MANIFEST_FILENAME, file_sha256
from optspread.runs.rundir import run_id, valid_run

SHORT = RunConfig(
    seed=4100,
    n_episodes=2,
    market=MarketConfig(n_days=10, tenor_days=(5, 10)),
    env=EnvConfig(episode_days=10),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(base=SHORT, premiums=(0.0, 1.0), arms=("delta_seller",), n_seeds=2)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    run_grid(GRID, tmp_path)
    return tmp_path


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    out: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _rehash(run_dir: Path) -> None:
    """Record run.json's present digest in the manifest, so no digest check can refuse it."""
    manifest = _read_json(run_dir / MANIFEST_FILENAME)
    manifest["artifacts"] = {
        **manifest["artifacts"],  # type: ignore[dict-item]
        RUN_FILENAME: file_sha256(run_dir / RUN_FILENAME),
    }
    _write_json(run_dir / MANIFEST_FILENAME, manifest)


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("name", "text"),
    [
        (RUN_FILENAME, '{"schema_version": 1, "run_id": "'),  # cut off half way
        (RUN_FILENAME, "[]"),  # JSON, but not a run record
        (MANIFEST_FILENAME, '{"run_id": '),
        (MANIFEST_FILENAME, '{"run_id": "x"}'),  # JSON, but not a manifest
    ],
)
def test_a_run_whose_files_cannot_be_read_is_recomputed_not_read_back(
    root: Path, name: str, text: str
) -> None:
    cell = grid_cells(GRID)[0]
    run_dir = root / cell.run_id
    assert valid_run(run_dir, cell.config) is not None
    (run_dir / name).write_text(text, encoding="utf-8")
    if name == RUN_FILENAME:
        _rehash(run_dir)  # the file itself is what is wrong, not its digest
    assert valid_run(run_dir, cell.config) is None
    assert plan_cells(grid_cells(GRID), root).to_run == 1


@pytest.mark.invariant
@pytest.mark.parametrize("name", [RUN_FILENAME, MANIFEST_FILENAME])
def test_a_run_whose_recorded_id_is_not_its_configs_is_recomputed(root: Path, name: str) -> None:
    cell = grid_cells(GRID)[0]
    run_dir = root / cell.run_id
    data = _read_json(run_dir / name)
    assert data["run_id"] == run_id(cell.config)
    # The config hash is kept, so only the id disagrees: another seed's id on this run.
    data["run_id"] = f"{data['config_hash']}-s{cell.config.seed + 1}"
    _write_json(run_dir / name, data)
    _rehash(run_dir)
    assert valid_run(run_dir, cell.config) is None


@pytest.mark.invariant
def test_a_rerun_recomputes_exactly_the_cells_it_could_not_read_or_whose_id_moved(
    root: Path,
) -> None:
    first = run_grid(GRID, root)
    cells = grid_cells(GRID)
    broken, relabelled = cells[1], cells[2]
    (root / broken.run_id / RUN_FILENAME).write_text("{", encoding="utf-8")
    record = _read_json(root / relabelled.run_id / RUN_FILENAME)
    record["run_id"] = f"{record['config_hash']}-s0"
    _write_json(root / relabelled.run_id / RUN_FILENAME, record)
    _rehash(root / relabelled.run_id)

    assert plan_cells(cells, root).to_run == 2
    lines: list[Progress] = []
    again = run_grid(GRID, root, report=lines.append)
    computed = {p.cell.run_id for p in lines if p.seconds is not None}
    assert computed == {broken.run_id, relabelled.run_id}
    assert again == first
    for cell in cells:
        assert valid_run(root / cell.run_id, cell.config) == read_run(root / cell.run_id)
