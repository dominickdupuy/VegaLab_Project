"""API payloads for the edge of a grid (``optspread.edge``): per arm and premium, each
component's and each structure's share of the paired contrast.

Lane: Ilan Danial (machinery).

Read from ``<attribution root>/edge`` as ``scripts/edge_attribution.py`` wrote it, with each
cell's run under the runs root. Moved from ``optspread.webviews``, which re-exports it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from optspread.api._base import Out, _finite_json, _source_suffix
from optspread.dashboard.jobs.spec import JOBS_DIR
from optspread.dashboard.runindex import GRID_PATTERN
from optspread.manifest import MANIFEST_FILENAME

# -- the edge: which components and structures carry it (optspread.edge) ------------ #

EDGE_DIR = "edge"  # under the attribution root, as scripts/edge_attribution.py writes it
_EDGE_FILE = re.compile(r"edge-([0-9a-f]+)(?:-first(\d+))?\.json")
_HASH = re.compile(r"[0-9a-f]+")
EDGE_COMMAND = 'make edge ARGS="--grid <runs root>/{grid} --workers 2"'
ACTIONMIX_COMMAND = 'make actionmix ARGS="--grid <runs root>/{grid} --workers 2"'


class EdgeFileOut(Out):
    """One ``edge-<grid hash>[-first<N>].json``, from its header; ``error`` when unreadable."""

    name: str
    grid_hash: str
    first: int | None
    source: str = ""
    arms: list[str] = Field(default_factory=list)
    premiums: list[float] = Field(default_factory=list)
    n_seeds: int = 0
    n_episodes: int = 0
    reproduction: str = ""
    error: str = ""


class EdgeGridOut(Out):
    """A grid under the runs root, with the commands that write its edge and action mix."""

    grid_hash: str
    key: str  # the grid file relative to the runs root, as the Sweep page keys it
    has_edge: bool
    edge_command: str
    actionmix_command: str


class EdgeListOut(Out):
    files: list[EdgeFileOut]
    grids: list[EdgeGridOut]
    folder: str  # where the files are read, relative to the attribution root
    command: str  # the command with a placeholder grid, for a root that has none


class EdgeOut(Out):
    """One grid's edge as ``optspread.edge`` wrote it, with each cell's run under the root."""

    name: str
    folder: str
    edge: dict[str, Any]  # GridEdge, field for field (the schema of optspread.edge)
    run_keys: dict[str, str | None]  # run id -> run key under the runs root, None if absent
    grid_key: str | None
    actionmix_command: str
    # run id -> {"n_episodes", "n_differ"} for a cell that differs, read from the run's own
    # attribution when one is cached: how many of its episodes are beyond PNL_RTOL
    episode_counts: dict[str, dict[str, int]] = Field(default_factory=dict)


def _episode_counts(attribution_root: Path, source: str, run_id: str) -> dict[str, int] | None:
    """Episodes and those beyond tolerance in a run's cached attribution, if it is there."""
    from optspread.attribution import read_attribution

    path = attribution_root / f"{run_id}{_source_suffix(source)}.json"
    if not path.is_file():
        return None
    try:
        att = read_attribution(path)
    except (OSError, ValueError):
        return None
    return {
        "n_episodes": len(att.episodes),
        "n_differ": sum(1 for e in att.episodes if not e.agrees),
    }


def _edge_folder(attribution_root: Path) -> Path:
    return attribution_root / EDGE_DIR


def _edge_file_name(grid_hash: str, first: int | None) -> str:
    return f"edge-{grid_hash}{'' if first is None else f'-first{first}'}.json"


def _edge_header(path: Path) -> EdgeFileOut:
    m = _EDGE_FILE.fullmatch(path.name)
    assert m is not None
    base = {"name": path.name, "grid_hash": m[1], "first": int(m[2]) if m[2] else None}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EdgeFileOut(
            **base,
            source=str(raw["source"]),
            arms=[str(a) for a in raw["arms"]],
            premiums=[float(p) for p in raw["premiums"]],
            n_seeds=int(raw["n_seeds"]),
            n_episodes=int(raw["n_episodes"]),
            reproduction=str(raw["reproduction"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return EdgeFileOut(**base, error=f"unreadable: {type(exc).__name__}")


def _grid_files(runs_root: Path) -> dict[str, list[Path]]:
    """Every ``grid-<hash>.json`` under the runs root by hash, the job queue's aside."""
    out: dict[str, list[Path]] = {}
    if not runs_root.is_dir():
        return out
    for p in sorted(runs_root.rglob(GRID_PATTERN)):
        gh = p.stem.removeprefix("grid-")
        if JOBS_DIR not in p.parts and p.is_file() and _HASH.fullmatch(gh):
            out.setdefault(gh, []).append(p)
    return out


def _grid_path(runs_root: Path, grid_hash: str, source: str) -> Path | None:
    """The grid's record: one off the synthetic world sits under its source's folder."""
    found = _grid_files(runs_root).get(grid_hash, [])
    real = [p for p in found if source in p.relative_to(runs_root).parts]
    pool = real if source not in ("", "synthetic") and real else found
    return pool[0] if pool else None


def edge_list(attribution_root: Path, runs_root: Path) -> EdgeListOut:
    """The edge files under ``<attribution root>/edge``, newest grid first, and the grids
    under the runs root with the commands that write theirs."""
    folder = _edge_folder(attribution_root)
    paths = (
        [p for p in folder.glob("edge-*.json") if _EDGE_FILE.fullmatch(p.name)]
        if folder.is_dir()
        else []
    )
    paths.sort(key=lambda p: (-p.stat().st_mtime, p.name))
    files = [_edge_header(p) for p in paths]
    have = {f.grid_hash for f in files}
    grids = [
        EdgeGridOut(
            grid_hash=gh,
            key=(key := ps[0].relative_to(runs_root).as_posix()),
            has_edge=gh in have,
            edge_command=EDGE_COMMAND.format(grid=key),
            actionmix_command=ACTIONMIX_COMMAND.format(grid=key),
        )
        for gh, ps in _grid_files(runs_root).items()
    ]
    return EdgeListOut(
        files=files,
        grids=grids,
        folder=f"{EDGE_DIR}/",
        command=EDGE_COMMAND.format(grid="grid-<hash>.json"),
    )


def edge_detail(
    attribution_root: Path, runs_root: Path, grid: str, first: int | None = None
) -> EdgeOut:
    """``edge-<grid>.json`` (or ``-first<N>``) as written. With no ``first``, the full
    read when there is one, else the partial read of the most episodes. ``ValueError``
    for a name that is not a grid hash; ``KeyError`` when no such file is written."""
    from optspread.edge import read_edge

    gh = grid.removeprefix("grid-").removeprefix("edge-").removesuffix(".json")
    if not _HASH.fullmatch(gh):
        raise ValueError(f"{grid!r} is not a grid hash")
    if first is not None and first < 2:
        raise ValueError(f"first must be at least 2, got {first}")
    folder = _edge_folder(attribution_root)
    if first is None and not (folder / _edge_file_name(gh, None)).is_file():
        partial = [
            int(m[2])
            for p in (folder.glob(f"edge-{gh}-first*.json") if folder.is_dir() else [])
            if (m := _EDGE_FILE.fullmatch(p.name)) and m[1] == gh and m[2]
        ]
        first = max(partial) if partial else None
    path = folder / _edge_file_name(gh, first)
    if not path.is_file():
        raise KeyError(gh)
    result = read_edge(path)
    grid_path = _grid_path(runs_root, gh, result.source)
    keys: dict[str, str | None] = {}
    for cell in result.cells:
        run_dir = grid_path.parent / cell.run_id if grid_path is not None else None
        ok = run_dir is not None and (run_dir / MANIFEST_FILENAME).is_file()
        keys[cell.run_id] = run_dir.relative_to(runs_root).as_posix() if ok and run_dir else None
    grid_key = grid_path.relative_to(runs_root).as_posix() if grid_path is not None else None
    counts = {
        c.run_id: n
        for c in result.cells
        if c.reproduction == "differs"
        and (n := _episode_counts(attribution_root, result.source, c.run_id)) is not None
    }
    return EdgeOut(
        name=path.name,
        folder=f"{EDGE_DIR}/",
        edge=_finite_json(result.model_dump(mode="json")),
        run_keys=keys,
        grid_key=grid_key,
        actionmix_command=ACTIONMIX_COMMAND.format(grid=grid_key or f"grid-{gh}.json"),
        episode_counts=counts,
    )
