"""The run index: where runs and docs live, one row per run directory, and one run opened.

Lane: Ilan Danial (machinery).

The dashboard only shows what the harness recorded, so every number on it traces
back to a run. Every loader here reads a file the harness wrote, and nothing here
computes a result that is not already on disk: the risk figures, the paired
comparison and the premium curve are arithmetic on the episodes a run recorded.

A run directory that does not read cleanly is listed with what is wrong rather
than dropped. A browser that silently skipped broken runs would make the results
folder look healthier than it is, which is the failure the manifest gate exists to
prevent.

The row cache (``_ROWS``) and the grid index (``_grid_index``) are this module's
state. ``optspread.dashboard_data`` re-exports the functions that use them, not the
state, so a caller on either path reads and fills the same cache.

Moved from ``optspread.dashboard_data``, which re-exports it; ``optspread.m2gate``
looks ``scan_runs`` up there by name.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from optspread.artifacts import RUN_FILENAME, RunRecord, read_run
from optspread.experiment.sweep import SWEEP_FILENAME
from optspread.fileio import env_path
from optspread.manifest import MANIFEST_FILENAME, DataRecord, RunManifest, file_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_ENV = "OPTSPREAD_RUNS"
DOCS_ENV = "OPTSPREAD_DOCS"


def runs_root(env: Mapping[str, str] | None = None) -> Path:
    """``$OPTSPREAD_RUNS`` if set, else ``runs/`` at the repository root."""
    return env_path(RUNS_ENV, REPO_ROOT / "runs", env)


def docs_dir(env: Mapping[str, str] | None = None) -> Path:
    """``$OPTSPREAD_DOCS`` if set, else ``docs/`` at the repository root."""
    return env_path(DOCS_ENV, REPO_ROOT / "docs", env)


# -- the run index ------------------------------------------------------------ #


@dataclass(frozen=True, slots=True)
class RunRow:
    """One run directory as the browser lists it: provenance, headline, and health.

    ``problems`` is empty for a run that reads cleanly. A row with problems is still
    listed, so a broken run is visible rather than missing. A run is OK only when the
    manifest records ``run.json`` among its artifacts and every recorded digest
    matches; one that records nothing, or not the results file, is UNVERIFIED, since
    nothing ties the numbers shown to the numbers the manifest was written over.

    ``data_source`` and ``panel`` are the manifest's data record (the source, and the
    id of the panel a real-data run read); both are empty for a manifest written
    before the source was recorded.
    """

    run_id: str
    path: Path
    config_hash: str
    seed: int | None
    policy: str
    premium: float | None
    git_sha: str
    git_dirty: bool | None
    created_at: str
    n_episodes: int | None
    mean_pnl: float | None
    se_pnl: float | None
    t_stat: float | None
    mean_cost: float | None
    mean_trades: float | None
    sweep: str = ""  # the parent sweep's directory name, when the run is a sweep cell
    digests_ok: bool | None = None  # None: the manifest records no artifacts
    recorded: tuple[str, ...] = ()  # the artifact paths the manifest records
    problems: tuple[str, ...] = ()
    data_source: str = ""  # empty: the manifest predates the data record
    panel: str = ""  # DataRecord.panel_id; empty unless the run read the real panel

    @property
    def commit(self) -> str:
        return self.git_sha[:8]

    @property
    def source(self) -> str:
        """The recorded source, else ``synthetic``: a manifest without the record was
        written by a build whose ``harness.build_env`` ran only the generator."""
        return self.data_source or "synthetic"

    @property
    def status(self) -> str:
        if self.problems:
            return "INVALID"
        if self.digests_ok is False:
            return "DIGEST MISMATCH"
        if not self.digests_ok or RUN_FILENAME not in self.recorded:
            return "UNVERIFIED"
        return "OK"


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _first_line(exc: Exception) -> str:
    text = str(exc).strip()
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        where = ".".join(str(p) for p in errors[0]["loc"]) if errors else ""
        return f"{where}: {errors[0]['msg']}" if errors else text
    return text.splitlines()[0] if text else type(exc).__name__


@dataclass(frozen=True, slots=True)
class ArtifactCheck:
    path: str
    recorded: str
    actual: str  # empty when the file is missing

    @property
    def ok(self) -> bool:
        return bool(self.actual) and self.actual == self.recorded


def check_artifacts(run_dir: Path, artifacts: Mapping[str, str]) -> tuple[ArtifactCheck, ...]:
    """Recompute every digest the manifest recorded and compare."""
    checks: list[ArtifactCheck] = []
    for rel, recorded in sorted(artifacts.items()):
        target = run_dir / rel
        actual = file_sha256(target) if target.is_file() else ""
        checks.append(ArtifactCheck(rel, recorded, actual))
    return tuple(checks)


def _data_source(raw: Any) -> tuple[str, str]:
    """``(source, panel id)`` from the manifest's data record; empty for a manifest
    without one. A record that does not parse keeps the source it names (the manifest
    check has listed it as a problem), so a broken record never reads as synthetic."""
    data = _get(raw, "data")
    if data is None:
        return "", ""
    try:
        record = DataRecord.model_validate(data)
    except ValidationError:
        return str(_get(data, "source") or "unreadable"), ""
    return record.source, record.panel_id


GRID_PATTERN = "grid-*.json"  # an orchestrated grid's record, beside its cells


@lru_cache(maxsize=64)
def _grid_index(folder: Path, stamp: tuple[tuple[str, float], ...]) -> dict[str, str]:
    """``{run id: grid hash}`` for the grid records in ``folder`` (``stamp`` keys the cache)."""
    out: dict[str, str] = {}
    for name, _ in stamp:
        try:
            raw: Any = json.loads((folder / name).read_text())
            grid_hash = str(raw["grid_hash"])
            for cell in raw["cells"]:
                out.setdefault(str(cell["run_id"]), grid_hash)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def grid_of(run_dir: Path) -> str:
    """The hash of the orchestrated grid that lists this run as a cell, or empty.

    A grid's cells sit beside its ``grid-<hash>.json``; the pilot and the sweep can share
    a root, so the record is read rather than the folder name taken.
    """
    folder = run_dir.parent
    try:
        stamp = tuple(sorted((p.name, p.stat().st_mtime) for p in folder.glob(GRID_PATTERN)))
    except OSError:
        return ""
    return _grid_index(folder, stamp).get(run_dir.name, "") if stamp else ""


def _row_stamp(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """What a run's row is read from: each file in the run directory, and the grid and
    sweep records beside it (its sweep), by name, modification time and size."""
    out: list[tuple[str, int, int]] = []
    for folder, keep in ((run_dir, None), (run_dir.parent, True)):
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for e in entries:
            beside = e.name == SWEEP_FILENAME or fnmatchcase(e.name, GRID_PATTERN)
            if keep and not beside:
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            if e.is_file():
                out.append((f"{'../' if keep else ''}{e.name}", st.st_mtime_ns, st.st_size))
    return tuple(sorted(out))


_ROWS: dict[Path, tuple[tuple[tuple[str, int, int], ...], RunRow]] = {}
_MAX_ROWS = 4096


def load_run_row(run_dir: Path) -> RunRow:
    """Read one run directory without trusting it: every failure becomes a problem.

    The row is kept while every file it was read from stands (``_row_stamp``), so the
    index of a few hundred runs is a stat per file rather than a read and a digest each;
    a file written, replaced or edited is read and checked again."""
    stamp = _row_stamp(run_dir)
    held = _ROWS.get(run_dir)
    if held is not None and held[0] == stamp:
        return held[1]
    row = _read_run_row(run_dir)
    _keep_row(run_dir, stamp, row)
    return row


def _keep_row(run_dir: Path, stamp: tuple[tuple[str, int, int], ...], row: RunRow) -> None:
    if len(_ROWS) >= _MAX_ROWS:
        _ROWS.clear()
    _ROWS[run_dir] = (stamp, row)


def _read_run_row(run_dir: Path) -> RunRow:
    return _read_run_dir(run_dir)[0]


def _read_run_dir(
    run_dir: Path,
) -> tuple[RunRow, Any, RunRecord | None, tuple[ArtifactCheck, ...]]:
    """The row, and what it was built from, each file read and hashed once: the manifest
    JSON as parsed (``{}`` when unreadable), the record (None when missing or unreadable)
    and the digest checks."""
    problems: list[str] = []
    try:
        raw: Any = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    except (OSError, ValueError) as exc:
        raw = {}
        problems.append(f"manifest.json unreadable: {_first_line(exc)}")
    if raw:
        try:
            RunManifest.model_validate(raw).validate_complete()
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            problems.append(f"manifest: {_first_line(exc)}")

    record: RunRecord | None = None
    if (run_dir / RUN_FILENAME).exists():
        try:
            record = read_run(run_dir)
        except (OSError, ValueError) as exc:  # includes SchemaVersionError
            problems.append(f"run.json: {_first_line(exc)}")
    else:
        problems.append("run.json is missing")

    results = _get(raw, "results") or {}
    artifacts = _get(raw, "artifacts")
    checks = (
        check_artifacts(run_dir, {str(k): str(v) for k, v in artifacts.items()})
        if isinstance(artifacts, dict)
        else ()
    )
    seed = _as_int(_get(raw, "config", "seed"))
    data_source, panel = _data_source(raw)
    row = RunRow(
        run_id=str(_get(raw, "run_id") or (record.run_id if record else run_dir.name)),
        path=run_dir,
        config_hash=str(_get(raw, "config_hash") or (record.config_hash if record else "")),
        seed=seed if seed is not None else (record.seed if record else None),
        policy=str(_get(raw, "config", "policy") or (record.policy if record else "")),
        premium=_as_float(_get(raw, "config", "market", "iv_premium_vol_pts")),
        git_sha=str(_get(raw, "git_sha") or ""),
        git_dirty=_get(raw, "git_dirty") if isinstance(_get(raw, "git_dirty"), bool) else None,
        created_at=str(_get(raw, "created_at") or ""),
        n_episodes=record.n_episodes if record else _as_int(_get(raw, "config", "n_episodes")),
        mean_pnl=record.mean_pnl if record else _as_float(_get(results, "mean_pnl")),
        se_pnl=record.se_pnl if record else _as_float(_get(results, "se_pnl")),
        t_stat=record.t_stat if record else _as_float(_get(results, "t_stat")),
        mean_cost=record.mean_cost if record else _as_float(_get(results, "mean_cost")),
        mean_trades=record.mean_trades if record else _as_float(_get(results, "mean_trades")),
        sweep=run_dir.parent.name
        if (run_dir.parent / SWEEP_FILENAME).exists()
        else grid_of(run_dir),
        digests_ok=all(c.ok for c in checks) if checks else None,
        recorded=tuple(c.path for c in checks),
        problems=tuple(problems),
        data_source=data_source,
        panel=panel,
    )
    return row, raw, record, checks


def is_matrix_manifest(run_dir: Path) -> bool:
    """The pilot's own manifest: at the pilot root, hashing ``.npy`` matrices, no ``run.json``.

    ``pilot.write_pilot_matrices`` writes it so the matrices can be checked; it
    describes no run, so the run index leaves it out rather than listing it INVALID.
    """
    if (run_dir / RUN_FILENAME).exists():
        return False
    try:
        artifacts: Any = json.loads((run_dir / MANIFEST_FILENAME).read_text()).get("artifacts")
    except (OSError, ValueError, AttributeError):
        return False
    return (
        isinstance(artifacts, dict)
        and bool(artifacts)
        and all(str(rel).endswith(".npy") for rel in artifacts)
    )


def scan_runs(root: Path) -> tuple[RunRow, ...]:
    """Every run directory under ``root`` holding a ``manifest.json``, newest first.

    The pilot's matrix manifest is not a run and is skipped (``is_matrix_manifest``).
    """
    if not root.is_dir():
        return ()
    found = (m.parent for m in root.rglob(MANIFEST_FILENAME) if m.is_file())
    rows = [load_run_row(d) for d in found if not is_matrix_manifest(d)]
    return tuple(sorted(rows, key=lambda r: (r.created_at, r.run_id), reverse=True))


@dataclass(frozen=True, slots=True)
class RunFacets:
    """The distinct values each filter can take, in display order."""

    config_hashes: tuple[str, ...]
    seeds: tuple[int, ...]
    policies: tuple[str, ...]
    premiums: tuple[float, ...]
    commits: tuple[str, ...]
    sweeps: tuple[str, ...]


def run_facets(rows: Iterable[RunRow]) -> RunFacets:
    rows = tuple(rows)
    return RunFacets(
        config_hashes=tuple(sorted({r.config_hash for r in rows if r.config_hash})),
        seeds=tuple(sorted({r.seed for r in rows if r.seed is not None})),
        policies=tuple(sorted({r.policy for r in rows if r.policy})),
        premiums=tuple(sorted({r.premium for r in rows if r.premium is not None})),
        commits=tuple(sorted({r.commit for r in rows if r.commit})),
        sweeps=tuple(sorted({r.sweep for r in rows if r.sweep})),
    )


def filter_runs(
    rows: Iterable[RunRow],
    *,
    config_hashes: Sequence[str] = (),
    seeds: Sequence[int] = (),
    policies: Sequence[str] = (),
    premiums: Sequence[float] = (),
    commits: Sequence[str] = (),
    sweeps: Sequence[str] = (),
    text: str = "",
) -> tuple[RunRow, ...]:
    """Rows matching every non-empty filter. An empty filter does not filter.

    ``text`` is a case-insensitive substring match on the run id, config hash and
    commit, for pasting a hash from a gate report or a figure caption.
    """
    needle = text.strip().lower()
    out: list[RunRow] = []
    for r in rows:
        if config_hashes and r.config_hash not in config_hashes:
            continue
        if seeds and r.seed not in seeds:
            continue
        if policies and r.policy not in policies:
            continue
        if premiums and r.premium not in premiums:
            continue
        if commits and r.commit not in commits:
            continue
        if sweeps and r.sweep not in sweeps:
            continue
        if needle and not any(needle in s.lower() for s in (r.run_id, r.config_hash, r.git_sha)):
            continue
        out.append(r)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class RunDetail:
    """One run opened: the manifest as written, the record, and per-episode arrays."""

    row: RunRow
    manifest: dict[str, Any]
    record: RunRecord | None
    pnl: NDArray[np.float64]
    cost: NDArray[np.float64]
    trades: NDArray[np.float64]
    artifacts: tuple[ArtifactCheck, ...]


def load_run_detail(run_dir: Path) -> RunDetail:
    """The row, plus the manifest, record and digest checks it was read from.

    The run directory is read once: the manifest, ``run.json`` and every recorded
    artifact are each read or hashed one time, and the row is built from that same
    read. It is stored in the row cache under the directory's stamp, as
    ``load_run_row`` stores it, unless the cache already holds an equal row under
    that stamp, which is then the one returned. So the row shown always agrees with
    the digest checks beside it, even after an edit the stamp cannot see (same size,
    modification time put back), where a cached row would have been shown.
    """
    stamp = _row_stamp(run_dir)
    row, manifest, record, checks = _read_run_dir(run_dir)
    held = _ROWS.get(run_dir)
    if held is not None and held[0] == stamp and held[1] == row:
        row = held[1]
    else:
        _keep_row(run_dir, stamp, row)
    episodes = record.episodes if record else ()
    return RunDetail(
        row=row,
        manifest=manifest if isinstance(manifest, dict) else {},
        record=record,
        pnl=np.asarray([e.pnl for e in episodes], dtype=np.float64),
        cost=np.asarray([e.cost for e in episodes], dtype=np.float64),
        trades=np.asarray([e.n_trades for e in episodes], dtype=np.float64),
        artifacts=checks,
    )


@dataclass(frozen=True, slots=True)
class RiskStats:
    """Per-episode risk figures for one run, computed from the episodes in its ``run.json``.

    Every figure is over episodes, the grain the run records. ``max_drawdown`` is the
    worst peak-to-trough of cumulative P&L with the episodes taken in seed order: it
    describes the sample, not a trading account, because the episodes are independent
    paths rather than consecutive ones. A figure the sample cannot support is None.
    """

    n: int
    mean: float | None
    se: float | None
    sortino: float | None  # mean over downside deviation against zero
    cvar5: float | None  # mean of the worst 5% of episodes (at least one)
    max_drawdown: float | None
    hit_rate: float | None  # share of episodes with positive P&L
    turnover: float | None  # mean trades per episode


def risk_stats(pnl: NDArray[np.float64], trades: NDArray[np.float64] | None = None) -> RiskStats:
    n = len(pnl)
    if n == 0:
        return RiskStats(0, None, None, None, None, None, None, None)
    downside = float(np.sqrt(np.mean(np.minimum(pnl, 0.0) ** 2)))
    worst = np.sort(pnl)[: max(1, int(np.ceil(0.05 * n)))]
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    return RiskStats(
        n=n,
        mean=float(pnl.mean()),
        se=float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        sortino=float(pnl.mean() / downside) if downside > 0.0 else None,
        cvar5=float(worst.mean()),
        max_drawdown=float((np.maximum.accumulate(equity) - equity).max()),
        hit_rate=float((pnl > 0.0).mean()),
        turnover=float(trades.mean()) if trades is not None and len(trades) else None,
    )


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    else:
        out[prefix] = value


def flatten_config(config: Any) -> dict[str, Any]:
    """``{"market.sigma": 0.2, ...}``: a config as dotted keys, for a diff a person can read."""
    out: dict[str, Any] = {}
    _flatten("", config if isinstance(config, dict) else {}, out)
    return out


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Two runs side by side, and their paired difference on the seeds they share.

    Runs on the same seeds saw the same paths, so ``b - a`` per seed is a paired
    difference and its standard error is the honest one. Unshared seeds are counted,
    not used.
    """

    a: RunDetail
    b: RunDetail
    config_diff: tuple[tuple[str, Any, Any], ...]  # (key, a's value, b's value)
    shared_seeds: tuple[int, ...]
    diff_mean: float | None
    diff_se: float | None

    @property
    def diff_t(self) -> float | None:
        if self.diff_mean is None or not self.diff_se:
            return None
        return self.diff_mean / self.diff_se


def compare_runs(a: RunDetail, b: RunDetail) -> RunComparison:
    fa, fb = flatten_config(a.manifest.get("config")), flatten_config(b.manifest.get("config"))
    diff = tuple(
        (k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)
    )
    pa = {e.seed: e.pnl for e in a.record.episodes} if a.record else {}
    pb = {e.seed: e.pnl for e in b.record.episodes} if b.record else {}
    shared = tuple(sorted(set(pa) & set(pb)))
    d = np.asarray([pb[s] - pa[s] for s in shared], dtype=np.float64)
    return RunComparison(
        a=a,
        b=b,
        config_diff=diff,
        shared_seeds=shared,
        diff_mean=float(d.mean()) if len(d) else None,
        diff_se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
    )


@dataclass(frozen=True, slots=True)
class CurvePoint:
    policy: str
    premium: float
    mean_pnl: float
    se_pnl: float
    run_id: str


def premium_curve(rows: Iterable[RunRow]) -> dict[str, tuple[CurvePoint, ...]]:
    """Mean P&L along the planted-premium axis, per arm, from runs that read cleanly.

    Only meaningful within one sweep (one base config, paired seeds); the caller
    passes the rows of a single sweep. Within an arm the points are in premium order.
    """
    by_arm: dict[str, list[CurvePoint]] = {}
    for r in rows:
        if r.status != "OK" or r.premium is None or r.mean_pnl is None or r.se_pnl is None:
            continue
        by_arm.setdefault(r.policy, []).append(
            CurvePoint(r.policy, r.premium, r.mean_pnl, r.se_pnl, r.run_id)
        )
    return {arm: tuple(sorted(pts, key=lambda p: p.premium)) for arm, pts in sorted(by_arm.items())}
