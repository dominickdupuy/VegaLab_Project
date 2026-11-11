"""The results-root lock (optspread.runs.rundir.root_lock) and the orchestrators that take it.

One orchestrator at a time computes under a root; a second one with runs to compute is
refused with RootInUseError naming the lock file, and writes nothing. The lock is released
however the body ends. Nothing is locked when every run is already on disk, and a lock
that cannot be taken for any reason but a live holder costs a warning, never the run. The
lock file is never taken for a run, and the results are the same with or without it.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.dashboard.runindex import scan_runs
from optspread.experiment import schedule, sweep
from optspread.experiment.grid import GridConfig, grid_cells
from optspread.experiment.schedule import Progress, run_grid, run_single
from optspread.experiment.sweep import SweepConfig, run_sweep
from optspread.replay import find_runs
from optspread.runs import rundir
from optspread.runs.rundir import LOCK_FILENAME, RootInUseError, root_lock, valid_run

TINY = RunConfig(
    seed=7,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(7, 14)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(base=TINY, premiums=(0.0, 2.0), arms=("flat", "delta_seller"), n_seeds=2)
SWEEP = SweepConfig(base=TINY, premiums=(0.0, 2.0), arms=("flat", "delta_seller"))
SINGLE = TINY.model_copy(update={"policy": "delta_seller"})

# Another process holding the root's lock: the operating system's lock on the same file,
# taken with filelock alone so the holder starts in a fraction of a second.
HOLD = """
import sys, time
from pathlib import Path
from filelock import FileLock
lock = FileLock(Path(sys.argv[1]) / sys.argv[2])
lock.acquire()
print("held", flush=True)
time.sleep(60)
"""


@contextlib.contextmanager
def held_by_another_process(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(root), LOCK_FILENAME], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        yield
    finally:
        holder.kill()
        holder.wait()


def short_lock(monkeypatch: pytest.MonkeyPatch, timeout: float = 0.2) -> None:
    """The orchestrators' lock with a short timeout, so a refusal takes a fraction of a second."""
    fast = partial(root_lock, timeout=timeout)
    monkeypatch.setattr(schedule, "root_lock", fast)
    monkeypatch.setattr(sweep, "root_lock", fast)


def forbid_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any orchestrator that takes the lock fails the test."""

    def taken(root: Path, timeout: float = 0.0) -> contextlib.AbstractContextManager[None]:
        raise AssertionError(f"the lock on {root} was taken")

    monkeypatch.setattr(schedule, "root_lock", taken)
    monkeypatch.setattr(sweep, "root_lock", taken)


def no_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestrators without any lock, as before it existed."""
    monkeypatch.setattr(schedule, "root_lock", lambda root, timeout=0.0: contextlib.nullcontext())
    monkeypatch.setattr(sweep, "root_lock", lambda root, timeout=0.0: contextlib.nullcontext())


def orchestrate(name: str, root: Path, report: list[Progress] | None = None) -> None:
    if name == "grid":
        run_grid(GRID, root, report=None if report is None else report.append)
    elif name == "sweep":
        run_sweep(SWEEP, root)
    else:
        run_single(SINGLE, root)


def results(root: Path) -> dict[str, Any]:
    """Every file under ``root`` but the lock file, manifests without their clock."""
    out: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == LOCK_FILENAME:
            continue
        rel = path.relative_to(root).as_posix()
        if path.name == "manifest.json":
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("created_at", None)
            out[rel] = data
        else:
            out[rel] = path.read_bytes()
    return out


@pytest.mark.behaviour
@pytest.mark.spawns_processes
@pytest.mark.parametrize("name", ["grid", "sweep", "single"])
def test_a_second_orchestrator_is_refused_while_another_process_holds_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    root = tmp_path / "root"
    short_lock(monkeypatch)
    with held_by_another_process(root), pytest.raises(RootInUseError) as refused:
        orchestrate(name, root)
    assert str(root / LOCK_FILENAME) in str(refused.value)
    assert [p.name for p in root.iterdir() if p.name != LOCK_FILENAME] == []


@pytest.mark.behaviour
@pytest.mark.parametrize("planted", [RuntimeError, KeyboardInterrupt])
def test_the_lock_is_released_when_the_body_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: type[BaseException]
) -> None:
    made: list[Any] = []
    real = rundir._HARD_LOCK

    def kept(*args: Any, **kwargs: Any) -> Any:
        made.append(real(*args, **kwargs))  # kept alive: only an explicit release frees it
        return made[-1]

    monkeypatch.setattr(rundir, "_HARD_LOCK", kept)
    with pytest.raises(planted), root_lock(tmp_path, timeout=0.2):
        raise planted("planted")
    assert len(made) == 1 and not made[0].is_locked
    with root_lock(tmp_path, timeout=0.2):  # a second lock object: refused if still held
        pass


@pytest.mark.behaviour
@pytest.mark.parametrize("name", ["grid", "sweep", "single"])
def test_nothing_is_locked_when_every_run_is_already_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    root = tmp_path / "root"
    orchestrate(name, root)
    before = results(root)
    forbid_lock(monkeypatch)
    orchestrate(name, root)  # reads every run back; a root it could not lock still works
    assert results(root) == before


@pytest.mark.behaviour
def test_cells_another_orchestrator_finished_while_waiting_are_read_back_not_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"

    @contextlib.contextmanager
    def lock_after_another_run(where: Path, timeout: float = 0.0) -> Iterator[None]:
        # While this orchestrator waited, another one computed every cell.
        no_lock(monkeypatch)
        run_grid(GRID, where)
        yield

    monkeypatch.setattr(schedule, "root_lock", lock_after_another_run)
    seen: list[Progress] = []
    run_grid(GRID, root, report=seen.append)
    assert len(seen) == len(grid_cells(GRID))
    assert all(p.seconds is None for p in seen)  # every one reused, none computed


@pytest.mark.behaviour
def test_a_lock_file_that_cannot_be_opened_costs_a_warning_not_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    (root / LOCK_FILENAME).mkdir(parents=True)  # no file can be opened at the lock's path
    run_grid(GRID, root)
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert err.startswith("warning: could not take the lock") and str(root / LOCK_FILENAME) in err
    assert all(valid_run(root / c.run_id, c.config) is not None for c in grid_cells(GRID))


@pytest.mark.behaviour
def test_a_lock_refused_by_permissions_costs_a_warning_not_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Denied:
        def __init__(self, path: Path, **kwargs: Any) -> None:
            self.path = path

        def acquire(self) -> None:
            raise PermissionError(errno.EACCES, "Permission denied", str(self.path))

        def release(self) -> None:
            raise AssertionError("released a lock that was never taken")

    monkeypatch.setattr(rundir, "_HARD_LOCK", Denied)
    root = tmp_path / "root"
    run_sweep(SWEEP, root)
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "PermissionError" in err and LOCK_FILENAME in err
    assert len(find_runs(root)) == len(SWEEP.premiums) * len(SWEEP.arms)


@pytest.mark.behaviour
def test_a_lock_file_left_in_a_root_is_never_taken_for_a_run(tmp_path: Path) -> None:
    root = tmp_path / "root"
    run_grid(GRID, root)
    (root / LOCK_FILENAME).write_text("", encoding="utf-8")  # as Linux and macOS leave it
    cells = sorted(root / c.run_id for c in grid_cells(GRID))
    assert find_runs(root) == cells
    rows = scan_runs(root)
    assert sorted(r.path for r in rows) == cells
    assert not any(r.problems for r in rows)


@pytest.mark.invariant
def test_results_are_byte_identical_with_and_without_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_grid(GRID, tmp_path / "locked")
    no_lock(monkeypatch)
    run_grid(GRID, tmp_path / "unlocked")
    locked, unlocked = results(tmp_path / "locked"), results(tmp_path / "unlocked")
    assert locked == unlocked
    assert sum(name.endswith("run.json") for name in locked) == len(grid_cells(GRID))


@pytest.mark.behaviour
@pytest.mark.skipif(sys.platform == "win32", reason="the lock file stays only on Linux and macOS")
def test_the_lock_file_left_behind_can_be_locked_by_every_account(tmp_path: Path) -> None:
    old = os.umask(0o077)
    try:
        with root_lock(tmp_path, timeout=0.2):
            pass
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / LOCK_FILENAME).stat().st_mode) == 0o666


@pytest.mark.behaviour
@pytest.mark.skipif(sys.platform == "win32", reason="flock and its soft fallback are Unix")
def test_a_filesystem_without_flock_never_turns_a_left_file_into_a_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import filelock._unix as unix_lock

    def missing(fd: int) -> bool:
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(unix_lock, "_lock_fd_nonblocking", missing)
    (tmp_path / LOCK_FILENAME).write_text("", encoding="utf-8")  # left by an earlier run
    for _ in range(2):
        with root_lock(tmp_path, timeout=0.2):
            pass
    assert capsys.readouterr().err.count("warning: could not take the lock") == 2
