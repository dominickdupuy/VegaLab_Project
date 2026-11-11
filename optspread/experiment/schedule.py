"""Computing a grid's cells: cost model, plan, a parallel resumable run, progress and ETA.

Lane: Ilan Danial (machinery).

Every cell is an ordinary run directory under ``root``, named by its run id and
written by ``write_run``, so a cell is the same object whichever command produced
it. ``run_cells`` reads back every cell that ``rundir.valid_run`` accepts and
computes the rest, most expensive first, across ``workers`` processes. A killed
run loses only the cells in flight; the same command resumes it. One orchestrator per
root at a time computes: ``run_cells`` and ``run_single`` hold the root's lock
(``rundir.root_lock``) while they compute, so a second one started on the same root while
the first computes stops with ``RootInUseError`` rather than computing the same cells.
The workers never take it. The lock is the parent's, so it does not outlive a parent
that is killed, while that parent's workers may still be finishing their cells.

A worker can die without raising: the kernel's out-of-memory killer on the sweep
machine is the likely one. ``multiprocessing.Pool`` replaces the process and waits
forever for the result it took with it (#30), so the pool is a
``concurrent.futures.ProcessPoolExecutor``, which notices the death and fails every
future in flight with ``BrokenProcessPool``. It also terminates the other workers, so
the cells in flight at that moment are lost together: each is reported as lost, and
one that had already been written in full is read back from disk instead. Every cell
finished before is on disk and stays there. The cells not yet started go on to a
fresh pool, provided the broken one finished at least one cell; a pool that dies
before finishing any is the machine saying it cannot fit the round (too many
workers for its memory, say), and the remaining cells are reported as not run
rather than fed to it again. Either way ``IncompleteGridError`` names every cell
missing, and the same command computes those and only those.

There is no per-cell timeout. A dead worker is detected without one, and a clock
cannot tell a stuck cell from a legitimately slow PPO cell on a loaded machine; a
cell that misfired would cost every cell in flight with it.

Workers are spawned rather than forked, so a worker never inherits a parent's
torch or BLAS thread pools, and the start method is the same on Linux and macOS. Each
is started with one BLAS and OpenMP thread (``SINGLE_THREAD_ENV``), set in the
environment it is spawned with: a BLAS library reads it once, when numpy is imported,
which in a spawned worker is before any initializer could run.

The trainer and the runner (config and policy in, episodes out) are injected, so a
learner, a market or a cost model of your own runs through the same grid
(``optspread.extend``, docs/EXTENDING.md). Both must pickle when ``workers > 1``. A cell
is reused only when its manifest names the trainer asked for (``rundir.valid_run``), so a
stand-in's cell is never taken for the real learner's, and only when it was made with the
same components (``rundir.components_identity``), so a cell of a world of one's own is never
taken for the default world's, nor the reverse; one is not written over the other either.
A reused cell may have been written at another commit: it is reused (the freeze keeps the
numbers of every commit after the cutoff the same) and the plan and the grid record say
which commits they hold.

The grid runs on the generator unless told otherwise (``source``). On the real panel
(``source="optionmetrics"``) every cell is evaluated on the ``eval`` split and a learned
cell is trained on ``train`` (``train.train_on``; the dates of pre-registration A8), so
no cell trains and evaluates on a common date. The source and the split are not in the
config hash, so a real-data cell has the run id of its synthetic twin: real-data runs
live under their own root (``source_root``), a cell on other data is never reused, and
a cell directory that holds another source's run of the same config is refused rather
than overwritten.

The pool pickles ``_execute`` by module path, so it stays module-level here. Moved from
``optspread.orchestrate``, which re-exports it.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from optspread.artifacts import RunRecord, read_run, write_run
from optspread.config import RunConfig
from optspread.data import SourceName
from optspread.experiment.grid import (
    Cell,
    GridConfig,
    GridRecord,
    _commits,
    grid_cells,
    grid_filename,
    grid_record,
    reusable,
)
from optspread.harness import run_episodes, summarize
from optspread.manifest import callable_name, git_sha
from optspread.runs.rundir import (
    Runner,
    expected_data,
    foreign_run,
    root_lock,
    run_id,
    valid_run,
)
from optspread.train import POLICY_FILENAME, Trainer, check_disjoint, train_on, train_ppo

# M1-measured rates, steps/s/core (README, Throughput). Only their ratio enters the ETA,
# which is rescaled by the measured wall clock as cells finish; the absolute values
# give the estimate printed before the first cell is done.
TRAIN_RATE = 1_000.0
EVAL_RATE = 1_800.0


def cell_seconds(config: RunConfig) -> float:
    """Core-seconds at the M1 rates. Training rounds up to whole rollouts, as SB3 does."""
    seconds = config.n_episodes * config.env.episode_days / EVAL_RATE
    if config.train is not None:
        n_steps = config.train.hyperparameters.n_steps
        steps = math.ceil(config.train.total_timesteps / n_steps) * n_steps
        seconds += steps / TRAIN_RATE
    return seconds


def format_seconds(seconds: float) -> str:
    s = round(seconds)
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s"


# -- one run ----------------------------------------------------------------- #


def source_runner(runner: Runner, source: SourceName) -> Runner:
    """The runner that evaluates on ``source``: ``runner`` itself on the generator, the
    harness on the ``eval`` split otherwise. A runner of your own brings
    its own data, so it is refused with another source rather than silently overridden."""
    if source == "synthetic":
        return runner
    if runner is not run_episodes:
        raise ValueError(
            f"a runner of your own brings its own data; --source {source} is for the harness"
        )
    return partial(run_episodes, source=source)


def execute_run(
    config: RunConfig,
    run_dir: Path,
    trainer: Trainer = train_ppo,
    runner: Runner = run_episodes,
    *,
    source: SourceName = "synthetic",
) -> None:
    """Train if the arm is learned, evaluate, and write the run directory.

    The trained policy is saved beside ``run.json`` and hashed into the manifest, so
    the expensive half of a PPO run is kept; the replay gate retrains from the config
    rather than trusting it. The manifest names the trainer, so a stand-in's cell is
    never taken for the real learner's (#31).

    On another ``source`` the run trains on its ``train`` split and is evaluated on
    ``eval``, and the manifest records the split. A run of one's own components records
    them (``sweep.expected_data``). A directory that holds this config's run on other
    data, or made with other components, is refused (``sweep.foreign_run``).
    """
    run = source_runner(runner, source)
    if why := foreign_run(run_dir, config, source, runner):
        raise FileExistsError(why)
    data = expected_data(runner, config, source)  # before anything is trained or written
    run_dir.mkdir(parents=True, exist_ok=True)
    extra: list[Path] = []
    trained_by = None
    if config.train is None:
        episodes = run(config, None)
    else:
        check_disjoint(config)
        policy = train_on(trainer, config, source)
        policy.save(run_dir / POLICY_FILENAME)
        extra.append(run_dir / POLICY_FILENAME)
        trained_by = callable_name(trainer)
        episodes = run(config, policy)
    write_run(
        run_dir,
        config,
        summarize(episodes),
        episodes,
        extra=extra,
        source=source,
        trainer=trained_by,
        data=data,
    )


def run_single(
    config: RunConfig,
    root: Path,
    *,
    trainer: Trainer = train_ppo,
    runner: Runner = run_episodes,
    source: SourceName = "synthetic",
) -> tuple[Path, RunRecord, bool]:
    """One run at ``root/<run id>``: ``(run_dir, record, reused)``.

    Read back when ``sweep.valid_run`` accepts what is on disk (same source, trainer and
    components), computed through ``execute_run`` otherwise: the grid's rule for one cell.
    The run is computed under ``root``'s lock (``rundir.root_lock``), as ``run_cells``
    computes; a run already on disk is read back without it, so a root this account cannot
    add files to still reads back. Under the lock the run is checked again, since another
    orchestrator may have finished it meanwhile.
    """
    run_dir = root / run_id(config)
    trained_by = callable_name(trainer)
    record = valid_run(run_dir, config, source, trainer=trained_by, runner=runner)
    if record is not None:
        return run_dir, record, True
    with root_lock(root):
        record = valid_run(run_dir, config, source, trainer=trained_by, runner=runner)
        if record is not None:
            return run_dir, record, True
        execute_run(config, run_dir, trainer, runner, source=source)
        return run_dir, read_run(run_dir), False


SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextmanager
def single_threaded_workers() -> Iterator[None]:
    """One BLAS and OpenMP thread in every process spawned inside the block.

    A spawned worker imports numpy (and torch) before it runs anything handed to it,
    and the libraries read these variables at that import, so they are set in the
    environment the worker is started with, and this process's own are put back after.
    """
    saved = {name: os.environ.get(name) for name in SINGLE_THREAD_ENV}
    os.environ.update(dict.fromkeys(SINGLE_THREAD_ENV, "1"))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


Task = tuple[RunConfig, Path, Trainer, Runner, SourceName]


def _execute(task: Task) -> tuple[str, float, str]:
    config, run_dir, trainer, runner, source = task
    start = time.perf_counter()
    try:
        execute_run(config, run_dir, trainer, runner, source=source)
    except Exception as exc:  # reported per cell; the other cells keep running
        return run_id(config), time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return run_id(config), time.perf_counter() - start, ""


# -- many runs ---------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Progress:
    """One line per cell: reused from disk, computed, or failed."""

    done: int
    total: int
    cell: Cell
    seconds: float | None  # None when the cell was read back
    eta_seconds: float | None
    mean_pnl: float | None
    error: str = ""

    def render(self) -> str:
        width = len(str(self.total))
        head = f"[{self.done:>{width}}/{self.total}] {self.cell.label}  {self.cell.run_id}"
        if self.error:
            return f"{head}  FAILED after {format_seconds(self.seconds or 0.0)}: {self.error}"
        pnl = f"mean {self.mean_pnl:>10,.0f}" if self.mean_pnl is not None else ""
        if self.seconds is None:
            return f"{head}  {pnl}  reused"
        eta = f"  eta {format_seconds(self.eta_seconds)}" if self.eta_seconds is not None else ""
        return f"{head}  {pnl}  {self.seconds:>7.1f}s{eta}"


Reporter = Callable[[Progress], None]


class IncompleteGridError(RuntimeError):
    """Some cells failed, or were lost with a dead worker or never started after one.

    ``failures`` maps every missing cell's run id to why; ``lost`` lists the ones a
    dead worker took with it. The rest are on disk; the same command resumes.
    """

    def __init__(self, failures: dict[str, str], lost: Sequence[str] = ()) -> None:
        self.failures = failures
        self.lost = tuple(sorted(lost))
        lines = "\n".join(f"  {k}: {v}" for k, v in sorted(failures.items()))
        head = f"{len(failures)} cell(s) failed"
        if self.lost:
            head += f", {len(self.lost)} of them lost with a dead worker"
        super().__init__(f"{head}; rerun to resume:\n{lines}")


# Why a cell is missing when its worker died rather than raised.
LOST = (
    "lost: a worker process died while this cell was in flight (killed, for example out "
    "of memory), and the pool with it"
)
NOT_RUN = (
    "not run: a fresh pool died before finishing any cell, so no further cells were "
    "started on this machine (fewer --workers may fit)"
)


@dataclass(frozen=True, slots=True)
class Plan:
    """What a run of ``run_cells`` will do before it does it."""

    total: int
    reused: int
    to_run: int
    core_seconds: float
    wall_seconds: float
    workers: int
    # Reused cells written at a commit other than this checkout's, ``(sha, cells)``: reused
    # all the same, since the freeze keeps every commit after the cutoff's numbers equal.
    other_commits: tuple[tuple[str, int], ...] = ()

    def render(self) -> str:
        text = (
            f"{self.total} cells: {self.reused} on disk, {self.to_run} to run, "
            f"~{self.core_seconds / 3600:,.2f} core-hours at the M1 rates "
            f"({TRAIN_RATE:,.0f} steps/s training, {EVAL_RATE:,.0f} evaluation), "
            f"~{format_seconds(self.wall_seconds)} wall on {self.workers} worker(s)"
        )
        if self.other_commits:
            shas = ", ".join(f"{sha[:8]} ({n})" for sha, n in self.other_commits)
            text += f"; reused cells written at other commits: {shas}"
        return text


def makespan(costs: Sequence[float], workers: int) -> float:
    """Wall clock of ``costs`` run longest first, each to the next free worker.

    The schedule ``run_cells`` follows, so the estimate respects that one cell
    cannot be split: eight PPO cells on sixteen workers take one cell's time."""
    loads = [0.0] * max(1, workers)
    for cost in sorted(costs, reverse=True):
        loads[loads.index(min(loads))] += cost
    return max(loads)


def plan_cells(
    cells: Sequence[Cell],
    root: Path,
    workers: int = 1,
    *,
    trainer: Trainer | None = train_ppo,
    source: SourceName = "synthetic",
    runner: Runner = run_episodes,
) -> Plan:
    """What ``run_cells`` would reuse and compute with the same arguments."""
    pending: list[Cell] = []
    reused: list[Cell] = []
    for c in cells:
        found = reusable(c, root, trainer=trainer, source=source, runner=runner)
        (pending if found is None else reused).append(c)
    costs = [cell_seconds(c.config) for c in pending]
    head = git_sha()
    others: dict[str, int] = {}
    for sha in _commits(root, reused):
        if sha != head:
            others[sha] = others.get(sha, 0) + 1
    return Plan(
        total=len(cells),
        reused=len(reused),
        to_run=len(pending),
        core_seconds=sum(costs),
        wall_seconds=makespan(costs, workers),
        workers=workers,
        other_commits=tuple(sorted(others.items())),
    )


def run_cells(
    cells: Sequence[Cell],
    root: Path,
    *,
    workers: int = 1,
    trainer: Trainer = train_ppo,
    report: Reporter | None = None,
    runner: Runner = run_episodes,
    source: SourceName = "synthetic",
) -> None:
    """Compute every cell not valid on disk under ``root``; raise if any is still missing.

    With one worker (or one cell) the cells run in this process. Otherwise they run on
    a pool of spawned processes that survives a dead worker (see the module docstring).
    ``source`` says which data every cell runs on (``execute_run``). The cells still to
    compute are computed under ``root``'s lock (``rundir.root_lock``), so a second
    orchestrator on the same root raises ``RootInUseError`` instead of computing them too.
    Cells valid on disk are read back first, without the lock: when every cell is, none is
    taken, so a root this account cannot add files to still reads back. Under the lock the
    pending cells are read back again, since another orchestrator may have finished some.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    source_runner(runner, source)  # refuse a runner of your own on other data now
    root.mkdir(parents=True, exist_ok=True)
    total, done = len(cells), 0

    def read_back(candidates: Sequence[Cell]) -> list[Cell]:
        """Report each of ``candidates`` valid on disk as reused; return the rest, in order."""
        nonlocal done
        pending: list[Cell] = []
        for cell in candidates:
            record = reusable(cell, root, trainer=trainer, source=source, runner=runner)
            if record is None:
                pending.append(cell)
                continue
            done += 1
            if report is not None:
                report(Progress(done, total, cell, None, None, record.mean_pnl))
        return pending

    pending = read_back(cells)
    if not pending:
        return
    with root_lock(root):
        pending = read_back(pending)
        _compute_cells(pending, root, workers, trainer, report, runner, source, done, total)


def _compute_cells(
    pending: list[Cell],
    root: Path,
    workers: int,
    trainer: Trainer,
    report: Reporter | None,
    runner: Runner,
    source: SourceName,
    done: int,
    total: int,
) -> None:
    """Compute ``pending`` (``done`` of ``total`` cells already read back), while
    ``run_cells`` holds the root's lock."""
    # Longest first, so the slow ppo cells do not end up as the tail of a parallel run.
    pending.sort(key=lambda c: cell_seconds(c.config), reverse=True)
    # The ETA is the plan's schedule over the cells not yet finished, rescaled by how
    # long finished cells took against their estimate. Cells in flight count in full,
    # so it errs long by at most one round.
    unfinished = [cell_seconds(c.config) for c in pending]
    predicted, measured = 0.0, 0.0
    failures: dict[str, str] = {}
    lost: list[str] = []

    def task(cell: Cell) -> Task:
        return cell.config, root / cell.run_id, trainer, runner, source

    def finish(cell: Cell, seconds: float | None, error: str) -> None:
        nonlocal done, predicted, measured
        done += 1
        cost = cell_seconds(cell.config)
        unfinished.remove(cost)
        if seconds is not None:
            predicted += cost
            measured += seconds
        eta = makespan(unfinished, workers) * measured / predicted if predicted else None
        mean = None
        if error:
            failures[cell.run_id] = error
        else:
            mean = read_run(root / cell.run_id).mean_pnl
        if report is not None:
            report(Progress(done, total, cell, seconds, eta, mean, error))

    if workers == 1 or len(pending) == 1:
        for cell in pending:
            _, seconds, error = _execute(task(cell))
            finish(cell, seconds, error)
    else:
        queue = deque(pending)
        while queue:
            dead, progressed, broken = _run_pool(queue, min(workers, len(queue)), task, finish)
            for cell in dead:
                if reusable(cell, root, trainer=trainer, source=source, runner=runner) is not None:
                    finish(cell, None, "")  # written in full before its result was lost
                else:
                    lost.append(cell.run_id)
                    finish(cell, None, LOST)
            if broken and not progressed:
                while queue:
                    finish(queue.popleft(), None, NOT_RUN)
    if failures:
        raise IncompleteGridError(failures, lost)


def _run_pool(
    queue: deque[Cell],
    workers: int,
    task: Callable[[Cell], Task],
    finish: Callable[[Cell, float | None, str], None],
) -> tuple[list[Cell], bool, bool]:
    """Feed ``queue`` to one pool, a cell per free worker, until it empties or the pool dies.

    Returns the cells in flight when it died, whether any cell came back from this
    pool, and whether it died. A cell is handed over only when a worker is free, so the
    longest-first order holds and a death costs at most ``workers`` cells.
    """
    with single_threaded_workers():
        return _feed_pool(queue, workers, task, finish)


def _feed_pool(
    queue: deque[Cell],
    workers: int,
    task: Callable[[Cell], Task],
    finish: Callable[[Cell, float | None, str], None],
) -> tuple[list[Cell], bool, bool]:
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
    in_flight: dict[Future[tuple[str, float, str]], Cell] = {}
    dead: list[Cell] = []
    progressed, broken = False, False

    def collect(futures: set[Future[tuple[str, float, str]]]) -> None:
        nonlocal progressed
        for future in futures:
            cell = in_flight.pop(future)
            try:
                _, seconds, error = future.result()
            except BrokenProcessPool:
                dead.append(cell)
                continue
            except Exception as exc:  # the task itself did not reach a worker, say
                seconds, error = 0.0, f"{type(exc).__name__}: {exc}"
            progressed = True
            finish(cell, seconds, error)

    try:
        while queue or in_flight:
            while queue and len(in_flight) < workers and not broken:
                try:
                    future = pool.submit(_execute, task(queue[0]))
                except BrokenProcessPool:
                    broken = True  # died between results; the futures in flight say so
                    break
                in_flight[future] = queue.popleft()
            if not in_flight:
                break
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            collect(finished)
            broken = broken or bool(dead)
            if broken:
                # The pool is gone: every future still in flight resolves now, most
                # with BrokenProcessPool, a few with a result that beat the death.
                collect(wait(in_flight).done)
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return dead, progressed, broken


def run_grid(
    grid: GridConfig,
    root: Path,
    *,
    workers: int = 1,
    trainer: Trainer = train_ppo,
    report: Reporter | None = None,
    runner: Runner = run_episodes,
    source: SourceName = "synthetic",
) -> GridRecord:
    """Run the grid under ``root`` and write ``grid-<hash>.json`` beside its cells.

    The grid file does not name the source (its hash is the design's); the cells'
    manifests do, which is why a real-data grid lives under its own root (``source_root``).
    """
    run_cells(
        grid_cells(grid),
        root,
        workers=workers,
        trainer=trainer,
        report=report,
        runner=runner,
        source=source,
    )
    record = grid_record(grid, root, trainer=trainer, source=source, runner=runner)
    (root / grid_filename(grid)).write_text(
        record.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    return record
