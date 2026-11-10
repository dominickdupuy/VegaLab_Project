"""Runs launched from the dashboard, the queue: a bounded pool of workers and its jobs.

Lane: Ilan Danial (machinery).

Jobs run in a bounded process pool so a long run never blocks the server; each worker
(``optspread.dashboard.jobs.worker``) writes its progress to files the queue reads.

The pool outlives a dead worker (#30). A worker killed outright (out of memory, say)
breaks a ``ProcessPoolExecutor`` for good: every job on it fails with
``BrokenProcessPool``, and so would every job submitted after. The manager notices
on its next call, builds a fresh pool, fails the jobs that had started (each says
its worker died) and puts the ones that had not back on the new pool. The queue is
capped (``max_pending`` jobs queued or running; ``QueueFullError`` past it, which
the server turns into a 429), and a job can be cancelled: a queued one never runs,
and a running one is asked to stop through a flag file, read between episodes and,
while a PPO job trains, every few hundred training steps (``train.STOP_CHECK_STEPS``), so
a cancelled training run gives its worker back at once. One that finishes anyway has its
result discarded. A run directory it had already written stays: it is an ordinary run,
possibly another job's too.

The pool's workers are spawned, as the orchestrator's are: the server is a multithreaded
process, and forking one is how a child inherits a lock some other thread held.

Moved from ``optspread.jobs``, which re-exports it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import secrets
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from optspread.dashboard.jobs.spec import (
    JOBS_DIR,
    JobSpec,
    available_sources,
    ppo_available,
    to_run_config,
)
from optspread.dashboard.jobs.worker import _Progress, execute_job

MAX_PENDING = 8  # jobs queued or running at once; a launch past it is refused, not queued

# -- the queue -------------------------------------------------------------------- #

JobState = Literal["queued", "running", "done", "failed", "cancelled"]

# What a job that was on the pool when a worker died says, unless it never started.
WORKER_DIED = (
    "BrokenProcessPool: a worker process died while this job was on it (killed, for "
    "example out of memory); the pool was rebuilt and later jobs run on the new one"
)


class QueueFullError(RuntimeError):
    """``max_pending`` jobs are already queued or running. The server answers 429."""


class JobNotCancellableError(RuntimeError):
    """The job has already finished, failed or been cancelled."""


@dataclass(slots=True)
class Job:
    job_id: str
    spec: JobSpec
    created_at: str
    future: Future[str] = field(repr=False)
    error: str = ""
    cancelled: bool = False  # asked to stop; whatever it returns is discarded
    pool: int = 0  # which pool it was handed to: JobManager.rebuilds at the time
    relaunched: bool = False


@dataclass(frozen=True, slots=True)
class JobStatus:
    job_id: str
    state: JobState
    phase: str
    done: int
    total: int
    run_id: str
    run_key: str  # the run directory relative to the runs root, once written
    created_at: str
    policy: str
    error: str
    log_tail: tuple[str, ...]
    source: str = "synthetic"

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


JobFunction = Callable[[str, str, str], str]


class JobManager:
    """A bounded pool of workers and the jobs it has been given, newest first.

    ``executor`` is the pool to start on; a broken one is replaced by ``factory()``,
    by default a process pool of ``workers``. ``target`` is the function a job runs,
    ``execute_job``; a test swaps in one whose worker dies.
    """

    def __init__(
        self,
        runs_root: Path,
        workers: int = 2,
        executor: Executor | None = None,
        *,
        max_pending: int = MAX_PENDING,
        factory: Callable[[], Executor] | None = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError(f"max_pending must be at least 1, got {max_pending}")
        self.runs_root = runs_root
        self.jobs_dir = runs_root / JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.factory: Callable[[], Executor] = (
            factory
            if factory is not None
            else lambda: ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
        )
        self.executor = executor if executor is not None else self.factory()
        self.max_pending = max_pending
        self.target: JobFunction = execute_job
        self.jobs: dict[str, Job] = {}
        self.rebuilds = 0  # pools replaced after a dead worker
        self._lock = threading.RLock()  # the server calls in from a thread pool

    def _launch(self, job_id: str, spec: JobSpec) -> tuple[Future[str], int]:
        """Hand the job to the pool, rebuilding it first if it is already broken."""
        args = (job_id, spec.model_dump_json(), str(self.runs_root))
        try:
            return self.executor.submit(self.target, *args), self.rebuilds
        except BrokenProcessPool:
            self._rebuild()
            return self.executor.submit(self.target, *args), self.rebuilds

    def _rebuild(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.executor = self.factory()
        self.rebuilds += 1

    def _recover(self) -> None:
        """After a dead worker: a new pool, the started jobs failed, the rest relaunched.

        A job counts as started once its worker has written its progress file. One
        that had not is relaunched once; if its pool dies again it is failed too, so
        a job that kills its worker on start cannot take pool after pool with it.
        """
        hit = [
            j
            for j in self.jobs.values()
            if not j.error
            and not j.cancelled
            and j.future.done()
            and not j.future.cancelled()
            and isinstance(j.future.exception(), BrokenProcessPool)
        ]
        if any(j.pool == self.rebuilds for j in hit):  # not a pool already replaced
            self._rebuild()
        for job in hit:
            started = _Progress(self.jobs_dir, job.job_id).state.exists()
            if started or job.relaunched:
                job.error = WORKER_DIED
                continue
            job.future, job.pool = self._launch(job.job_id, job.spec)
            job.relaunched = True

    def pending(self) -> int:
        """Jobs queued or running: what the cap counts."""
        return sum(not j.future.done() for j in self.jobs.values())

    def submit(self, spec: JobSpec) -> str:
        if spec.policy == "ppo" and not ppo_available():
            raise ValueError("the PPO arm is not available in this checkout")
        if spec.source not in available_sources():
            raise ValueError(f"the {spec.source} data is not available here (set VEGALAB_DATA)")
        to_run_config(spec)  # refuse here, synchronously, anything the worker would refuse
        with self._lock:
            self._recover()
            if self.pending() >= self.max_pending:
                raise QueueFullError(
                    f"the job queue is full ({self.max_pending} jobs queued or running); "
                    "launch again when one finishes, or cancel one"
                )
            job_id = secrets.token_hex(6)
            future, pool = self._launch(job_id, spec)
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            self.jobs[job_id] = Job(job_id, spec, stamp, future, pool=pool)
        return job_id

    def cancel(self, job_id: str) -> JobStatus:
        """Cancel a queued job, or ask a running one to stop and discard what it returns."""
        with self._lock:
            self._recover()
            job = self.jobs[job_id]
            if job.cancelled or job.future.done():
                raise JobNotCancellableError(f"job {job_id} is already {self.status(job_id).state}")
            job.cancelled = True
            if not job.future.cancel():
                _Progress(self.jobs_dir, job_id).cancel_flag.touch()
            return self.status(job_id)

    def status(self, job_id: str, tail: int = 20) -> JobStatus:
        with self._lock:
            self._recover()
            job = self.jobs[job_id]
            future = job.future
        progress = _Progress(self.jobs_dir, job_id)
        try:
            state: dict[str, Any] = json.loads(progress.state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        try:
            lines = progress.log_path.read_text(encoding="utf-8").splitlines()[-tail:]
        except OSError:
            lines = []
        run_key = ""
        if job.cancelled:
            status: JobState = "cancelled"
        elif job.error:
            status = "failed"
        elif future.done():
            exc = future.exception()
            if exc is not None:
                job.error = "".join(traceback.format_exception_only(exc)).strip()
                status = "failed"
            else:
                status, run_key = "done", future.result()
        else:
            status = "running" if state else "queued"
        phase = str(state.get("phase", "queued"))
        if status in ("failed", "cancelled"):
            phase = status
        return JobStatus(
            job_id=job_id,
            state=status,
            phase=phase,
            done=int(state.get("done", 0)),
            total=int(state.get("total", job.spec.n_episodes)),
            run_id=str(state.get("run_id", "")),
            run_key=run_key,
            created_at=job.created_at,
            policy=job.spec.policy,
            error=job.error,
            log_tail=tuple(lines),
            source=job.spec.source,
        )

    def all(self) -> tuple[JobStatus, ...]:
        with self._lock:
            self._recover()
            ids = list(reversed(list(self.jobs)))
        return tuple(self.status(j) for j in ids)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
