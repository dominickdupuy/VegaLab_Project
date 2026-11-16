"""Benchmarks launched from the dashboard: a spec, a small queue, and the worker.

Lane: Dominick Dupuy.

A benchmark trains and evaluates for minutes, so it runs on the server's process pool
(``optspread.dashboard.jobs.queue.JobManager.executor``, read at each submit, so a pool
the manager rebuilt after a dead worker is the one used) and reports through the same
progress files a launched run writes (``optspread.dashboard.jobs.worker._Progress``).
Building a dataset is quick and is not a job.
"""

from __future__ import annotations

import secrets
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from optspread.alpha.benchmark import (
    MAX_SEEDS,
    Arm,
    bench_path,
    check_arm,
    is_learned,
    run_benchmark,
    stages,
)
from optspread.alpha.dataset import load_dataset
from optspread.alpha.plugins import NAME_PATTERN
from optspread.alpha.sweep import AlphaSweepSpec, edge_knob, run_sweep, sweep_path
from optspread.config import LEARNED_ARMS
from optspread.dashboard.jobs.queue import JobState, JobStatus, QueueFullError
from optspread.dashboard.jobs.spec import JOBS_DIR, ppo_available
from optspread.dashboard.jobs.worker import _Progress
from optspread.fileio import read_json

DEFAULT_TIMESTEPS = 100_000
MAX_PENDING = 4  # benchmarks and sweeps queued or running at once: each trains learners
DEFAULT_SEEDS = 5


class BenchSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(pattern=r"^[0-9a-f]{16}$")
    arm: Arm = Field(default="ppo", pattern=NAME_PATTERN)
    total_timesteps: int | None = Field(default=DEFAULT_TIMESTEPS, ge=2048, le=5_000_000)
    n_seeds: int = Field(default=1, ge=1, le=MAX_SEEDS)

    @field_validator("arm")
    @classmethod
    def _known(cls, arm: str) -> str:
        return check_arm(arm)

    @model_validator(mode="after")
    def _budget(self) -> BenchSpec:
        if is_learned(self.arm) != (self.total_timesteps is not None):
            raise ValueError("total_timesteps is set exactly when the arm is learned")
        if not is_learned(self.arm) and self.n_seeds != 1:
            raise ValueError("a scripted arm is deterministic: n_seeds is 1")
        return self

    @property
    def stages(self) -> int:
        return stages(self.arm, self.n_seeds)


def execute_bench(job_id: str, spec_json: str, runs_root: str) -> str:
    """Run one benchmark; return its ``benchmark.json`` path relative to ``runs_root``."""
    root = Path(runs_root)
    progress = _Progress(root / JOBS_DIR, job_id)
    spec = BenchSpec.model_validate_json(spec_json)
    ds = load_dataset(root, spec.dataset)
    target = bench_path(root, ds.hash, spec.arm, spec.total_timesteps, spec.n_seeds)
    rel = target.relative_to(root).as_posix()
    done = 0
    progress.write(done=0, total=spec.stages, phase="starting", run_id=ds.hash, key=rel)

    def log(line: str) -> None:
        nonlocal done
        done = min(done + 1, spec.stages)
        progress.log(line)
        progress.write(done=done, total=spec.stages, phase="running", run_id=ds.hash, key=rel)

    if (root / rel).is_file():
        progress.log("this benchmark is already on disk; not recomputed")
    run_benchmark(ds, spec.arm, spec.total_timesteps, root, log=log, n_seeds=spec.n_seeds)
    progress.write(done=spec.stages, total=spec.stages, phase="written", run_id=ds.hash, key=rel)
    return rel


def execute_sweep(job_id: str, spec_json: str, runs_root: str) -> str:
    """Run one sweep; return its ``sweep.json`` path relative to ``runs_root``."""
    root = Path(runs_root)
    progress = _Progress(root / JOBS_DIR, job_id)
    spec = AlphaSweepSpec.model_validate_json(spec_json)
    rel = sweep_path(root, spec).relative_to(root).as_posix()
    done = 0
    progress.write(done=0, total=spec.stages, phase="starting", run_id=spec.dataset, key=rel)

    def log(line: str) -> None:
        nonlocal done
        done = min(done + 1, spec.stages)
        progress.log(line)
        progress.write(done=done, total=spec.stages, phase="running", run_id=spec.dataset, key=rel)

    run_sweep(spec, root, log=log)
    progress.write(
        done=spec.stages, total=spec.stages, phase="written", run_id=spec.dataset, key=rel
    )
    return rel


AlphaSpec = BenchSpec | AlphaSweepSpec


@dataclass(slots=True)
class BenchJob:
    job_id: str
    spec: AlphaSpec
    created_at: str
    future: Future[str] = field(repr=False)


class AlphaJobManager:
    """Benchmark jobs on an executor shared with the run queue; never shuts it down."""

    def __init__(self, runs_root: Path, executor: Callable[[], Executor]) -> None:
        self.runs_root = runs_root
        self.jobs_dir = runs_root / JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.jobs: dict[str, BenchJob] = {}
        self._lock = threading.RLock()

    def submit(self, spec: AlphaSpec) -> str:
        if spec.arm in LEARNED_ARMS and not ppo_available():
            raise ValueError(f"the {spec.arm.upper()} arm is not available in this checkout")
        ds = load_dataset(self.runs_root, spec.dataset)  # refuse an unknown dataset here
        if isinstance(spec, AlphaSweepSpec) and spec.axis == "edge":
            edge_knob(ds.spec.generator)  # refuse a generator with no edge size here
        with self._lock:
            if sum(not j.future.done() for j in self.jobs.values()) >= MAX_PENDING:
                raise QueueFullError(
                    f"{MAX_PENDING} benchmarks or sweeps are already queued or running; "
                    "launch again when one finishes"
                )
            job_id = "a" + secrets.token_hex(6)
            args = (job_id, spec.model_dump_json(), str(self.runs_root))
            work = execute_sweep if isinstance(spec, AlphaSweepSpec) else execute_bench
            future = self.executor().submit(work, *args)
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            self.jobs[job_id] = BenchJob(job_id, spec, stamp, future)
        return job_id

    def status(self, job_id: str, tail: int = 20) -> JobStatus:
        job = self.jobs[job_id]
        # The future first: a worker writes its last progress before it returns, so a
        # finished future means the files read next are final, not a stale snapshot.
        finished = job.future.done()
        progress = _Progress(self.jobs_dir, job_id)
        state: dict[str, Any] = read_json(progress.state, {})
        try:
            lines = progress.log_path.read_text(encoding="utf-8").splitlines()[-tail:]
        except OSError:
            lines = []
        error, run_key = "", ""
        if finished:
            exc = job.future.exception()
            if exc is not None:
                error = "".join(traceback.format_exception_only(exc)).strip()
                status: JobState = "failed"
            else:
                status, run_key = "done", job.future.result()
        else:
            status = "running" if state else "queued"
        return JobStatus(
            job_id=job_id,
            state=status,
            phase="failed" if status == "failed" else str(state.get("phase", "queued")),
            done=int(state.get("done", 0)),
            total=int(state.get("total", job.spec.stages)),
            run_id=str(state.get("run_id", "")),
            run_key=run_key,
            created_at=job.created_at,
            policy=job.spec.arm,
            error=error,
            log_tail=tuple(lines),
        )
