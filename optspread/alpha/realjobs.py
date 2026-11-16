"""Real-data tests launched from the dashboard: the worker and a small queue.

Lane: Dominick Dupuy.

A test rebuilds a learned arm's policies (one training per seed) and plays every real
window, so it runs for minutes on the server's process pool, as a benchmark does
(``optspread.alpha.jobs``), and reports through the same progress files. A test names
the benchmark it scores by the benchmark's own spec (``BenchRef``, the fields of a
``BenchSpec`` without the check that the arm is registered now, so a plug-in that has
since gone is refused with its reason rather than as an unknown arm).
"""

from __future__ import annotations

import re
import secrets
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optspread.alpha import plugins as plug
from optspread.alpha.benchmark import (
    ARMS,
    BENCH_DIR,
    BENCH_FILE,
    MAX_SEEDS,
    Benchmark,
    bench_path,
    load_benchmark,
)
from optspread.alpha.dataset import load_dataset
from optspread.alpha.jobs import BenchSpec
from optspread.alpha.realtest import (
    RealDataUnavailableError,
    plugin_for,
    real_available,
    real_path,
    run_real_test,
    stages,
    unavailable_message,
)
from optspread.config import LEARNED_ARMS
from optspread.dashboard.jobs.queue import JobState, JobStatus, QueueFullError
from optspread.dashboard.jobs.spec import JOBS_DIR, ppo_available
from optspread.dashboard.jobs.worker import _Progress
from optspread.fileio import read_json

MAX_PENDING = 2  # real-data tests queued or running at once: each retrains the arm


class BenchRef(BaseModel):
    """Which benchmark a real-data test scores: its dataset, arm, budget and seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(pattern=r"^[0-9a-f]{16}$")
    arm: str = Field(pattern=plug.NAME_PATTERN)
    total_timesteps: int | None = Field(default=None, ge=2048, le=5_000_000)
    n_seeds: int = Field(default=1, ge=1, le=MAX_SEEDS)

    @model_validator(mode="after")
    def _budget(self) -> BenchRef:
        # a built-in arm's kind is known; a plug-in's is its benchmark's (it may be gone)
        if self.arm in LEARNED_ARMS and self.total_timesteps is None:
            raise ValueError("total_timesteps is set exactly when the arm is learned")
        scripted = self.arm in ARMS and self.arm not in LEARNED_ARMS
        if scripted and (self.total_timesteps is not None or self.n_seeds != 1):
            raise ValueError("a scripted arm is deterministic: no budget, n_seeds is 1")
        return self


def find_benchmark(root: Path, spec: BenchRef | BenchSpec) -> Benchmark:
    """The benchmark ``spec`` names, or a ``FileNotFoundError`` saying it has not run.

    A plug-in's benchmarks are named by its source digest: the one its code has now first,
    else the newest written under another digest or while it was registered (which
    ``realtest.plugin_for`` then refuses, saying why)."""
    found = None if spec.arm in ARMS else plug.find(spec.arm)
    if spec.arm in ARMS or found is not None:
        path = bench_path(root, spec.dataset, spec.arm, spec.total_timesteps, spec.n_seeds)
        if path.is_file():
            return load_benchmark(path)
    if spec.arm not in ARMS:
        name = re.compile(
            rf"{spec.dataset}-{spec.arm}\.[0-9a-f]{{8}}-{spec.total_timesteps or 0}-s{spec.n_seeds}"
        )
        others = sorted(
            (
                p
                for p in (root / BENCH_DIR).glob(f"{spec.dataset}-{spec.arm}.*/{BENCH_FILE}")
                if name.fullmatch(p.parent.name)
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if others:
            return load_benchmark(others[0])
    raise FileNotFoundError(
        f"no benchmark of {spec.arm} on dataset {spec.dataset} with this budget and these "
        "seeds; run it first"
    )


def execute_real(job_id: str, spec_json: str, runs_root: str, data_root: str) -> str:
    """Run one real-data test; return its ``realtest.json`` path relative to ``runs_root``."""
    root = Path(runs_root)
    progress = _Progress(root / JOBS_DIR, job_id)
    spec = BenchSpec.model_validate_json(spec_json)
    bench = find_benchmark(root, spec)
    ds = load_dataset(root, spec.dataset)
    rel = real_path(root, bench).relative_to(root).as_posix()
    total = stages(spec.arm, spec.n_seeds)
    done = 0
    progress.write(done=0, total=total, phase="starting", run_id=ds.hash, key=rel)

    def log(line: str) -> None:
        nonlocal done
        done = min(done + 1, total)
        progress.log(line)
        progress.write(done=done, total=total, phase="running", run_id=ds.hash, key=rel)

    run_real_test(bench, ds, root, Path(data_root), log=log)
    progress.write(done=total, total=total, phase="written", run_id=ds.hash, key=rel)
    return rel


@dataclass(slots=True)
class RealJob:
    job_id: str
    spec: BenchSpec
    created_at: str
    future: Future[str] = field(repr=False)


class RealJobManager:
    """Real-data test jobs on an executor shared with the run queue; never shuts it down."""

    def __init__(self, runs_root: Path, data_root: Path, executor: Callable[[], Executor]) -> None:
        self.runs_root = runs_root
        self.data_root = data_root
        self.jobs_dir = runs_root / JOBS_DIR
        self.executor = executor
        self.jobs: dict[str, RealJob] = {}
        self._lock = threading.RLock()

    def submit(self, ref: BenchRef | BenchSpec) -> str:
        """Queue the test; ``RealDataUnavailableError`` without a panel, ``FileNotFoundError``
        without the benchmark, ``ValueError`` without the learner or for a plug-in gone or
        changed since its benchmark (``PluginChangedError``), ``QueueFullError``."""
        if not real_available(self.data_root):
            raise RealDataUnavailableError(unavailable_message(self.data_root))
        if ref.arm in LEARNED_ARMS and not ppo_available():
            raise ValueError(f"the {ref.arm.upper()} arm is not available in this checkout")
        plugin_for(find_benchmark(self.runs_root, ref))
        spec = BenchSpec.model_validate(ref.model_dump())  # registered and unchanged by now
        with self._lock:
            if sum(not j.future.done() for j in self.jobs.values()) >= MAX_PENDING:
                raise QueueFullError(
                    f"{MAX_PENDING} real-data tests are already queued or running; "
                    "launch again when one finishes"
                )
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            job_id = "r" + secrets.token_hex(6)
            args = (job_id, spec.model_dump_json(), str(self.runs_root), str(self.data_root))
            future = self.executor().submit(execute_real, *args)
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            self.jobs[job_id] = RealJob(job_id, spec, stamp, future)
        return job_id

    def status(self, job_id: str, tail: int = 20) -> JobStatus:
        job = self.jobs[job_id]
        finished = job.future.done()  # first: a finished future means final files
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
            total=int(state.get("total", stages(job.spec.arm, job.spec.n_seeds))),
            run_id=str(state.get("run_id", "")),
            run_key=run_key,
            created_at=job.created_at,
            policy=job.spec.arm,
            error=error,
            log_tail=tuple(lines),
            source="optionmetrics",
        )
