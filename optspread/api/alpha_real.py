"""API payloads and routes for the Synthetic alpha real-data test (#46, item 5).

Lane: Dominick Dupuy.

A benchmark's policies played on every real SPX window of A8's evaluation dates
(``optspread.alpha.realtest``), beside the same benchmark's synthetic held-out result.
Where the panel is absent the read says so in a sentence and a launch is refused with it;
nothing falls back to synthetic chains. A plug-in benchmark whose plug-in is no longer
registered, or has changed since, is refused the same way, with its reason. The routes
are added with ``add_api_route`` by ``register``, which the server calls once before it
mounts the site.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from optspread.alpha.dataset import DatasetError
from optspread.alpha.jobs import PoolUnavailableError, Rebuild
from optspread.alpha.realjobs import BenchRef, RealJobManager, find_benchmark
from optspread.alpha.realtest import (
    REAL_SPLIT,
    PluginChangedError,
    RealDataUnavailableError,
    RealTest,
    load_real,
    plugin_for,
    real_available,
    real_dates,
    real_path,
    unavailable_message,
)
from optspread.api._base import Out
from optspread.api.jobs import JobOut, job_out
from optspread.dashboard.jobs.queue import JobStatus, QueueFullError

if TYPE_CHECKING:
    from fastapi import FastAPI


class AlphaRealOut(Out):
    available: bool  # whether this machine has the real panel
    message: str  # why not, in a sentence; empty when it does
    split: str
    first_date: str  # the dates a test reads (A8's evaluation range)
    last_date: str
    result: RealTest | None  # the test, once it has run
    # why this benchmark cannot be tested here (its plug-in gone or changed); empty if it can
    refused: str = ""
    source: str


class AlphaRealJobOut(JobOut):
    result: RealTest | None = None


def alpha_real(runs: Path, data: Path | None, spec: BenchRef) -> AlphaRealOut:
    """The real-data test of the benchmark ``spec`` names, and whether it can run here.
    ``FileNotFoundError`` when that benchmark has not been run."""
    bench = find_benchmark(runs, spec)
    try:
        plugin_for(bench)
        refused = ""
    except PluginChangedError as exc:
        refused = f"{str(exc)[0].upper()}{str(exc)[1:]}."
    path = real_path(runs, bench)
    try:
        result: RealTest | None = load_real(path)
    except (OSError, ValueError):
        result = None
    available = real_available(data)
    first, last = real_dates()
    return AlphaRealOut(
        available=available,
        message="" if available else unavailable_message(data),
        split=REAL_SPLIT,
        first_date=first.isoformat(),
        last_date=last.isoformat(),
        result=result,
        refused=refused,
        source=f"<runs root>/{path.relative_to(runs).as_posix()}",
    )


def alpha_real_job(status: JobStatus, runs: Path) -> AlphaRealJobOut:
    result = None
    if status.state == "done" and status.run_key:
        result = load_real(runs / status.run_key)
    return AlphaRealJobOut(**job_out(status).model_dump(), result=result)


def register(
    app: FastAPI,
    runs: Path,
    data: Path,
    executor: Callable[[], Executor],
    rebuild: Rebuild | None = None,
) -> RealJobManager:
    """Add the three routes to ``app``; the tests reach the queue through the return."""
    from fastapi import HTTPException

    manager = RealJobManager(runs, data, executor, rebuild)

    def spec_of(raw: dict[str, object]) -> BenchRef:
        try:
            return BenchRef.model_validate(raw)
        except ValidationError as exc:
            raise HTTPException(422, [e["msg"] for e in exc.errors()]) from exc

    def alpha_real_read(
        dataset: str, arm: str, total_timesteps: int | None = None, n_seeds: int = 1
    ) -> AlphaRealOut:
        """The benchmark's real-data test, if it has run, and whether it can run here."""
        spec = spec_of(
            {"dataset": dataset, "arm": arm, "total_timesteps": total_timesteps, "n_seeds": n_seeds}
        )
        try:
            return alpha_real(runs, data, spec)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    def alpha_real_start(spec: dict[str, object]) -> AlphaRealJobOut:
        """Score a benchmark's policies on the real SPX evaluation windows, as a job."""
        parsed = spec_of(spec)
        try:
            job_id = manager.submit(parsed)
        except RealDataUnavailableError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (FileNotFoundError, DatasetError) as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except QueueFullError as exc:
            raise HTTPException(429, str(exc)) from exc
        except PoolUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        return alpha_real_job(manager.status(job_id), runs)

    def alpha_real_status(job_id: str) -> AlphaRealJobOut:
        if job_id not in manager.jobs:
            raise HTTPException(404, f"no real-data test job {job_id!r}")
        return alpha_real_job(manager.status(job_id), runs)

    app.add_api_route("/api/alpha/real", alpha_real_read, methods=["GET"])
    app.add_api_route("/api/alpha/real", alpha_real_start, methods=["POST"], status_code=201)
    app.add_api_route("/api/alpha/real/jobs/{job_id}", alpha_real_status, methods=["GET"])
    return manager
