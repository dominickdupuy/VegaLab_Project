"""The alpha job queues on the run queue's shared pool: cancelled and dead jobs, broken pools.

The pool is the run queue's, which cancels pending futures when it is rebuilt or shut
down and breaks when a worker dies; an alpha job's status must say so, never raise.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

from optspread.alpha import benchmark as bm
from optspread.alpha.dataset import SynthSpec, build_dataset
from optspread.alpha.generators import ConstVrpParams
from optspread.alpha.jobs import (
    CANCELLED,
    WORKER_DIED,
    AlphaJobManager,
    BenchSpec,
    PoolUnavailableError,
    submit_to_pool,
)
from optspread.alpha.realjobs import RealJob, RealJobManager

SPEC = SynthSpec(generator=ConstVrpParams(premium_pts=5.0), n_days=10, n_train=2, n_heldout=4)


class FixedFuture(Executor):
    """Hands back the future ``make`` returns, already settled; never runs anything."""

    def __init__(self, make: Callable[[], Future[str]]) -> None:
        self.make = make

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[str]:
        return self.make()


class Broken(Executor):
    """A pool that refuses every submit, as a broken process pool does."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[str]:
        raise BrokenProcessPool("a worker died")


def _cancelled() -> Future[str]:
    f: Future[str] = Future()
    f.cancel()
    return f


def _died() -> Future[str]:
    f: Future[str] = Future()
    f.set_exception(BrokenProcessPool("a worker died"))
    return f


@pytest.fixture()
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(bm, "FRESH_N", 16)
    return build_dataset(SPEC, tmp_path).hash


@pytest.mark.behaviour
@pytest.mark.parametrize(
    ("make", "state", "error"),
    [(_cancelled, "cancelled", CANCELLED), (_died, "failed", WORKER_DIED)],
)
def test_a_cancelled_or_dead_alpha_job_reads_as_such(
    tmp_path: Path,
    dataset: str,
    make: Callable[[], Future[str]],
    state: str,
    error: str,
) -> None:
    pool = FixedFuture(make)
    benches = AlphaJobManager(tmp_path, lambda: pool)
    job_id = benches.submit(BenchSpec(dataset=dataset, arm="flat", total_timesteps=None))
    status = benches.status(job_id)  # once raised CancelledError: a 500 on every poll
    assert (status.state, status.phase, status.error) == (state, state, error)
    assert "launch it again" in status.error


@pytest.mark.behaviour
@pytest.mark.parametrize(
    ("make", "state", "error"),
    [(_cancelled, "cancelled", CANCELLED), (_died, "failed", WORKER_DIED)],
)
def test_a_cancelled_or_dead_real_data_job_reads_as_such(
    tmp_path: Path,
    dataset: str,
    make: Callable[[], Future[str]],
    state: str,
    error: str,
) -> None:
    real = RealJobManager(tmp_path, tmp_path / "nodata", lambda: ThreadPoolExecutor(1))
    spec = BenchSpec(dataset=dataset, arm="flat", total_timesteps=None)
    real.jobs["r1"] = RealJob("r1", spec, "2026-01-01T00:00:00+00:00", make())
    status = real.status("r1")
    assert (status.state, status.phase, status.error) == (state, state, error)


@pytest.mark.behaviour
def test_a_broken_pool_is_rebuilt_once_or_refused(tmp_path: Path, dataset: str) -> None:
    fresh = ThreadPoolExecutor(1)
    pools: list[Executor] = [Broken()]
    asked: list[Executor] = []

    def rebuild(old: Executor) -> None:
        asked.append(old)
        pools[0] = fresh

    benches = AlphaJobManager(tmp_path, lambda: pools[0], rebuild)
    job_id = benches.submit(BenchSpec(dataset=dataset, arm="flat", total_timesteps=None))
    assert len(asked) == 1 and isinstance(asked[0], Broken)
    assert benches.jobs[job_id].future.result(timeout=60).endswith("benchmark.json")
    fresh.shutdown()

    # no way to a fresh pool, or a rebuilt pool that is broken too: refused, not a 500
    with pytest.raises(PoolUnavailableError, match="restart the server"):
        submit_to_pool(Broken, None, str)
    with pytest.raises(PoolUnavailableError, match="even after a rebuild"):
        submit_to_pool(Broken, lambda _: None, str)
    shut = ThreadPoolExecutor(1)
    shut.shutdown()
    with pytest.raises(PoolUnavailableError):
        submit_to_pool(lambda: shut, None, str)
