"""The work the run page starts and polls, on one thread of the server, keyed by kind and run key.

Lane: Ilan Danial (machinery).

``Tasks`` runs the API's re-simulations of one run (attribution, capital, its action mix,
replay); ``TaskState`` is what a poll of one reads back. Neither is served as it is. Moved
from ``optspread.webviews``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskState:
    state: str  # "idle" | "running" | "done" | "failed"
    error: str = ""
    seconds: float | None = None


class Tasks:
    """Work the API starts and the page polls: one thread, keyed by kind and run key.

    Not ``optspread.dashboard.jobs``: these read a run and write nowhere but the attribution root,
    so they need no process pool and no run directory of their own.
    """

    def __init__(self, workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="optspread-task")
        self._futures: dict[tuple[str, str], tuple[Future[Any], float]] = {}
        self._finished: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()  # a done future runs its callback inside submit

    def submit(self, kind: str, key: str, work: Callable[[], Any]) -> None:
        """Start ``work`` unless the same kind is already running for this key."""
        with self._lock:
            held = self._futures.get((kind, key))
            if held is not None and not held[0].done():
                return
            start = time.perf_counter()
            future = self._pool.submit(work)
            future.add_done_callback(lambda _: self._finish((kind, key)))
            self._futures[(kind, key)] = (future, start)

    def _finish(self, task: tuple[str, str]) -> None:
        with self._lock:
            self._finished[task] = time.perf_counter()

    def state(self, kind: str, key: str) -> tuple[TaskState, Any]:
        with self._lock:
            held = self._futures.get((kind, key))
            end = self._finished.get((kind, key))
        if held is None:
            return TaskState(state="idle"), None
        future, start = held
        if not future.done():
            return TaskState(state="running", seconds=time.perf_counter() - start), None
        seconds = (end or time.perf_counter()) - start
        exc = future.exception()
        if exc is not None:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return TaskState(state="failed", error=text, seconds=seconds), None
        return TaskState(state="done", seconds=seconds), future.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
