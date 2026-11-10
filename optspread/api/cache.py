"""The dashboard's answer cache: a rendered JSON body kept while the files it read stand.

Lane: Ilan Danial (machinery).

Most of what the dashboard serves is a pure function of its query and of a few files:
the runs under the runs root, the panel's tables, the reports in docs/. Rebuilding it on
every request is what made the pages slow, and the panel is the worst of it, since it
sits in a synced cloud folder whose first read of a file can take minutes. So a body is
kept, keyed by the route, its query and a *stamp* of the files it depends on (each
file's path, modification time and size): when any of them changes the key changes and
the body is built again. Nothing is ever served from a file that changed since.

Two tiers. In memory, the last ``size`` bodies, least recently used out first. On disk,
for the routes whose inputs are slow to read (the panel), one file per key under
``<folder>/<route>/``: a restarted server, or one whose cloud folder dropped its local
copies, still answers at once. The disk tier is opt-in (``folder`` None keeps it off),
so a test server never writes beside the repository. Writes are atomic, and a body that
does not read back is ignored and rebuilt.

The ETag of a body is its key's digest: a browser that sends it back (``If-None-Match``)
is told 304 without the body being rebuilt or resent. Moved from ``optspread.webcache``,
which re-exports it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from optspread.fileio import atomic_write_bytes

Stamp = tuple[tuple[str, int, int], ...]


def file_stamp(paths: Iterable[Path]) -> Stamp:
    """Each file's path, modification time (ns) and size; (path, -1, -1) when absent."""
    out: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            st = path.stat()
            out.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(path), -1, -1))
    return tuple(out)


def tree_stamp(root: Path, skip: Iterable[str] = (".jobs",)) -> str:
    """A digest of every file under ``root`` (relative path, mtime, size), ``skip`` left out.

    A run written, removed or edited anywhere under the runs root changes it; reading it
    is a directory walk and a stat per file, milliseconds for a few hundred runs.
    """
    skipped = set(skip)
    h = hashlib.sha256()
    if not root.is_dir():
        return h.hexdigest()
    for here, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skipped)
        for name in sorted(files):
            path = os.path.join(here, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            h.update(f"{os.path.relpath(path, root)}\0{st.st_mtime_ns}\0{st.st_size}\n".encode())
    return h.hexdigest()


def answer_key(route: str, query: Mapping[str, Any], deps: Any) -> str:
    """The cache key: the route, its query and the stamp of what it read, digested."""
    text = json.dumps(
        [route, sorted((str(k), repr(v)) for k, v in query.items()), deps], default=str
    )
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class Answers:
    """Rendered bodies by key: in memory (LRU), and on disk for the routes that ask."""

    def __init__(self, folder: Path | None = None, size: int = 256) -> None:
        self.folder = folder
        self.size = size
        self._memory: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _disk_path(self, route: str, key: str) -> Path | None:
        if self.folder is None:
            return None
        safe = route.strip("/").replace("/", "_") or "root"
        return self.folder / safe / f"{key}.json"

    def _remember(self, key: str, body: bytes) -> None:
        with self._lock:
            self._memory[key] = body
            self._memory.move_to_end(key)
            while len(self._memory) > self.size:
                self._memory.popitem(last=False)

    def get(
        self,
        route: str,
        query: Mapping[str, Any],
        deps: Any,
        build: Callable[[], bytes],
        *,
        disk: bool = False,
    ) -> tuple[bytes, str]:
        """``(body, etag)``: the kept body for this key, else ``build()``'s, then kept."""
        key = answer_key(route, query, deps)
        with self._lock:
            body = self._memory.get(key)
            if body is not None:
                self._memory.move_to_end(key)
                self.hits += 1
                return body, f'"{key}"'
        path = self._disk_path(route, key) if disk else None
        if path is not None and path.is_file():
            try:
                body = path.read_bytes()
                json.loads(body)  # a torn or foreign file is rebuilt, never served
            except (OSError, ValueError):
                body = None
            if body is not None:
                self._remember(key, body)
                self.hits += 1
                return body, f'"{key}"'
        self.misses += 1
        body = build()
        self._remember(key, body)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # one try: a file another reader holds open on Windows is left as it is,
                # and the request is not kept waiting on a retry
                atomic_write_bytes(path, body, attempts=1)
            except OSError:
                pass  # the disk tier is a convenience; the answer stands without it
        return body, f'"{key}"'
