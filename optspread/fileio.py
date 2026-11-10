"""Small file helpers: atomic writes, file digests, env-var paths, forgiving JSON.

Lane: Ilan Danial (machinery).

Standard library only, and nothing here imports the rest of ``optspread``, so any
module can use it without an import cycle. Text is read and written as UTF-8 unless a
caller names another encoding, as every run writer and reader here does, so a helper
never falls back to the locale's encoding (cp1252 on a Windows machine).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def atomic_write_text(
    path: Path,
    text: str,
    *,
    newline: str | None = None,
    encoding: str = "utf-8",
    attempts: int = 20,
    delay: float = 0.05,
) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the new one, never half.

    The text goes, in ``encoding`` and with ``newline`` as ``open`` takes it, to a
    temporary file beside ``path``, which ``os.replace`` then moves over it. Windows
    refuses the replace (``PermissionError``) while another process has ``path`` open,
    so it is tried ``attempts`` times, ``delay`` seconds apart, before the error is
    raised. The temporary file never outlives a failed write.
    """

    def write(tmp: Path) -> None:
        with tmp.open("x", encoding=encoding, newline=newline) as fh:
            fh.write(text)

    _replace_atomically(path, write, attempts, delay)


def atomic_write_bytes(path: Path, data: bytes, *, attempts: int = 20, delay: float = 0.05) -> None:
    """Write ``data`` to ``path`` so a reader sees the old file or the new one, never half.

    As ``atomic_write_text``, for bytes: a temporary file beside ``path``, moved over
    it by ``os.replace``, retried ``attempts`` times ``delay`` seconds apart while
    Windows refuses it (``PermissionError``), and removed if the write fails.
    """

    def write(tmp: Path) -> None:
        with tmp.open("xb") as fh:
            fh.write(data)

    _replace_atomically(path, write, attempts, delay)


def _replace_atomically(
    path: Path, write: Callable[[Path], None], attempts: int, delay: float
) -> None:
    """``write`` a fresh temporary file beside ``path``, then replace ``path`` with it."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        write(tmp)
        for attempt in range(attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of the file's bytes, read in chunks."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def changed_files(root: Path, recorded: Mapping[str, str]) -> tuple[str, ...]:
    """Paths in ``recorded`` (relative to ``root``) that are missing or no longer match
    their recorded SHA-256, in sorted order."""
    return tuple(
        rel
        for rel, digest in sorted(recorded.items())
        if not (root / rel).is_file() or sha256_file(root / rel) != digest
    )


def env_path(name: str, default: Path, env: Mapping[str, str] | None = None) -> Path:
    """``$name`` as a user-expanded path when set and non-empty, else ``default``.

    ``env`` replaces ``os.environ``, so a caller can be tested without touching it.
    """
    value = (os.environ if env is None else env).get(name, "")
    return Path(value).expanduser() if value else default


def read_json(path: Path, default: Any = None, *, encoding: str = "utf-8") -> Any:
    """The parsed JSON in ``path``, read in ``encoding``, or ``default`` when the file is
    missing, cannot be read or decoded, or is not valid JSON."""
    try:
        return json.loads(path.read_text(encoding=encoding))
    except (OSError, ValueError):  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        return default
