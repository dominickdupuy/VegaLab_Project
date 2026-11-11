"""Ilan Danial: the shared file helpers in ``optspread.fileio``.

The job worker's progress file and the API's answer cache are written through the atomic
write; the reuse check compares run files with their recorded digests through
``changed_files``; the runs, docs and data roots come from ``env_path``; the job queue
reads progress through ``read_json``. Each is held here to what its docstring says.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optspread.fileio import (
    atomic_write_bytes,
    atomic_write_text,
    changed_files,
    env_path,
    read_json,
    sha256_file,
)

CHECKOUT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
# Not ASCII, and U+2264 is not in cp1252 (the locale encoding on a Windows machine). No
# newline, which a text write turns into the platform's own.
TEXT = "premium \u2264 2 vol pts, \u03c4\u00b2 = 0, caf\u00e9"


# -- the atomic write ------------------------------------------------------------- #


def _write(kind: str, path: Path, text: str, **kw: object) -> None:
    if kind == "text":
        atomic_write_text(path, text, **kw)  # type: ignore[arg-type]
    else:
        atomic_write_bytes(path, text.encode("utf-8"), **kw)  # type: ignore[arg-type]


@pytest.mark.behaviour
@pytest.mark.parametrize("kind", ["text", "bytes"])
def test_an_atomic_write_replaces_the_whole_file_and_leaves_nothing_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"the old content")
    replace = os.replace
    moves: list[tuple[str, bytes, bytes]] = []

    def watched(src: str | Path, dst: str | Path) -> None:
        # At the moment of the move the target is still the old file, whole, and the new
        # content is complete beside it: a reader sees one or the other, never half.
        moves.append((Path(src).name, Path(src).read_bytes(), Path(dst).read_bytes()))
        replace(src, dst)

    monkeypatch.setattr(os, "replace", watched)
    _write(kind, path, TEXT)
    assert path.read_bytes() == TEXT.encode("utf-8")
    [(name, new, old)] = moves
    assert (new, old) == (TEXT.encode("utf-8"), b"the old content")
    assert name.startswith(".state.json.") and name.endswith(".tmp")
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.behaviour
@pytest.mark.parametrize("kind", ["text", "bytes"])
def test_a_refused_replace_is_retried_and_then_raised_with_the_old_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    replace = os.replace
    refusals = {"left": 2}
    calls: list[str] = []

    def refusing(src: str | Path, dst: str | Path) -> None:
        # Windows refuses to replace a file another process holds open.
        calls.append(Path(src).name)
        if refusals["left"]:
            refusals["left"] -= 1
            raise PermissionError(13, "held open by a reader", str(dst))
        replace(src, dst)

    monkeypatch.setattr(os, "replace", refusing)
    _write(kind, path, "new", attempts=3, delay=0.0)
    assert path.read_bytes() == b"new" and len(calls) == 3
    assert len(set(calls)) == 1, "one temporary file, retried"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]

    refusals["left"], calls[:] = 99, []
    with pytest.raises(PermissionError):
        _write(kind, path, "newer", attempts=4, delay=0.0)
    assert len(calls) == 4
    assert path.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.behaviour
def test_a_write_that_fails_leaves_the_old_file_whole_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"old")
    with pytest.raises(UnicodeEncodeError):  # fails while the temporary file is written
        atomic_write_text(path, TEXT, encoding="ascii")
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]

    calls: list[str | Path] = []

    def gone(src: str | Path, dst: str | Path) -> None:
        calls.append(src)
        raise FileNotFoundError(2, "the folder went away", str(dst))

    monkeypatch.setattr(os, "replace", gone)
    with pytest.raises(FileNotFoundError):  # only a refusal is retried
        atomic_write_bytes(path, b"new", attempts=5, delay=0.0)
    assert len(calls) == 1
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
    with pytest.raises(ValueError, match="attempts"):
        atomic_write_text(path, "new", attempts=0)
    assert path.read_bytes() == b"old"


@pytest.mark.behaviour
@pytest.mark.spawns_processes
def test_text_round_trips_as_utf_8_whatever_the_locale(tmp_path: Path) -> None:
    # Under -X warn_default_encoding, made an error, any open() that falls back to the
    # locale's encoding fails; the bytes on disk must be UTF-8 on every platform.
    script = """
import json, sys
from pathlib import Path
from optspread.fileio import atomic_write_text, read_json
path, text = Path(sys.argv[1]), sys.argv[2]
atomic_write_text(path / "a.txt", text)
atomic_write_text(path / "b.json", json.dumps({"text": text}, ensure_ascii=False))
assert read_json(path / "b.json") == {"text": text}, read_json(path / "b.json")
"""
    done = subprocess.run(
        [
            sys.executable,
            *("-X", "warn_default_encoding", "-W", "error::EncodingWarning"),
            *("-c", script, str(tmp_path), TEXT),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(CHECKOUT), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "a.txt").read_bytes() == TEXT.encode("utf-8")
    assert json.loads((tmp_path / "b.json").read_bytes().decode("utf-8")) == {"text": TEXT}


# -- digests ------------------------------------------------------------------------ #


@pytest.mark.invariant
def test_changed_files_names_the_missing_and_the_changed_and_ignores_the_rest(
    tmp_path: Path,
) -> None:
    for rel, body in {"run.json": b"{}", "sub/policy.zip": b"PK", "b.txt": b"b"}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(body)
    assert sha256_file(tmp_path / "run.json") == hashlib.sha256(b"{}").hexdigest()
    recorded = {rel: sha256_file(tmp_path / rel) for rel in ("run.json", "sub/policy.zip", "b.txt")}
    assert changed_files(tmp_path, recorded) == ()
    (tmp_path / "extra.txt").write_bytes(b"not recorded, not this check's concern")
    assert changed_files(tmp_path, recorded) == ()

    (tmp_path / "sub" / "policy.zip").write_bytes(b"PK!")  # changed under its record
    (tmp_path / "b.txt").unlink()  # missing
    recorded["a.txt"] = recorded["run.json"]  # recorded, never there
    assert changed_files(tmp_path, recorded) == ("a.txt", "b.txt", "sub/policy.zip")
    (tmp_path / "b.txt").mkdir()  # a folder where a recorded file was is not that file
    assert changed_files(tmp_path, recorded) == ("a.txt", "b.txt", "sub/policy.zip")
    assert changed_files(tmp_path, {"run.json": ""}) == ("run.json",)  # no digest, no match


# -- paths from the environment, and forgiving JSON -------------------------------------- #


@pytest.mark.behaviour
def test_env_path_reads_the_variable_or_falls_back_to_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "default"
    assert env_path("ROOT", default, {}) == default  # unset
    assert env_path("ROOT", default, {"ROOT": ""}) == default  # set but empty
    assert env_path("ROOT", default, {"ROOT": "runs/here"}) == Path("runs/here")  # as given
    assert env_path("ROOT", default, {"ROOT": str(tmp_path / "abs")}) == tmp_path / "abs"
    home = Path("~").expanduser()
    assert home != Path("~")
    assert env_path("ROOT", default, {"ROOT": "~"}) == home
    assert env_path("ROOT", default, {"ROOT": "~/runs"}) == home / "runs"
    monkeypatch.setenv("OPTSPREAD_TEST_ROOT", str(tmp_path / "env"))
    assert env_path("OPTSPREAD_TEST_ROOT", default) == tmp_path / "env"  # os.environ
    monkeypatch.delenv("OPTSPREAD_TEST_ROOT")
    assert env_path("OPTSPREAD_TEST_ROOT", default) == default


@pytest.mark.behaviour
def test_read_json_gives_the_default_for_what_it_cannot_read(tmp_path: Path) -> None:
    default: dict[str, int] = {}
    (tmp_path / "ok.json").write_text('{"done": 3, "phase": "\u2264"}', encoding="utf-8")
    assert read_json(tmp_path / "ok.json") == {"done": 3, "phase": "\u2264"}
    assert read_json(tmp_path / "missing.json", default) is default
    (tmp_path / "folder.json").mkdir()  # cannot be read as a file
    assert read_json(tmp_path / "folder.json", default) is default
    (tmp_path / "cut.json").write_text('{"done": ', encoding="utf-8")
    assert read_json(tmp_path / "cut.json", default) is default
    (tmp_path / "latin.json").write_bytes(b'{"phase": "\xe9"}')  # not UTF-8
    assert read_json(tmp_path / "latin.json", default) is default
    assert read_json(tmp_path / "latin.json", default, encoding="latin-1") == {"phase": "\u00e9"}
    assert read_json(tmp_path / "missing.json") is None
