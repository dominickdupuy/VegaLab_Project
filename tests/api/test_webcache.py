"""Ilan Danial: the dashboard keeps an answer only while the files it read stand.

A kept body is keyed by its route, its query and a stamp of the files it depends on, so
an edited, added or removed file is never answered from before; the disk tier survives a
restart and ignores a body that does not read back; a browser holding a body is told 304.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from optspread.webcache import Answers, file_stamp, tree_stamp


def _touch(path: Path, text: str, ns: int) -> None:
    path.write_text(text)
    os.utime(path, ns=(ns, ns))


@pytest.mark.behaviour
def test_a_stamp_changes_with_any_file_under_the_root(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    _touch(tmp_path / "a" / "run.json", "{}", 10**18)
    first = tree_stamp(tmp_path)
    assert tree_stamp(tmp_path) == first
    _touch(tmp_path / "a" / "run.json", "{} ", 10**18)  # same time, another size
    assert tree_stamp(tmp_path) != first
    edited = tree_stamp(tmp_path)
    (tmp_path / "b").mkdir()
    _touch(tmp_path / "b" / "manifest.json", "{}", 10**18)
    assert tree_stamp(tmp_path) != edited
    # the job queue's own bookkeeping is not an input
    added = tree_stamp(tmp_path)
    (tmp_path / ".jobs").mkdir()
    _touch(tmp_path / ".jobs" / "j.json", "{}", 1)
    assert tree_stamp(tmp_path) == added
    assert file_stamp([tmp_path / "missing"]) == ((str(tmp_path / "missing"), -1, -1),)


@pytest.mark.behaviour
def test_a_kept_answer_is_rebuilt_when_its_stamp_or_query_changes() -> None:
    answers = Answers()
    built: list[str] = []

    def build(text: str) -> bytes:
        built.append(text)
        return json.dumps(text).encode()

    a, tag = answers.get("r", {"x": 1}, "s1", lambda: build("one"))
    assert answers.get("r", {"x": 1}, "s1", lambda: build("two")) == (a, tag)
    assert built == ["one"]
    assert answers.get("r", {"x": 2}, "s1", lambda: build("two"))[0] == b'"two"'
    b, other = answers.get("r", {"x": 1}, "s2", lambda: build("three"))
    assert b == b'"three"' and other != tag


@pytest.mark.behaviour
def test_the_disk_tier_survives_a_restart_and_ignores_a_torn_body(tmp_path: Path) -> None:
    Answers(tmp_path).get("data/x", {}, "s", lambda: b'{"v": 1}', disk=True)
    (kept,) = (tmp_path / "data_x").glob("*.json")
    calls: list[int] = []
    assert (
        Answers(tmp_path).get("data/x", {}, "s", lambda: calls.append(1) or b"{}", disk=True)[0]
        == b'{"v": 1}'
    )
    assert not calls
    kept.write_bytes(b'{"v": ')  # torn
    assert (
        Answers(tmp_path).get("data/x", {}, "s", lambda: b'{"v": 2}', disk=True)[0] == b'{"v": 2}'
    )
    # without a folder nothing is written anywhere
    Answers(None).get("data/y", {}, "s", lambda: b"{}", disk=True)
    assert not (tmp_path / "data_y").exists()


@pytest.mark.behaviour
def test_the_server_answers_from_the_cache_until_a_run_changes(
    run_root: Path, gate_docs: Path, panel_root: Path, tmp_path: Path
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from optspread.dashboard_data import REPO_ROOT
    from optspread.jobs import JobManager
    from optspread.server import Settings, create_app

    runs = tmp_path / "runs"
    shutil.copytree(run_root, runs)
    settings = Settings(
        runs,
        gate_docs,
        panel_root,
        tmp_path / "noweb",
        REPO_ROOT / "README.md",
        attribution=tmp_path / "attr",
        cache=tmp_path / "cache",
    )
    app = create_app(settings, JobManager(runs, executor=ThreadPoolExecutor(1)))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        first = client.get("/api/runs")
        assert first.status_code == 200 and first.headers["cache-control"] == "no-cache"
        tag = first.headers["etag"]
        again = client.get("/api/runs", headers={"If-None-Match": tag})
        assert again.status_code == 304 and not again.content
        n = len(first.json()["rows"])
        # a run removed: the stamp moves, the answer is rebuilt
        victim = next(
            p.parent for p in runs.rglob("manifest.json") if (p.parent / "run.json").is_file()
        )
        shutil.rmtree(victim)
        after = client.get("/api/runs", headers={"If-None-Match": tag})
        assert after.status_code == 200 and after.headers["etag"] != tag
        assert len(after.json()["rows"]) == n - 1
        # a panel answer is kept on disk as well
        assert client.get("/api/data/status").status_code == 200
        assert client.get("/api/data/coverage").status_code == 200
        assert list((tmp_path / "cache" / "data_coverage").glob("*.json"))


@pytest.mark.invariant
def test_a_kept_run_row_is_read_again_when_a_file_changes(run_root: Path, tmp_path: Path) -> None:
    from optspread.dashboard_data import load_run_row

    runs = tmp_path / "runs"
    shutil.copytree(run_root, runs)
    # A cell of the fixture's sweep, a clean run. Not the first run.json in directory order:
    # that order is the filesystem's, and it can be the tampered copy under adhoc/.
    run_dir = sorted((runs / "sweeps" / "tiny").glob("*-s7"))[0]
    row = load_run_row(run_dir)
    assert row.status == "OK" and load_run_row(run_dir) is row  # kept
    (run_dir / "run.json").write_text((run_dir / "run.json").read_text() + " ")
    assert load_run_row(run_dir).digests_ok is False  # edited: read and checked again


@pytest.mark.behaviour
def test_a_warmed_server_reads_its_tables_at_start_and_answers_as_before(
    run_root: Path, gate_docs: Path, panel_root: Path, tmp_path: Path
) -> None:
    pytest.importorskip("fastapi")
    import threading
    import time

    from fastapi.testclient import TestClient

    from optspread.dashboard_data import REPO_ROOT
    from optspread.jobs import JobManager
    from optspread.server import Settings, create_app

    settings = Settings(
        run_root,
        gate_docs,
        panel_root,
        tmp_path / "noweb",
        REPO_ROOT / "README.md",
        attribution=tmp_path / "attr",
        warm=True,
    )
    app = create_app(settings, JobManager(tmp_path / "runs", executor=ThreadPoolExecutor(1)))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        deadline = time.monotonic() + 60
        while any(t.name == "optspread-warm" for t in threading.enumerate()):
            assert time.monotonic() < deadline, "the warm-up never ended"
            time.sleep(0.05)
        assert client.get("/api/data/grid/dates").status_code == 200
        assert client.get("/api/runs").status_code == 200
