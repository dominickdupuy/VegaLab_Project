"""The real-data test routes: read, launch and poll, with a fixture panel and without one.

Needs the ``dashboard`` extra. Jobs run on a thread pool here, so the oracle's fresh
sample shrinks with monkeypatch and the worker sees it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("pyarrow")

from fastapi.testclient import TestClient
from test_alpha_realtest import TIMER, write_panel

from optspread.alpha import benchmark as bm
from optspread.alpha import plugins as plug
from optspread.api.server import Settings, create_app
from optspread.dashboard.jobs.queue import JobManager
from optspread.dashboard.runindex import REPO_ROOT

SPEC = {
    "generator": {"kind": "const_vrp", "premium_pts": 5},
    "n_days": 10,
    "n_train": 2,
    "n_heldout": 4,
}


def _client(tmp_path: Path, gate_docs: Path, data: Path) -> TestClient:
    runs = tmp_path / "runs"
    settings = Settings(
        runs,
        gate_docs,
        data,
        tmp_path / "noweb",
        REPO_ROOT / "README.md",
        attribution=tmp_path / "attr",
    )
    manager = JobManager(runs, executor=ThreadPoolExecutor(1))
    return TestClient(create_app(settings, manager), base_url="http://127.0.0.1")


@pytest.fixture()
def client(
    tmp_path: Path, gate_docs: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(bm, "FRESH_N", 16)
    with _client(tmp_path, gate_docs, write_panel(tmp_path / "data")) as c:
        yield c


@pytest.fixture()
def bare(tmp_path: Path, gate_docs: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(bm, "FRESH_N", 16)
    with _client(tmp_path, gate_docs, tmp_path / "nodata") as c:
        yield c


def _finish(client: TestClient, path: str) -> dict[str, object]:
    deadline = time.monotonic() + 120
    while (out := client.get(path).json())["state"] in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    return out


def _bench(client: TestClient) -> dict[str, object]:
    ds = client.post("/api/alpha/datasets", json=SPEC).json()
    spec = {"dataset": ds["hash"], "arm": "delta_seller", "total_timesteps": None}
    job = client.post("/api/alpha/benchmarks", json=spec).json()
    assert _finish(client, f"/api/alpha/jobs/{job['job_id']}")["state"] == "done"
    return spec


@pytest.mark.behaviour
def test_launch_poll_and_read_back(client: TestClient) -> None:
    spec = _bench(client)
    query = {k: v for k, v in spec.items() if v is not None}
    before = client.get("/api/alpha/real", params=query)
    assert before.status_code == 200, before.text
    body = before.json()
    assert body["available"] is True and body["message"] == "" and body["result"] is None
    assert (body["split"], body["first_date"], body["last_date"]) == (
        "eval",
        "2019-01-01",
        "2022-08-24",
    )
    assert body["source"].startswith("<runs root>/alpha/real/")
    job = client.post("/api/alpha/real", json=spec)
    assert job.status_code == 201, job.text
    assert job.json()["job_id"].startswith("r") and job.json()["source"] == "optionmetrics"
    out = _finish(client, f"/api/alpha/real/jobs/{job.json()['job_id']}")
    assert out["state"] == "done", out["error"]
    result = out["result"]
    assert isinstance(result, dict) and result["arm"] == "delta_seller" and result["exploratory"]
    assert out["done"] == out["total"]
    assert client.get("/api/alpha/real", params=query).json()["result"] == result


@pytest.mark.behaviour
def test_without_the_panel_the_read_says_so_and_a_launch_is_refused(bare: TestClient) -> None:
    spec = _bench(bare)
    body = bare.get(
        "/api/alpha/real", params={k: v for k, v in spec.items() if v is not None}
    ).json()
    assert body["available"] is False and body["result"] is None
    assert body["message"].startswith("Real data is not available on this machine")
    assert "<data root>/panel" in body["message"], "the serving machine's path is not shown"
    refused = bare.post("/api/alpha/real", json=spec)
    assert refused.status_code == 409 and "not available on this machine" in refused.text


@pytest.mark.behaviour
def test_refusals(client: TestClient) -> None:
    missing = {"dataset": "0123456789abcdef", "arm": "flat", "total_timesteps": None}
    assert client.post("/api/alpha/real", json=missing).status_code == 404
    assert (
        client.get(
            "/api/alpha/real", params={"dataset": "0123456789abcdef", "arm": "flat"}
        ).status_code
        == 404
    )
    assert (
        client.post("/api/alpha/real", json={"dataset": "nope", "arm": "flat"}).status_code == 422
    )
    assert (
        client.get(
            "/api/alpha/real", params={"dataset": "0123456789abcdef", "arm": "ppo"}
        ).status_code
        == 422
    )
    assert client.get("/api/alpha/real/jobs/r000000000000").status_code == 404


@pytest.mark.behaviour
def test_a_plug_in_arm_is_tested_and_refused_once_it_is_gone(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(plug.PLUGINS_ENV, TIMER)
    ds = client.post("/api/alpha/datasets", json=SPEC).json()
    spec = {"dataset": ds["hash"], "arm": "iv_timer", "total_timesteps": None}
    job = client.post("/api/alpha/benchmarks", json=spec).json()
    assert _finish(client, f"/api/alpha/jobs/{job['job_id']}")["state"] == "done"
    query = {"dataset": ds["hash"], "arm": "iv_timer"}
    body = client.get("/api/alpha/real", params=query).json()
    assert body["refused"] == "" and body["result"] is None
    started = client.post("/api/alpha/real", json=spec)
    assert started.status_code == 201, started.text
    out = _finish(client, f"/api/alpha/real/jobs/{started.json()['job_id']}")
    assert out["state"] == "done", out["error"]
    assert isinstance(out["result"], dict) and out["result"]["arm"] == "iv_timer"
    # the server restarted without it: the saved result still reads, a new test is refused
    monkeypatch.setenv(plug.PLUGINS_ENV, "")
    gone = client.get("/api/alpha/real", params=query)
    assert gone.status_code == 200, gone.text
    assert gone.json()["result"] == out["result"]
    assert "no longer registered" in gone.json()["refused"]
    refused = client.post("/api/alpha/real", json=spec)
    assert refused.status_code == 422 and "no longer registered" in refused.text
