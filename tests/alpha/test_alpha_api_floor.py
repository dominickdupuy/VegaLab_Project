"""The routes for plug-in arms and edge-size sweeps: what the page's picker and floor read.

Needs the ``dashboard`` extra. Jobs run on a thread pool, so the plug-ins this process
registers and the oracle's shrunk sample are the worker's too.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from optspread.alpha import benchmark as bm
from optspread.alpha import plugins as plug
from optspread.api.server import Settings, create_app
from optspread.dashboard.jobs.queue import JobManager
from optspread.dashboard.runindex import REPO_ROOT

EXAMPLE = REPO_ROOT / "examples" / "alpha_plugin.py"
SPEC = {
    "generator": {"kind": "const_vrp", "premium_pts": 5},
    "n_days": 20,
    "n_train": 2,
    "n_heldout": 10,
}


@pytest.fixture()
def client(
    tmp_path: Path, gate_docs: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(bm, "FRESH_N", 16)
    monkeypatch.setenv(plug.PLUGINS_ENV, f"{EXAMPLE}:iv_timer,{EXAMPLE}:iv_picker")
    monkeypatch.delenv(plug.PLUGINS_FILE_ENV, raising=False)
    runs = tmp_path / "runs"
    settings = Settings(
        runs,
        gate_docs,
        tmp_path / "nodata",
        tmp_path / "noweb",
        REPO_ROOT / "README.md",
        attribution=tmp_path / "attr",
    )
    manager = JobManager(runs, executor=ThreadPoolExecutor(1))
    with TestClient(create_app(settings, manager), base_url="http://127.0.0.1") as c:
        yield c


def _finish(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 120
    while (out := client.get(f"/api/alpha/jobs/{job_id}").json())["state"] in (
        "queued",
        "running",
    ):
        assert time.monotonic() < deadline
        time.sleep(0.1)
    return out


def test_the_picker_lists_plugins_labelled(client: TestClient) -> None:
    options = client.get("/api/alpha/options").json()
    assert options["arms"][-2:] == ["iv_timer", "iv_picker"]
    assert options["learned_arms"][:3] == ["ppo", "a2c", "dqn"]
    assert "iv_picker" in options["learned_arms"] and "iv_timer" not in options["learned_arms"]
    details = {a["name"]: a for a in options["arm_details"]}
    assert details["iv_timer"]["label"] == "IV timer (plug-in)"
    assert details["iv_timer"]["kind"] == "plugin" and not details["iv_timer"]["trainable"]
    # the server never shows a path on its machine: the checkout reads as <repo>
    assert details["iv_picker"]["source"].startswith("<repo>/examples/alpha_plugin.py:iv_picker@")
    assert details["delta_seller"]["kind"] == "scripted" and options["plugin_error"] is None
    const = next(g for g in options["generators"] if g["kind"] == "const_vrp")
    assert (const["edge_knob"], const["edge_unit"]) == ("premium_pts", "vol points")
    assert next(g for g in options["generators"] if g["kind"] == "noise")["edge_knob"] is None
    arms = client.get("/api/alpha/arms").json()
    assert [a["name"] for a in arms] == options["arms"]


def test_a_broken_plugin_list_is_reported_not_hidden(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(plug.PLUGINS_ENV, "nowhere.at_all:x")
    options = client.get("/api/alpha/options").json()
    assert "cannot be imported" in options["plugin_error"]
    assert "iv_timer" not in options["arms"] and "flat" in options["arms"]
    assert client.get("/api/alpha/arms").status_code == 500


def test_a_plugin_and_an_edge_sweep_through_the_routes(client: TestClient) -> None:
    ds = client.post("/api/alpha/datasets", json=SPEC).json()
    job = client.post(
        "/api/alpha/benchmarks",
        json={"dataset": ds["hash"], "arm": "iv_timer", "total_timesteps": None},
    )
    assert job.status_code == 201, job.text
    out = _finish(client, job.json()["job_id"])
    assert out["state"] == "done", out["error"]
    bench = out["benchmark"]
    assert isinstance(bench, dict) and bench["arm"] == "iv_timer" and bench["plugin"]
    saved = client.get(f"/api/alpha/datasets/{ds['hash']}/benchmarks").json()
    assert [b["arm"] for b in saved] == ["iv_timer"]

    sweep = {"dataset": ds["hash"], "arm": "delta_seller", "axis": "edge", "values": [0, 2, 5]}
    job = client.post("/api/alpha/sweeps", json=sweep)
    assert job.status_code == 201, job.text
    out = _finish(client, job.json()["job_id"])
    assert out["state"] == "done", out["error"]
    result = out["sweep"]
    assert isinstance(result, dict) and result["edge_knob"] == "premium_pts"
    floor = result["floor"]
    assert floor["status"] in {"reached", "below_grid", "not_reached"} and floor["curve"]
    assert floor["unit"] == "path blocks" and "rough" in floor["note"]
    assert [p["test"]["edge"] for p in result["points"]] == [0, 2, 5]
    listed = client.get(f"/api/alpha/datasets/{ds['hash']}/sweeps").json()
    assert [s["key"] for s in listed] == [result["key"]]


def test_edge_sweeps_are_refused_with_the_reason(client: TestClient) -> None:
    noise = client.post("/api/alpha/datasets", json={**SPEC, "generator": {"kind": "noise"}}).json()
    refused = client.post(
        "/api/alpha/sweeps",
        json={"dataset": noise["hash"], "arm": "flat", "axis": "edge", "values": [0, 1]},
    )
    assert refused.status_code == 422 and "no edge size" in refused.text
    one_seed = {
        "dataset": noise["hash"],
        "arm": "iv_picker",
        "axis": "edge",
        "values": [0, 1],
        "total_timesteps": 2048,
    }
    response = client.post("/api/alpha/sweeps", json=one_seed)
    assert response.status_code == 422 and "2 training seeds" in response.text
    unknown = {"dataset": noise["hash"], "arm": "mystery", "total_timesteps": None}
    response = client.post("/api/alpha/benchmarks", json=unknown)
    assert response.status_code == 422 and "unknown arm" in response.text
