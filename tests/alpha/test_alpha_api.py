"""The Synthetic alpha routes: build a dataset, benchmark an arm on it, read it back.

Needs the ``dashboard`` extra. Benchmarks run on a thread pool here, so a test can shrink
the oracle's fresh sample with monkeypatch and the worker sees it.
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
from optspread.api.server import Settings, create_app
from optspread.dashboard.jobs.queue import JobManager
from optspread.dashboard.runindex import REPO_ROOT

SPEC = {
    "generator": {"kind": "const_vrp", "premium_pts": 5},
    "n_days": 20,
    "n_train": 2,
    "n_heldout": 4,
}


@pytest.fixture()
def client(
    tmp_path: Path, gate_docs: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(bm, "FRESH_N", 16)
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


def test_options_name_every_generator_and_arm(client: TestClient) -> None:
    options = client.get("/api/alpha/options").json()
    assert [g["kind"] for g in options["generators"]] == [
        "noise",
        "const_vrp",
        "mr_vrp",
        "spot_mr",
        "drift",
    ]
    mr = next(g for g in options["generators"] if g["kind"] == "mr_vrp")
    assert {k["name"] for k in mr["knobs"]} == {"mean_pts", "vol_pts", "half_life_days"}
    assert all(k["label"] for g in options["generators"] for k in g["knobs"])
    assert {"flat", "delta_seller"} <= set(options["arms"])
    assert options["learned_arms"] == ["ppo", "a2c", "dqn"]


def test_build_benchmark_and_read_back(client: TestClient) -> None:
    built = client.post("/api/alpha/datasets", json=SPEC)
    assert built.status_code == 201, built.text
    ds = built.json()
    assert len(ds["preview"]["signal"]["spot"]) == 4 and ds["title"]
    assert ds["source"].startswith("<runs root>/alpha/datasets/")
    assert client.get(f"/api/alpha/datasets/{ds['hash']}").json()["hash"] == ds["hash"]
    listed = client.get("/api/alpha/datasets").json()
    assert [(m["hash"], m["kind"]) for m in listed] == [(ds["hash"], "const_vrp")]
    assert client.get(f"/api/alpha/datasets/{ds['hash']}/benchmarks").json() == []

    job = client.post(
        "/api/alpha/benchmarks",
        json={"dataset": ds["hash"], "arm": "delta_seller", "total_timesteps": None},
    )
    assert job.status_code == 201, job.text
    out = _finish(client, job.json()["job_id"])
    assert out["state"] == "done", out["error"]
    bench = out["benchmark"]
    assert isinstance(bench, dict)
    assert bench["dataset"] == ds["hash"] and bench["arm"] == "delta_seller"
    assert {r["policy"] for r in bench["rows"]} == {"arm", "oracle", "flat"}
    assert out["done"] == out["total"]  # the final progress, never a stale snapshot
    assert client.get(f"/api/alpha/datasets/{ds['hash']}/benchmarks").json() == [bench]
    # a dataset is not a run: the run browser still lists nothing
    assert client.get("/api/overview").json()["n_runs"] == 0


def test_refusals(client: TestClient) -> None:
    bad = client.post(
        "/api/alpha/datasets", json={"generator": {"kind": "drift", "bps_per_day": 999}}
    )
    assert bad.status_code == 422
    assert (
        client.post("/api/alpha/datasets", json={"generator": {"kind": "nope"}}).status_code == 422
    )
    assert client.get("/api/alpha/datasets/0123456789abcdef").status_code == 404
    missing = {"dataset": "0123456789abcdef", "arm": "flat", "total_timesteps": None}
    assert client.post("/api/alpha/benchmarks", json=missing).status_code == 404
    unbudgeted = {"dataset": "0123456789abcdef", "arm": "flat"}
    assert client.post("/api/alpha/benchmarks", json=unbudgeted).status_code == 422
    assert client.get("/api/alpha/jobs/nope").status_code == 404


def test_a_sweep_runs_as_a_job_and_is_listed_on_every_dataset_it_touched(
    client: TestClient,
) -> None:
    ds = client.post("/api/alpha/datasets", json=SPEC).json()
    sweep = {"dataset": ds["hash"], "arm": "delta_seller", "axis": "n_train", "values": [2, 3]}
    job = client.post("/api/alpha/sweeps", json=sweep)
    assert job.status_code == 201, job.text
    assert job.json()["kind"] == "sweep"
    out = _finish(client, job.json()["job_id"])
    assert out["state"] == "done", out["error"]
    result = out["sweep"]
    assert isinstance(result, dict) and out["benchmark"] is None
    assert [p["n_train"] for p in result["points"]] == [2, 3]
    for point in result["points"]:
        listed = client.get(f"/api/alpha/datasets/{point['dataset']}/sweeps").json()
        assert [s["key"] for s in listed] == [result["key"]]


def test_seeds_and_sweeps_are_refused_with_the_reason(client: TestClient) -> None:
    ds = client.post("/api/alpha/datasets", json=SPEC).json()
    seeded = {"dataset": ds["hash"], "arm": "flat", "total_timesteps": None, "n_seeds": 3}
    assert "deterministic" in str(client.post("/api/alpha/benchmarks", json=seeded).json())
    flat_budget = {
        "dataset": ds["hash"],
        "arm": "flat",
        "axis": "total_timesteps",
        "values": [2048, 4096],
    }
    response = client.post("/api/alpha/sweeps", json=flat_budget)
    assert response.status_code == 422 and "learned arm" in str(response.json())
