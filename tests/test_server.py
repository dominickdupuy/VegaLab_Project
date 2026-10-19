"""Ilan Danial: the server is wiring, and the wiring holds (week 6).

Every route over fixture roots through FastAPI's TestClient: the reads return
the builders' payloads, a missing panel is a 404 that says what is missing, bad
input is a 422 with the reason, and a job POSTed with knobs comes back as an
ordinary run directory the run list then shows. Needs the ``dashboard`` extra.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from optspread.dashboard_data import REPO_ROOT
from optspread.jobs import JobManager
from optspread.server import WEB_DIR, Settings, allowed_hosts, create_app


def _client(settings: Settings, runs: Path) -> Iterator[TestClient]:
    manager = JobManager(runs, executor=ThreadPoolExecutor(1))
    with TestClient(create_app(settings, manager), base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture()
def client(
    run_root: Path, gate_docs: Path, panel_root: Path, tmp_path: Path
) -> Iterator[TestClient]:
    runs = tmp_path / "runs"  # a copy: the job test writes into it
    shutil.copytree(run_root, runs)
    web = tmp_path / "out"
    (web / "runs").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>optspread</title>")
    (web / "runs" / "index.html").write_text("<!doctype html><title>runs</title>")
    settings = Settings(runs, gate_docs, panel_root, web, REPO_ROOT / "README.md")
    yield from _client(settings, runs)


@pytest.fixture()
def bare(tmp_path: Path, gate_docs: Path) -> Iterator[TestClient]:
    """No runs, no panel, no built site."""
    runs = tmp_path / "runs"
    settings = Settings(runs, gate_docs, tmp_path / "nodata", tmp_path / "noweb", tmp_path / "x")
    yield from _client(settings, runs)


def test_health_and_the_site(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["ok"] and health["version"]
    assert "optspread" in client.get("/").text
    assert "runs" in client.get("/runs/").text


PAGES = ("/", "/runs/", "/launch/", "/power/", "/data/", "/leaks/", "/sweep/")


@pytest.mark.skipif(not (WEB_DIR / "index.html").is_file(), reason="needs `make web` first")
def test_every_page_of_the_built_site_is_served(tmp_path: Path, gate_docs: Path) -> None:
    settings = Settings(tmp_path / "runs", gate_docs, tmp_path, WEB_DIR, REPO_ROOT / "README.md")
    for client in _client(settings, tmp_path / "runs"):
        for page in PAGES:
            assert client.get(page).status_code == 200, page
        assert client.get("/gates/").status_code == 404  # the gate board is API-only


def test_without_a_built_site_the_root_says_how_to_build_it(bare: TestClient) -> None:
    assert "make web" in bare.get("/").json()["detail"]
    assert bare.get("/api/runs").json()["rows"] == []


def test_runs_list_filter_detail_and_compare(client: TestClient) -> None:
    listed = client.get("/api/runs").json()
    assert listed["trials"]["n_runs"] == 6
    flat = client.get(
        "/api/runs", params={"policy": "flat", "premium": 2.0, "sweep": "tiny"}
    ).json()
    assert flat["n_shown"] == 1
    key = flat["rows"][0]["key"]
    detail = client.get("/api/runs/detail", params={"key": key}).json()
    assert detail["row"]["key"] == key and len(detail["episodes"]) == 3
    assert detail["manifest"]["git_sha"] == detail["row"]["git_sha"]
    seller = client.get(
        "/api/runs", params={"policy": "delta_seller", "premium": 2.0, "sweep": "tiny"}
    ).json()["rows"][0]["key"]
    cmp = client.get("/api/runs/compare", params={"a": key, "b": seller}).json()
    assert cmp["shared_seeds"] == 3 and cmp["config_diff"][0]["key"] == "policy"
    assert client.get("/api/runs/detail", params={"key": "../x"}).status_code == 404
    assert client.get("/api/runs/compare", params={"a": key, "b": "nope"}).status_code == 404


def test_gates(client: TestClient) -> None:
    boards = client.get("/api/gates").json()["boards"]
    assert [b["milestone"] for b in boards] == ["M1", "M2"]
    assert client.get("/api/gates/m2").json()["verdict"] == "NOT CLEARED"
    assert client.get("/api/gates/M7").status_code == 404


def test_a_job_launched_over_http_lands_as_a_listed_run(client: TestClient) -> None:
    options = client.get("/api/jobs/options").json()
    assert "delta_seller" in options["arms"] and options["presets"]
    knobs = {"policy": "always_on", "n_days": 21, "tenor_days": [7, 14], "n_episodes": 2}
    created = client.post("/api/jobs", json=knobs)
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 60
    while (job := client.get(f"/api/jobs/{job_id}").json())["state"] in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert job["state"] == "done" and job["run_key"].startswith("launched/")
    assert [j["job_id"] for j in client.get("/api/jobs").json()] == [job_id]
    rows = client.get("/api/runs", params={"q": job["run_id"]}).json()["rows"]
    assert [r["key"] for r in rows] == [job["run_key"]] and rows[0]["status"] == "OK"
    detail = client.get("/api/runs/detail", params={"key": job["run_key"]}).json()
    assert detail["equity"]["seeds"] == [0, 1]
    assert client.get("/api/jobs/nope").status_code == 404


@pytest.mark.parametrize(
    "knobs",
    [{"n_episodes": 0}, {"tenor_days": [42, 21]}, {"bogus": 1}, {"policy": "ppo"}],
)
def test_bad_knobs_are_refused_with_the_reason(
    client: TestClient, knobs: dict[str, object]
) -> None:
    response = client.post("/api/jobs", json=knobs)
    assert response.status_code == 422 and response.json()["detail"]


def test_ppo_without_the_learner_is_refused(client: TestClient) -> None:
    if client.get("/api/jobs/options").json()["ppo_available"]:
        pytest.skip("the learner is installed here")
    response = client.post("/api/jobs", json={"policy": "ppo", "total_timesteps": 100})
    assert response.status_code == 422 and "PPO" in response.json()["detail"]


def test_the_power_lab(client: TestClient) -> None:
    out = client.get("/api/power", params={"n_seeds": 5, "n_episodes": 50}).json()
    assert out["design"]["n_seeds"] == 5 and len(out["curve"]["delta"]) == 81
    assert client.get("/api/power", params={"alpha": 0.9}).status_code == 422


def test_the_leak_demo(client: TestClient) -> None:
    out = client.post("/api/leaks/demo", json={"eps": 0.5, "n_seeds": 4}).json()
    assert out["caught"] and out["poisoned"]["worst_feature"] == "ret_1d"
    assert client.post("/api/leaks/demo", json={"eps": 0}).status_code == 422


def test_sweeps(client: TestClient) -> None:
    listed = client.get("/api/sweeps").json()
    assert [s["key"] for s in listed["sweeps"]] == ["sweeps/tiny"]
    detail = client.get("/api/sweeps/detail", params={"key": "sweeps/tiny"}).json()
    assert detail["power_curve"]["status"] == "waiting for sweep"
    assert client.get("/api/sweeps/detail", params={"key": "nope"}).status_code == 404


def test_the_panel_routes(client: TestClient) -> None:
    assert client.get("/api/data/status").json()["built"]
    assert client.get("/api/data/coverage").json()["overlap"] == ["2025-07-01", "2025-07-03", 2]
    assert "2025-07-02" not in client.get("/api/data/dates", params={"source": "databento"}).json()
    day = {"date": "2025-07-01"}
    chain = client.get("/api/data/chain", params=day).json()
    assert chain["sources"] == ["databento", "optionmetrics"]
    smile = client.get("/api/data/smile", params={**day, "source": "optionmetrics"}).json()
    assert smile["has_vendor"] and smile["dte"] == 30
    exp = client.get("/api/data/smile", params={**day, "expiry": chain["expiries"][0]["expiry"]})
    assert exp.json()["dte"] == 7
    surface = client.get("/api/data/surface", params={**day, "roots": ["SPX"]}).json()
    assert surface["dte"] == [30, 60, 120]
    assert len(client.get("/api/data/term", params=day).json()["dte"]) == 4
    daily = client.get("/api/data/daily", params={"start": "2025-07-01"}).json()
    assert all(d >= "2025-07-01" for s in daily["series"] for d in s["date"])
    assert client.get("/api/data/seam").json()["available"]
    assert client.get("/api/data/smile", params={"date": "07/01/2025"}).status_code == 422
    assert client.get("/api/overview").json()["overlap"] == ["2025-07-01", "2025-07-03", 2]


def test_an_unbuilt_panel_is_a_404_that_names_the_missing_files(bare: TestClient) -> None:
    response = bare.get("/api/data/coverage")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"].startswith("panel not built yet") and len(body["missing"]) == 3
    assert bare.get("/api/data/smile", params={"date": "2025-07-01"}).status_code == 404
    assert not bare.get("/api/data/status").json()["built"]
    assert bare.get("/api/overview").json()["coverage"] == []


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/power", {"delta": 1e200}),
        ("/api/power", {"delta": 1e-300}),
        ("/api/power", {"tau2": "inf"}),
        ("/api/power", {"sigma2": "nan"}),
        ("/api/runs/detail", {"key": "\x00"}),
        ("/api/runs/compare", {"a": "\x00", "b": "x"}),
        ("/api/sweeps/detail", {"key": "\x00"}),
        ("/api/data/surface", {"date": "2025-07-01", "min_dte": 90, "max_dte": 30}),
    ],
)
def test_bad_input_is_a_4xx_with_a_reason_never_a_500(
    client: TestClient, path: str, params: dict[str, object]
) -> None:
    response = client.get(path, params=params)
    assert 400 <= response.status_code < 500 and response.json()["detail"]


def test_a_job_whose_implied_vol_is_not_positive_is_refused(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"sigma": 0.01, "iv_premium_vol_pts": -10.0})
    assert response.status_code == 422 and "implied vol" in str(response.json()["detail"])


def test_a_foreign_host_header_is_refused(client: TestClient) -> None:
    rebound = client.get("/api/health", headers={"host": "attacker.example"})
    assert rebound.status_code == 400
    for host in ("localhost:8600", "127.0.0.1:8600", "[::1]:8600"):
        assert client.get("/api/health", headers={"host": host}).status_code == 200
    assert allowed_hosts("192.168.1.5")[-1] == "192.168.1.5"
    assert allowed_hosts("fe80::1")[-2:] == ["fe80::1", "[fe80::1]"]
