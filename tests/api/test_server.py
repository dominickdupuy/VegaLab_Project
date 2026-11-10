"""Ilan Danial: the server is wiring, and the wiring holds (week 6).

Every route over fixture roots through FastAPI's TestClient: the reads return
the builders' payloads, a missing panel is a 404 that says what is missing, bad
input is a 422 with the reason, and a job POSTed with knobs comes back as an
ordinary run directory the run list then shows. Needs the ``dashboard`` extra.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from optspread.dashboard_data import REPO_ROOT
from optspread.jobs import JobManager, execute_job
from optspread.server import WEB_DIR, Settings, allowed_hosts, create_app, redactor

HANG = 600.0  # seconds: bounds a job that never ends, never a slow one


def _client(settings: Settings, manager: JobManager) -> Iterator[TestClient]:
    with TestClient(create_app(settings, manager), base_url="http://127.0.0.1") as client:
        yield client


@pytest.fixture()
def jobs(run_root: Path, tmp_path: Path) -> JobManager:
    """The queue behind ``client``, so a test can wait on a job rather than on a clock."""
    runs = tmp_path / "runs"  # a copy: the job test writes into it
    shutil.copytree(run_root, runs)
    return JobManager(runs, executor=ThreadPoolExecutor(1))


@pytest.fixture()
def client(
    jobs: JobManager, gate_docs: Path, panel_root: Path, tmp_path: Path
) -> Iterator[TestClient]:
    runs = jobs.runs_root
    web = tmp_path / "out"
    (web / "runs").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><title>optspread</title>")
    (web / "runs" / "index.html").write_text("<!doctype html><title>runs</title>")
    settings = Settings(
        runs, gate_docs, panel_root, web, REPO_ROOT / "README.md", attribution=tmp_path / "attr"
    )
    yield from _client(settings, jobs)


@pytest.fixture()
def bare(tmp_path: Path, gate_docs: Path) -> Iterator[TestClient]:
    """No runs, no panel, no built site."""
    runs = tmp_path / "runs"
    settings = Settings(runs, gate_docs, tmp_path / "nodata", tmp_path / "noweb", tmp_path / "x")
    yield from _client(settings, JobManager(runs, executor=ThreadPoolExecutor(1)))


@pytest.mark.behaviour
def test_health_and_the_site(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["ok"] and health["version"]
    assert "optspread" in client.get("/").text
    assert "runs" in client.get("/runs/").text


PAGES = ("/", "/runs/", "/launch/", "/power/", "/data/", "/leaks/", "/sweep/", "/attribution/")


@pytest.mark.behaviour
@pytest.mark.skipif(not (WEB_DIR / "index.html").is_file(), reason="needs `make web` first")
def test_every_page_of_the_built_site_is_served(tmp_path: Path, gate_docs: Path) -> None:
    settings = Settings(tmp_path / "runs", gate_docs, tmp_path, WEB_DIR, REPO_ROOT / "README.md")
    for client in _client(settings, JobManager(tmp_path / "runs", executor=ThreadPoolExecutor(1))):
        for page in PAGES:
            assert client.get(page).status_code == 200, page
        assert client.get("/gates/").status_code == 404  # the gate board is API-only
    # the site calls the routes the server serves for the edge and the action mix
    chunks = "".join(p.read_text(errors="ignore") for p in (WEB_DIR / "_next").rglob("*.js"))
    for route in ("/api/edge", "/api/edge/detail", "/api/runs/actionmix"):
        assert route in chunks, route


@pytest.mark.behaviour
def test_without_a_built_site_the_root_says_how_to_build_it(bare: TestClient) -> None:
    assert "make web" in bare.get("/").json()["detail"]
    assert bare.get("/api/runs").json()["rows"] == []


@pytest.mark.behaviour
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


@pytest.mark.behaviour
def test_the_checks_panel_is_the_one_route_to_the_reports(client: TestClient) -> None:
    # the board's own routes went with the board: no page read them
    for path in ("/api/gates", "/api/gates/m2"):
        response = client.get(path)
        assert response.status_code == 404 and response.json()["detail"], path
    rows = client.get("/api/checks").json()
    assert {r["sidecar"] for r in rows} >= {"M2_GATE_REPORT.json"}
    replay = next(r for r in rows if r["name"] == "Replay")
    assert replay["report"] == "M2_GATE_REPORT.md"  # the readable one, not the sidecar
    assert (
        client.get(f"/api/docs/{replay['report']}").headers["content-type"].startswith("text/plain")
    )


@pytest.mark.behaviour
def test_an_unknown_api_path_is_a_json_404_not_the_site(client: TestClient) -> None:
    for method in ("get", "post"):
        response = getattr(client, method)("/api/nope/really")
        assert response.status_code == 404
        assert response.headers["content-type"] == "application/json"
        assert "/api/nope/really" in response.json()["detail"]
    assert client.get("/runs/").status_code == 200  # the site still answers its pages
    wrong = client.post("/api/runs")  # a route that is there, asked the wrong way
    assert wrong.status_code == 405 and "GET" in wrong.headers["allow"]


@pytest.mark.invariant
def test_payloads_name_roots_not_where_they_are(client: TestClient, tmp_path: Path) -> None:
    """``root``, ``file`` and ``run_dir`` fields read ``<runs root>/...``, never a home."""
    bodies = [
        client.get("/api/sweeps").text,
        client.get("/api/overview").text,
        client.get("/api/data/status").text,
        client.get("/api/edge").text,
    ]
    key = client.get("/api/runs").json()["rows"][0]["key"]
    bodies += [client.get("/api/runs/detail", params={"key": key}).text]
    bodies += [client.get("/api/runs/attribution", params={"key": key}).text]
    for text in bodies:
        assert str(tmp_path) not in text and str(Path.home()) not in text, text[:300]
    assert client.get("/api/sweeps").json()["root"] == "<runs root>"
    assert (
        client.get("/api/runs/attribution", params={"key": key})
        .json()["file"]
        .startswith("<attribution root>/")
    )


@pytest.mark.behaviour
def test_a_job_launched_over_http_lands_as_a_listed_run(
    client: TestClient, jobs: JobManager
) -> None:
    options = client.get("/api/jobs/options").json()
    assert "delta_seller" in options["arms"] and options["presets"]
    knobs = {"policy": "always_on", "n_days": 21, "tenor_days": [7, 14], "n_episodes": 2}
    created = client.post("/api/jobs", json=knobs)
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    done, _ = wait([jobs.jobs[job_id].future], timeout=HANG)
    assert done, f"job {job_id} still unfinished after {HANG:.0f}s"
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["state"] == "done" and job["run_key"].startswith("launched/")
    assert [j["job_id"] for j in client.get("/api/jobs").json()] == [job_id]
    rows = client.get("/api/runs", params={"q": job["run_id"]}).json()["rows"]
    assert [r["key"] for r in rows] == [job["run_key"]] and rows[0]["status"] == "OK"
    detail = client.get("/api/runs/detail", params={"key": job["run_key"]}).json()
    assert detail["equity"]["seeds"] == [0, 1]
    assert client.get("/api/jobs/nope").status_code == 404


@pytest.mark.behaviour
@pytest.mark.parametrize(
    "knobs",
    [{"n_episodes": 0}, {"tenor_days": [42, 21]}, {"bogus": 1}, {"policy": "ppo"}],
)
def test_bad_knobs_are_refused_with_the_reason(
    client: TestClient, knobs: dict[str, object]
) -> None:
    response = client.post("/api/jobs", json=knobs)
    assert response.status_code == 422 and response.json()["detail"]


@pytest.mark.behaviour
def test_ppo_without_the_learner_is_refused(client: TestClient) -> None:
    if client.get("/api/jobs/options").json()["ppo_available"]:
        pytest.skip("the learner is installed here")
    response = client.post("/api/jobs", json={"policy": "ppo", "total_timesteps": 100})
    assert response.status_code == 422 and "PPO" in response.json()["detail"]


@pytest.mark.behaviour
def test_the_power_lab(client: TestClient) -> None:
    out = client.get("/api/power", params={"n_seeds": 5, "n_episodes": 50}).json()
    assert out["design"]["n_seeds"] == 5 and len(out["curve"]["delta"]) == 81
    assert client.get("/api/power", params={"alpha": 0.9}).status_code == 422


@pytest.mark.behaviour
def test_an_undefined_figure_is_sent_as_null_never_a_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN and infinity reach the browser as null on every path a body takes: the answer
    cache, a returned model and an error's detail."""
    from fastapi import HTTPException

    from optspread import webviews as v

    # kept by the answer cache: with no variance at all the floor ratio is NaN by design
    power = client.get("/api/power", params={"tau2": 0, "sigma2": 0})
    assert power.status_code == 200 and power.json()["floor_vs_seeds"]["ratio"] is None
    # a model the route returns, rendered by the app's response class
    inf = float("inf")
    estimate = v.EstimateOut(
        eval_steps=1,
        train_steps=0,
        eval_hours=float("nan"),
        train_hours=inf,
        total_hours=-inf,
        train_rate=1.0,
        eval_rate=2.5,
        source="synthetic",
    )
    monkeypatch.setattr("optspread.api.builders.launch_estimate", lambda *_: estimate)
    got = client.get("/api/launch/estimate", params={"n_episodes": 1, "n_days": 1})
    assert got.status_code == 200
    assert got.json() == {
        **estimate.model_dump(),
        "eval_hours": None,
        "train_hours": None,
        "total_hours": None,
    }

    # an error whose detail holds one
    def refuse(*_: object) -> None:
        raise HTTPException(409, {"value": float("nan")})

    monkeypatch.setattr("optspread.api.builders.launch_estimate", refuse)
    got = client.get("/api/launch/estimate", params={"n_episodes": 1, "n_days": 1})
    assert got.status_code == 409 and got.json() == {"detail": {"value": None}}


@pytest.mark.behaviour
def test_the_leak_demo(client: TestClient) -> None:
    out = client.post("/api/leaks/demo", json={"eps": 0.5, "n_seeds": 4}).json()
    assert out["caught"] and out["poisoned"]["worst_feature"] == "ret_1d"
    assert client.post("/api/leaks/demo", json={"eps": 0}).status_code == 422


@pytest.mark.behaviour
def test_sweeps(client: TestClient) -> None:
    listed = client.get("/api/sweeps").json()
    assert [s["key"] for s in listed["sweeps"]] == ["sweeps/tiny"]
    detail = client.get("/api/sweeps/detail", params={"key": "sweeps/tiny"}).json()
    assert "power_curve" not in detail and "waiting for sweep" not in json.dumps(detail)
    assert client.get("/api/sweeps/detail", params={"key": "nope"}).status_code == 404


@pytest.mark.behaviour
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


@pytest.mark.behaviour
def test_the_3d_routes(client: TestClient) -> None:
    dates = client.get("/api/data/grid/dates").json()
    assert dates[0] == "2025-06-27" and "2025-07-08" in dates
    grid = client.get("/api/data/grid", params={"date": "2025-07-01"}).json()
    assert grid["tenor_days"][0] == 7 and len(grid["iv"]) == len(grid["tenor_days"])
    frames = client.get("/api/data/grid/frames", params={"start": "2025-07-01", "max_frames": 2})
    assert [f["date"] for f in frames.json()["frames"]] == ["2025-07-01", "2025-07-02"]
    assert frames.json()["more"]
    sim = client.get(
        "/api/data/simulated", params={"date": "2025-07-01", "premium": 2, "sigma_from": "atm"}
    ).json()
    assert sim["sigma_from"] == "atm" and len(sim["iv"]) == 2 and len(sim["real"][0]) == 33
    land = client.get("/api/data/landscape", params={"max_dates": 3}).json()
    assert land["stride"] == 3 and len(land["date"]) == 3
    surface = client.get("/api/power/surface", params={"n_episodes": 500, "max_seeds": 20}).json()
    assert surface["seeds"][-1] == 20 and len(surface["power"][0]) == len(surface["premium"])
    assert client.get("/api/data/grid", params={"date": "2025-07-04"}).status_code == 404


@pytest.mark.behaviour
def test_an_unbuilt_panel_is_a_404_that_names_the_missing_files(bare: TestClient) -> None:
    response = bare.get("/api/data/coverage")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"].startswith("panel not built yet") and len(body["missing"]) == 3
    assert bare.get("/api/data/smile", params={"date": "2025-07-01"}).status_code == 404
    assert not bare.get("/api/data/status").json()["built"]
    assert bare.get("/api/overview").json()["coverage"] == []


@pytest.mark.behaviour
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
        ("/api/power/surface", {"tau2": "inf"}),
        ("/api/power/surface", {"max_premium": 0}),
        ("/api/data/grid", {"date": "July 1"}),
        ("/api/data/grid/frames", {"start": "2025-07-01", "max_frames": 0}),
        ("/api/data/grid/frames", {"start": "2025-07-01", "max_frames": 253}),
        ("/api/data/grid/frames", {"start": "2025-07-01", "max_frames": 10**9}),
        ("/api/data/simulated", {"date": "2025-07-01", "premium": -1}),
        ("/api/data/simulated", {"date": "2025-07-01", "premium": "nan"}),
        ("/api/data/simulated", {"date": "2025-07-01", "sigma_from": "vix"}),
        ("/api/data/simulated", {"date": "2025-07-01", "premium": 19, "sigma_from": "atm"}),
        ("/api/data/landscape", {"max_dates": 1}),
        ("/api/data/landscape", {"max_dates": 4_201}),
        ("/api/data/landscape", {"start": "2025-07-03", "end": "2025-07-01"}),
    ],
)
def test_bad_input_is_a_4xx_with_a_reason_never_a_500(
    client: TestClient, path: str, params: dict[str, object]
) -> None:
    response = client.get(path, params=params)
    assert 400 <= response.status_code < 500 and response.json()["detail"]


@pytest.mark.behaviour
def test_a_full_queue_is_a_429_and_a_job_can_be_cancelled(
    client: TestClient, jobs: JobManager
) -> None:
    started, gate = threading.Event(), threading.Event()

    def gated(*args: str) -> str:
        started.set()
        assert gate.wait(HANG)
        return execute_job(*args)

    jobs.target, jobs.max_pending = gated, 1
    knobs = {"n_days": 21, "tenor_days": [7, 14], "n_episodes": 1}
    try:
        first = client.post("/api/jobs", json=knobs)
        assert first.status_code == 201
        job_id = first.json()["job_id"]
        full = client.post("/api/jobs", json={**knobs, "seed": 5})
        assert full.status_code == 429 and "queue is full" in full.json()["detail"]
        assert started.wait(HANG)
        cancelled = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled"
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409
        assert client.post("/api/jobs/nope/cancel").status_code == 404
    finally:
        gate.set()
    done, _ = wait([jobs.jobs[job_id].future], timeout=HANG)
    assert done, f"job {job_id} still unfinished after {HANG:.0f}s"
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["state"] == "cancelled" and job["run_key"] == ""
    assert client.post("/api/jobs", json={**knobs, "seed": 5}).status_code == 201


@pytest.mark.behaviour
def test_a_job_whose_implied_vol_is_not_positive_is_refused(client: TestClient) -> None:
    response = client.post("/api/jobs", json={"sigma": 0.01, "iv_premium_vol_pts": -10.0})
    assert response.status_code == 422 and "implied vol" in str(response.json()["detail"])


@pytest.mark.behaviour
def test_a_foreign_host_header_is_refused(client: TestClient) -> None:
    rebound = client.get("/api/health", headers={"host": "attacker.example"})
    assert rebound.status_code == 400
    for host in ("localhost:8600", "127.0.0.1:8600", "[::1]:8600"):
        assert client.get("/api/health", headers={"host": host}).status_code == 200
    assert allowed_hosts("192.168.1.5")[-1] == "192.168.1.5"
    assert allowed_hosts("fe80::1")[-2:] == ["fe80::1", "[fe80::1]"]


@pytest.mark.behaviour
def test_the_servers_default_runs_root_is_the_orchestrators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from optspread.orchestrate import default_runs_root

    monkeypatch.delenv("OPTSPREAD_RUNS", raising=False)
    monkeypatch.setenv("VEGALAB_DATA", str(tmp_path))
    assert Settings.from_env().runs == default_runs_root() == tmp_path / "runs"
    monkeypatch.setenv("OPTSPREAD_RUNS", str(tmp_path / "elsewhere"))
    assert Settings.from_env().runs == tmp_path / "elsewhere"  # an explicit root still wins


def _until_done(client: TestClient, path: str, key: str, states: tuple[str, ...]) -> dict[str, Any]:
    for _ in range(int(HANG * 10)):
        out: dict[str, Any] = client.get(path, params={"key": key}).json()
        if out["state"] in states:
            return out
        time.sleep(0.1)
    raise AssertionError(f"{path} never reached {states}")


@pytest.mark.behaviour
def test_the_study_its_checks_and_its_documents(client: TestClient) -> None:
    study = client.get("/api/study").json()
    assert (study["n_seeds"], study["n_episodes"]) == (5, 500) and study["doc"]
    rows = client.get("/api/checks").json()
    assert {"Replay", "Analysis pipeline validated"} <= {r["name"] for r in rows}
    doc = client.get("/api/docs/M2_GATE_REPORT.json")
    assert doc.status_code == 200 and doc.headers["content-type"] == "application/json"
    assert (
        client.get("/api/docs/M2_GATE_REPORT.md").headers["content-type"].startswith("text/plain")
    )
    assert client.get("/api/docs/nope.md").status_code == 404
    assert client.get("/api/docs/..%2Fconftest.py").status_code == 404
    assert client.get("/api/vrp").json()["available"] is False  # the fixture docs have none
    m4 = client.get("/api/m4").json()
    assert not m4["available"] and "make m4" in m4["waiting_for"]


@pytest.mark.invariant
def test_attribution_capital_and_replay_start_and_are_polled(client: TestClient) -> None:
    rows = client.get("/api/runs", params={"policy": "delta_seller", "sweep": "tiny"}).json()
    key = rows["rows"][0]["key"]
    assert client.get("/api/runs/attribution", params={"key": key}).json()["state"] == "absent"
    started = client.post("/api/runs/attribution", params={"key": key})
    assert started.status_code == 202
    done = _until_done(client, "/api/runs/attribution", key, ("cached", "failed"))
    assert done["state"] == "cached" and done["result"]["reproduces"] and done["steps"]
    assert done["file"].startswith("<attribution root>/")  # the test's root, by its name
    client.post("/api/runs/capital", params={"key": key})
    cap = _until_done(client, "/api/runs/capital", key, ("done", "failed"))
    assert cap["state"] == "done" and cap["reproduces"] and cap["n"] == 3
    client.post("/api/runs/replay", params={"key": key})
    rep = _until_done(client, "/api/runs/replay", key, ("done", "failed"))
    assert rep["verdict"] == "IDENTICAL" and rep["reason"] and rep["note"]
    assert rep["rtol"] == 1e-9 and not rep["retrains"] and rep["comparable"]
    assert cap["agrees"] and cap["n_differ"] == 0 and cap["rtol"] == 1e-9
    tampered = next(r for r in client.get("/api/runs").json()["rows"] if r["status"] != "OK")
    for kind in ("attribution", "capital"):
        assert client.post(f"/api/runs/{kind}", params={"key": tampered["key"]}).status_code == 422
    assert client.get("/api/runs/replay", params={"key": "nope"}).status_code == 404


@pytest.mark.behaviour
def test_the_leak_floor_and_field_routes(client: TestClient) -> None:
    fields = client.get("/api/leaks/fields").json()
    assert "ret_1d" in fields and len(fields) == 14
    floor = client.post("/api/leaks/floor", json={"n_seeds": 3, "eps_grid": [1.0, 0.5]}).json()
    assert floor["eps"] == [1.0, 0.5] and len(floor["flagged"]) == 2
    bad = client.post("/api/leaks/floor", json={"n_seeds": 3, "eps_grid": [5.0, 1.0]})
    assert bad.status_code == 422
    one = client.post("/api/leaks/field", json={"field": "atm_iv", "eps": 1.0, "n_seeds": 4})
    assert one.json()["field"] == "atm_iv"
    assert client.post("/api/leaks/field", json={"field": "nope"}).status_code == 422


@pytest.mark.behaviour
def test_the_seller_on_a_real_close_and_the_launch_extras(client: TestClient) -> None:
    seller = client.get("/api/data/seller", params={"date": "2025-07-01"}).json()
    assert seller["structure"] == "IRON_CONDOR" and len(seller["legs"]) == 4
    assert seller["credit"] > 0 and seller["margin"] > 0 and "Illustration" in seller["note"]
    assert client.get("/api/data/seller", params={"date": "2001-01-02"}).status_code == 404
    opts = client.get("/api/launch/options").json()
    assert opts["ppo_preset"]["spec"]["policy"] == "ppo"
    est = client.get(
        "/api/launch/estimate", params={"n_episodes": 10, "n_days": 63, "total_timesteps": 804}
    ).json()
    assert est["train_hours"] == pytest.approx(1 / 3600)
    assert (
        client.get("/api/launch/estimate", params={"n_episodes": 0, "n_days": 1}).status_code == 422
    )


# -- hardening: bounded windows, no pilot as a run, no absolute paths --------- #


@pytest.mark.behaviour
def test_the_largest_windows_are_served_and_one_past_them_is_a_422(client: TestClient) -> None:
    from optspread.webapi import MAX_DATES, MAX_FRAMES

    frames = client.get(
        "/api/data/grid/frames", params={"start": "2025-07-01", "max_frames": MAX_FRAMES}
    )
    assert frames.status_code == 200
    assert client.get("/api/data/landscape", params={"max_dates": MAX_DATES}).status_code == 200
    over = client.get("/api/data/landscape", params={"max_dates": MAX_DATES + 1})
    assert over.status_code == 422 and str(MAX_DATES) in over.json()["detail"]


@pytest.mark.behaviour
def test_the_pilots_matrix_manifest_is_not_served_as_a_run(
    client: TestClient, jobs: JobManager
) -> None:
    pilot = jobs.runs_root / "pilot"
    pilot.mkdir()
    (pilot / "matrix.npy").write_bytes(b"\x93NUMPY")
    (pilot / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": {"matrix.npy": "0" * 64}})
    )
    key = client.get("/api/runs").json()["rows"][0]["key"]
    for path, params in (
        ("/api/runs/detail", {"key": "pilot"}),
        ("/api/runs/detail", {"key": "pilot/"}),
        ("/api/runs/detail", {"key": "pilot/../pilot"}),
        ("/api/runs/compare", {"a": "pilot", "b": key}),
        ("/api/runs/attribution", {"key": "pilot"}),
        ("/api/runs/capital", {"key": "pilot"}),
        ("/api/runs/replay", {"key": "pilot"}),
    ):
        response = client.get(path, params=params)
        assert response.status_code == 404, (path, params)
    assert client.post("/api/runs/replay", params={"key": "pilot"}).status_code == 404
    assert all(r["key"] != "pilot" for r in client.get("/api/runs").json()["rows"])


@pytest.mark.invariant
def test_errors_and_health_name_no_absolute_path(
    client: TestClient, bare: TestClient, tmp_path: Path
) -> None:
    health = client.get("/api/health").json()
    assert set(health) == {"ok", "version", "git_sha"}  # the header shows the sha
    responses = [
        client.get("/api/runs/detail", params={"key": "nope"}),
        client.get("/api/runs/compare", params={"a": "nope", "b": "nada"}),
        client.get("/api/sweeps/detail", params={"key": "nope"}),
        client.get("/api/gates/M9"),
        client.get("/api/no/such/route"),
        client.get("/api/docs/NOPE.md"),
        client.get("/api/data/grid", params={"date": "2025-07-04"}),
        client.get("/api/edge/detail", params={"grid": "abc123"}),
        client.get("/api/runs/actionmix", params={"grid": "abc123"}),
        bare.get("/api/data/coverage"),
        bare.get("/api/data/grid", params={"date": "2025-07-01"}),
        bare.get("/"),
    ]
    for response in responses:
        text = response.text
        assert response.status_code >= 400 or response.request.url.path == "/", text
        assert str(tmp_path) not in text and str(Path.home()) not in text, text
        assert str(REPO_ROOT) not in text, text


@pytest.mark.invariant
def test_a_redacted_path_reads_with_forward_slashes_on_every_platform(tmp_path: Path) -> None:
    """After a root's name the path is written as Linux writes it, also where the
    separator is a backslash (Windows); a path under no root is left as it is."""
    redact = redactor([(tmp_path / "runs", "<runs root>")])
    cell = tmp_path / "runs" / "sweeps" / "tiny" / "run.json"
    assert redact(str(cell)) == "<runs root>/sweeps/tiny/run.json"
    assert redact(f"no file {cell}: gone") == "no file <runs root>/sweeps/tiny/run.json: gone"
    assert redact([str(tmp_path / "runs")]) == ["<runs root>"]
    elsewhere = str(tmp_path / "elsewhere" / "run.json")
    assert redact(elsewhere) == elsewhere


@pytest.mark.behaviour
def test_the_data_pages_question_routes(client: TestClient, bare: TestClient) -> None:
    premium = client.get("/api/data/premium").json()
    assert len(premium["date"]) == len(premium["premium"]) and premium["reading"].startswith("From")
    shape = client.get("/api/data/shape", params={"date": "2025-07-01"}).json()
    assert shape["date"] == "2025-07-01" and shape["reading"].startswith("On 2025-07-01")
    assert client.get("/api/data/shape", params={"date": "2025-07-04"}).status_code == 404
    sim = client.get("/api/data/simulated", params={"date": "2025-07-01", "premium": 1}).json()
    assert [r["name"] for r in sim["regions"]] == ["put", "atm", "call"] and sim["reading"]
    seller = client.get("/api/data/seller", params={"date": "2025-07-01"}).json()
    assert seller["reading"].startswith("At the close of 2025-07-01")
    quality = client.get("/api/data/quality").json()
    assert quality["panel_version"] is None and "<data root>" not in quality["reading"]
    assert [c["path"] for c in quality["checked"]] == [
        "panel/spx_daily.parquet",
        "panel/surface_grid.parquet",
    ]
    assert bare.get("/api/data/premium").status_code == 404
    assert bare.get("/api/data/quality").json()["checked"] == []


@pytest.mark.invariant
def test_the_run_pages_new_routes(client: TestClient) -> None:
    """Premiums, the catalog (absent here), capital cached, the mix view and the preview."""
    prem = client.get("/api/runs/premiums", params={"sweep": "tiny"}).json()
    assert prem["n_runs"] == 4 and {g["arm"] for g in prem["groups"]} == {"flat", "delta_seller"}
    assert all(c["premium"] == 2.0 for c in prem["contrasts"])
    assert client.get("/api/runs/catalog").json()["available"] is False
    key = client.get("/api/runs", params={"policy": "delta_seller", "sweep": "tiny"}).json()[
        "rows"
    ][0]["key"]
    assert client.post("/api/runs/capital", params={"key": key}).status_code == 202
    cap = _until_done(client, "/api/runs/capital", key, ("done", "failed"))
    assert cap["state"] == "done" and len(cap["paths"]) == 3 and cap["equity_band"]
    assert cap["file"].startswith("<attribution root>/capital/")
    assert client.get("/api/runs/mixview", params={"key": key}).json()["state"] == "absent"
    assert client.post("/api/runs/mixview", params={"key": key}).status_code == 202
    mix = _until_done(client, "/api/runs/mixview", key, ("cached", "failed"))
    assert mix["state"] == "cached" and mix["detail"]["n_holdings"] > 0
    tampered = next(r for r in client.get("/api/runs").json()["rows"] if r["status"] != "OK")
    assert client.post("/api/runs/mixview", params={"key": tampered["key"]}).status_code == 422
    bad = client.post("/api/launch/preview", json={"policy": "ppo", "seed": 40000}).json()
    assert bad["ok"] is False and len(bad["errors"]) >= 1
    ok = client.post("/api/launch/preview", json={"n_episodes": 3}).json()
    assert ok["ok"] and ok["day0"]["structure"] == "IRON_CONDOR" and ok["estimate"]
    cmp = client.get("/api/runs/compare", params={"a": key, "b": key}).json()
    assert cmp["verdict"] and cmp["diff_df"] == 2 and cmp["question"] == ""
