"""Ilan Danial: the API payloads carry what the harness wrote, and say where from (week 6).

The builders behind every route, tested without the web framework, except a job's
status, which no public builder serialises: that one goes through its route. The SPX
panel payloads run on a stand-in reader that serves the fixture columns, so they are
covered on the ``dev`` install; the real parquet round trip is in
``test_dashboard_panel.py`` and the wiring in ``test_server.py``.

Where a figure can be undefined, a check that it is sent as null is made on the bytes
the server sends (``_sent``, which needs the ``dashboard`` extra), whatever the model
holds; a check that a figure is there is ``_finite``, true of a number and false of
None or NaN alike.
"""

from __future__ import annotations

import json
import math
import tempfile
from concurrent.futures import Executor, Future
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dashboard_fixtures import (
    DB_DAYS,
    OM_DAYS,
    chain_columns,
    coverage_columns,
    daily_columns,
    merged,
    seam_columns,
)
from pydantic import BaseModel

from optspread.api.catalog import CATALOG_FILENAME, catalog
from optspread.api.compare import compare, compare_verdict
from optspread.api.gates import gates
from optspread.api.jobs import job_options
from optspread.api.leaks import leak_demo, leak_t_critical
from optspread.api.overview import overview
from optspread.api.panel import PanelNotBuiltError, PanelReader
from optspread.api.power import (
    POWER_DEFAULTS,
    SURFACE_PREMIUMS,
    PowerParams,
    PowerSurfaceParams,
    power_lab,
    power_surface,
    surface_seeds,
)
from optspread.api.premiums import _premium_contrast, premium_view
from optspread.api.runs import (
    RunFilters,
    histogram,
    index,
    load_detail,
    resolve_run,
    run_detail,
    run_list,
)
from optspread.api.sweeps import sweep_detail, sweeps
from optspread.dashboard.jobs.queue import JobManager
from optspread.dashboard.jobs.spec import EQUITY_FILENAME, JOBS_DIR, JobSpec
from optspread.dashboard.jobs.worker import _Progress, execute_job
from optspread.dashboard.leakdemo import run_leak_demo
from optspread.dashboard.panel_views import Columns, normalise
from optspread.dashboard.runindex import REPO_ROOT, load_run_detail
from optspread.experiment.sweep import SWEEP_FILENAME


def _sent(out: BaseModel) -> Any:
    """``out`` as the server sends it, parsed as strictly as a browser parses JSON.

    The model is returned from a route of the server's own app, so the bytes are the ones
    its redacting response writes: an undefined figure is null there, whatever the model
    holds. A NaN or an Infinity in the bytes fails the parse."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from optspread.api.server import Settings, create_app

    def refuse(token: str) -> Any:
        raise ValueError(f"{token} is not JSON")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            root / "runs",
            root / "docs",
            root / "data",
            root / "web",
            root / "README.md",
            attribution=root / "attribution",
        )
        app = create_app(settings)
        app.add_api_route("/sent", lambda: out, response_model=type(out))
        response = TestClient(app, base_url="http://127.0.0.1").get("/sent")
    assert response.status_code == 200, response.text
    return json.loads(response.content, parse_constant=refuse)


def _finite(x: float | None) -> bool:
    """A figure is there: a finite number, not None (undefined today) or NaN (undefined
    once the models keep what the views computed)."""
    return x is not None and math.isfinite(x)


# -- runs ---------------------------------------------------------------------------- #


@pytest.mark.invariant
def test_the_run_list_discloses_every_run_and_filters(run_root: Path) -> None:
    rows = index(run_root)
    out = run_list(rows, run_root, RunFilters())
    assert out.n_shown == len(rows) == out.trials.n_runs == 6
    assert out.trials.n_ok == 4 and out.trials.n_configs == 4 and out.trials.n_seeds == 1
    assert out.facets.sources == ["synthetic"] and out.facets.sweeps == ["tiny"]
    keys = [r.key for r in out.rows]
    assert len(set(keys)) == len(keys)  # unique even where two runs share a run id
    narrowed = run_list(rows, run_root, RunFilters(sweep=["tiny"], policy=["flat"], premium=[2.0]))
    assert narrowed.n_shown == 1 and narrowed.trials.n_runs == 6  # trials stay disclosed
    assert run_list(rows, run_root, RunFilters(source=["optionmetrics"])).n_shown == 0
    clean = next(r for r in out.rows if r.status == "OK")
    assert clean.source == "synthetic" and clean.source_recorded and clean.panel == ""
    tampered = next(r for r in out.rows if r.status == "DIGEST MISMATCH")
    assert tampered.mean_pnl == 1e9 and tampered.key.startswith("adhoc/")


@pytest.mark.behaviour
def test_a_row_outside_the_root_keeps_its_absolute_path(run_root: Path, tmp_path: Path) -> None:
    rows = index(run_root)
    listed = run_list(rows, tmp_path, RunFilters())  # rows that are not under the root given
    assert [r.key for r in listed.rows] == [r.path.as_posix() for r in rows]


@pytest.mark.behaviour
def test_run_keys_cannot_escape_the_root(run_root: Path) -> None:
    with pytest.raises(KeyError):
        resolve_run(run_root, "../../etc")
    with pytest.raises(KeyError):
        resolve_run(run_root, "sweeps/tiny")  # a sweep directory is not a run
    key = run_list(index(run_root), run_root, RunFilters()).rows[0].key
    assert resolve_run(run_root, key).is_dir()


@pytest.mark.behaviour
def test_a_run_detail_has_risk_histogram_provenance(run_root: Path) -> None:
    rows = index(run_root)
    seller = next(r for r in rows if r.policy == "delta_seller" and r.status == "OK")
    key = seller.path.relative_to(run_root).as_posix()
    out = run_detail(load_detail(run_root, key), run_root, rows)
    assert out.row.key == key and out.manifest["config_hash"] == seller.config_hash
    assert len(out.episodes) == 3 and out.risk.n == 3
    assert out.risk.mean == pytest.approx(seller.mean_pnl)
    assert sum(out.histogram.counts) == 3
    assert len(out.histogram.edges) == len(out.histogram.counts) + 1
    assert out.equity is None  # a sweep cell records no equity path
    assert [a.path for a in out.artifacts] == ["run.json"] and out.artifacts[0].ok
    assert out.provenance.replicas == 1 and out.provenance.sweep_cells == 4
    assert out.provenance.manifest.endswith("manifest.json")
    broken = next(r for r in rows if r.run_id == "broken-run")
    bad = run_detail(load_run_detail(broken.path), run_root, rows)
    assert bad.risk.n == 0 and bad.histogram.counts == [] and bad.provenance.sweep_cells == 0


@pytest.mark.behaviour
def test_a_launched_run_sends_its_equity_paths(tmp_path: Path) -> None:
    (tmp_path / ".jobs").mkdir()
    spec = JobSpec(n_days=21, tenor_days=(7, 14), n_episodes=2, seed=5)
    key = execute_job("j", spec.model_dump_json(), str(tmp_path))
    rows = index(tmp_path)
    assert len(rows) == 1  # the queue's own files are not runs
    out = run_detail(load_detail(tmp_path, key), tmp_path, rows)
    assert out.equity is not None and out.equity.seeds == [5, 6]
    assert out.equity.n_recorded == 2 and len(out.equity.paths[0]) == 22
    assert out.equity.paths[0][-1] == pytest.approx(out.episodes[0].pnl)
    assert {a.path for a in out.artifacts} == {"run.json", EQUITY_FILENAME}
    (tmp_path / key / EQUITY_FILENAME).write_text("[]")
    assert run_detail(load_detail(tmp_path, key), tmp_path, rows).equity is None


@pytest.mark.behaviour
def test_histograms_handle_a_constant_sample_and_cap_their_bins() -> None:
    flat = histogram(np.zeros(5))
    assert flat.counts == [5] and flat.edges == [-0.5, 0.5]
    wide = histogram(np.random.default_rng(0).standard_normal(100_000), max_bins=12)
    assert len(wide.counts) == 12 and sum(wide.counts) == 100_000


@pytest.mark.behaviour
def test_two_runs_compare_with_a_paired_difference(run_root: Path) -> None:
    rows = index(run_root)
    ok = [r for r in rows if r.status == "OK" and r.premium == 2.0]
    a, b = (next(r for r in ok if r.policy == p) for p in ("flat", "delta_seller"))
    out = compare(load_run_detail(a.path), load_run_detail(b.path), run_root, rows)
    assert [d.key for d in out.config_diff] == ["policy"]
    assert out.config_diff[0].a == "flat" and out.config_diff[0].b == "delta_seller"
    assert out.shared_seeds == 3 and out.diff_mean == pytest.approx(b.mean_pnl)
    assert sum(out.diff_histogram.counts) == 3


# -- gates ------------------------------------------------------------------------------ #


@pytest.mark.behaviour
def test_gates_payload_carries_both_boards_and_their_files(gate_docs: Path) -> None:
    out = gates(gate_docs)
    assert [b.milestone for b in out.boards] == ["M1", "M2"]
    m1, m2 = out.boards
    assert not m1.from_json and m1.source.endswith("M1_GATE_REPORT.md")
    assert m1.counts["PASS"] == 8 and m1.items[5].cited_runs == ["adddb0b7e5f604ad-s1"]
    assert m2.from_json and m2.verdict == "NOT CLEARED"
    assert m2.items[1].failed_first == "no replay has been run"


# -- jobs --------------------------------------------------------------------------------- #


@pytest.mark.behaviour
def test_job_options_describe_every_knob_with_its_bounds() -> None:
    out = job_options()
    knobs = {k.name: k for k in out.knobs}
    assert out.ppo_available == ("ppo" in out.arms)
    if not out.ppo_available:
        assert "total_timesteps" not in knobs
    assert knobs["policy"].kind == "choice" and knobs["policy"].choices == out.arms
    assert knobs["tenor_days"].kind == "tenors" and knobs["tenor_days"].default == [21, 42]
    assert knobs["costs_enabled"].kind == "boolean"
    assert knobs["n_episodes"].kind == "integer" and knobs["n_episodes"].minimum == 1
    assert knobs["n_episodes"].maximum == 2000
    assert knobs["sigma"].kind == "number" and knobs["sigma"].exclusive_minimum == 0.0
    assert knobs["sigma"].group == "Market" and knobs["sigma"].help
    # the form's own words, and which knobs a real chain does not read
    assert all(k.label and "_" not in k.label for k in out.knobs)
    assert knobs["n_days"].label == "Days per episode"
    assert {k.name for k in out.knobs if k.synthetic_only} == {"iv_premium_vol_pts", "sigma", "r"}
    words = " ".join([k.help for k in out.knobs] + [p.description for p in out.presets])
    for internal in ("Wave 0", "(A4)", "decision D2", "(D2)", "gate"):
        assert internal not in words, internal
    assert out.defaults["policy"] == "delta_seller"
    assert {p.name for p in out.presets} >= {"null_world", "planted_1", "costs_off"}
    planted = next(p for p in out.presets if p.name == "planted_1")
    assert planted.spec["iv_premium_vol_pts"] == 1.0
    # the source toggle: the generator always, where each source's jobs land, A8's dates
    assert out.sources[0] == "synthetic" and set(out.sources) <= set(knobs["source"].choices)
    assert out.landing == {"synthetic": "launched", "optionmetrics": "optionmetrics/launched"}
    assert out.splits == {
        "train": ["2010-01-01", "2018-12-31"],
        "eval": ["2019-01-01", "2022-08-24"],
    }


class _NeverStarted(Executor):
    """A pool that holds every job queued, so the test can write its progress as a worker."""

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        return Future()


@pytest.mark.behaviour
def test_a_job_status_serialises(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from optspread.api.server import Settings, create_app

    manager = JobManager(tmp_path, executor=_NeverStarted())
    settings = Settings(tmp_path, tmp_path, tmp_path / "nodata", tmp_path / "noweb", tmp_path)
    with TestClient(create_app(settings, manager), base_url="http://127.0.0.1") as client:
        knobs = {"policy": "flat", "n_episodes": 10}
        job_id = client.post("/api/jobs", json=knobs).json()["job_id"]
        progress = _Progress(tmp_path / JOBS_DIR, job_id)  # what a running worker writes
        progress.write(done=3, total=10, phase="evaluating")
        progress.log("evaluated 3 of 10")
        out = client.get(f"/api/jobs/{job_id}").json()
    assert out["state"] == "running" and out["phase"] == "evaluating"
    assert out["fraction"] == pytest.approx(0.3)
    assert len(out["log_tail"]) == 1 and out["log_tail"][0].endswith("evaluated 3 of 10")


# -- the power lab ------------------------------------------------------------------------ #


@pytest.mark.behaviour
def test_the_power_lab_agrees_with_the_formula() -> None:
    p = PowerParams(tau2=0.0, sigma2=1.0, n_seeds=4, n_episodes=25, delta=1.0)
    out = power_lab(p)
    assert out.var_grand_mean == pytest.approx(0.01) and out.se == pytest.approx(0.1)
    # the exact t floor (#20): the noncentrality at S = 4, 3.2994, times se, not
    # (z_.95 + z_.80) se = 0.2486
    assert out.floor == pytest.approx(3.2994 * 0.1, rel=1e-4)
    # budget (1 / 3.2994)^2 = 0.0919, so N >= 1 / (4 x 0.0919) = 2.72 -> 3 (the normal form: 2)
    assert out.variance_budget == pytest.approx(0.09186, rel=1e-3)
    assert out.power == pytest.approx(1.0) and out.episodes_needed == 3
    assert out.curve.power[0] == pytest.approx(0.05) and out.curve.power[-1] > 0.99
    assert len(out.curve.delta) == len(out.curve.power_s5) == len(out.curve.power_s50) == 81
    assert out.design.total_hours is not None and out.design.n_seeds == 4
    assert "tau^2/S" in out.source


@pytest.mark.behaviour
def test_the_floor_falls_faster_than_one_over_root_s_when_seeds_dominate() -> None:
    out = power_lab(PowerParams(tau2=1e6, sigma2=0.0, delta=1000.0))
    # sqrt(10) by the normal form; the exact t adds the fall of its noncentrality (#20)
    assert out.floor_vs_seeds.ratio == pytest.approx(3.812, abs=1e-3)
    assert out.floor_vs_seeds.ratio > math.sqrt(10.0)
    s = out.floor_vs_seeds
    assert s.seeds[0] == 2 and s.floor == sorted(s.floor, reverse=True)
    assert any(not d.n_episodes for d in out.frontier)  # few seeds: infeasible at any N
    infeasible = next(d for d in out.frontier if d.n_episodes is None)
    assert infeasible.power is None and infeasible.total_hours is None


@pytest.mark.behaviour
def test_power_params_are_bounded() -> None:
    with pytest.raises(ValueError):
        PowerParams(alpha=0.6)
    with pytest.raises(ValueError):
        PowerParams(delta=0.0)
    with pytest.raises(ValueError):  # one seed mean has no t test
        PowerParams(n_seeds=1)
    assert PowerParams().tau2 == POWER_DEFAULTS["tau2"]


@pytest.mark.invariant
def test_the_power_surface_is_exact_t_with_the_contour_from_its_inverse() -> None:
    from optspread import power as pw

    p = PowerSurfaceParams(tau2=51_838.0, sigma2=1.06e8, n_episodes=500, delta=824.3)
    out = power_surface(p)
    assert len(out.premium) == SURFACE_PREMIUMS and out.premium[-1] == pytest.approx(4.0)
    assert out.seeds[:3] == [2, 3, 4] and out.seeds[-1] == 100 and len(out.power) == len(out.seeds)
    s5 = out.seeds.index(5)
    assert out.power[s5][0] == pytest.approx(0.05)  # no premium: the test's own alpha
    cell = pw.power_t(824.3 * out.premium[10], 51_838.0, 1.06e8, 5, 500)
    assert out.power[s5][10] == pytest.approx(cell)
    row = [v for v in out.power[s5] if v is not None]
    assert row == sorted(row)  # more premium, more power
    # A7: S = 5, N = 500 has an exact-t floor of 698.3 dollars, 0.847 vol points here
    assert out.contour_premium[s5] == pytest.approx(698.3 / 824.3, abs=1e-3)
    floors = [c for c in out.contour_premium if c is not None]
    assert floors == sorted(floors, reverse=True)  # more seeds, smaller floor
    assert "power_t" in out.source
    assert surface_seeds(12) == list(range(2, 13)) and surface_seeds(41)[-2:] == [40, 41]
    with pytest.raises(ValueError):
        PowerSurfaceParams(max_premium=0.0)


# -- the leak demo ----------------------------------------------------------------------- #


@pytest.mark.behaviour
def test_the_leak_payload_has_every_field_for_both_envs() -> None:
    out = leak_demo(run_leak_demo(eps=0.5, n_seeds=4))
    assert out.caught and not out.clean.flagged and out.poisoned.flagged
    assert out.clean.past_unchanged and not out.poisoned.past_unchanged
    assert len(out.clean.features) == len(out.poisoned.features) == 14
    sent = _sent(out)  # no NaN reaches the wire
    untested = [f for f in sent["clean"]["features"] if not f["tested"]]
    assert untested and all(f["t_stat"] is None and f["reason"] for f in untested)
    assert out.poisoned.worst_feature == "ret_1d" and out.seeds == [0, 1, 2, 3]


@pytest.mark.behaviour
def test_the_leak_threshold_in_t_flags_exactly_the_fields_p_flags() -> None:
    out = leak_demo(run_leak_demo(eps=0.5, n_seeds=4))
    for side in (out.clean, out.poisoned):
        assert side.t_critical is not None and side.t_critical > 0
        for f in side.features:
            if f.tested and f.t_stat is not None and f.p_value is not None:
                assert (abs(f.t_stat) > side.t_critical) == (f.p_value < side.threshold)
    assert leak_t_critical(0.01, 2) is None


# -- sweeps ------------------------------------------------------------------------------ #


@pytest.mark.behaviour
def test_sweeps_list_and_open_with_the_oracle_curve(run_root: Path) -> None:
    out = sweeps(run_root)
    assert [s.key for s in out.sweeps] == ["sweeps/tiny"]
    assert out.sweeps[0].n_cells == 4
    # no stale "waiting for sweep" placeholder rides along: the analyses have pages of their own
    assert "power_curve" not in out.model_dump() and "attribution" not in out.model_dump()
    detail = sweep_detail(run_root, "sweeps/tiny")
    assert set(detail.oracle) == {"flat", "delta_seller"}
    flat = detail.oracle["flat"]
    assert [p.premium for p in flat] == [0.0, 2.0] and all(p.lift == 0.0 for p in flat)
    assert detail.source.endswith(SWEEP_FILENAME)
    assert "waiting" not in detail.model_dump_json()
    with pytest.raises(KeyError):
        sweep_detail(run_root, "adhoc")
    assert sweeps(run_root / "absent").sweeps == []


@pytest.mark.behaviour
def test_an_unreadable_sweep_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / SWEEP_FILENAME).write_text("{}")
    assert sweeps(tmp_path).sweeps == []


# -- the SPX panel, on a stand-in reader ------------------------------------------------- #


def _chain_for(day: date) -> Columns:
    parts = []
    if day in OM_DAYS:
        parts.append(chain_columns((day,), "optionmetrics"))
    if day in DB_DAYS:
        parts.append(chain_columns((day,), "databento"))
    return normalise(merged(*parts)) if parts else {}


def _fake_read(path: Path, columns: Any = None, day: date | None = None) -> Columns:
    name = path.name
    if name == "spx_daily.parquet":
        return normalise(
            merged(
                daily_columns(date(2025, 1, 2), 130, "optionmetrics", seed=1),
                daily_columns(date(2025, 7, 1), 30, "databento", seed=2),
            )
        )
    if name == "coverage.parquet":
        return normalise(coverage_columns())
    if name == "seam_overlap.parquet":
        return normalise(seam_columns())
    assert day is not None and columns is not None
    return _chain_for(day)


@pytest.fixture()
def reader(tmp_path: Path) -> PanelReader:
    panel = tmp_path / "panel"
    (panel / "spx" / "year=2025").mkdir(parents=True)
    for rel in ("spx_daily.parquet", "coverage.parquet", "seam_overlap.parquet"):
        (panel / rel).write_bytes(b"")
    (panel / "spx" / "year=2025" / "part.parquet").write_bytes(b"")
    return PanelReader(tmp_path, chain_cache=2, read=_fake_read)


@pytest.mark.behaviour
def test_an_unbuilt_panel_raises_not_built(tmp_path: Path) -> None:
    r = PanelReader(tmp_path)
    status = r.status()
    assert not status.built and status.years == [] and len(status.missing) == 3
    assert [f.present for f in status.files] == [False, False, False]
    with pytest.raises(PanelNotBuiltError):
        r.coverage_out()
    with pytest.raises(PanelNotBuiltError):
        r.chain(date(2025, 7, 1))
    assert r.seam() == {}


@pytest.mark.behaviour
def test_the_status_lists_every_panel_file(reader: PanelReader) -> None:
    status = reader.status()
    assert status.built and status.years == [2025] and status.missing == []
    assert all(f.present and f.modified for f in status.files) and len(status.files) == 4


@pytest.mark.behaviour
def test_the_coverage_payload_has_strips_missing_days_and_the_overlap(
    reader: PanelReader,
) -> None:
    out = reader.coverage_out()
    assert [s.source for s in out.spans] == ["databento", "optionmetrics"]
    assert out.missing == [("2025-07-02", "databento")]
    assert out.overlap == ("2025-07-01", "2025-07-03", 2)
    db = out.strips[0]
    assert db.date == sorted(db.date) and db.missing.count(True) == 1
    assert reader.dates("optionmetrics") == [d.isoformat() for d in OM_DAYS]
    assert reader.coverage() is reader.coverage()  # cached until the file changes


@pytest.mark.behaviour
def test_the_chain_payloads_for_one_date(reader: PanelReader) -> None:
    day = date(2025, 7, 1)
    listing = reader.chain_index(day, None, [])
    assert listing.sources == ["databento", "optionmetrics"] and listing.roots == ["SPX", "SPXW"]
    assert [e.dte for e in listing.expiries] == [7, 30, 60, 120]
    assert listing.default_expiry == (day + timedelta(days=30)).isoformat()
    assert listing.expiries[1].carry_fit == "near+r_ref"

    smile = reader.smile_out(day, None, "optionmetrics", [])
    assert smile.dte == 30 and smile.has_vendor and smile.data_source == "optionmetrics"
    assert _finite(smile.forward) and len(smile.iv) == len(smile.cp)
    db = reader.smile_out(day, date(2025, 7, 31), "databento", ["SPX"])
    vendor = _sent(db)["iv_vendor"]  # Databento carries no vendor IV: null on the wire
    assert not db.has_vendor and vendor and all(v is None for v in vendor)

    surface = reader.surface_out(day, "optionmetrics", [], min_dte=20)
    assert surface.dte == [30, 60, 120] and len(surface.iv[0]) == len(surface.moneyness)

    term = reader.term_out(day, None, [])
    assert term.dte == [7, 30, 60, 120] and all(_finite(v) for v in term.atm_iv)
    assert term.data_source == "optionmetrics"  # the default source when both quote


@pytest.mark.behaviour
def test_a_date_with_no_quotes_has_no_smile(reader: PanelReader) -> None:
    with pytest.raises(PanelNotBuiltError):
        reader.smile_out(date(2025, 3, 3), None, None, [])
    with pytest.raises(PanelNotBuiltError):
        reader.chain(date(2019, 3, 4))  # no partition for the year


@pytest.mark.internal
def test_the_chain_cache_keeps_the_last_few_dates(reader: PanelReader) -> None:
    reads: list[date] = []

    def counted(path: Path, columns: Any = None, day: date | None = None) -> Columns:
        if day is not None:
            reads.append(day)
        return _fake_read(path, columns, day)

    cached = PanelReader(reader.layout.root, chain_cache=2, read=counted)
    first, second, third = OM_DAYS[:3]
    for day in (first, second, third, first):  # two dates kept: the first had gone
        cached.chain(day)
    assert reads == [first, second, third, first]
    cached.chain(third)
    cached.chain(first)  # the last two dates come from the cache
    assert reads == [first, second, third, first]


@pytest.mark.behaviour
def test_the_daily_payload_per_source_with_a_range(reader: PanelReader) -> None:
    out = reader.daily_out()
    assert [s.data_source for s in out.series] == ["databento", "optionmetrics"]
    om = out.stats["optionmetrics"]
    assert om.n_days == 127 and om.mean_vol_pts is not None and 1.0 < om.mean_vol_pts < 5.0
    clipped = reader.daily_out(date(2025, 3, 1), date(2025, 3, 31))
    assert all("2025-03" in d for s in clipped.series for d in s.date)
    assert out.series[1].spot_parity[0] == pytest.approx(5000.5)
    sent = _sent(out)  # the trailing NaNs went out as null
    assert [s["rv21_fwd"][-3:] for s in sent["series"]] == [[None, None, None]] * 2


@pytest.mark.behaviour
def test_the_seam_payload(reader: PanelReader) -> None:
    out = reader.seam_out()
    assert out.available and out.date == ["2025-07-01", "2025-07-03"]
    assert out.medians["iv_diff_median_volpts"] == pytest.approx(0.2)
    # the headline is each day's 95th percentile averaged over the days, as docs/DATA.md says
    p95 = [x for x in out.stats["iv_diff_p95_abs_volpts"] if x is not None]
    assert out.p95_mean == pytest.approx(sum(p95) / len(p95)) and out.p95_worst == pytest.approx(
        max(p95)
    )
    assert out.iv30_sources == ["databento", "optionmetrics"] and out.iv30_date


@pytest.mark.behaviour
def test_without_a_seam_file_the_seam_is_unavailable(tmp_path: Path) -> None:
    r = PanelReader(tmp_path, read=_fake_read)
    out = r.seam_out()
    assert not out.available and out.iv30_date == [] and out.medians == {}


# -- the 3D views over the panel ------------------------------------------------------- #


@pytest.fixture()
def built(panel_root: Path) -> PanelReader:
    """The fixture panel on disk, every file the build writes, read through pyarrow."""
    return PanelReader(panel_root)


@pytest.mark.behaviour
def test_the_grid_payloads_one_day_and_a_window(built: PanelReader) -> None:
    dates = built.grid_dates()
    assert dates == sorted({d.isoformat() for d in (*OM_DAYS, *DB_DAYS)})
    day = built.grid_out(date(2025, 7, 1))
    assert day.tenor_days[0] == 7 and day.tenor_days[-1] == 365 and len(day.k) == 33
    assert len(day.iv) == len(day.tenor_days) and all(len(r) == len(day.k) for r in day.iv)
    sent = _sent(day)
    assert sent["iv"][0][0] is None  # nothing quoted at k = -0.5, sent as null
    assert any(_finite(v) for row in day.iv for v in row)
    with pytest.raises(PanelNotBuiltError):
        built.grid_out(date(2025, 7, 4))  # a holiday: not in the grid, not filled
    frames = built.grid_frames(date(2025, 6, 30), None, max_frames=3)
    assert [f.date for f in frames.frames] == dates[1:4] and frames.more
    assert _sent(frames)["frames"][1]["iv"] == sent["iv"] and frames.k == day.k
    rest = built.grid_frames(date(2025, 7, 3), date(2025, 7, 7))
    assert [f.date for f in rest.frames] == ["2025-07-03", "2025-07-07"] and not rest.more
    assert built.grid_frames(date(2026, 1, 1), None).frames == []
    with pytest.raises(ValueError):
        built.grid_frames(date(2025, 7, 1), None, max_frames=0)


@pytest.mark.behaviour
def test_the_landscape_payload(built: PanelReader) -> None:
    land = built.landscape_out()
    assert land.stride == 1 and land.n_days == len(land.date) == 7
    assert land.tenor_days[0] == 7 and len(land.iv[0]) == len(land.tenor_days)
    atm30 = land.iv[0][land.tenor_days.index(30)]
    assert atm30 is not None and 0.1 < atm30 < 0.3
    # no expiry past 120 days: the 365-day point is not invented, and is sent as null
    assert _sent(land)["iv"][0][-1] is None
    short = built.landscape_out(max_dates=3)
    assert short.stride == 3 and len(short.date) == 3 and short.date[-1] == land.date[-1]
    window = built.landscape_out(date(2025, 7, 1), date(2025, 7, 3))
    assert window.first == "2025-07-01" and window.last == "2025-07-03" and window.n_days == 3
    assert built.landscape_out(date(2030, 1, 1)).first is None
    with pytest.raises(ValueError):
        built.landscape_out(max_dates=1)
    with pytest.raises(ValueError):
        built.landscape_out(date(2025, 7, 3), date(2025, 7, 1))


@pytest.mark.behaviour
def test_the_simulated_surface_is_the_simulators_own_chain(built: PanelReader) -> None:
    from optspread.config import MarketConfig

    cfg = MarketConfig()
    day = date(2025, 7, 1)
    sim = built.simulated_out(day, 1.0)
    assert sim.sigma_from == "config" and sim.sigma == cfg.sigma and sim.premium == 1.0
    assert sim.level == pytest.approx(cfg.sigma + 0.01)
    assert len(sim.strike) == 2 * cfg.n_strikes_each_side + 1
    assert sim.tenor_trading_days == list(cfg.tenor_days)
    assert sim.tenor_days == pytest.approx([t * 365 / 252 for t in cfg.tenor_days])
    assert all(v == pytest.approx(sim.level) for row in sim.iv for v in row)  # one number
    assert sim.spot_from == "spx_daily.parquet" and sim.strike[16] == round(sim.spot / 50) * 50
    assert sim.k[0][16] == pytest.approx(math.log(sim.strike[16] / sim.spot) - cfg.r * 21 / 252)
    # the real surface at the same contracts: skewed, so it differs most in the put wing
    assert _finite(sim.real[0][16]) and _finite(sim.mean_abs_diff)
    assert sim.max_abs_diff is not None and sim.max_abs_diff >= sim.mean_abs_diff
    assert "SyntheticChainProvider" in sim.source and "iv_premium_vol_pts=1" in sim.source
    matched = built.simulated_out(day, 1.0, "atm")
    iv30 = built._daily_row(day)["iv30"]
    assert matched.level == pytest.approx(iv30) and matched.sigma == pytest.approx(iv30 - 0.01)
    assert built._daily_row(date(2019, 1, 2)) == {}  # a day the daily table does not hold
    realised = built.simulated_out(day, 0.0, "rv21")
    assert realised.sigma == pytest.approx(built._daily_row(day)["rv21"])
    with pytest.raises(ValueError, match="sigma_from"):
        built.simulated_out(day, 1.0, "vix")
    with pytest.raises(ValueError, match="not positive"):
        built.simulated_out(day, 19.0, "atm")
    with pytest.raises(PanelNotBuiltError):
        built.simulated_out(date(2025, 7, 4), 1.0)


@pytest.mark.behaviour
def test_the_simulated_surface_without_the_daily_table(tmp_path: Path, panel_root: Path) -> None:
    import shutil

    shutil.copytree(panel_root, tmp_path / "copy")
    (tmp_path / "copy" / "panel" / "spx_daily.parquet").unlink()
    r = PanelReader(tmp_path / "copy")
    sim = r.simulated_out(date(2025, 7, 1), 0.0)
    assert sim.spot_from == "config" and sim.spot == 5000.0
    with pytest.raises(ValueError, match="no rv21"):
        r.simulated_out(date(2025, 7, 1), 0.0, "rv21")


@pytest.mark.behaviour
def test_the_3d_views_need_their_files(tmp_path: Path) -> None:
    r = PanelReader(tmp_path)
    with pytest.raises(PanelNotBuiltError):
        r.grid_dates()
    with pytest.raises(PanelNotBuiltError):
        r.landscape_out()


# -- the overview ------------------------------------------------------------------------- #


@pytest.mark.behaviour
def test_the_overview_brings_the_tiles_together(
    run_root: Path, gate_docs: Path, reader: PanelReader
) -> None:
    rows = index(run_root)
    out = overview(REPO_ROOT / "README.md", gate_docs, run_root, reader, rows)
    assert out.summary[0].startswith("A leak-tested SPX options simulator")
    assert [(g.milestone, g.verdict) for g in out.gates] == [("M1", "PASS"), ("M2", "NOT CLEARED")]
    assert out.n_runs == 6 and out.n_ok == 4 and out.sweeps == 1 and len(out.latest) == 6
    assert out.overlap == ("2025-07-01", "2025-07-03", 2) and len(out.coverage) == 2
    bare = overview(REPO_ROOT / "README.md", gate_docs, run_root, PanelReader(run_root), ())
    assert bare.coverage == [] and bare.overlap is None and not bare.data.built


@pytest.mark.behaviour
def test_the_run_payload_carries_the_recorded_source_and_panel(
    run_root: Path, tmp_path: Path
) -> None:
    clean = next(r for r in index(run_root) if r.status == "OK")
    manifest = json.loads((clean.path / "manifest.json").read_text())
    for name, data in (("old", None), ("real", {"source": "optionmetrics", "panel": {"a": "b"}})):
        (tmp_path / name).mkdir()
        if data is None:
            manifest.pop("data")
        else:
            manifest["data"] = data
        (tmp_path / name / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / name / "run.json").write_bytes((clean.path / "run.json").read_bytes())
    out = run_list(index(tmp_path), tmp_path, RunFilters())
    by_key = {r.key: r for r in out.rows}
    assert (by_key["old"].source, by_key["old"].source_recorded) == ("synthetic", False)
    assert by_key["real"].source == "optionmetrics" and by_key["real"].source_recorded
    assert len(by_key["real"].panel) == 16 and by_key["old"].panel == ""
    assert out.facets.sources == ["optionmetrics", "synthetic"]
    real = run_list(index(tmp_path), tmp_path, RunFilters(source=["optionmetrics"]))
    assert [r.key for r in real.rows] == ["real"]


# -- the runs list's premium view, the paired compare and the catalog ------------------ #


@pytest.mark.invariant
def test_premiums_pool_each_arm_and_pair_it_with_itself_at_zero(run_root: Path) -> None:
    out = premium_view(index(run_root), run_root, RunFilters(sweep=["tiny"]))
    assert out.n_runs == 4 and out.n_skipped == 0 and len(out.worlds) == 1
    assert out.worlds[0].label == "" and out.worlds[0].arms == ["delta_seller", "flat"]
    cells = {(g.arm, g.premium): g for g in out.groups}
    seller = cells[("delta_seller", 2.0)]
    assert seller.n_runs == 1 and seller.n_episodes == 3 and seller.interval == "episodes"
    assert (
        seller.p5 is not None and seller.p95 is not None and seller.p5 <= seller.p50 <= seller.p95
    )
    assert seller.ci_low is not None and seller.ci_low < (seller.mean or 0) < (seller.ci_high or 0)
    contrast = next(c for c in out.contrasts if c.arm == "delta_seller")
    pnl = {
        p: {
            e.seed: e.pnl
            for e in load_run_detail(
                run_root / cells[("delta_seller", p)].run_keys[0]
            ).record.episodes
        }  # type: ignore[union-attr]
        for p in (0.0, 2.0)
    }
    diffs = [pnl[2.0][s] - pnl[0.0][s] for s in sorted(pnl[0.0])]
    assert contrast.n_blocks == 1 and contrast.n_pairs == 3 and contrast.interval == "episodes"
    assert contrast.estimate == pytest.approx(float(np.mean(diffs)))
    flat = next(c for c in out.contrasts if c.arm == "flat")
    assert flat.estimate == 0.0 and not flat.detected  # flat earns nothing anywhere
    assert {c.premium for c in out.contrasts} == {2.0}  # zero is the control, never a contrast


@pytest.mark.invariant
def test_premium_contrasts_over_runs_use_the_decision_rule() -> None:
    first = [(f"k{i}", 100 * i, {100 * i + j: float(j + i) for j in range(4)}) for i in range(3)]
    treated = [(k, s, {e: v + 10.0 + s / 100 for e, v in eps.items()}) for k, s, eps in first]
    c = _premium_contrast("delta_seller", "", "x", "synthetic", 1.0, treated, first)
    assert c.interval == "runs" and c.n_blocks == 3 and c.df == 2
    assert c.estimate == pytest.approx(11.0) and c.se == pytest.approx(1 / math.sqrt(3))
    assert c.ci_low is not None and c.ci_low < 11.0 < (c.ci_high or 0) and c.detected
    alone = _premium_contrast("ppo", "", "x", "synthetic", 1.0, treated, [])
    assert alone.estimate is None and "no run at zero" in alone.reason


@pytest.mark.behaviour
def test_a_compare_says_in_one_sentence_what_the_pair_shows(run_root: Path) -> None:
    rows = index(run_root)
    flat, seller = (
        next(r for r in rows if r.policy == p and r.premium == 2.0 and r.sweep == "tiny")
        for p in ("flat", "delta_seller")
    )
    cmp = compare(load_run_detail(flat.path), load_run_detail(seller.path), run_root, rows)
    assert cmp.diff_df == 2 and len(cmp.diffs) == 3 and cmp.diffs[0].a == 0.0
    assert cmp.diff_ci_low is not None and cmp.diff_ci_high is not None
    assert cmp.diff_ci_low < (cmp.diff_mean or 0) < cmp.diff_ci_high
    assert cmp.verdict.endswith(".") and "3" in cmp.verdict and cmp.a_label == ""
    clear = compare_verdict(flat, seller, 50, 100.0, 20.0, 180.0, ("A one", "B one"))
    assert clear.startswith("B one made $100 more per episode than A one on the same 50 paths")
    assert "(\u00b1 $80, 95%): a clear difference." in clear
    unclear = compare_verdict(flat, seller, 50, -10.0, -30.0, 10.0, ("A one", "B one"))
    assert unclear.startswith("B one made $10 less per episode than A one")
    assert "not distinguishable from noise at this sample size; run more episodes" in unclear
    assert "exactly the same on all 5" in compare_verdict(flat, seller, 5, 0.0, 0.0, 0.0)
    assert "share no seeds" in compare_verdict(flat, seller, 0, None, None, None)


@pytest.mark.behaviour
def test_a_catalog_groups_runs_by_question_and_pairs_them(run_root: Path, tmp_path: Path) -> None:
    rows = index(run_root)
    assert not catalog(run_root, rows).available  # no catalog: every page as before
    root = tmp_path / "runs"
    import shutil

    shutil.copytree(run_root, root)
    rows = index(root)
    listed = run_list(rows, root, RunFilters(sweep=["tiny"])).rows  # the keys the run list sends
    keys = {(r.policy, r.premium): r.key for r in listed if r.status == "OK"}
    entries = [
        {
            "key": keys[(p, prem)],
            "group": "arms",
            "question": "Which arm earns the premium?",
            "label": f"{p} at {prem:g}",
            "settings": {"policy": p, "iv_premium_vol_pts": prem, "seed": 7},
        }
        for p in ("flat", "delta_seller")
        for prem in (0.0, 2.0)
    ] + [{"key": "launched/missing-s1", "group": "arms", "label": "gone", "settings": {}}]
    (root / CATALOG_FILENAME).write_text(json.dumps({"seed": 7, "runs": entries}))
    out = catalog(root, rows)
    assert out.available and out.seed == 7 and len(out.groups) == 1
    group = out.groups[0]
    assert group.question == "Which arm earns the premium?" and len(group.runs) == 5
    assert [r.present for r in group.runs].count(False) == 1
    # at each premium, the arm away from the launch defaults against the seller (the default)
    assert {(p.premium, p.varies[0]) for p in group.pairs} == {(0.0, "policy"), (2.0, "policy")}
    pair = next(p for p in group.pairs if p.premium == 2.0)
    assert (
        pair.a == keys[("delta_seller", 2.0)]
        and pair.title == "flat at 2 against delta_seller at 2"
    )
    assert pair.n_shared == 3 and pair.diff_mean is not None and pair.diff_ci_low is not None
    cmp = compare(load_run_detail(root / pair.a), load_run_detail(root / pair.b), root, rows, out)
    assert cmp.question == group.question and cmp.a_label == "delta_seller at 2"
    (root / CATALOG_FILENAME).write_text("{not json")
    broken = catalog(root, rows)
    assert not broken.available and broken.problem
