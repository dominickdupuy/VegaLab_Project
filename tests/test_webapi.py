"""Ilan Danial: the API payloads carry what the harness wrote, and say where from (week 6).

The builders behind every route, tested without the web framework. The SPX panel
payloads run on a stand-in reader that serves the fixture columns, so they are
covered on the ``dev`` install; the real parquet round trip is in
``test_dashboard_panel.py`` and the wiring in ``test_server.py``.
"""

from __future__ import annotations

import json
import math
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

from optspread import webapi as w
from optspread.dashboard_data import REPO_ROOT, load_run_detail, run_leak_demo
from optspread.dashboard_panel import Columns, normalise
from optspread.jobs import EQUITY_FILENAME, JobSpec, JobStatus, execute_job
from optspread.sweep import SWEEP_FILENAME

# -- runs ---------------------------------------------------------------------------- #


def test_the_run_list_discloses_every_run_and_filters(run_root: Path) -> None:
    rows = w.index(run_root)
    out = w.run_list(rows, run_root, w.RunFilters())
    assert out.n_shown == len(rows) == out.trials.n_runs == 6
    assert out.trials.n_ok == 4 and out.trials.n_configs == 4 and out.trials.n_seeds == 1
    assert out.facets.sources == ["synthetic"] and out.facets.sweeps == ["tiny"]
    keys = [r.key for r in out.rows]
    assert len(set(keys)) == len(keys)  # unique even where two runs share a run id
    narrowed = w.run_list(
        rows, run_root, w.RunFilters(sweep=["tiny"], policy=["flat"], premium=[2.0])
    )
    assert narrowed.n_shown == 1 and narrowed.trials.n_runs == 6  # trials stay disclosed
    assert w.run_list(rows, run_root, w.RunFilters(source=["optionmetrics"])).n_shown == 0
    tampered = next(r for r in out.rows if r.status == "DIGEST MISMATCH")
    assert tampered.mean_pnl == 1e9 and tampered.key.startswith("adhoc/")


def test_a_row_outside_the_root_keeps_its_absolute_path(run_root: Path, tmp_path: Path) -> None:
    row = w.index(run_root)[0]
    assert w.row_out(row, tmp_path).key == row.path.as_posix()


def test_run_keys_cannot_escape_the_root(run_root: Path) -> None:
    with pytest.raises(KeyError):
        w.resolve_run(run_root, "../../etc")
    with pytest.raises(KeyError):
        w.resolve_run(run_root, "sweeps/tiny")  # a sweep directory is not a run
    key = w.run_list(w.index(run_root), run_root, w.RunFilters()).rows[0].key
    assert w.resolve_run(run_root, key).is_dir()


def test_a_run_detail_has_risk_histogram_provenance(run_root: Path) -> None:
    rows = w.index(run_root)
    seller = next(r for r in rows if r.policy == "delta_seller" and r.status == "OK")
    key = seller.path.relative_to(run_root).as_posix()
    out = w.run_detail(w.load_detail(run_root, key), run_root, rows)
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
    bad = w.run_detail(load_run_detail(broken.path), run_root, rows)
    assert bad.risk.n == 0 and bad.histogram.counts == [] and bad.provenance.sweep_cells == 0


def test_a_launched_run_sends_its_equity_paths(tmp_path: Path) -> None:
    (tmp_path / ".jobs").mkdir()
    spec = JobSpec(n_days=21, tenor_days=(7, 14), n_episodes=2, seed=5)
    key = execute_job("j", spec.model_dump_json(), str(tmp_path))
    rows = w.index(tmp_path)
    assert len(rows) == 1  # the queue's own files are not runs
    out = w.run_detail(w.load_detail(tmp_path, key), tmp_path, rows)
    assert out.equity is not None and out.equity.seeds == [5, 6]
    assert out.equity.n_recorded == 2 and len(out.equity.paths[0]) == 22
    assert out.equity.paths[0][-1] == pytest.approx(out.episodes[0].pnl)
    assert {a.path for a in out.artifacts} == {"run.json", EQUITY_FILENAME}
    (tmp_path / key / EQUITY_FILENAME).write_text("[]")
    assert w.run_detail(w.load_detail(tmp_path, key), tmp_path, rows).equity is None


def test_histograms_handle_a_constant_sample_and_cap_their_bins() -> None:
    flat = w.histogram(np.zeros(5))
    assert flat.counts == [5] and flat.edges == [-0.5, 0.5]
    wide = w.histogram(np.random.default_rng(0).standard_normal(100_000), max_bins=12)
    assert len(wide.counts) == 12 and sum(wide.counts) == 100_000


def test_two_runs_compare_with_a_paired_difference(run_root: Path) -> None:
    rows = w.index(run_root)
    ok = [r for r in rows if r.status == "OK" and r.premium == 2.0]
    a, b = (next(r for r in ok if r.policy == p) for p in ("flat", "delta_seller"))
    out = w.compare(load_run_detail(a.path), load_run_detail(b.path), run_root, rows)
    assert [d.key for d in out.config_diff] == ["policy"]
    assert out.config_diff[0].a == "flat" and out.config_diff[0].b == "delta_seller"
    assert out.shared_seeds == 3 and out.diff_mean == pytest.approx(b.mean_pnl)
    assert sum(out.diff_histogram.counts) == 3


# -- gates ------------------------------------------------------------------------------ #


def test_gates_payload_carries_both_boards_and_their_files(gate_docs: Path) -> None:
    out = w.gates(gate_docs)
    assert [b.milestone for b in out.boards] == ["M1", "M2"]
    m1, m2 = out.boards
    assert not m1.from_json and m1.source.endswith("M1_GATE_REPORT.md")
    assert m1.counts["PASS"] == 8 and m1.items[5].cited_runs == ["adddb0b7e5f604ad-s1"]
    assert m2.from_json and m2.verdict == "NOT CLEARED"
    assert m2.items[1].failed_first == "no replay has been run"


# -- jobs --------------------------------------------------------------------------------- #


def test_job_options_describe_every_knob_with_its_bounds() -> None:
    out = w.job_options()
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
    assert out.defaults["policy"] == "delta_seller"
    assert {p.name for p in out.presets} >= {"null_world", "planted_1", "costs_off"}
    planted = next(p for p in out.presets if p.name == "planted_1")
    assert planted.spec["iv_premium_vol_pts"] == 1.0


def test_a_job_status_serialises() -> None:
    status = JobStatus("abc", "running", "evaluating", 3, 10, "x-s0", "", "t", "flat", "", ("l",))
    out = w.job_out(status)
    assert out.fraction == pytest.approx(0.3) and out.log_tail == ["l"]


# -- the power lab ------------------------------------------------------------------------ #


def test_the_power_lab_agrees_with_the_formula() -> None:
    p = w.PowerParams(tau2=0.0, sigma2=1.0, n_seeds=4, n_episodes=25, delta=1.0)
    out = w.power_lab(p)
    assert out.var_grand_mean == pytest.approx(0.01) and out.se == pytest.approx(0.1)
    # with no seed variance the floor is (z_.95 + z_.80) * se
    assert out.floor == pytest.approx((1.6449 + 0.8416) * 0.1, rel=1e-3)
    assert out.power == pytest.approx(1.0) and out.episodes_needed == 2  # 1/(4 x 0.1618) -> 2
    assert out.curve.power[0] == pytest.approx(0.05) and out.curve.power[-1] > 0.99
    assert len(out.curve.delta) == len(out.curve.power_s5) == len(out.curve.power_s50) == 81
    assert out.design.total_hours is not None and out.design.n_seeds == 4
    assert "tau^2/S" in out.source


def test_the_floor_falls_as_one_over_root_s_when_seeds_dominate() -> None:
    out = w.power_lab(w.PowerParams(tau2=1e6, sigma2=0.0, delta=1000.0))
    assert out.floor_vs_seeds.ratio == pytest.approx(math.sqrt(10.0))
    s = out.floor_vs_seeds
    assert s.seeds[0] == 1 and s.floor == sorted(s.floor, reverse=True)
    assert any(not d.n_episodes for d in out.frontier)  # few seeds: infeasible at any N
    infeasible = next(d for d in out.frontier if d.n_episodes is None)
    assert infeasible.power is None and infeasible.total_hours is None


def test_power_params_are_bounded() -> None:
    with pytest.raises(ValueError):
        w.PowerParams(alpha=0.6)
    with pytest.raises(ValueError):
        w.PowerParams(delta=0.0)
    assert w.PowerParams().tau2 == w.POWER_DEFAULTS["tau2"]


# -- the leak demo ----------------------------------------------------------------------- #


def test_the_leak_payload_has_every_field_for_both_envs() -> None:
    out = w.leak_demo(run_leak_demo(eps=0.5, n_seeds=4))
    assert out.caught and not out.clean.flagged and out.poisoned.flagged
    assert out.clean.past_unchanged and not out.poisoned.past_unchanged
    assert len(out.clean.features) == len(out.poisoned.features) == 14
    untested = [f for f in out.clean.features if not f.tested]
    assert all(f.t_stat is None and f.reason for f in untested)
    assert out.poisoned.worst_feature == "ret_1d" and out.seeds == [0, 1, 2, 3]
    json.dumps(out.model_dump())  # no NaN reaches the wire


# -- sweeps ------------------------------------------------------------------------------ #


def test_sweeps_list_and_open_with_the_oracle_curve(run_root: Path) -> None:
    out = w.sweeps(run_root)
    assert [s.key for s in out.sweeps] == ["sweeps/tiny"]
    assert out.sweeps[0].n_cells == 4 and out.power_curve.status == "waiting for sweep"
    detail = w.sweep_detail(run_root, "sweeps/tiny")
    assert set(detail.oracle) == {"flat", "delta_seller"}
    flat = detail.oracle["flat"]
    assert [p.premium for p in flat] == [0.0, 2.0] and all(p.lift == 0.0 for p in flat)
    assert detail.attribution.waiting_for and detail.source.endswith(SWEEP_FILENAME)
    with pytest.raises(KeyError):
        w.sweep_detail(run_root, "adhoc")
    assert w.sweeps(run_root / "absent").sweeps == []


def test_an_unreadable_sweep_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / SWEEP_FILENAME).write_text("{}")
    assert w.sweeps(tmp_path).sweeps == []


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
def reader(tmp_path: Path) -> w.PanelReader:
    panel = tmp_path / "panel"
    (panel / "spx" / "year=2025").mkdir(parents=True)
    for rel in ("spx_daily.parquet", "coverage.parquet", "seam_overlap.parquet"):
        (panel / rel).write_bytes(b"")
    (panel / "spx" / "year=2025" / "part.parquet").write_bytes(b"")
    return w.PanelReader(tmp_path, chain_cache=2, read=_fake_read)


def test_an_unbuilt_panel_raises_not_built(tmp_path: Path) -> None:
    r = w.PanelReader(tmp_path)
    status = r.status()
    assert not status.built and status.years == [] and len(status.missing) == 3
    assert [f.present for f in status.files] == [False, False, False]
    with pytest.raises(w.PanelNotBuiltError):
        r.coverage_out()
    with pytest.raises(w.PanelNotBuiltError):
        r.chain(date(2025, 7, 1))
    assert r.seam() == {}


def test_the_status_lists_every_panel_file(reader: w.PanelReader) -> None:
    status = reader.status()
    assert status.built and status.years == [2025] and status.missing == []
    assert all(f.present and f.modified for f in status.files) and len(status.files) == 4


def test_the_coverage_payload_has_strips_missing_days_and_the_overlap(
    reader: w.PanelReader,
) -> None:
    out = reader.coverage_out()
    assert [s.source for s in out.spans] == ["databento", "optionmetrics"]
    assert out.missing == [("2025-07-02", "databento")]
    assert out.overlap == ("2025-07-01", "2025-07-03", 2)
    db = out.strips[0]
    assert db.date == sorted(db.date) and db.missing.count(True) == 1
    assert reader.dates("optionmetrics") == [d.isoformat() for d in OM_DAYS]
    assert reader.coverage() is reader.coverage()  # cached until the file changes


def test_the_chain_payloads_for_one_date(reader: w.PanelReader) -> None:
    day = date(2025, 7, 1)
    index = reader.chain_index(day, None, [])
    assert index.sources == ["databento", "optionmetrics"] and index.roots == ["SPX", "SPXW"]
    assert [e.dte for e in index.expiries] == [7, 30, 60, 120]
    assert index.default_expiry == (day + timedelta(days=30)).isoformat()
    assert index.expiries[1].carry_fit == "near+r_ref"

    smile = reader.smile_out(day, None, "optionmetrics", [])
    assert smile.dte == 30 and smile.has_vendor and smile.data_source == "optionmetrics"
    assert smile.forward is not None and len(smile.iv) == len(smile.cp)
    db = reader.smile_out(day, date(2025, 7, 31), "databento", ["SPX"])
    assert not db.has_vendor and all(v is None for v in db.iv_vendor)

    surface = reader.surface_out(day, "optionmetrics", [], min_dte=20)
    assert surface.dte == [30, 60, 120] and len(surface.iv[0]) == len(surface.moneyness)

    term = reader.term_out(day, None, [])
    assert term.dte == [7, 30, 60, 120] and all(v is not None for v in term.atm_iv)
    assert term.data_source == "optionmetrics"  # the default source when both quote


def test_a_date_with_no_quotes_has_no_smile(reader: w.PanelReader) -> None:
    with pytest.raises(w.PanelNotBuiltError):
        reader.smile_out(date(2025, 3, 3), None, None, [])
    with pytest.raises(w.PanelNotBuiltError):
        reader.chain(date(2019, 3, 4))  # no partition for the year


def test_the_chain_cache_keeps_the_last_few_dates(reader: w.PanelReader) -> None:
    for day in (*OM_DAYS[:3], OM_DAYS[0]):
        reader.chain(day)
    assert len(reader._chains) == 2


def test_the_daily_payload_per_source_with_a_range(reader: w.PanelReader) -> None:
    out = reader.daily_out()
    assert [s.data_source for s in out.series] == ["databento", "optionmetrics"]
    om = out.stats["optionmetrics"]
    assert om.n_days == 127 and om.mean_vol_pts is not None and 1.0 < om.mean_vol_pts < 5.0
    clipped = reader.daily_out(date(2025, 3, 1), date(2025, 3, 31))
    assert all("2025-03" in d for s in clipped.series for d in s.date)
    assert out.series[1].spot_parity[0] == pytest.approx(5000.5)
    json.dumps(out.model_dump())  # the trailing NaNs went out as null


def test_the_seam_payload(reader: w.PanelReader) -> None:
    out = reader.seam_out()
    assert out.available and out.date == ["2025-07-01", "2025-07-03"]
    assert out.medians["iv_diff_median_volpts"] == pytest.approx(0.2)
    assert out.iv30_sources == ["databento", "optionmetrics"] and out.iv30_date


def test_without_a_seam_file_the_seam_is_unavailable(tmp_path: Path) -> None:
    r = w.PanelReader(tmp_path, read=_fake_read)
    out = r.seam_out()
    assert not out.available and out.iv30_date == [] and out.medians == {}


# -- the overview ------------------------------------------------------------------------- #


def test_the_overview_brings_the_tiles_together(
    run_root: Path, gate_docs: Path, reader: w.PanelReader
) -> None:
    rows = w.index(run_root)
    out = w.overview(REPO_ROOT / "README.md", gate_docs, run_root, reader, rows)
    assert out.summary[0].startswith("A leak-tested SPX options simulator")
    assert [(g.milestone, g.verdict) for g in out.gates] == [("M1", "PASS"), ("M2", "NOT CLEARED")]
    assert out.n_runs == 6 and out.n_ok == 4 and out.sweeps == 1 and len(out.latest) == 6
    assert out.overlap == ("2025-07-01", "2025-07-03", 2) and len(out.coverage) == 2
    bare = w.overview(REPO_ROOT / "README.md", gate_docs, run_root, w.PanelReader(run_root), ())
    assert bare.coverage == [] and bare.overlap is None and not bare.data.built
