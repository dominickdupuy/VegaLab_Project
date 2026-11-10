"""Ilan Danial: the study-level payloads say what the pre-registration and the analyses say.

The design is the A7 one with its power recomputed; the checks are the gate reports'
items; the M4 view reads a sidecar the real generator writes on a tiny grid the real
orchestrator runs with the fake learner (``tests/fake_learner.py``), so the view is
tested on the schema ``optspread.m4analysis`` writes and cannot drift from it. The
grid also drives the Sweep page's grid reader. Attribution, capital and replay run on
the fixture's own runs. Every number is synthetic.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from fake_learner import fake_trainer

from optspread import m4analysis as m4
from optspread import webapi as w
from optspread import webviews as v
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.dashboard_data import REPO_ROOT, RunRow, grid_of, load_run_row
from optspread.jobs import available_arms
from optspread.market import SyntheticChainProvider
from optspread.orchestrate import GridConfig, grid_filename, read_grid, run_grid
from optspread.rng import SeededStepRNG

SHORT = RunConfig(
    seed=7000,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
GRID = GridConfig(
    base=SHORT,
    premiums=(0.0, 1.0, 2.0, 4.0),
    n_seeds=3,
    train=TrainConfig(total_timesteps=64, seed=10),
)


@pytest.fixture(scope="module")
def grid_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("grid") / "runs"
    run_grid(GRID, root, trainer=fake_trainer)
    return root


@pytest.fixture(scope="module")
def m4_docs(grid_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """docs/ with the real VRP mapping and an M4 sidecar from the real generator."""
    docs = tmp_path_factory.mktemp("docs")
    shutil.copy(REPO_ROOT / "docs" / v.VRP_FILE, docs / v.VRP_FILE)
    record = read_grid(grid_root / grid_filename(GRID))
    result = m4.analyse(record, grid_root, exploratory=True, n_boot=200, replications=40)
    prov = m4.Provenance(date(2026, 10, 28), "runs", grid_filename(GRID), "0" * 40, False, {})
    (docs / v.M4_FILE).write_text(result.render_json(prov))
    return docs


# -- the study design --------------------------------------------------------- #


def test_the_study_is_the_a7_design_with_its_power_recomputed() -> None:
    s = v.study()
    assert (s.n_seeds, s.n_episodes, s.train_steps, s.episode_days) == (5, 500, 1_000_000, 63)
    assert s.grid_hash == m4.CONFIRMATORY_GRID_HASH and s.arms == ["delta_seller", "ppo"]
    assert round(s.power, 3) == 0.898 and round(s.floor, 1) == 698.3  # A7's own numbers
    assert round(s.core_hours, 1) == 10.7 and s.floor_vol_pts == pytest.approx(698.3 / 824.3, 1e-4)
    assert (s.train_rate, s.eval_rate, s.delta_per_vol_pt) == (804.0, 1538.0, 824.3)
    assert s.doc == "PRE_REGISTRATION.md" and (REPO_ROOT / "docs" / s.doc).is_file()
    assert s.measured is None  # no analysis asked for, none read


def test_the_study_quotes_the_measured_power_beside_the_plan(m4_docs: Path, tmp_path: Path) -> None:
    """M4's design check is the power the design had; A7's 0.898 stays the plan."""
    s = v.study(m4_docs)
    raw = json.loads((m4_docs / v.M4_FILE).read_text())["design_check"]
    row = next(r for r in raw["arms"]["ppo"] if r["premium"] == raw["reference_premium"])
    assert s.measured is not None and s.measured.arm == "ppo"
    assert (s.measured.power, s.measured.floor) == (row["power"], row["floor"])
    assert s.measured.floor_vol_pts == pytest.approx(row["floor"] / v.A6_DELTA)
    assert round(s.power, 3) == 0.898  # the plan is unchanged beside it
    assert v.study(tmp_path).measured is None  # no analysis under these docs
    (tmp_path / v.M4_FILE).write_text('{"design_check": {}}')
    assert v.study(tmp_path).measured is None  # one without a design check


def test_the_launch_estimate_includes_training_at_the_a7_rates() -> None:
    scripted = v.launch_estimate(100, 63, None)
    assert scripted.train_hours == 0.0
    assert scripted.total_hours == pytest.approx(100 * 63 / 1538 / 3600)
    ppo = v.launch_estimate(100, 63, 100_000)
    assert ppo.train_hours == pytest.approx(100_000 / 804 / 3600)
    assert ppo.total_hours == pytest.approx(scripted.eval_hours + ppo.train_hours)
    opts = v.launch_options()
    assert opts.ppo_preset is not None and opts.ppo_preset.spec["policy"] == "ppo"
    assert opts.ppo_preset.spec["total_timesteps"] == 100_000
    assert [p.name for p in opts.presets] == [p.name for p in w.job_options().presets]


# -- checks and docs ---------------------------------------------------------- #


def test_checks_are_every_gate_item_once_with_its_report(gate_docs: Path) -> None:
    rows = v.checks(gate_docs)
    names = [r.name for r in rows]
    assert len(names) == len(set(names)) and "Replay" in names
    replay = next(r for r in rows if r.name == "Replay")
    assert replay.verdict == "NOT CLEARED" and replay.failed_first == "no replay has been run"
    assert replay.sidecar == "M2_GATE_REPORT.json"  # the latest report comes first
    assert replay.report == "M2_GATE_REPORT.md"  # and the link opens the readable one
    assert replay.criterion and replay.label == "Replay"
    passed = next(r for r in rows if r.verdict == "PASS")
    assert passed.evidence and not passed.failed_first  # a pass shows its evidence instead
    assert rows[0].report_date >= rows[-1].report_date


def test_check_labels_say_what_is_checked_not_which_gate() -> None:
    rows = v.checks(REPO_ROOT / "docs")
    labels = {r.name: r.label for r in rows}
    assert labels["Wave 0 no-edge gate"] == "No edge in the fair world"
    assert labels["Dashboard run browser and gate board"] == "Dashboard run browser and check board"
    assert not any("gate" in r.label.lower() or "wave" in r.label.lower() for r in rows)
    assert all(r.report.endswith(".md") for r in rows)  # both reports have their markdown


def test_only_report_files_in_docs_are_served(gate_docs: Path) -> None:
    assert v.doc_file(gate_docs, "M2_GATE_REPORT.json").name == "M2_GATE_REPORT.json"
    for bad in ("../conftest.py", "M2_GATE_REPORT.py", "nope.md", ".hidden.md"):
        with pytest.raises(KeyError):
            v.doc_file(gate_docs, bad)


# -- the VRP mapping and the M4 analysis -------------------------------------- #


def test_the_vrp_mapping_is_the_real_premium_or_says_it_is_missing(
    m4_docs: Path, tmp_path: Path
) -> None:
    out = v.vrp_mapping(m4_docs)
    assert out.available and out.mapping is not None
    assert round(out.mapping["headline"]["mean"], 2) == 0.64  # the v4 panel
    assert out.mapping["bracket"] == [0.5, 1.0]
    missing = v.vrp_mapping(tmp_path)
    assert not missing.available and missing.mapping is None and missing.error == ""
    (tmp_path / v.VRP_FILE).write_text("[1, 2]")
    assert "not a JSON object" in v.vrp_mapping(tmp_path).error


def test_without_the_sidecar_the_m4_view_says_what_produces_it(tmp_path: Path) -> None:
    out = v.m4_analysis(tmp_path)
    assert not out.available and out.analysis is None and out.vrp is None and not out.mapped
    assert "make m4" in out.waiting_for


def test_the_m4_view_reads_the_generators_sidecar_and_the_real_premium_off_it(
    m4_docs: Path,
) -> None:
    out = v.m4_analysis(m4_docs)
    assert out.available and out.analysis is not None and out.error == ""
    a = out.analysis
    assert a["schema_version"] == m4.SCHEMA_VERSION and a["confirmatory"] is False
    assert a["arm_order"] == ["delta_seller", "ppo"]
    for arm in a["arm_order"]:
        curve = a["arms"][arm]["curve"]
        assert curve["interpolated"][0]["premium"] == 0.0
        assert {"power", "power_lo", "power_hi"} <= set(curve["interpolated"][0])
        assert len(a["arms"][arm]["decisions"]) == 3  # every premium above zero
    assert out.vrp is not None and round(out.vrp["estimate"], 2) == 0.64
    assert [m.arm for m in out.mapped] == a["arm_order"]
    # The committed mapping's interval, so a rebuilt panel does not stale the test.
    head = json.loads((REPO_ROOT / "docs" / v.VRP_FILE).read_text())
    expected = (head["headline"]["mean_low"], head["headline"]["mean_high"])
    for m in out.mapped:
        assert (m.low, m.high) == pytest.approx(expected, abs=1e-4)
        # read from its knots as the M4 gate reads it; a curve that dips is read as it is
        assert m.reason == "" and m.power is not None and m.clipped  # -0.34 is below 0
        assert m.power_min is not None and m.power_max is not None
        assert m.power_min <= m.power <= m.power_max
        assert m.band_min is not None and m.band_max is not None
        assert m.band_min <= m.power_min and m.power_max <= m.band_max


def test_a_sidecar_of_another_schema_is_refused_by_name(m4_docs: Path, tmp_path: Path) -> None:
    raw = json.loads((m4_docs / v.M4_FILE).read_text())
    raw["schema_version"] = 99
    (tmp_path / v.M4_FILE).write_text(json.dumps(raw))
    out = v.m4_analysis(tmp_path)
    assert not out.available and "schema v99" in out.error


def test_a_curve_is_read_at_the_premium_or_the_reason_it_cannot_be_is_given() -> None:
    good = {"interpolated": [{"premium": x, "power": 0.05 + 0.2 * x} for x in (0.0, 1.0, 2.0)]}
    m = v._mapped("delta_seller", good, 1.0, (0.5, 3.0))
    assert m.power == pytest.approx(0.25) and m.clipped and m.power_high == pytest.approx(0.45)
    dips = {"interpolated": [{"premium": 0.0, "power": 0.5}, {"premium": 1.0, "power": 0.2}]}
    m = v._mapped("ppo", dips, 0.5, (0.0, 1.0))  # read as it is, not smoothed or refused
    assert m.reason == "" and m.power == pytest.approx(0.35)
    assert (m.power_min, m.power_max) == pytest.approx((0.2, 0.5))
    bad = {"interpolated": [{"premium": 0.0, "power": 0.5}, {"premium": 1.0, "power": 1.2}]}
    assert "[0, 1]" in v._mapped("ppo", bad, 0.5, (0.0, 1.0)).reason
    assert v._mapped("ppo", {}, 0.5, (0.0, 1.0)).reason


# -- the orchestrated grid on the Sweep page ---------------------------------- #


def test_the_sweep_list_carries_orchestrated_grids(grid_root: Path) -> None:
    listed = w.sweeps(grid_root)
    grids = [s for s in listed.sweeps if s.kind == "grid"]
    assert len(grids) == 1
    g = grids[0]
    assert g.key == grid_filename(GRID) and g.n_seeds == 3 and g.n_cells == 4 * 2 * 3
    assert g.n_present == g.n_cells and g.premiums == [0.0, 1.0, 2.0, 4.0]


def test_a_grid_reads_as_means_by_premium_cells_and_the_pre_registered_contrasts(
    grid_root: Path,
) -> None:
    out = w.sweep_detail(grid_root, grid_filename(GRID))
    assert out.kind == "grid" and out.grid is not None
    g = out.grid
    assert g.grid_hash == read_grid(grid_root / grid_filename(GRID)).grid_hash
    assert len(g.cells) == 24 and all(c.present for c in g.cells) and not g.missing
    seller = g.means["delta_seller"]
    assert [m.premium for m in seller] == [0.0, 1.0, 2.0, 4.0]
    first = seller[0]
    blocks = [c.mean_pnl or 0.0 for c in g.cells if c.arm == "delta_seller" and c.premium == 0.0]
    assert first.n_blocks == 3 and first.mean == pytest.approx(float(np.mean(blocks)))
    assert first.se == pytest.approx(float(np.std(blocks, ddof=1) / np.sqrt(3)))
    record = read_grid(grid_root / grid_filename(GRID))
    for arm in ("delta_seller", "ppo"):
        assert [c.premium for c in g.contrasts[arm]] == [1.0, 2.0, 4.0]
        c = g.contrasts[arm][-1]
        contrast = record.contrast(grid_root, 4.0, arm)
        assert c.estimate == pytest.approx(float(contrast.mean()))
        assert c.df == 2 and c.block_means == pytest.approx(list(contrast.mean(axis=1)))
        assert c.lower_bound is not None and (c.lower_bound > 0) == c.detected
    assert out.oracle["ppo"][0].lift == 0.0 and g.git_shas
    cell = g.cells[0]
    assert (grid_root / cell.key / "manifest.json").is_file()


def test_a_grid_with_a_cell_gone_says_which_and_keeps_the_rest(
    grid_root: Path, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    shutil.copytree(grid_root, root)
    record = read_grid(root / grid_filename(GRID))
    gone = next(c for c in record.cells if c.arm == "ppo" and c.premium == 2.0)
    shutil.rmtree(root / gone.run_id)
    out = w.sweep_detail(root, grid_filename(GRID))
    assert (
        out.grid is not None and len(out.grid.missing) == 1 and gone.run_id in out.grid.missing[0]
    )
    assert w.sweeps(root).sweeps[0].n_present == 23
    at2 = next(c for c in out.grid.contrasts["ppo"] if c.premium == 2.0)
    assert at2.estimate is None and at2.reason and not at2.detected
    assert next(c for c in out.grid.contrasts["delta_seller"] if c.premium == 2.0).estimate


def test_grid_cells_are_their_grid_in_the_run_index(grid_root: Path) -> None:
    record = read_grid(grid_root / grid_filename(GRID))
    cell = record.cells[0]
    assert grid_of(grid_root / cell.run_id) == record.grid_hash
    assert load_run_row(grid_root / cell.run_id).sweep == record.grid_hash
    assert grid_of(grid_root / "not-a-cell") == ""
    facets = w.run_list(w.index(grid_root), grid_root, w.RunFilters()).facets
    assert facets.sweeps == [record.grid_hash]


def test_a_bad_grid_key_is_not_found(grid_root: Path, tmp_path: Path) -> None:
    (tmp_path / "grid-broken.json").write_text("{")
    for root, key in ((tmp_path, "grid-broken.json"), (grid_root, "../x"), (grid_root, "nope")):
        with pytest.raises(KeyError):
            w.sweep_detail(root, key)
    assert w.sweeps(tmp_path).sweeps == []  # an unreadable record is left out


@pytest.mark.parametrize("spelling", ["relative", "dot-dot"])
def test_a_grid_and_its_runs_open_however_the_runs_root_is_spelled(
    grid_root: Path, spelling: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runs root not spelled as its resolved path (relative, through ``..``, a short
    name or a link) opens the grid, and a cell's run keeps its key and its sweep."""
    if spelling == "relative":
        monkeypatch.chdir(grid_root.parent)
        root = Path(grid_root.name)
    else:
        root = grid_root / ".." / grid_root.name
    assert root != grid_root and root.resolve() == grid_root.resolve()
    reference = w.sweep_detail(grid_root, grid_filename(GRID))
    out = w.sweep_detail(root, grid_filename(GRID))
    assert out.grid is not None and reference.grid is not None
    assert [c.key for c in out.grid.cells] == [c.key for c in reference.grid.cells]
    key = out.grid.cells[0].key
    detail = w.run_detail(w.load_detail(root, key), root, w.index(root))
    assert detail.row.key == key and detail.provenance.sweep_cells == len(out.grid.cells)


# -- background tasks --------------------------------------------------------- #


def _wait(tasks: v.Tasks, kind: str, key: str) -> tuple[v.TaskState, object]:
    for _ in range(3000):
        state, result = tasks.state(kind, key)
        if state.state in ("done", "failed"):
            return state, result
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_tasks_run_once_per_key_and_report_their_failure() -> None:
    tasks = v.Tasks()
    try:
        assert tasks.state("k", "a")[0].state == "idle"
        tasks.submit("k", "a", lambda: 41 + 1)
        state, result = _wait(tasks, "k", "a")
        assert state.state == "done" and result == 42 and state.seconds is not None

        def boom() -> None:
            raise RuntimeError("it broke\nsecond line")

        tasks.submit("k", "b", boom)
        state, _ = _wait(tasks, "k", "b")
        assert state.state == "failed" and state.error == "it broke"
    finally:
        tasks.shutdown()


def _scripted_only() -> tuple[str, ...]:
    return tuple(a for a in available_arms() if a != "ppo")


def _ok_row(run_root: Path, policy: str) -> RunRow:
    return next(
        r
        for r in w.index(run_root)
        if r.status == "OK" and r.policy == policy and r.path.parent.name == "tiny"
    )


def test_attribution_is_computed_outside_the_run_and_read_back(
    run_root: Path, tmp_path: Path
) -> None:
    row = _ok_row(run_root, "delta_seller")
    idle = v.TaskState(state="idle")
    absent = v.attribution_view(tmp_path, row, idle)
    assert absent.state == "absent" and absent.can_compute and absent.result is None
    target = v.attribution_file(tmp_path, row)
    assert target.name == f"{row.run_id}.json" and target.parent == tmp_path
    v.compute_attribution(row.path, target)
    assert not (row.path / target.name).exists()  # nothing lands in the run directory
    out = v.attribution_view(tmp_path, row, idle)
    assert out.state == "cached" and out.result is not None and out.result["reproduces"]
    episodes = out.result["episodes"]
    assert len(episodes) == 3 and all(e["steps"] == [] for e in episodes)
    assert out.seed == episodes[0]["seed"] and len(out.steps) == 21
    t = out.result["totals"]
    parts = sum(t[k] for k in ("delta", "gamma", "vega", "theta", "carry", "cost", "residual"))
    assert parts == pytest.approx(t["pnl"], abs=1e-6)
    later = v.attribution_view(tmp_path, row, idle, seed=episodes[-1]["seed"])
    assert later.seed == episodes[-1]["seed"]
    assert v.attribution_view(tmp_path, row, v.TaskState(state="running")).state == "cached"


def test_attribution_says_why_a_run_cannot_be_attributed_here(
    run_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v, "available_arms", _scripted_only)  # as without the train extra
    row = _ok_row(run_root, "flat")
    from dataclasses import replace

    real = replace(row, data_source="optionmetrics")
    assert v.attribution_file(tmp_path, real).name == f"{row.run_id}.optionmetrics.json"
    # a real-data run is attributed on its recorded panel: never the old CLI-only refusal
    assert "scripts.attribution" not in v.attribution_reason(real)
    assert "train extra" in v.attribution_reason(replace(row, policy="ppo"))
    tampered = next(r for r in w.index(run_root) if r.status == "DIGEST MISMATCH")
    assert "DIGEST MISMATCH" in v.attribution_reason(tampered)
    running = v.attribution_view(tmp_path, row, v.TaskState(state="running", seconds=1.0))
    assert running.state == "computing" and running.seconds == 1.0
    failed = v.attribution_view(tmp_path, row, v.TaskState(state="failed", error="no"))
    assert failed.state == "failed" and failed.reason == "no"
    (tmp_path / f"{row.run_id}.json").write_text("{}")
    assert v.attribution_view(tmp_path, row, v.TaskState(state="idle")).state == "failed"


def test_capital_re_simulates_the_run_and_reads_margin_and_drawdown(run_root: Path) -> None:
    row = _ok_row(run_root, "delta_seller")
    out = v.compute_capital(row.path)
    assert out.state == "done" and out.reproduces and out.n == 3
    assert out.initial_cash == 1_000_000.0
    for e in out.episodes:
        assert e.max_drawdown >= 0.0 and e.peak_margin > 0.0  # the seller always holds
        assert e.return_on_cash == pytest.approx(e.pnl / 1_000_000.0)
        assert e.return_on_margin == pytest.approx(e.pnl / e.peak_margin)
    assert out.worst_drawdown == max(e.max_drawdown for e in out.episodes)
    flat = v.compute_capital(_ok_row(run_root, "flat").path)
    assert all(e.peak_margin == 0.0 and e.return_on_margin is None for e in flat.episodes)
    assert flat.mean_return_on_margin is None
    view = v.capital_view(row, v.TaskState(state="done", seconds=2.0), out)
    assert view.seconds == 2.0 and view.n == 3
    absent = v.capital_view(row, v.TaskState(state="idle"), None)
    assert absent.state == "absent" and absent.can_compute and absent.method
    assert v.capital_view(row, v.TaskState(state="running"), None).state == "computing"
    assert v.capital_view(row, v.TaskState(state="failed", error="x"), None).reason == "x"
    from dataclasses import replace

    assert out.agrees and out.n_differ == 0 and out.rtol == v.SCRIPTED_RTOL
    assert out.max_abs_diff == 0.0
    ppo = replace(row, policy="ppo")
    if "ppo" in available_arms():  # with the learner the reason is the missing policy.zip
        assert "policy.zip" in v.capital_reason(ppo) and "extra" not in v.capital_reason(ppo)
    else:
        assert "train extra" in v.capital_reason(ppo)
    # a real-data run is re-simulated on its recorded panel, not refused as synthetic-only
    assert "synthetic world only" not in v.capital_reason(replace(row, data_source="optionmetrics"))


def _stub_run(tmp_path: Path) -> RunRow:
    """A seller run on the real-data adapter's stub source (the generator behind the real
    source's shape), recorded as the harness records a real-data run."""
    from optspread.artifacts import write_run
    from optspread.harness import run_episodes, summarize

    config = RunConfig(
        policy="delta_seller",
        n_episodes=2,
        seed=7,
        market=MarketConfig(n_days=21, tenor_days=(7, 14)),
        env=EnvConfig(episode_days=21),
    )
    episodes = run_episodes(config, source="optionmetrics_stub")
    run_dir = tmp_path / "runs" / "stub"
    write_run(run_dir, config, summarize(episodes), episodes, source="optionmetrics_stub")
    return load_run_row(run_dir)


def test_capital_and_attribution_re_simulate_a_real_data_run_on_its_recorded_source(
    tmp_path: Path,
) -> None:
    row = _stub_run(tmp_path)
    assert row.status == "OK" and row.source == "optionmetrics_stub"
    assert v.capital_reason(row) == "" and v.attribution_reason(row) == ""
    capital = v.compute_capital(row.path, v.capital_file(tmp_path / "attr", row))
    assert capital.reproduces is True and capital.n == 2
    assert v.capital_file(tmp_path / "attr", row).name == f"{row.run_id}.optionmetrics_stub.json"
    target = v.attribution_file(tmp_path / "attr", row)
    v.compute_attribution(row.path, target)
    out = v.attribution_view(tmp_path / "attr", row, v.TaskState(state="idle"))
    assert out.state == "cached" and out.result is not None
    assert out.result["reproduces"] is True and out.result["source"] == "optionmetrics_stub"


def test_a_real_data_run_whose_panel_was_rebuilt_is_refused_with_the_reason(
    tmp_path: Path,
) -> None:
    from optspread.manifest import MANIFEST_FILENAME

    row = _stub_run(tmp_path)
    path = row.path / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["data"]["panel"] = {"panel/surface_grid.parquet": "0" * 64}
    path.write_text(json.dumps(manifest))
    rebuilt = load_run_row(row.path)
    for reason in (v.capital_reason(rebuilt), v.attribution_reason(rebuilt), v.mix_reason(rebuilt)):
        assert "panel is not the one" in reason
    with pytest.raises(ValueError, match="panel is not the one"):
        v.compute_capital(row.path)


def test_capital_counts_the_episodes_beyond_the_tolerance(run_root: Path, tmp_path: Path) -> None:
    """A P&L a billionth off is within tolerance; a dollar off is one episode that differs."""
    from optspread.artifacts import RUN_FILENAME
    from optspread.manifest import RunManifest, file_sha256

    row = _ok_row(run_root, "delta_seller")
    run = tmp_path / row.path.name
    shutil.copytree(row.path, run)
    record = json.loads((run / RUN_FILENAME).read_text())
    eps = record["episodes"]
    eps[0]["pnl"] = eps[0]["pnl"] * (1 + 1e-12)  # last bits
    eps[1]["pnl"] = eps[1]["pnl"] + 1.0  # a dollar
    (run / RUN_FILENAME).write_text(json.dumps(record))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["artifacts"][RUN_FILENAME] = file_sha256(run / RUN_FILENAME)
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert RunManifest.read(run).config == RunManifest.read(row.path).config
    out = v.compute_capital(run)
    assert out.reproduces is False and out.agrees is False and out.n_differ == 1
    assert out.max_abs_diff == pytest.approx(1.0)


def test_replay_reports_identical_with_the_reason(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v, "available_arms", _scripted_only)  # as without the train extra
    row = _ok_row(run_root, "delta_seller")
    out = v.run_replay(row.path)
    assert out.verdict == "IDENTICAL" and "reproduced" in out.reason and out.seconds >= 0.0
    assert out.rtol == v.SCRIPTED_RTOL and out.comparable  # a scripted run, at the seller's
    done = v.replay_view(row, v.TaskState(state="done"), out)
    assert done.verdict == "IDENTICAL" and done.can_replay and "bit for bit" in done.note
    assert not done.retrains and done.estimate_hours is not None
    tampered = next(r for r in w.index(run_root) if r.status == "DIGEST MISMATCH")
    bad = v.run_replay(tampered.path)
    assert bad.verdict == "DIFFERS" and bad.reason
    assert v.replay_view(row, v.TaskState(state="idle"), None).state == "idle"
    assert v.replay_view(row, v.TaskState(state="running"), None).state == "running"
    assert v.replay_view(row, v.TaskState(state="failed", error="e"), None).reason == "e"
    from dataclasses import replace

    assert "train extra" in v.replay_reason(replace(row, policy="ppo"))


def test_a_replay_within_tolerance_agrees_and_never_reads_as_a_failure(run_root: Path) -> None:
    """Three states: bit for bit, within the seller's 1e-9, beyond it. A retrained PPO run
    off its own platform is not comparable, which is not a failure either."""
    row = _ok_row(run_root, "delta_seller")
    within = v.ReplayDone(
        verdict="WITHIN", reason="r", seconds=1.0, rtol=1e-9, max_rel_diff=1.3e-11, comparable=True
    )
    out = v.replay_view(row, v.TaskState(state="done"), within)
    assert out.verdict == "AGREES" and out.rtol == 1e-9 and out.max_rel_diff == 1.3e-11
    assert "1e-09 relative" in out.note and "1.3e-11" in out.note
    beyond = within.model_copy(update={"verdict": "DIFFERS", "max_rel_diff": 0.2})
    assert v.replay_view(row, v.TaskState(state="done"), beyond).verdict == "DIFFERS"
    retrain = within.model_copy(update={"verdict": "DIFFERS", "rtol": None, "comparable": False})
    ppo = v.replay_view(row, v.TaskState(state="done"), retrain)
    assert ppo.verdict == "DIFFERS" and not ppo.comparable and "not comparable" in ppo.note


def test_a_ppo_replay_says_it_retrains_and_how_long_before_it_starts(grid_root: Path) -> None:
    ppo = next(r for r in w.index(grid_root) if r.policy == "ppo" and r.status == "OK")
    idle = v.replay_view(ppo, v.TaskState(state="idle"), None)
    assert idle.retrains and idle.train_steps == GRID.train.total_timesteps  # type: ignore[union-attr]
    assert idle.estimate_hours == pytest.approx(
        v.launch_estimate(ppo.n_episodes or 1, SHORT.env.episode_days, 64).total_hours
    )
    assert "not comparable across platforms" in idle.note


# -- leaks -------------------------------------------------------------------- #


def test_the_leak_floor_is_the_smallest_eps_still_caught() -> None:
    out = v.leak_floor_out(v.LeakFloorParams(n_seeds=4, eps_grid=[1.0, 0.5, 0.01]))
    assert out.eps == [1.0, 0.5, 0.01] and len(out.flagged) == 3
    assert out.flagged[0] and out.floor is not None and out.floor in out.eps
    with pytest.raises(ValueError):
        v.leak_floor_out(v.LeakFloorParams(n_seeds=3, eps_grid=[3.0, 1.0]))


def test_a_leak_planted_in_one_field_is_caught_on_that_field() -> None:
    out = v.leak_field_out(v.LeakFieldParams(field="atm_iv", eps=1.0, n_seeds=6))
    assert out.flagged and out.named_correctly and out.worst_feature == "atm_iv"
    assert out.scale > 0 and len(out.features) == len(v.leak_fields())
    assert out.t_critical is not None and out.t_critical > 0
    with pytest.raises(ValueError):
        v.leak_field_out(v.LeakFieldParams(field="nope"))


@pytest.mark.slow
def test_field_coverage_runs_every_field() -> None:
    out = v.leak_coverage_out(1.0, 4)
    assert [c.field for c in out.controls] == v.leak_fields()
    assert not out.clean_flagged and out.passed == (not out.uncovered and not out.misattributed)


# -- the seller on a real chain ----------------------------------------------- #


def test_the_seller_opens_the_condor_with_its_credit_risk_and_breakevens() -> None:
    cfg = MarketConfig()
    chain = SyntheticChainProvider(cfg).reset(SeededStepRNG(0))
    out = v.seller_on(chain, date(2022, 6, 13), cfg)
    assert out.structure == "IRON_CONDOR" and len(out.legs) == 4
    assert [leg.qty for leg in out.legs] == [1, -1, -1, 1]  # wing, short, short, wing
    strikes = [leg.strike for leg in out.legs]
    assert strikes == sorted(strikes) and out.credit > 0
    width = max(strikes[1] - strikes[0], strikes[3] - strikes[2]) * 100
    assert out.max_loss == pytest.approx(width - out.credit) and out.margin == out.max_loss
    assert out.breakevens == pytest.approx(
        [strikes[1] - out.credit / 100, strikes[2] + out.credit / 100]
    )
    assert out.tenor_trading_days == cfg.tenor_days[EnvConfig().expiry_idx]
    assert "Illustration" in out.note


def test_a_real_chain_is_built_from_the_days_grid() -> None:
    from optspread.realdata.panel import K_GRID, TENOR_GRID

    k = np.asarray(K_GRID, dtype=np.float64)
    tenors = np.asarray(TENOR_GRID, dtype=np.int64)
    iv = np.full((len(tenors), len(k)), np.nan)
    iv[2:5, 10:25] = 0.2 - 0.1 * k[None, 10:25]  # a skew, holes around it
    cfg = MarketConfig()
    chain = v.real_chain(date(2022, 6, 13), k, tenors, iv, 3750.0, 0.01, cfg)
    assert chain.spot == 3750.0 and chain.ivs.shape == (len(cfg.tenor_days), len(chain.strikes))
    assert np.isfinite(chain.ivs).all()
    puts = chain.ivs[0][: len(chain.strikes) // 2]
    assert puts[0] > puts[-1]  # the skew survives the fill
    with pytest.raises(w.PanelNotBuiltError):
        v.real_chain(date(2022, 6, 13), k, tenors, np.full_like(iv, np.nan), 3750.0, 0.0, cfg)


# -- the run page's re-simulations, cached outside the run; the launch preview -------- #


def test_capital_is_cached_outside_the_run_with_its_equity_and_margin_by_day(
    run_root: Path, tmp_path: Path
) -> None:
    row = _ok_row(run_root, "delta_seller")
    target = v.capital_file(tmp_path, row)
    assert target == tmp_path / "capital" / f"{row.run_id}.json"
    idle = v.TaskState(state="idle")
    assert v.capital_view(row, idle, None, tmp_path).state == "absent"
    out = v.compute_capital(row.path, target)
    assert target.is_file() and not (row.path / "capital").exists()
    assert out.file == str(target) and len(out.paths) == 3 and out.path_seeds == [7, 8, 9]
    assert all(len(p) == 22 and p[0] == 0.0 for p in out.paths)  # the reset, then 21 days
    assert [p[-1] for p in out.paths] == pytest.approx([e.pnl for e in out.episodes])
    assert out.equity_band is not None and out.margin_band is not None
    assert out.equity_band.p10[-1] <= out.equity_band.p50[-1] <= out.equity_band.p90[-1]
    assert max(x or 0.0 for x in out.margin_band.mean) > 0.0  # the seller posts margin
    cached = v.capital_view(row, idle, None, tmp_path)
    assert cached.state == "done" and cached.paths == out.paths
    with pytest.raises(ValueError, match="holds runs"):
        v.compute_capital(row.path, row.path / "capital.json")
    assert v.band([]) is None


def test_the_action_mix_is_re_simulated_once_and_read_with_its_holdings(
    run_root: Path, tmp_path: Path
) -> None:
    row = _ok_row(run_root, "delta_seller")
    idle = v.TaskState(state="idle")
    lean_root = tmp_path / "lean"
    absent = v.mix_view(tmp_path, row, idle, lean_root)
    assert absent.state == "absent" and absent.can_compute and absent.mix is None
    from optspread import actionmix as am

    am.write_mix(am.mix_run(row.path), lean_root / f"{row.run_id}.json", episodes=False)
    lean = v.mix_view(tmp_path, row, idle, lean_root)
    assert lean.state == "lean" and lean.mix is not None and lean.detail is None
    running = v.mix_view(tmp_path, row, v.TaskState(state="running", seconds=1.0), lean_root)
    assert running.state == "computing" and running.mix is not None
    target = v.mix_file(tmp_path, row)
    v.compute_mix(row.path, target)
    assert target.parent == tmp_path / "actionmix" and not (row.path / target.name).exists()
    out = v.mix_view(tmp_path, row, idle, lean_root)
    assert out.state == "cached" and out.mix is not None and out.detail is not None
    assert all(not e.holdings for e in out.mix.episodes)  # lean on the wire
    d = out.detail
    assert d.n_holdings == out.mix.summary.n_opens == len(d.holdings)
    assert sum(sum(c) for c in d.days_held.values()) == d.n_holdings
    assert d.example is not None and len(d.example.held) == 21 and len(d.example.events) == 21
    h = d.holdings[0]
    assert h.structure == "IRON_CONDOR" and h.short_delta is not None and len(h.strikes) == 4
    assert h.short_delta == pytest.approx(0.16, abs=0.08) and h.days >= 1
    from dataclasses import replace

    tampered = next(r for r in w.index(run_root) if r.status == "DIGEST MISMATCH")
    assert "DIGEST MISMATCH" in v.mix_reason(tampered)
    assert "policy.zip" in v.mix_reason(replace(row, policy="ppo")) or "ppo" not in available_arms()


def test_the_launch_preview_shows_the_run_or_every_reason_it_would_be_refused(
    run_root: Path,
) -> None:
    rows = w.index(run_root)
    bad = v.launch_preview({"policy": "ppo", "total_timesteps": 10, "seed": 40_000}, run_root, rows)
    assert not bad.ok and any("below 32,768" in e for e in bad.errors)
    assert not any(e.startswith("Value error") for e in bad.errors)
    spec = {
        "seed": 7,
        "n_episodes": 3,
        "n_days": 21,
        "tenor_days": [7, 14],
        "iv_premium_vol_pts": 2.0,
    }
    out = v.launch_preview(spec, run_root, rows)
    assert out.ok and out.run_key == f"launched/{out.run_id}" and not out.exists
    assert "planted premium of 2 vol points" in out.world and out.implied_vol == pytest.approx(0.22)
    assert out.day0 is not None and out.day0.structure == "IRON_CONDOR" and len(out.day0.legs) == 4
    assert out.day0.credit is not None and out.day0.credit > 0 and "iron condor" in out.day0.plain
    assert out.estimate is not None and out.estimate.eval_steps == 3 * 21
    # the sweep's seller at zero is this run's paired control: the same world and seeds
    control = next(p for p in out.peers if p.relation == "control")
    assert control.policy == "delta_seller" and control.premium == 0.0 and control.paired
    flat = v.launch_preview({**spec, "policy": "flat"}, run_root, rows)
    assert flat.day0 is not None and flat.day0.structure == "FLAT" and not flat.day0.legs
    ppo = v.launch_preview({**spec, "policy": "ppo", "total_timesteps": 64}, run_root, rows)
    assert ppo.ok and ppo.day0 is not None and ppo.day0.reference  # the seller, for scale


def test_the_seller_view_shows_the_loss_its_wings_cap_when_the_frozen_check_says_naked() -> None:
    """The dashboard's 2026-10-02 condor (#39): the wings cap the loss at $30,364, but the
    frozen margin check margins it as naked; the view says both, not "not capped"."""
    from optspread.margin import is_defined_risk
    from optspread.types import OptionLeg

    legs = (
        OptionLeg("P", 6900.0, 21, 1, 26.989371069455615),
        OptionLeg("P", 7250.0, 21, -1, 50.9507515689761),
        OptionLeg("C", 8150.0, 21, -1, 29.48013413588751),
        OptionLeg("C", 8400.0, 21, 1, 7.077296280691314),
    )
    assert not is_defined_risk(legs)  # the open defect, left as the harness ran it
    capped, naked = v.capped_loss(legs)
    assert naked and capped == pytest.approx(30_363.58, abs=0.01)
    # a condor the check reads right is capped and not naked; a strangle is neither
    calm = tuple(OptionLeg(x.right, x.strike / 2, 21, x.qty, x.entry_price / 2) for x in legs)
    assert v.capped_loss(calm)[1] is False and v.capped_loss(calm)[0] is not None
    assert v.capped_loss(legs[1:3]) == (None, False)
    line = v.panel.seller_reading(
        "2026-10-02",
        7721.12,
        21,
        7250,
        8150,
        6900,
        8400,
        4636.4,
        capped,
        85_254.33,
        [7203.6, 8196.4],
        0.16,
        naked=True,
    )
    assert (
        "can lose at most $30,364" in line and "margins this condor as naked shorts (#39)" in line
    )
