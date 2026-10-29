"""Dominick Dupuy: the M4 gate items (week 8, #35).

Every item is tested both ways: with its deliverable absent or wrong it must read NOT
CLEARED and say what failed first, and with evidence that meets its criterion it must
pass. The analysis the first four items read is a real ``docs/M4_ANALYSIS.json``, made
by ``m4analysis`` on a tiny grid the real orchestrator runs with the fake learner of
``tests/fake_learner.py``; for the length of the build that grid stands in for the A7
one, so the analysis is confirmatory by the generator's own rule. The failures are
that file edited one field at a time. The VRP mapping and the data manifest are
synthetic; the attribution is the real one of the tiny grid's cells. The dashboard
server is stood in for by a module registered under the name the gate imports, as
the M2 tests do, and the real one is called once when it is installed. Every number
is synthetic; nothing here reads the shared folder.
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fake_learner import FakeLearned, fake_trainer

from optspread import m4analysis as m4
from optspread import m4gate
from optspread.attribution import attribute_run, output_name, write_attribution
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig, TrainConfig
from optspread.config import config_hash as hash_of
from optspread.gatereport import GateItem, GateReport
from optspread.orchestrate import GridConfig, GridRecord, grid_filename, read_grid, run_grid

SHORT = RunConfig(
    seed=7000,
    n_episodes=3,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)
TINY = GridConfig(
    base=SHORT, premiums=(0.0, 2.0, 4.0), n_seeds=3, train=TrainConfig(total_timesteps=64, seed=10)
)
# Enough normal replications that a two-level arm's threshold (0.025) can be reached.
REPLICATIONS = 200
DIGEST = "ab" * 32


def blocked(item: GateItem, contains: str) -> None:
    assert not item.passed and item.verdict == "FAIL"
    assert contains in item.failed_first, item.failed_first
    assert item.has_criterion


def passes(item: GateItem) -> None:
    assert item.passed and item.verdict == "PASS", item.failed_first
    assert item.evidence


@dataclass(frozen=True)
class Sweep:
    root: Path  # the runs root: the grid record and its cells
    record: GridRecord
    analysis: dict[str, Any]  # the confirmatory sidecar, as written
    exploratory: dict[str, Any]  # the same grid analysed against the real A7 hash


def _sidecar(result: m4.M4Analysis) -> dict[str, Any]:
    prov = m4.Provenance(date(2026, 11, 19), "$VEGALAB_DATA/runs", "grid.json", "a" * 40, False, {})
    data = json.loads(result.render_json(prov))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Sweep:
    root = tmp_path_factory.mktemp("m4gate") / "runs"
    run_grid(TINY, root, trainer=fake_trainer)
    record = read_grid(root / grid_filename(TINY))
    exploratory = m4.analyse(record, root, exploratory=True, replications=REPLICATIONS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m4, "CONFIRMATORY_GRID_HASH", hash_of(TINY))
        confirmatory = m4.analyse(record, root, replications=REPLICATIONS)
        analysis = _sidecar(
            confirmatory
        )  # the sidecar names the confirmatory hash it was read under
    assert confirmatory.confirmatory and not exploratory.confirmatory
    return Sweep(root, record, analysis, _sidecar(exploratory))


def write_docs(
    docs: Path,
    analysis: dict[str, Any] | None,
    *,
    mapping: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> Path:
    """A docs folder with the analysis, a VRP mapping and a data manifest that agree."""
    docs.mkdir(parents=True, exist_ok=True)
    if analysis is not None:
        (docs / m4gate.ANALYSIS_FILE).write_text(json.dumps(analysis))
    (docs / m4gate.VRP_FILE).write_text(json.dumps(mapping or vrp_mapping()))
    (docs / m4gate.DATA_MANIFEST_FILE).write_text(json.dumps(manifest or data_manifest()))
    return docs


def vrp_mapping(
    estimate: float = 2.5, interval: tuple[float, float] = (1.0, 3.5)
) -> dict[str, Any]:
    return {
        "generated": "2026-10-28",
        "source": "panel/spx_daily.parquet, as recorded in docs/DATA_MANIFEST.json",
        "source_sha256": DIGEST,
        "power_inputs": {"estimate": estimate, "interval": list(interval)},
        "sweep_axis": list(TINY.premiums),
    }


def data_manifest(digest: str = DIGEST) -> dict[str, Any]:
    return {"files": {m4gate.PANEL_DAILY: {"bytes": 1, "sha256": digest}}, "panel_version": 3}


def edited(data: dict[str, Any], edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    out = copy.deepcopy(data)
    edit(out)
    return out


# -- the analysis every analysis item reads ------------------------------------- #


def test_the_real_sidecar_of_the_stand_in_grid_is_the_analysis_the_items_accept(
    sweep: Sweep,
) -> None:
    assert m4gate.analysis_problem(sweep.analysis, TINY) == ""
    assert "exploratory, not confirmatory" in m4gate.analysis_problem(sweep.exploratory, TINY)
    # the real A7 grid is not the stand-in's, and its hash has not moved
    assert "not 7d9d53d340006f02" in m4gate.analysis_problem(sweep.analysis, m4.CONFIRMATORY_GRID)
    assert hash_of(RunConfig()) == "17c15aa74ce473a9"


def test_read_json_refuses_what_strict_json_does_not_allow(tmp_path: Path) -> None:
    assert m4gate.read_json(tmp_path / "absent.json") == (None, f"no {tmp_path / 'absent.json'}")
    (tmp_path / "nan.json").write_text('{"x": NaN}')
    assert "not readable JSON" in m4gate.read_json(tmp_path / "nan.json")[1]
    (tmp_path / "list.json").write_text("[1]")
    assert "not a JSON object" in m4gate.read_json(tmp_path / "list.json")[1]


# -- 1. the power curve ----------------------------------------------------------- #


def test_the_power_curve_item_passes_on_the_confirmatory_analysis(
    sweep: Sweep, tmp_path: Path
) -> None:
    item = m4gate.power_curve_item(write_docs(tmp_path, sweep.analysis), TINY)
    passes(item)
    floors = {a: sweep.analysis["arms"][a]["floor"] for a in sweep.analysis["arm_order"]}
    for arm, floor in floors.items():
        assert f"{arm} floor" in item.evidence and floor["status"] in m4gate.FLOOR_STATUSES
    gap = sweep.analysis["gap"]
    assert f"gap (ppo minus delta_seller) {gap['kind']}" in item.evidence
    assert f"grid {hash_of(TINY)} (18 of 18 cells, S = 3 x N = 3)" in item.evidence
    assert "60 of 60" in m4gate.CURVE_CRITERION and "7d9d53d340006f02" in m4gate.CURVE_CRITERION


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda d: d.update(schema_version=2), "not schema 1 of M4"),
        (lambda d: d.update(milestone="M2"), "not schema 1 of M4"),
        (lambda d: d.update(deviations=["incomplete: 1 of 18 cells"]), "incomplete: 1 of 18"),
        (lambda d: d["provenance"].update(grid_hash="0" * 16), "is of grid 0000000000000000"),
        (lambda d: d["provenance"].update(cells_used=12), "incomplete: 12 of 18 cells used"),
        (lambda d: d["design"].update(n_seeds=2), "is not the grid's"),
        (lambda d: d["arms"].pop("ppo"), "no analysis of the ppo arm"),
        (lambda d: d["arms"]["ppo"]["curve"]["points"].pop(), "not at every premium"),
        (lambda d: d["arms"]["delta_seller"]["floor"].update(status="?"), "status '?'"),
        (lambda d: d.update(gap=None), "no gap with a kind"),
    ],
)
def test_the_power_curve_item_names_what_is_wrong_with_the_analysis(
    sweep: Sweep, tmp_path: Path, edit: Callable[[dict[str, Any]], None], why: str
) -> None:
    blocked(m4gate.power_curve_item(write_docs(tmp_path, edited(sweep.analysis, edit)), TINY), why)


def test_the_power_curve_item_refuses_a_missing_or_exploratory_analysis(
    sweep: Sweep, tmp_path: Path
) -> None:
    blocked(m4gate.power_curve_item(write_docs(tmp_path / "a", None), TINY), "no ")
    item = m4gate.power_curve_item(write_docs(tmp_path / "b", sweep.exploratory), TINY)
    blocked(item, "exploratory, not confirmatory: not the confirmatory design")
    # the real A7 grid's report does not accept the stand-in's analysis
    blocked(m4gate.power_curve_item(write_docs(tmp_path / "c", sweep.analysis)), "7d9d53d3")


# -- 2. the bootstrap ------------------------------------------------------------- #


def test_the_bootstrap_item_reports_the_rule_in_force_per_arm(sweep: Sweep, tmp_path: Path) -> None:
    item = m4gate.bootstrap_item(write_docs(tmp_path / "t", sweep.analysis), TINY)
    passes(item)
    for arm in ("delta_seller", "ppo"):
        boot = sweep.analysis["arms"][arm]["bootstrap"]
        if boot["rule_in_force"] == "t":
            assert f"{arm}: all 2 levels agree, the t test is in force" in item.evidence
    assert "10,000 resamples, 200 normal replications" in item.evidence

    def diverge(d: dict[str, Any]) -> None:
        boot = d["arms"]["ppo"]["bootstrap"]
        boot["levels"][1].update(p_value=0.001, agrees=False)
        boot["levels"][0].update(p_value=0.5, agrees=True)
        boot.update(rule_in_force="bootstrap", diverged_at=[4.0])

    item = m4gate.bootstrap_item(write_docs(tmp_path / "b", edited(sweep.analysis, diverge)), TINY)
    passes(item)  # a divergence is a result, not a failure: the bootstrap rule is in force
    assert "ppo: diverges at 4 vol points of 2 levels, the bootstrap rule" in item.evidence


def _level(arm: str, k: int, **fields: Any) -> Callable[[dict[str, Any]], None]:
    return lambda d: d["arms"][arm]["bootstrap"]["levels"][k].update(**fields)


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (lambda d: d["settings"].update(n_boot=200), "200 resamples, fewer than 10,000"),
        (lambda d: d["settings"].update(agreement_alpha=0.1), "agreement alpha 0.1"),
        (lambda d: d["settings"].update(reference_replications=40), "could never be applied"),
        (lambda d: d["arms"]["ppo"]["bootstrap"]["levels"].pop(), "ppo: bootstrap levels at"),
        (_level("ppo", 0, threshold=0.05), "not p against 0.025"),
        (_level("ppo", 0, p_value=None), "not p against 0.025"),
        (_level("delta_seller", 1, agrees=False, p_value=0.9), "agrees is False with p = 0.9"),
        (lambda d: d["arms"]["ppo"]["bootstrap"].update(rule_in_force="bootstrap"), "give 't'"),
    ],
)
def test_the_bootstrap_item_refuses_a_rule_not_the_pre_stated_one(
    sweep: Sweep, tmp_path: Path, edit: Callable[[dict[str, Any]], None], why: str
) -> None:
    for arm in ("delta_seller", "ppo"):  # the edits assume levels that agree
        assert sweep.analysis["arms"][arm]["bootstrap"]["rule_in_force"] == "t"
    blocked(m4gate.bootstrap_item(write_docs(tmp_path, edited(sweep.analysis, edit)), TINY), why)


def test_the_bootstrap_item_needs_the_confirmatory_analysis(sweep: Sweep, tmp_path: Path) -> None:
    item = m4gate.bootstrap_item(write_docs(tmp_path, sweep.exploratory), TINY)
    blocked(item, "the M4 analysis is not usable: the analysis is exploratory")


# -- 3. tau^2 homogeneity ---------------------------------------------------------- #


def test_the_homogeneity_item_passes_on_the_test_being_run_whatever_it_finds(
    sweep: Sweep, tmp_path: Path
) -> None:
    item = m4gate.homogeneity_item(write_docs(tmp_path / "a", sweep.analysis), TINY)
    passes(item)
    assert "whatever it finds" in item.criterion and "not a failure" in item.criterion
    h = sweep.analysis["arms"]["ppo"]["homogeneity"]["contrast"]
    assert f"ppo on the contrast W = {h['statistic']:.3f}, df 1, 4" in item.evidence

    def reject(d: dict[str, Any]) -> None:
        d["arms"]["ppo"]["homogeneity"]["arm"].update(p_value=0.001, rejected=True)
        d["arms"]["delta_seller"]["homogeneity"]["contrast"].update(
            statistic=None, p_value=None, rejected=None
        )

    item = m4gate.homogeneity_item(write_docs(tmp_path / "b", edited(sweep.analysis, reject)), TINY)
    passes(item)  # a rejection is a finding, and an undefined test is reported as such
    assert "p = 0.001, equal variance rejected" in item.evidence
    assert "W = --, df 1, 4, p = --, undefined" in item.evidence


def _homog(arm: str, on: str, **fields: Any) -> Callable[[dict[str, Any]], None]:
    return lambda d: d["arms"][arm]["homogeneity"][on].update(**fields)


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (
            lambda d: d["arms"]["ppo"]["homogeneity"].pop("arm"),
            "ppo: no homogeneity test on the arm",
        ),
        (_homog("ppo", "contrast", test="bartlett"), "not Brown-Forsythe"),
        (_homog("ppo", "contrast", premiums=[0.0, 2.0, 4.0]), "run on premiums"),
        (_homog("delta_seller", "arm", df=[2, 2]), "not [2, 6] for 3 groups of 3"),
        (_homog("ppo", "arm", levels=[]), "no per-level split"),
        (_homog("ppo", "arm", p_value=0.001, rejected=False), "rejected is False with p = 0.001"),
    ],
)
def test_the_homogeneity_item_refuses_a_test_not_run_or_misreported(
    sweep: Sweep, tmp_path: Path, edit: Callable[[dict[str, Any]], None], why: str
) -> None:
    blocked(m4gate.homogeneity_item(write_docs(tmp_path, edited(sweep.analysis, edit)), TINY), why)


def test_the_homogeneity_item_needs_an_analysis(tmp_path: Path) -> None:
    blocked(m4gate.homogeneity_item(write_docs(tmp_path, None), TINY), "not usable: no ")


# -- 4. the VRP mapping, read off the curves --------------------------------------- #


def _exact(points: list[dict[str, Any]], key: str, at: float) -> float:
    """The analysis's own reading, from m4analysis: lambda linear in premium, nct power."""
    xs = [c["premium"] for c in points]
    lam = float(np.interp(min(max(at, xs[0]), xs[-1]), xs, [c[key] for c in points]))
    return m4.power_at(lam, TINY.n_seeds)


def test_the_vrp_item_reads_every_arms_curve_at_the_point_and_the_interval(
    sweep: Sweep, tmp_path: Path
) -> None:
    item = m4gate.vrp_item(write_docs(tmp_path, sweep.analysis), TINY)
    passes(item)
    assert "sha256 abababababababab, the build DATA_MANIFEST.json records" in item.evidence
    assert "mapped premium +2.5 vol points, 95% [+1, +3.5], between sweep points 2 and 4" in (
        item.evidence
    )
    for arm in ("delta_seller", "ppo"):
        points = sweep.analysis["arms"][arm]["curve"]["points"]
        mid, band = m4gate.read_curve(points, TINY.n_seeds, 0.05, 1.0, 3.5, 2.5)
        assert mid.power == pytest.approx(_exact(points, "noncentrality", 2.5), abs=1e-12)
        assert mid.power_low == pytest.approx(_exact(points, "noncentrality", 1.0), abs=1e-12)
        assert mid.power_high == pytest.approx(_exact(points, "noncentrality", 3.5), abs=1e-12)
        edges = [_exact(points, "noncentrality_lo", x) for x in (1.0, 2.0, 3.5)]
        assert band[0] == pytest.approx(min(edges), abs=1e-12)
        assert (
            f"{arm} power {mid.power:.3f} at the mapped premium, {mid.power_low:.3f} to "
            f"{mid.power_high:.3f} at its interval's ends and {mid.power_min:.3f} to "
            f"{mid.power_max:.3f} over it ({band[0]:.3f} to {band[1]:.3f} within the "
            "curve's band)"
        ) in item.evidence
    assert "held at the curve's end value" not in item.evidence


def test_the_vrp_item_says_when_an_interval_end_lies_off_the_curve(
    sweep: Sweep, tmp_path: Path
) -> None:
    docs = write_docs(tmp_path, sweep.analysis, mapping=vrp_mapping(0.63, (-0.34, 1.48)))
    item = m4gate.vrp_item(docs, TINY)
    passes(item)
    assert "between sweep points 0 and 2" in item.evidence
    assert "held at the curve's end value" in item.evidence


# The real PPO curve of the confirmatory sweep (docs/M4_ANALYSIS.json at 834ffb3): it
# dips at 0.5 and at 4 vol points by sampling noise, which the old reader refused.
REAL_PPO = [
    (0.25, 1.9110, -0.4219, 4.3577),
    (0.5, 1.4996, -0.7026, 3.7708),
    (1.0, 3.5307, 0.5414, 6.9485),
    (2.0, 6.9226, 2.1618, 12.8703),
    (4.0, 5.0256, 1.2973, 9.5323),
]


def _real_curve(edit: dict[str, Any], S: int = 5) -> None:
    knots = [m4.CurvePoint(0.0, False, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, None)]
    for p, mid, lo, hi in REAL_PPO:
        powers = (m4.power_at(mid, S), m4.power_at(lo, S), m4.power_at(hi, S))
        knots.append(m4.CurvePoint(p, True, mid, lo, hi, *powers, None))
    curve = edit["arms"]["ppo"]["curve"]
    curve["points"] = [
        {
            "premium": c.premium,
            "noncentrality": c.noncentrality,
            "noncentrality_lo": c.noncentrality_lo,
            "noncentrality_hi": c.noncentrality_hi,
            "power": c.power,
        }
        for c in knots
    ]
    curve["interpolated"] = [
        {"premium": x, "power": w, "power_lo": lo, "power_hi": hi}
        for x, w, lo, hi in m4.interpolate(knots, S)
    ]


def test_the_vrp_item_reads_a_curve_that_dips_as_it_is(sweep: Sweep, tmp_path: Path) -> None:
    data = edited(sweep.analysis, _real_curve)
    points = data["arms"]["ppo"]["curve"]["points"]
    dense = [c["power"] for c in data["arms"]["ppo"]["curve"]["interpolated"]]
    assert any(b < a for a, b in itertools.pairwise(dense))  # it does dip
    mid, _ = m4gate.read_curve(points, 5, 0.05, -0.344, 1.474, 0.6395)
    lam = float(
        np.interp(0.6395, [c["premium"] for c in points], [c["noncentrality"] for c in points])
    )
    assert mid.power == pytest.approx(m4.power_at(lam, 5), abs=1e-12)
    assert 0.5 < mid.power < 0.55 and mid.power_low == 0.05 and mid.clipped
    assert mid.power_min == 0.05 and mid.power_max == mid.power_high > 0.98
    assert (
        m4gate.dense_mismatch(points, data["arms"]["ppo"]["curve"]["interpolated"], 5, 0.05) == ""
    )
    # and the item passes on it, read at the stand-in grid's own S
    at_s = edited(sweep.analysis, lambda d: _real_curve(d, TINY.n_seeds))
    docs = write_docs(tmp_path, at_s, mapping=vrp_mapping(0.6395, (-0.344, 1.474)))
    item = m4gate.vrp_item(docs, TINY)
    passes(item)
    assert "ppo power " in item.evidence and "held at the curve's end value" in item.evidence


def test_the_vrp_item_refuses_a_sidecar_that_is_not_one_curve(sweep: Sweep, tmp_path: Path) -> None:
    def dense_off(d: dict[str, Any]) -> None:
        d["arms"]["ppo"]["curve"]["interpolated"][-1]["power"] -= 0.01

    item = m4gate.vrp_item(write_docs(tmp_path / "a", edited(sweep.analysis, dense_off)), TINY)
    blocked(item, "the ppo sidecar is not one curve: at 4 the interpolated curve holds")

    def no_knot(d: dict[str, Any]) -> None:
        del d["arms"]["ppo"]["curve"]["points"][-1]["noncentrality_hi"]

    item = m4gate.vrp_item(write_docs(tmp_path / "b", edited(sweep.analysis, no_knot)), TINY)
    blocked(item, "the ppo curve's knots cannot be read")
    assert "ppo on the interpolated curve, linearly: " in item.evidence
    assert " at +2.5" in item.evidence  # the number is in the report either way

    def unsorted(d: dict[str, Any]) -> None:
        d["arms"]["ppo"]["curve"]["points"][-1]["premium"] = 1.0

    item = m4gate.vrp_item(write_docs(tmp_path / "c", edited(sweep.analysis, unsorted)), TINY)
    blocked(item, "premiums must be strictly increasing")


def test_the_linear_reading_says_when_there_is_no_curve_to_read() -> None:
    assert m4gate._linear([], (1.0,)) == "--"
    assert m4gate._linear([{"premium": 0.0, "power": None}], (1.0,)) == "--"


@pytest.mark.parametrize(
    ("mapping", "manifest", "why"),
    [
        (vrp_mapping(), {"files": {}}, "records no sha256 for panel/spx_daily.parquet"),
        (vrp_mapping(), data_manifest("cd" * 32), "the build DATA_MANIFEST.json records is cdcd"),
        ({**vrp_mapping(), "power_inputs": {"estimate": None}}, None, "no finite mapped premium"),
        (
            {**vrp_mapping(), "power_inputs": {"estimate": 1.0, "interval": [0.0, None]}},
            None,
            "no finite mapped premium",
        ),
        (vrp_mapping(5.0, (1.0, 3.5)), None, "lies outside [+1, +3.5]"),
        ({**vrp_mapping(), "sweep_axis": [0.0, 1.0]}, None, "sweep axis [0.0, 1.0] is not"),
    ],
)
def test_the_vrp_item_refuses_a_mapping_not_from_the_recorded_panel_or_unreadable(
    sweep: Sweep,
    tmp_path: Path,
    mapping: dict[str, Any],
    manifest: dict[str, Any] | None,
    why: str,
) -> None:
    docs = write_docs(tmp_path, sweep.analysis, mapping=mapping, manifest=manifest)
    blocked(m4gate.vrp_item(docs, TINY), why)


def test_the_vrp_item_needs_its_files_and_the_analysis(sweep: Sweep, tmp_path: Path) -> None:
    docs = write_docs(tmp_path / "a", None)
    blocked(m4gate.vrp_item(docs, TINY), "the M4 analysis is not usable: no ")
    (docs / m4gate.DATA_MANIFEST_FILE).unlink()
    blocked(m4gate.vrp_item(docs, TINY), f"no {docs / m4gate.DATA_MANIFEST_FILE}")
    (docs / m4gate.VRP_FILE).unlink()
    blocked(m4gate.vrp_item(docs, TINY), f"no {docs / m4gate.VRP_FILE}")


def test_the_committed_mapping_is_from_the_panel_the_data_manifest_records() -> None:
    docs = Path(__file__).resolve().parent.parent / "docs"
    mapping, _ = m4gate.read_json(docs / m4gate.VRP_FILE)
    manifest, _ = m4gate.read_json(docs / m4gate.DATA_MANIFEST_FILE)
    assert mapping is not None and manifest is not None
    assert mapping["source_sha256"] == manifest["files"][m4gate.PANEL_DAILY]["sha256"]


# -- 5. the dashboard's views ---------------------------------------------------- #


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Registers a stand-in ``optspread.server`` whose app routes ``paths``."""

    def install(*paths: str, raises: Exception | None = None, factory: bool = True) -> None:
        module = types.ModuleType(m4gate.SERVER_MODULE)

        @dataclass
        class Settings:
            runs: Path
            docs: Path
            data: Path
            web: Path
            readme: Path

        def create_app(settings: Settings) -> types.SimpleNamespace:
            if raises is not None:
                raise raises
            routes = [types.SimpleNamespace(path=p) for p in paths]
            return types.SimpleNamespace(routes=[*routes, types.SimpleNamespace(name="mount")])

        setattr(module, "Settings", Settings)  # noqa: B010
        if factory:
            setattr(module, "create_app", create_app)  # noqa: B010
        monkeypatch.setitem(sys.modules, m4gate.SERVER_MODULE, module)

    yield install


def built_site(web: Path, *calls: str) -> Path:
    """A static export: an index page and a chunk calling ``calls``."""
    chunks = web / "_next" / "static" / "chunks" / "app" / "sweep"
    chunks.mkdir(parents=True, exist_ok=True)
    (web / "index.html").write_text("<html></html>")
    (chunks / "page-1a2b.js").write_text(";".join(f'get("{c}")' for c in calls))
    (web / "favicon.ico").write_bytes(b"\x00/api/m4")  # not a page or a script: not read
    return web


ROUTES = ("/api/health", "/api/runs/attribution", "/api/m4", "/api/power")


def test_the_dashboard_item_matches_the_views_by_name_and_finds_them_called(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(*ROUTES)
    web = built_site(tmp_path / "web", "/api/runs/attribution", "/api/m4")
    item = m4gate.dashboard_item(tmp_path / "runs", tmp_path / "docs", web)
    passes(item)
    assert "4 API routes; attribution view: /api/runs/attribution; power curve view: /api/m4" in (
        item.evidence
    )
    assert "/api/m4 from _next/static/chunks/app/sweep/page-1a2b.js" in item.evidence


def test_the_power_curve_view_is_an_m4_segment_or_a_power_curve_not_the_power_lab() -> None:
    match = m4gate.POWER_CURVE_ROUTE.search
    assert all(
        match(p)
        for p in ("/api/m4", "/api/m4/curve", "/api/power-curve", "/api/sweeps/power_curve")
    )
    assert not any(match(p) for p in ("/api/power", "/api/power/surface", "/api/m40"))
    assert m4gate.ATTRIBUTION_ROUTE.search("/api/runs/{key}/attribution")


@pytest.mark.parametrize(
    ("routes", "why"),
    [
        (("/api/health", "/api/m4"), "routes no attribution endpoint"),
        (("/api/attribution", "/api/power"), "routes no power curve endpoint"),
        (("/api/health",), "no attribution and no power curve endpoint"),
    ],
)
def test_the_dashboard_item_names_the_view_with_no_route(
    server: Callable[..., None], tmp_path: Path, routes: tuple[str, ...], why: str
) -> None:
    server(*routes)
    item = m4gate.dashboard_item(tmp_path, tmp_path, built_site(tmp_path / "web"))
    blocked(item, why)


def test_the_dashboard_item_needs_the_built_site_to_call_both_views(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(*ROUTES)
    blocked(m4gate.dashboard_item(tmp_path, tmp_path, tmp_path / "noweb"), "no built site at")
    web = built_site(tmp_path / "web", "/api/m4")
    item = m4gate.dashboard_item(tmp_path, tmp_path, web)
    blocked(item, "never calls the attribution route (/api/runs/attribution)")
    server("/api/runs/{key}/attribution", "/api/m4")
    passes(m4gate.dashboard_item(tmp_path, tmp_path, built_site(web, "/api/runs/", "/api/m4")))


def test_the_dashboard_item_without_a_server_or_with_a_broken_one(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(factory=False)
    blocked(m4gate.dashboard_item(tmp_path, tmp_path, tmp_path), "has no create_app")
    server(raises=RuntimeError("half wired"))
    blocked(m4gate.dashboard_item(tmp_path, tmp_path, tmp_path), "half wired")


def test_the_dashboard_item_builds_the_real_app(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from optspread.server import Settings, create_app

    item = m4gate.dashboard_item(tmp_path / "runs", tmp_path / "docs", tmp_path / "noweb")
    app = create_app(Settings(tmp_path, tmp_path, tmp_path, tmp_path / "noweb", tmp_path))
    n = len(m4gate._api_paths(app))
    assert n > 10 and item.evidence.startswith(f"{n} API routes; attribution view: ")
    assert not item.passed  # no built site in a temporary folder, whatever the routes


# -- 6. the collaboration report ---------------------------------------------------- #


def test_the_collaboration_report_must_exist_and_hold_something(tmp_path: Path) -> None:
    blocked(m4gate.collaboration_item(tmp_path), "no COLLABORATION.md or COLLABORATION_REPORT.md")
    (tmp_path / "COLLABORATION_REPORT.md").write_text("  \n\n")
    blocked(m4gate.collaboration_item(tmp_path), "COLLABORATION_REPORT.md is empty")
    (tmp_path / "COLLABORATION.md").write_text("# Collaboration report\n\nWho did what.\n")
    item = m4gate.collaboration_item(tmp_path)
    passes(item)
    assert (
        "3 lines, beginning 'Collaboration report'" in item.evidence and item.owner == "All members"
    )


# -- 7. the attribution residual ----------------------------------------------------- #


def fake_policy(path: Path, config: RunConfig) -> FakeLearned:
    """The fake learner's policy for a PPO cell, standing in for loading its policy.zip."""
    assert config.train is not None
    return FakeLearned(config.train.seed)


@pytest.fixture(scope="module")
def attributed(sweep: Sweep, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real attribution per arm of the stand-in grid, from the cells' run directories."""
    out = tmp_path_factory.mktemp("attribution")
    for arm, premium in (("delta_seller", 2.0), ("ppo", 4.0)):
        cell = sweep.record.cell(premium, arm, 0)
        result = attribute_run(sweep.root / cell.run_id, load_policy=fake_policy)
        assert result.reproduces is True and result.agrees is True
        write_attribution(result, out / output_name(result), steps=False)
    (out / "broken.json").write_text("{}")
    return out


def test_the_residual_item_reads_one_reproducing_attribution_per_arm(
    sweep: Sweep, attributed: Path
) -> None:
    item = m4gate.residual_item(sweep.root, attributed, TINY)
    passes(item)
    assert (
        "2 attribution(s) of grid-" in item.evidence and "(2 readable, 1 refused)" in item.evidence
    )
    assert "delta_seller: 1 cell(s) at premiums 2" in item.evidence
    assert "ppo: 1 cell(s) at premiums 4" in item.evidence
    assert "every one agreeing with its run.json (1 of 1 bit for bit)" in item.evidence
    assert "vega is zero on every attributed sweep cell" in item.evidence
    assert "no other attribution under the root exercises vega" in item.evidence


def test_the_residual_item_passes_a_cell_that_agrees_without_reproducing(
    sweep: Sweep, attributed: Path, tmp_path: Path
) -> None:
    """A cell attributed on another platform: its last digits differ, its trades do not."""
    import shutil

    root = tmp_path / "other_platform"
    shutil.copytree(attributed, root)
    _rewrite(
        root,
        lambda d: d.update(reproduces=False, agrees=True),
        sweep.record.cell(4.0, "ppo", 0).run_id,
    )
    item = m4gate.residual_item(sweep.root, root, TINY)
    passes(item)
    assert (
        "ppo: 1 cell(s) at premiums 4" in item.evidence and "(0 of 1 bit for bit)" in item.evidence
    )


def test_the_residual_item_says_what_exercises_vega(
    sweep: Sweep, attributed: Path, tmp_path: Path
) -> None:
    import shutil

    root = tmp_path / "moving"
    shutil.copytree(attributed, root)
    data = json.loads(
        next(root.glob(f"{sweep.record.cell(2.0, 'delta_seller', 0).run_id}*.json")).read_text()
    )
    data.update(run_id="real-chain-run", source="optionmetrics", residual_share=0.034)
    data["totals"]["vega"] = -1234.5
    (root / "real-chain-run.optionmetrics.json").write_text(json.dumps(data))
    item = m4gate.residual_item(sweep.root, root, TINY)
    passes(item)
    assert (
        "1 other attribution(s) under the root exercise vega (optionmetrics), residual share "
        "at most 3.40% (real-chain-run)"
    ) in item.evidence


def _rewrite(root: Path, edit: Callable[[dict[str, Any]], None], match: str = "") -> None:
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("run_id", "").startswith(match):
            edit(data)
            path.write_text(json.dumps(data))
            return
    raise AssertionError("nothing to rewrite")


def test_the_residual_item_refuses_what_is_not_evidence(
    sweep: Sweep, attributed: Path, tmp_path: Path
) -> None:
    import shutil

    blocked(m4gate.residual_item(tmp_path / "noruns", attributed, TINY), "no grid record at")
    blocked(m4gate.residual_item(sweep.root, tmp_path / "none", TINY), "no attribution of a")
    blocked(m4gate.residual_item(sweep.root, attributed), "no grid record at")  # the A7 grid's
    bad_grid = tmp_path / "badgrid"
    bad_grid.mkdir()
    (bad_grid / grid_filename(TINY)).write_text("{}")
    blocked(m4gate.residual_item(bad_grid, attributed, TINY), "is refused")

    ppo = sweep.record.cell(4.0, "ppo", 0).run_id
    cases: list[tuple[Callable[[dict[str, Any]], None], str]] = [
        (
            lambda d: d.update(reproduces=False, agrees=False),
            f"{ppo} does not agree with its run.json",
        ),
        (lambda d: d.pop("agrees"), "written before the agrees verdict"),
        (lambda d: d.update(run_dir=None), "no attribution of a ppo cell"),
        (lambda d: d.update(residual_share=0.08), "has a residual share of 8.00%, not under 5%"),
    ]
    for k, (edit, why) in enumerate(cases):
        root = tmp_path / f"case{k}"
        shutil.copytree(attributed, root)
        _rewrite(root, edit, ppo)
        blocked(m4gate.residual_item(sweep.root, root, TINY), why)


# -- the report command ------------------------------------------------------------ #


def test_the_report_names_every_root_portably_and_round_trips(
    sweep: Sweep, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import m4_gate_report

    assert m4_gate_report.default_web({}) == m4_gate_report.REPO_ROOT / "web" / "out"
    assert m4_gate_report.default_web({"OPTSPREAD_WEB": "/srv/site"}) == Path("/srv/site")
    data = tmp_path / "shared"
    monkeypatch.setenv("VEGALAB_DATA", str(data))
    docs = write_docs(data / "docs", sweep.exploratory)
    out = data / "out" / "M4_GATE_REPORT.md"
    argv = ["--out", str(out), "--docs", str(docs), "--attribution", str(data / "attribution")]
    argv += ["--web", str(data / "web")]
    assert m4_gate_report.main(argv, on=date(2026, 11, 19)) == 1  # the A7 grid is not here
    text = out.read_text()
    assert "Runs root: `$VEGALAB_DATA/runs` (grid `grid-7d9d53d340006f02.json`)" in text
    assert "docs: `$VEGALAB_DATA/docs`; attribution root: `$VEGALAB_DATA/attribution`" in text
    assert "Built by `python -m scripts.m4_gate_report` at " in text and str(data) not in text
    report = GateReport.from_markdown(text)  # the committed file re-renders byte for byte
    assert report.milestone == "M4" and len(report.items) == 7 and not report.passed
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["report_date"] == "2026-11-19" and sidecar["verdict"] == "NOT CLEARED"
    assert [i["name"] for i in sidecar["items"]] == [i.name for i in report.items]
    assert "exploratory, not confirmatory" in sidecar["items"][0]["failed_first"]


def test_the_report_clears_when_every_item_does(
    sweep: Sweep,
    attributed: Path,
    tmp_path: Path,
    server: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import m4_gate_report

    server(*ROUTES)
    docs = write_docs(tmp_path / "docs", sweep.analysis)
    (docs / "COLLABORATION.md").write_text("# Collaboration report\n")
    web = built_site(tmp_path / "web", "/api/runs/attribution", "/api/m4")
    for name in ("power_curve_item", "bootstrap_item", "homogeneity_item", "vrp_item"):
        item = getattr(m4gate, name)
        monkeypatch.setattr(m4_gate_report, name, lambda d, f=item: f(d, TINY))
    monkeypatch.setattr(
        m4_gate_report, "residual_item", lambda r, a: m4gate.residual_item(r, a, TINY)
    )
    argv = ["--docs", str(docs), "--runs-root", str(sweep.root), "--attribution", str(attributed)]
    argv += ["--web", str(web), "--out", "-"]
    assert m4_gate_report.main(argv) == 0
    text = capsys.readouterr().out
    assert "**Verdict: PASS** (7 items, 0 failed, 0 without a stated criterion)" in text
