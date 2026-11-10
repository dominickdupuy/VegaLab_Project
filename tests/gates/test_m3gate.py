"""Dominick Dupuy: the M3 gate items (week 8, #11).

Every item is tested both ways: with its deliverable absent or wrong it must read NOT
CLEARED and say what failed first, and with evidence that meets its criterion it must
pass. The sweep is a tiny grid the real orchestrator runs with the fake learner, written
on a throwaway git history with its own cutoff tag (``tests/m3_fixtures.py``); for the
length of a test it stands in for the A7 grid, and the fake learner for the real one.
The failures are copies of it broken one way at a time. The floor runs the real detector
on fewer seeds against a sidecar written to match. The dashboard server is stood in for
by a small FastAPI app registered under the name the gate imports, as the M4 tests do,
and the real one is called once. The partner's record is made by
``optspread.reproduction``'s own models. Nothing here reads the shared folder.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from m3_fixtures import FAKE_TRAINER, TAG, TINY, World, build_world, git

from optspread import m3gate
from optspread import reproduction as rp
from optspread.artifacts import RUN_FILENAME
from optspread.gatereport import GateItem, GateReport
from optspread.leaks import default_env, leak_floor
from optspread.manifest import MANIFEST_FILENAME, file_sha256
from optspread.orchestrate import grid_cells, grid_filename
from optspread.replay import ReplayResult


def blocked(item: GateItem, contains: str) -> None:
    assert not item.passed and item.verdict == "FAIL"
    assert contains in item.failed_first, item.failed_first
    assert item.has_criterion


def passes(item: GateItem) -> None:
    assert item.passed and item.verdict == "PASS", item.failed_first
    assert item.evidence


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return build_world(tmp_path_factory.mktemp("m3gate"))


@pytest.fixture(autouse=True)
def _fake_learner_is_the_real_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m3gate, "REAL_TRAINER", FAKE_TRAINER)


def copy_root(world: World, tmp_path: Path) -> Path:
    return Path(shutil.copytree(world.root, tmp_path / "runs"))


sweep = partial(m3gate.sweep_item, grid=TINY, tag=TAG)


# -- 1. the sweep ------------------------------------------------------------------- #


@pytest.mark.behaviour
def test_the_sweep_item_passes_on_a_complete_valid_grid(world: World) -> None:
    item = sweep(world.root, world.repo)
    passes(item)
    assert "18 of 18 cells (3 premiums x 2 arms x 3 blocks), all valid under" in item.evidence
    assert f"9 PPO cells by {FAKE_TRAINER}" in item.evidence
    assert f"written at {world.at_tag[:8]} (clean, at or after {TAG}" in item.evidence
    assert "no replay spot-check" in item.evidence


def replay_line(root: Path, run_id: str, *, sha: str | None = None, ok: bool = True) -> str:
    digest = sha or file_sha256(root / run_id / RUN_FILENAME)
    if not ok:
        return f"DIFFERS  {run_id}  (1.0s)  episode seed 7000: saved pnl 1.0, replayed 2.0"
    return f"IDENTICAL  {run_id}  (17.5s)  run.json sha256 {digest[:16]} reproduced"


@pytest.mark.invariant
def test_the_sweep_item_reads_the_replay_log_and_holds_it_to_the_files(
    world: World, tmp_path: Path
) -> None:
    root = copy_root(world, tmp_path)
    cells = grid_cells(TINY)
    seller, ppo = cells[0], cells[1]
    log = [replay_line(root, seller.run_id), replay_line(root, ppo.run_id)]
    log += ["IDENTICAL  someone-elses-run  (1.0s)  run.json sha256 00 reproduced", "2/2 runs"]
    (root / m3gate.REPLAY_LOG).write_text("\n".join(log) + "\n")
    item = sweep(root, world.repo)
    passes(item)
    assert "2 cell(s) bit-identical (1 delta_seller, 1 ppo; 2 in replay.log)" in item.evidence
    (root / m3gate.REPLAY_LOG).write_text(replay_line(root, seller.run_id, sha="ab" * 8) + "\n")
    blocked(sweep(root, world.repo), "stale")
    (root / m3gate.REPLAY_LOG).write_text(replay_line(root, seller.run_id, ok=False) + "\n")
    blocked(sweep(root, world.repo), f"the replay (log) of {seller.run_id} differs: episode seed")
    (root / m3gate.REPLAY_LOG).write_text(f"IDENTICAL  {seller.run_id}  (1.0s)  no digest\n")
    blocked(sweep(root, world.repo), "(none named)")


@pytest.mark.invariant
def test_the_sweep_item_replays_a_sample_here_when_asked(world: World) -> None:
    calls: list[str] = []

    def replay(run_dir: Path, matches: bool = True) -> ReplayResult:
        calls.append(run_dir.name)
        return ReplayResult(run_dir, run_dir.name, matches, "run.json reproduced", 0.1)

    item = sweep(world.root, world.repo, replays=3, replay=replay)
    passes(item)
    assert "3 cell(s) bit-identical (2 delta_seller, 1 ppo; 3 replayed here)" in item.evidence
    arms = {c.run_id: c.arm for c in grid_cells(TINY)}
    assert [arms[c] for c in calls] == ["delta_seller", "ppo", "delta_seller"]
    item = sweep(world.root, world.repo, replays=1, replay=partial(replay, matches=False))
    blocked(item, "the replay (here) of")
    item = sweep(world.root, world.repo, replays=1, replay=replay)
    assert "no ppo cell replayed" in item.evidence


def _written_elsewhere(run_dir: Path, platform: dict[str, str] | None) -> None:
    path = run_dir / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    if platform is None:
        manifest.pop("platform", None)
    else:
        manifest["platform"] = platform
    path.write_text(json.dumps(manifest))


WINDOWS = {"os": "Windows", "arch": "AMD64", "python": "3.12.0", "numpy": "2.5.3"}


@pytest.mark.invariant
def test_cells_written_on_another_platform_replay_within_the_tolerance_ppo_not_at_all(
    world: World, tmp_path: Path
) -> None:
    root = copy_root(world, tmp_path)
    for cell in grid_cells(TINY):
        _written_elsewhere(root / cell.run_id, WINDOWS)
    seller = grid_cells(TINY)[0]
    # Here the bytes are this machine's own, so a tolerance-mode replay says IDENTICAL; the
    # point is that it was asked with the seller tolerance, and the reason names both.
    found = m3gate.replay_here(root / seller.run_id)
    assert found.rtol == rp.SELLER_REL_TOL and found.verdict == "IDENTICAL"
    assert "recorded on Windows AMD64" in found.reason
    calls: list[str] = []

    def replay(run_dir: Path) -> ReplayResult:
        calls.append(run_dir.name)
        return ReplayResult(
            run_dir, run_dir.name, False, "within", 0.1, rtol=1e-9, within=True, max_rel_diff=2e-12
        )

    item = sweep(root, world.repo, replays=3, replay=replay)
    passes(item)
    arms = {c.run_id: c.arm for c in grid_cells(TINY)}
    assert [arms[c] for c in calls] == ["delta_seller"] * 3, "no PPO cell trained elsewhere"
    assert "3 cell(s) reproduced (3 delta_seller; 3 replayed here, 3 of those within 1e-09" in (
        item.evidence
    )
    assert "9 PPO cell(s) trained on another platform are not comparable here" in item.evidence
    # A manifest that records no platform (written before it was recorded) counts as another.
    _written_elsewhere(root / seller.run_id, None)
    assert m3gate.replay_here(root / seller.run_id).rtol == rp.SELLER_REL_TOL


@pytest.mark.behaviour
def test_the_replay_sample_is_fixed_and_takes_the_arms_in_turn() -> None:
    cells = grid_cells(TINY)
    first = m3gate.replay_sample(cells, 4)
    assert first == m3gate.replay_sample(cells, 4)
    assert [c.arm for c in first] == ["delta_seller", "ppo", "delta_seller", "ppo"]
    assert len(set(first)) == 4 and m3gate.replay_sample(cells, 0) == []
    assert len(m3gate.replay_sample(cells, 99)) == len(cells)


@pytest.mark.invariant
def test_the_sweep_item_without_a_sweep_or_with_another_grid(world: World, tmp_path: Path) -> None:
    blocked(sweep(tmp_path, world.repo), "the sweep has not run")
    other = TINY.model_copy(update={"n_seeds": 2})
    root = copy_root(world, tmp_path)
    shutil.copy(root / grid_filename(TINY), root / grid_filename(other))
    blocked(m3gate.sweep_item(root, world.repo, grid=other, tag=TAG), "the record is of grid")
    (root / grid_filename(TINY)).write_text("{")
    blocked(sweep(root, world.repo), "is refused")


@pytest.mark.invariant
def test_the_sweep_item_needs_every_cell_listed(world: World, tmp_path: Path) -> None:
    root = copy_root(world, tmp_path)
    record = json.loads((root / grid_filename(TINY)).read_text())
    dropped = record["cells"].pop()
    (root / grid_filename(TINY)).write_text(json.dumps(record))
    blocked(sweep(root, world.repo), "lists 17 cells, the grid has 18; first missing: ")
    assert dropped["run_id"] in sweep(root, world.repo).failed_first


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("damage", "why"),
    [
        ("delete", "no run.json and manifest.json"),
        ("results", "run.json no longer match the manifest's digests"),
        ("unreadable", "unreadable"),
        ("headline", "the record's headline is not its run.json's"),
    ],
)
def test_the_sweep_item_names_the_first_invalid_cell(
    world: World, tmp_path: Path, damage: str, why: str
) -> None:
    root = copy_root(world, tmp_path)
    cell = grid_cells(TINY)[0]
    if damage == "delete":
        shutil.rmtree(root / cell.run_id)
    elif damage == "results":
        run = json.loads((root / cell.run_id / RUN_FILENAME).read_text())
        run["mean_pnl"] += 1.0
        (root / cell.run_id / RUN_FILENAME).write_text(json.dumps(run))
    elif damage == "unreadable":
        (root / cell.run_id / MANIFEST_FILENAME).write_text("{")
    else:
        record = json.loads((root / grid_filename(TINY)).read_text())
        record["cells"][0]["mean_pnl"] += 1.0
        (root / grid_filename(TINY)).write_text(json.dumps(record))
    item = sweep(root, world.repo)
    blocked(item, "1 of 18 cells are not valid; first: ")
    assert why in item.failed_first and cell.run_id in item.failed_first


@pytest.mark.invariant
def test_why_invalid_names_the_key_the_run_id_and_the_trainer(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = copy_root(world, tmp_path)
    seller, ppo = grid_cells(TINY)[0], grid_cells(TINY)[1]
    monkeypatch.setattr(m3gate, "REAL_TRAINER", "optspread.train.train_ppo")
    assert "not optspread.train.train_ppo" in m3gate.why_invalid(root / ppo.run_id, ppo)
    blocked(sweep(root, world.repo), f"trained by {FAKE_TRAINER}, not optspread.train.train_ppo")
    other = grid_cells(TINY)[2]
    assert "keyed by" in m3gate.why_invalid(root / seller.run_id, other)
    manifest = json.loads((root / seller.run_id / MANIFEST_FILENAME).read_text())
    manifest["run_id"] = "elsewhere"
    (root / seller.run_id / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert "run id" in m3gate.why_invalid(root / seller.run_id, seller)
    manifest["run_id"] = seller.run_id
    manifest["data"] = {"source": "optionmetrics"}
    (root / seller.run_id / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    assert "ran on optionmetrics" in m3gate.why_invalid(root / seller.run_id, seller)


@pytest.mark.invariant
def test_the_sweep_item_needs_clean_cells_written_at_or_after_the_tag(
    world: World, tmp_path: Path
) -> None:
    root = copy_root(world, tmp_path)
    blocked(m3gate.sweep_item(root, world.repo, grid=TINY, tag="no-such-tag"), "no no-such-tag")
    git(world.repo, "tag", "later-test", world.after)
    item = m3gate.sweep_item(root, world.repo, grid=TINY, tag="later-test")
    blocked(item, f"commit {world.at_tag[:8]} is not at or after later-test")
    cell = grid_cells(TINY)[0]
    manifest = json.loads((root / cell.run_id / MANIFEST_FILENAME).read_text())
    manifest["git_dirty"] = True
    (root / cell.run_id / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    blocked(sweep(root, world.repo), f"1 cell(s) written from a dirty tree, first {cell.run_id}")


# -- 2. the leak detection floor ------------------------------------------------------ #

SEEDS = 3


def m1_sidecar(docs: Path, floor: float | None) -> Path:
    docs.mkdir(parents=True, exist_ok=True)
    evidence = "nothing was caught" if floor is None else f"grid (...), floor eps = {floor}"
    items = [{"name": m3gate.M1_FLOOR_ITEM, "evidence": evidence}]
    (docs / m3gate.M1_SIDECAR).write_text(
        json.dumps({"milestone": "M1", "report_date": "2026-09-28", "items": items})
    )
    return docs


@pytest.fixture(scope="module")
def measured_floor() -> float:
    floor = leak_floor(default_env, m3gate.FLOOR_GRID, range(SEEDS)).floor
    assert floor is not None
    return floor


@pytest.mark.invariant
@pytest.mark.slow
def test_the_floor_item_measures_the_floor_and_matches_m1(
    tmp_path: Path, measured_floor: float
) -> None:
    docs = m1_sidecar(tmp_path / "docs", measured_floor)
    item = m3gate.floor_item(docs, seeds=SEEDS)
    passes(item)
    assert f"floor eps = {measured_floor:g}" in item.evidence
    assert "clean world quiet" in item.evidence and "(2026-09-28)" in item.evidence
    blocked(m3gate.floor_item(m1_sidecar(docs, 0.5), seeds=SEEDS), "records floor eps = 0.5")
    blocked(m3gate.floor_item(m1_sidecar(docs, None), seeds=SEEDS), "stating a floor")
    blocked(m3gate.floor_item(tmp_path / "none", seeds=SEEDS), "no ")


@pytest.mark.invariant
def test_the_committed_m1_sidecar_states_the_readme_floor() -> None:
    assert m3gate.m1_floor(Path(__file__).resolve().parent.parent.parent / "docs") == (
        0.1,
        "2026-09-28",
    )


def _never(eps: float) -> Any:
    return default_env(0.0)


def _always(eps: float) -> Any:
    return default_env(0.5)


def _only_middle(eps: float) -> Any:
    return default_env(0.5 if eps == 0.25 else 0.0)


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("make_env", "why"),
    [
        (_never, "no planted leak was caught"),
        (_always, "the clean world is flagged"),
        (_only_middle, "is missed above the smallest caught 0.25: not a threshold"),
    ],
)
def test_the_floor_item_refuses_a_detector_without_a_threshold(
    tmp_path: Path, make_env: Callable[[float], Any], why: str
) -> None:
    item = m3gate.floor_item(m1_sidecar(tmp_path, 0.1), seeds=SEEDS, make_env=make_env)
    blocked(item, why)


# -- 3. the poisoned-environment demo ------------------------------------------------- #


def demo_payload(**override: Any) -> dict[str, Any]:
    side = {"flagged": False, "worst_feature": "ret_1d", "p_value": 0.2, "threshold": 0.001}
    payload: dict[str, Any] = {
        "poisoned": {**side, "flagged": True, "p_value": 1e-50, "past_unchanged": False},
        "clean": {**side, "past_unchanged": True},
        "caught": True,
    }
    for key, value in override.items():
        where, _, field = key.partition("__")
        if field:
            payload[where][field] = value
        else:
            payload[where] = value
    return payload


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Registers a stand-in ``optspread.server`` whose app answers the demo with ``payload``."""
    fastapi = pytest.importorskip("fastapi")

    def install(
        payload: dict[str, Any] | None = None,
        *,
        path: str = "/api/leaks/demo",
        status: int = 200,
        raises: Exception | None = None,
        factory: bool = True,
    ) -> None:
        module = types.ModuleType(m3gate.SERVER_MODULE)

        @dataclass
        class Settings:
            runs: Path
            docs: Path
            data: Path
            web: Path
            readme: Path

        def create_app(settings: Settings) -> Any:
            if raises is not None:
                raise raises
            app = fastapi.FastAPI()

            @app.get("/api/health")
            def health() -> dict[str, str]:
                return {"ok": "yes"}

            @app.post(path)
            def demo() -> Any:
                if status != 200:
                    raise fastapi.HTTPException(status_code=status, detail="refused")
                return payload if payload is not None else demo_payload()

            return app

        setattr(module, "Settings", Settings)  # noqa: B010
        if factory:
            setattr(module, "create_app", create_app)  # noqa: B010
        monkeypatch.setitem(sys.modules, m3gate.SERVER_MODULE, module)

    yield install


def built_site(web: Path, *calls: str, page: str | None = "leaks/index.html") -> Path:
    chunks = web / "_next" / "static" / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    (web / "index.html").write_text("<html></html>")
    if page is not None:
        (web / page).parent.mkdir(parents=True, exist_ok=True)
        (web / page).write_text("<html>leaks</html>")
    (chunks / "page-1a2b.js").write_text(";".join(f'post("{c}")' for c in calls))
    return web


@pytest.mark.behaviour
def test_the_demo_item_calls_the_route_and_finds_the_page(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server()
    web = built_site(tmp_path / "web", "/api/leaks/demo")
    item = m3gate.demo_item(tmp_path, tmp_path, web)
    passes(item)
    assert "POST /api/leaks/demo (eps 0.5, 10 seeds) answered 200" in item.evidence
    assert "poisoned flagged True (worst ret_1d, p = 1e-50), past moved True" in item.evidence
    assert "has leaks/index.html and calls /api/leaks/demo from _next/static" in item.evidence


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"poisoned__flagged": False}, "does not flag the poisoned variant"),
        ({"poisoned__past_unchanged": True}, "perturbation check does not catch"),
        ({"clean__flagged": True}, "the clean variant does not pass"),
        ({"clean__past_unchanged": False}, "the clean variant does not pass"),
        ({"caught": False}, "does not report the leak caught"),
        ({"poisoned": "nothing"}, "does not flag the poisoned variant"),
    ],
)
def test_the_demo_item_fails_unless_poisoned_fails_and_clean_passes(
    server: Callable[..., None], tmp_path: Path, override: dict[str, Any], why: str
) -> None:
    server(demo_payload(**override))
    blocked(m3gate.demo_item(tmp_path, tmp_path, built_site(tmp_path / "web")), why)


@pytest.mark.behaviour
def test_the_demo_item_needs_the_route_answering_and_the_site(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(path="/api/other")
    blocked(m3gate.demo_item(tmp_path, tmp_path, tmp_path), "routes no leak demo endpoint")
    server(status=422)
    blocked(m3gate.demo_item(tmp_path, tmp_path, tmp_path), "answered 422")
    server()
    blocked(m3gate.demo_item(tmp_path, tmp_path, tmp_path / "noweb"), "no built site at")
    web = built_site(tmp_path / "web", page=None)
    blocked(m3gate.demo_item(tmp_path, tmp_path, web), "has no leak page")
    web = built_site(tmp_path / "web2")
    blocked(m3gate.demo_item(tmp_path, tmp_path, web), "never calls /api/leaks/demo")
    passes(
        m3gate.demo_item(tmp_path, tmp_path, built_site(web, "/api/leaks/demo", page="leaks.html"))
    )


@pytest.mark.behaviour
def test_the_demo_item_without_a_server_or_with_a_broken_one(
    server: Callable[..., None], tmp_path: Path
) -> None:
    server(factory=False)
    blocked(m3gate.demo_item(tmp_path, tmp_path, tmp_path), "has no create_app")
    server(raises=RuntimeError("half wired"))
    blocked(m3gate.demo_item(tmp_path, tmp_path, tmp_path), "half wired")


@pytest.mark.invariant
def test_the_demo_item_calls_the_real_app(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    web = built_site(tmp_path / "web", "/api/leaks/demo")
    item = m3gate.demo_item(tmp_path / "runs", tmp_path / "docs", web)
    passes(item)
    assert "poisoned flagged True" in item.evidence and "clean flagged False" in item.evidence


# -- 4. the partner reproduction ------------------------------------------------------ #


def comparison(cell: rp.ExpectedCell, **override: Any) -> rp.CellComparison:
    fields: dict[str, Any] = {
        "run_id": cell.run_id,
        "arm": cell.arm,
        "premium": cell.premium,
        "block": cell.block,
        "rel_tol": rp.SELLER_REL_TOL if cell.arm == rp.SELLER else 0.0,
        "byte_identical": False,
        "within_tolerance": True,
        "max_rel_diff": 8.6e-13,
        "values_compared": 2005,
        "episodes_compared": 3,
        "seconds": 40.0,
    }
    if cell.arm == rp.PPO:  # compared statistically: training is not bit-reproducible
        fields.update(check="statistical", z=-0.41, k_se=rp.PPO_K_SE, structure="20 episodes")
    return rp.CellComparison(**{**fields, **override})


def partner_record(world: World, **override: Any) -> rp.PartnerRecord:
    sellers = world.expected.sellers(2)
    steps = [rp.StepResult(name=n, passed=True, seconds=10.0) for n in rp.STEPS]
    fields: dict[str, Any] = {
        "team": "Partner U",
        "generated": "2026-11-13T15:00:00+00:00",
        "commit": world.after,
        "dirty": False,
        "tag": TAG,
        "tag_commit": world.at_tag,
        "platform": {"platform": "macOS-15.1-arm64", "python": "3.12.7"},
        "versions": {"numpy": "2.5.3"},
        "expected_file": f"docs/{rp.EXPECTED_FILE}",
        "expected_sha256": rp.sha256_bytes(world.expected_text.encode()),
        "tolerance": rp.tolerances(),
        "steps": steps,
        "cells": [comparison(c) for c in sellers],
        "seconds": 900.0,
    }
    return rp.PartnerRecord(**{**fields, **override})


def send(docs: Path, record: rp.PartnerRecord) -> Path:
    docs.mkdir(parents=True, exist_ok=True)
    (docs / rp.RECORD_FILE).write_text(record.model_dump_json(indent=2))
    return docs


def partner(docs: Path, world: World, root: Path | None = None) -> GateItem:
    return m3gate.partner_item(docs, root or world.root, world.repo, grid=TINY, tag=TAG)


@pytest.mark.behaviour
def test_the_partner_item_passes_on_a_complete_record(world: World, tmp_path: Path) -> None:
    item = partner(send(tmp_path, partner_record(world)), world)
    passes(item)
    assert "from Partner U on macOS-15.1-arm64, Python 3.12.7" in item.evidence
    assert f"(clean, at or after {TAG} {world.at_tag[:8]})" in item.evidence
    assert "8 of 8 steps passed" in item.evidence
    assert "2 sweep cell(s) regenerated (2 delta_seller; premiums 0, 4)" in item.evidence
    assert "0 byte-identical" in item.evidence and "18 run.json digests are the sweep's" in (
        item.evidence
    )


@pytest.mark.behaviour
def test_the_partner_item_without_a_record_or_an_unreadable_one(
    world: World, tmp_path: Path
) -> None:
    blocked(partner(tmp_path, world), "the partner team has not sent back")
    (tmp_path / rp.RECORD_FILE).write_text("{}")
    blocked(partner(tmp_path, world), "is refused")
    send(tmp_path, partner_record(world, schema_version=2))
    blocked(partner(tmp_path, world), "schema 2, not 1")


@pytest.mark.invariant
@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"commit": "f" * 40}, "is not in this repository"),
        ({"dirty": True}, "uncommitted changes"),
        ({"tolerance": {"seller_rel": 1e-3, "abs": 1.0, "ppo_k_se": 3.0}}, "compared at tolerance"),
        ({"expected_sha256": "0" * 64}, "compared against an expected file with sha256 0000"),
    ],
)
def test_the_partner_item_refuses_a_record_that_is_not_the_repository(
    world: World, tmp_path: Path, override: dict[str, Any], why: str
) -> None:
    blocked(partner(send(tmp_path, partner_record(world, **override)), world), why)


@pytest.mark.invariant
def test_the_partner_item_needs_a_commit_at_or_after_the_tag_with_the_expected_file(
    world: World, tmp_path: Path
) -> None:
    item = partner(send(tmp_path, partner_record(world, commit=world.before)), world)
    blocked(item, f"{world.before[:8]} is not at or after {TAG}")
    item = partner(send(tmp_path, partner_record(world, commit=world.at_tag)), world)
    blocked(item, f"docs/{rp.EXPECTED_FILE} is not in commit {world.at_tag[:8]}")
    item = m3gate.partner_item(
        send(tmp_path, partner_record(world)), world.root, world.repo, grid=TINY, tag="nope"
    )
    blocked(item, "no nope tag")


@pytest.mark.invariant
def test_the_partner_item_needs_every_step_passed(world: World, tmp_path: Path) -> None:
    good = partner_record(world)
    fewer = good.model_copy(update={"steps": good.steps[:-1]})
    blocked(partner(send(tmp_path, fewer), world), "step(s) not run: sweep cells")
    failed = rp.StepResult(name="mypy --strict", passed=False, seconds=3.0, detail="exited 1")
    steps = [failed if s.name == failed.name else s for s in good.steps]
    item = partner(send(tmp_path, good.model_copy(update={"steps": steps})), world)
    blocked(item, "mypy --strict failed: exited 1")
    ppo = rp.StepResult(name=rp.PPO_STEP, passed=False, seconds=1200.0, returncode=None)
    item = partner(send(tmp_path, good.model_copy(update={"steps": [*good.steps, ppo]})), world)
    blocked(item, "ppo cell failed: exit None")


@pytest.mark.invariant
def test_the_partner_item_holds_every_cell_to_the_grid_and_the_tolerance(
    world: World, tmp_path: Path
) -> None:
    good = partner_record(world)
    cell = world.expected.sellers(1)[0]
    ppo = world.expected.ppo_reference()
    assert ppo is not None
    cases = [
        ([comparison(cell, block=cell.block + 1)], "is not the A7 cell"),
        ([comparison(cell, rel_tol=1e-3)], "was compared at 0.001"),
        ([comparison(cell, within_tolerance=False, first_difference="pnl")], "outside the tol"),
        ([comparison(ppo)], "no fixed-delta seller cell was regenerated"),
        (
            [*good.cells, comparison(ppo, check="numeric", rel_tol=1e-6)],
            "was not compared statistically at 3 se",
        ),
        ([*good.cells, comparison(ppo, k_se=10.0)], "was not compared statistically"),
        ([comparison(cell, check="statistical")], "was compared at"),
    ]
    for cells, why in cases:
        blocked(partner(send(tmp_path, good.model_copy(update={"cells": cells})), world), why)
    both = good.model_copy(update={"cells": [*good.cells, comparison(ppo, max_rel_diff=3e-7)]})
    item = partner(send(tmp_path, both), world)
    passes(item)
    assert "(2 delta_seller, 1 ppo; premiums 0, 4)" in item.evidence and "3e-07" in item.evidence


@pytest.mark.behaviour
def test_a_windows_checkout_with_crlf_hashes_the_expected_file_as_the_gate_does(
    world: World, tmp_path: Path
) -> None:
    # git for Windows with core.autocrlf checks the file out with CRLF; the gate reads the
    # committed blob (LF). Both hash it with git's line endings, so the record passes.
    crlf = world.expected_text.replace("\n", "\r\n").encode()
    assert rp.sha256_bytes(crlf) != rp.sha256_bytes(world.expected_text.encode())
    assert rp.text_sha256(crlf) == rp.text_sha256(world.expected_text.encode())
    record = partner_record(world, expected_sha256=rp.text_sha256(crlf))
    passes(partner(send(tmp_path, record), world))


@pytest.mark.behaviour
def test_the_committed_json_keeps_lf_in_every_checkout() -> None:
    repo = Path(__file__).resolve().parent.parent.parent
    done = subprocess.run(
        ["git", "check-attr", "eol", "text", "--", f"docs/{rp.EXPECTED_FILE}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    assert "eol: lf" in done.stdout and "text: set" in done.stdout


@pytest.mark.invariant
def test_the_partner_item_ties_the_expected_file_to_the_sweep(world: World, tmp_path: Path) -> None:
    docs = send(tmp_path / "docs", partner_record(world))
    blocked(partner(docs, world, tmp_path / "nosweep"), "no sweep at")
    root = copy_root(world, tmp_path)
    cell = grid_cells(TINY)[4]
    (root / cell.run_id / RUN_FILENAME).write_text("{}")
    blocked(partner(docs, world, root), f"{cell.run_id}'s run.json under")
    other = TINY.model_copy(update={"n_seeds": 2})
    assert "is of grid" in m3gate.expected_problem(world.expected, world.root, other)


@pytest.mark.invariant
def test_expected_problem_needs_exactly_the_grids_cells(world: World) -> None:
    fewer = world.expected.model_copy(update={"cells": world.expected.cells[:-1]})
    assert "does not list exactly" in m3gate.expected_problem(fewer, world.root, TINY)
    assert m3gate.expected_problem(world.expected, world.root, TINY) == ""


# -- the report command ------------------------------------------------------------ #


@pytest.mark.behaviour
@pytest.mark.slow
def test_the_report_names_every_root_portably_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import m3_gate_report

    data = tmp_path / "shared"
    monkeypatch.setenv("VEGALAB_DATA", str(data))
    docs = m1_sidecar(data / "docs", 0.1)
    out = data / "out" / "M3_GATE_REPORT.md"
    argv = ["--out", str(out), "--docs", str(docs), "--web", str(data / "web"), "--seeds", "3"]
    assert m3_gate_report.main(argv, on=date(2026, 11, 13)) == 1  # no sweep, no partner
    text = out.read_text()
    assert "Runs root: `$VEGALAB_DATA/runs` (grid `grid-7d9d53d340006f02.json`)" in text
    assert "docs: `$VEGALAB_DATA/docs`; built site: `$VEGALAB_DATA/web`" in text
    assert "Built by `python -m scripts.m3_gate_report` at " in text and str(data) not in text
    report = GateReport.from_markdown(text)
    assert report.milestone == "M3" and len(report.items) == 4 and not report.passed
    names = [i.name for i in report.items]
    assert names == [
        "Sweep complete",
        "Leak detection floor measured",
        "Poisoned-environment demo live",
        "Partner team reproduces the gates",
    ]
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["report_date"] == "2026-11-13" and sidecar["verdict"] == "NOT CLEARED"
    assert "the sweep has not run" in sidecar["items"][0]["failed_first"]
    with pytest.raises(SystemExit):
        m3_gate_report.main(["--replay", "-1"])


@pytest.mark.behaviour
def test_the_report_clears_when_every_item_does(
    world: World,
    tmp_path: Path,
    server: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    measured_floor: float,
) -> None:
    from scripts import m3_gate_report

    server()
    docs = send(m1_sidecar(tmp_path / "docs", measured_floor), partner_record(world))
    web = built_site(tmp_path / "web", "/api/leaks/demo")
    monkeypatch.setattr(
        m3_gate_report,
        "sweep_item",
        lambda r, repo, replays: m3gate.sweep_item(r, world.repo, grid=TINY, tag=TAG),
    )
    monkeypatch.setattr(
        m3_gate_report,
        "partner_item",
        lambda d, r, repo: m3gate.partner_item(d, r, world.repo, grid=TINY, tag=TAG),
    )
    argv = ["--docs", str(docs), "--runs-root", str(world.root), "--web", str(web)]
    argv += ["--seeds", str(SEEDS), "--replay", "1", "--out", "-"]
    assert m3_gate_report.main(argv) == 0
    text = capsys.readouterr().out
    assert "**Verdict: PASS** (4 items, 0 failed, 0 without a stated criterion)" in text
    assert "replaying 1 cell(s)" in text


def _make_dry_run(target: str, *args: str) -> str:
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
    env.pop("PY", None)
    repo = Path(__file__).resolve().parent.parent.parent
    out = subprocess.run(
        ["make", "-n", "-f", str(repo / "Makefile"), target, *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


@pytest.mark.behaviour
@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_the_m3gate_and_reproduce_targets_pass_their_args_through() -> None:
    line = _make_dry_run("m3gate", "ARGS=--runs-root /tmp/r", "PY=py")
    assert line == "py -m scripts.m3_gate_report --runs-root /tmp/r"
    line = _make_dry_run("reproduce", "ARGS=--with-ppo", "BOOTSTRAP_PY=py")
    assert line == "py scripts/reproduce.py --with-ppo"
