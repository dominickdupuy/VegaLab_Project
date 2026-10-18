"""Ilan Danial: the dashboard's loaders read what the harness wrote, and nothing else (week 6).

The dashboard's promise is that every number on it traces back to a run. These
tests hold the loaders to it on fixture run directories written by the real
sweep, plus two that are wrong on purpose: a manifest the gate refuses, and a
results file edited after its manifest was written. Both must be listed with what
is wrong, never dropped and never shown as healthy.

The gate board reads the JSON sidecar when there is one and falls back to the
markdown table, which is what lets it work on the M1 report as committed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from optspread.dashboard_data import (
    DOCS_ENV,
    REPO_ROOT,
    RUNS_ENV,
    GateReportFormatError,
    compare_runs,
    docs_dir,
    filter_runs,
    flatten_config,
    gate_milestones,
    load_gate_board,
    load_gate_boards,
    load_run_detail,
    load_run_row,
    parse_gate_json,
    parse_gate_markdown,
    premium_curve,
    readme_summary,
    risk_stats,
    run_facets,
    run_leak_demo,
    runs_root,
    scan_runs,
)
from optspread.leaks import PASS_CRITERION
from optspread.sweep import SweepConfig

# -- where things are read from ------------------------------------------------ #


def test_roots_come_from_the_environment_or_the_repository(tmp_path: Path) -> None:
    assert runs_root({}) == REPO_ROOT / "runs"
    assert docs_dir({}) == REPO_ROOT / "docs"
    assert runs_root({RUNS_ENV: str(tmp_path)}) == tmp_path
    assert docs_dir({DOCS_ENV: str(tmp_path)}) == tmp_path
    assert runs_root() == runs_root(dict(os.environ))


# -- the run browser ------------------------------------------------------------ #


def test_every_run_directory_is_listed_including_the_broken_ones(
    run_root: Path, tiny_sweep: SweepConfig
) -> None:
    rows = scan_runs(run_root)
    cells = len(tiny_sweep.premiums) * len(tiny_sweep.arms)
    assert len(rows) == cells + 2
    statuses = sorted(r.status for r in rows)
    assert statuses.count("OK") == cells
    assert "INVALID" in statuses and "DIGEST MISMATCH" in statuses
    # newest first
    stamps = [r.created_at for r in rows]
    assert stamps == sorted(stamps, reverse=True)


def test_a_clean_cell_carries_its_provenance_and_headline(
    run_root: Path, tiny_sweep: SweepConfig
) -> None:
    ok = [r for r in scan_runs(run_root) if r.status == "OK"]
    row = next(r for r in ok if r.policy == "delta_seller" and r.premium == 2.0)
    assert row.run_id == f"{row.config_hash}-s{row.seed}"
    assert row.seed == tiny_sweep.base.seed
    assert row.sweep == "tiny"
    assert len(row.git_sha) == 40 and row.commit == row.git_sha[:8]
    assert row.n_episodes == tiny_sweep.base.n_episodes
    assert row.mean_pnl is not None and row.se_pnl is not None and row.t_stat is not None
    assert row.digests_ok is True and row.problems == ()


def test_a_manifest_the_gate_refuses_is_listed_with_the_reason(run_root: Path) -> None:
    row = next(r for r in scan_runs(run_root) if r.run_id == "broken-run")
    assert row.status == "INVALID"
    assert any(p.startswith("manifest:") for p in row.problems)
    assert "run.json is missing" in row.problems
    assert row.seed is None and row.premium is None and row.mean_pnl is None
    assert row.digests_ok is None  # nothing recorded, so nothing to check


def test_a_results_file_edited_after_its_manifest_is_caught(run_root: Path) -> None:
    row = next(r for r in scan_runs(run_root) if r.status == "DIGEST MISMATCH")
    assert row.mean_pnl == 1e9  # the edited number is shown, flagged
    detail = load_run_detail(row.path)
    assert [a.ok for a in detail.artifacts] == [False]


def test_unreadable_files_become_problems_not_exceptions(tmp_path: Path) -> None:
    run = tmp_path / "bad"
    run.mkdir()
    (run / "manifest.json").write_text("{not json")
    (run / "run.json").write_text(json.dumps({"schema_version": 99}))
    row = load_run_row(run)
    assert row.status == "INVALID"
    assert row.problems[0].startswith("manifest.json unreadable")
    assert any(p.startswith("run.json:") for p in row.problems)
    assert row.run_id == "bad"
    detail = load_run_detail(run)
    assert detail.manifest == {} and detail.record is None and detail.pnl.size == 0


def test_a_missing_root_lists_nothing(tmp_path: Path) -> None:
    assert scan_runs(tmp_path / "absent") == ()


def test_filters_narrow_by_every_facet(run_root: Path, tiny_sweep: SweepConfig) -> None:
    rows = scan_runs(run_root)
    facets = run_facets(rows)
    assert facets.policies == ("delta_seller", "flat")
    assert facets.premiums == (0.0, 2.0)
    assert facets.seeds == (tiny_sweep.base.seed,)
    assert facets.sweeps == ("tiny",)
    assert len(facets.config_hashes) == 4 and len(facets.commits) == 1

    assert filter_runs(rows) == rows
    assert {r.policy for r in filter_runs(rows, policies=["flat"])} == {"flat"}
    assert {r.premium for r in filter_runs(rows, premiums=[2.0])} == {2.0}
    assert len(filter_runs(rows, sweeps=["tiny"])) == 4
    assert filter_runs(rows, seeds=[-1]) == ()
    assert len(filter_runs(rows, commits=list(facets.commits))) == len(rows) - 1  # broken: none
    one = facets.config_hashes[0]
    assert all(r.config_hash == one for r in filter_runs(rows, config_hashes=[one]))
    assert filter_runs(rows, text=one.upper()) == filter_runs(rows, config_hashes=[one])


def test_a_run_opens_with_its_manifest_and_episodes(
    run_root: Path, tiny_sweep: SweepConfig
) -> None:
    row = next(r for r in scan_runs(run_root) if r.status == "OK" and r.policy == "delta_seller")
    detail = load_run_detail(row.path)
    assert detail.record is not None
    assert detail.manifest["run_id"] == row.run_id
    assert set(detail.manifest["versions"]) >= {"python", "numpy", "optspread"}
    assert detail.pnl.shape == (tiny_sweep.base.n_episodes,)
    assert detail.pnl.mean() == pytest.approx(row.mean_pnl)
    assert detail.cost.shape == detail.trades.shape == detail.pnl.shape
    assert all(a.ok for a in detail.artifacts) and detail.artifacts[0].path == "run.json"


def test_risk_figures_on_a_hand_worked_sample() -> None:
    pnl = np.asarray([100.0, -50.0, 30.0, -200.0, 120.0])
    stats = risk_stats(pnl, np.asarray([2.0, 2.0, 4.0, 0.0, 2.0]))
    assert stats.n == 5 and stats.mean == pytest.approx(0.0)
    assert stats.sortino == pytest.approx(0.0)
    assert stats.cvar5 == -200.0  # 5% of 5 rounds up to one episode, the worst
    # cumulative 0, 100, 50, 80, -120, 0: the worst fall is from 100 to -120
    assert stats.max_drawdown == pytest.approx(220.0)
    assert stats.hit_rate == pytest.approx(0.6) and stats.turnover == pytest.approx(2.0)
    up = risk_stats(np.asarray([1.0, 2.0]))
    assert up.sortino is None and up.turnover is None and up.max_drawdown == 0.0
    one = risk_stats(np.asarray([-3.0]))
    assert one.se is None and one.sortino == pytest.approx(-1.0)
    assert risk_stats(np.asarray([])).mean is None


def test_two_runs_compare_on_their_shared_seeds(run_root: Path) -> None:
    ok = [r for r in scan_runs(run_root) if r.status == "OK" and r.premium == 2.0]
    flat = load_run_detail(next(r for r in ok if r.policy == "flat").path)
    seller = load_run_detail(next(r for r in ok if r.policy == "delta_seller").path)
    cmp = compare_runs(flat, seller)
    assert [k for k, _, _ in cmp.config_diff] == ["policy"]
    assert cmp.shared_seeds == tuple(int(e.seed) for e in seller.record.episodes)  # type: ignore[union-attr]
    assert cmp.diff_mean == pytest.approx(float(seller.pnl.mean()))  # flat is zero
    assert cmp.diff_t is not None
    broken = load_run_detail(next(r for r in scan_runs(run_root) if r.run_id == "broken-run").path)
    none = compare_runs(flat, broken)
    assert none.shared_seeds == () and none.diff_mean is None and none.diff_t is None
    assert flatten_config({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    assert flatten_config(None) == {}


def test_the_premium_curve_uses_clean_runs_only(run_root: Path) -> None:
    curve = premium_curve(scan_runs(run_root))
    assert set(curve) == {"delta_seller", "flat"}
    assert [p.premium for p in curve["delta_seller"]] == [0.0, 2.0]
    assert all(p.mean_pnl == 0.0 for p in curve["flat"])  # P1 of the sweep: flat is zero
    assert all(p.mean_pnl != 1e9 for pts in curve.values() for p in pts)  # tampered excluded


# -- the gate board -------------------------------------------------------------- #


def test_the_committed_m1_report_parses_from_markdown() -> None:
    board = load_gate_board(REPO_ROOT / "docs", "M1")
    assert board is not None and not board.from_json
    assert board.milestone == "M1" and board.report_date == "2026-09-28"
    assert board.verdict == "PASS" and len(board.items) == 8
    assert board.counts["PASS"] == 8 and board.not_passing == ()
    wave0 = board.items[0]
    # a pipe inside a cell ("|t| <= 3") is not a column separator
    assert "|t| <= 3" in wave0.criterion and wave0.evidence.startswith("100 episodes")
    assert wave0.failed_first == ""
    manifests = next(i for i in board.items if i.name == "Run artifacts and manifests")
    assert manifests.cited_runs == ("adddb0b7e5f604ad-s1",)


def test_the_json_sidecar_wins_over_the_markdown(gate_docs: Path) -> None:
    assert gate_milestones(gate_docs) == ("M1", "M2")
    m2 = load_gate_board(gate_docs, "M2")
    assert m2 is not None and m2.from_json and m2.verdict == "NOT CLEARED"
    assert m2.counts["NOT CLEARED"] == 1 and m2.counts["PASS"] == 1
    assert [i.name for i in m2.not_passing] == ["Replay"]
    assert m2.not_passing[0].failed_first == "no replay has been run"
    assert m2.items[0].cited_runs == ("0123456789abcdef-s3",)
    assert [b.milestone for b in load_gate_boards(gate_docs)] == ["M1", "M2"]
    assert load_gate_board(gate_docs, "M9") is None


def test_milestones_sort_numerically(tmp_path: Path) -> None:
    for name in ("M10", "M2", "M1"):
        (tmp_path / f"{name}_GATE_REPORT.md").write_text("")
    (tmp_path / "M3_GATE_REPORT.txt").write_text("")
    assert gate_milestones(tmp_path) == ("M1", "M2", "M10")
    assert gate_milestones(tmp_path / "absent") == ()


def test_a_failing_markdown_report_records_what_failed_first(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "# M2 gate report",
            "",
            "Report date: 2026-10-26",
            "",
            "**Verdict: NOT CLEARED** (2 items, 1 failed, 0 without a stated criterion)",
            "",
            "| Item | Owner | Verdict | Criterion | Evidence | What failed first |",
            "| --- | --- | --- | --- | --- | --- |",
            "| Pilot | Ilan Danial | FAIL | a | b | c | the pilot did not finish |",
            "| Docs | Dominick Dupuy | NO CRITERION |  | written | -- |",
            "",
            "## What failed first",
        ]
    )
    board = parse_gate_markdown(text, tmp_path / "M2_GATE_REPORT.md")
    assert board.verdict == "NOT CLEARED" and board.report_date == "2026-10-26"
    pilot, docs = board.items
    assert pilot.criterion == "a | b" and pilot.evidence == "c"
    assert pilot.failed_first == "the pilot did not finish"
    assert docs.verdict == "NO CRITERION" and docs.criterion == ""
    assert board.counts == {"PASS": 0, "FAIL": 1, "NO CRITERION": 1, "NOT CLEARED": 0}


@pytest.mark.parametrize(
    "text",
    [
        "# M1 gate report\n\nno table here\n",
        "**Verdict: PASS**\n\n| Item | Owner | Verdict |\n| --- |\n| a | b | c |\n",
    ],
)
def test_an_unparseable_markdown_report_is_an_error(text: str, tmp_path: Path) -> None:
    with pytest.raises(GateReportFormatError):
        parse_gate_markdown(text, tmp_path / "M1_GATE_REPORT.md")


@pytest.mark.parametrize("text", ["{", json.dumps({"milestone": "M2"}), json.dumps([1])])
def test_an_unparseable_json_report_is_an_error(text: str, tmp_path: Path) -> None:
    with pytest.raises(GateReportFormatError):
        parse_gate_json(text, tmp_path / "M2_GATE_REPORT.json")


# -- the poisoned-environment demo ----------------------------------------------- #


def test_the_demo_passes_the_clean_world_and_catches_the_poisoned_one() -> None:
    demo = run_leak_demo(eps=0.5, n_seeds=5)
    assert not demo.clean.flagged and demo.clean_past_unchanged
    assert demo.poisoned.flagged and not demo.poisoned_past_unchanged
    assert demo.poisoned.worst_feature == "ret_1d"
    assert demo.caught and demo.seeds == tuple(range(5)) and demo.seconds > 0.0
    assert demo.criterion == PASS_CRITERION


def test_the_demo_refuses_a_poison_of_zero() -> None:
    with pytest.raises(ValueError, match="eps > 0"):
        run_leak_demo(eps=0.0)


# -- the landing page --------------------------------------------------------------- #


def test_the_summary_is_the_readme_opening_without_self_references(tmp_path: Path) -> None:
    paras = readme_summary(REPO_ROOT / "README.md")
    assert paras[0].startswith("A leak-tested SPX options simulator")
    assert not any("README" in p or "project document" in p for p in paras)
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\nOne line.\nSame paragraph.\n\nSee the README. Kept.")
    assert readme_summary(readme) == ("One line. Same paragraph.", "Kept.")
    assert readme_summary(tmp_path / "absent.md") == ()
