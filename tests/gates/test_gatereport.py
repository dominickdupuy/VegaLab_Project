"""Dominick Dupuy: the gate report format (week 2).

A gate report exists to stop a milestone claiming more than the evidence
supports, so the tests here are all about what it refuses to render as a pass.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from optspread.gatereport import GateItem, GateReport, GateReportSidecar, build_report

PASSING = GateItem(
    name="Wave 0 no-edge gate",
    owner="Hemdutt Rao",
    criterion="flat P&L exactly 0; |t| <= 3 for no-cost always-on",
    evidence="100 episodes/arm, all four orderings hold",
    passed=True,
)
FAILING = GateItem(
    name="Undefined-risk margin",
    owner="Hemdutt Rao",
    criterion="Reg-T requirement matches a hand-worked strangle to the cent",
    evidence="short strangle at 16 delta",
    passed=False,
    failed_first="the lesser-side premium was added twice on a same-right pair",
)
NO_CRITERION = GateItem(
    name="Leak detection suite",
    owner="Dominick Dupuy",
    criterion="",
    evidence="clean env passes, poisoned env fails, 20 seeds",
    passed=True,
)


@pytest.mark.behaviour
def test_a_report_of_passes_clears() -> None:
    report = build_report("M1", [PASSING], date(2026, 10, 1))
    assert report.passed
    assert "PASS" in report.render()


@pytest.mark.behaviour
def test_a_failure_blocks_the_gate_and_is_named() -> None:
    report = build_report("M1", [PASSING, FAILING], date(2026, 10, 1))
    assert not report.passed
    assert report.failures == (FAILING,)
    rendered = report.render()
    assert "NOT CLEARED" in rendered
    assert "What failed first" in rendered
    assert "added twice" in rendered


@pytest.mark.behaviour
def test_evidence_without_a_stated_criterion_is_not_a_pass() -> None:
    report = build_report("M1", [PASSING, NO_CRITERION], date(2026, 10, 1))
    assert not report.passed, "a check with no stated rule must not clear a gate"
    assert report.without_criterion == (NO_CRITERION,)
    assert NO_CRITERION.verdict == "NO CRITERION"
    assert "Items without a stated criterion" in report.render()


@pytest.mark.behaviour
def test_an_empty_report_does_not_clear() -> None:
    assert not build_report("M1", [], date(2026, 10, 1)).passed


@pytest.mark.behaviour
def test_a_pass_cannot_also_record_a_failure() -> None:
    with pytest.raises(ValueError, match="cannot record a first failure"):
        GateItem("x", "Dominick Dupuy", "c", "e", passed=True, failed_first="something")


@pytest.mark.behaviour
def test_a_failure_must_say_what_failed() -> None:
    with pytest.raises(ValueError, match="must record what failed first"):
        GateItem("x", "Dominick Dupuy", "c", "e", passed=False)


@pytest.mark.behaviour
def test_every_item_reaches_the_table() -> None:
    report = build_report("M1", [PASSING, FAILING, NO_CRITERION], date(2026, 10, 1))
    rendered = report.render()
    for item in report.items:
        assert item.name in rendered
        assert item.owner in rendered


@pytest.mark.behaviour
def test_a_pass_must_show_its_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        GateItem("x", "Dominick Dupuy", "c", "  ", passed=True)


# -- the JSON sidecar and reading a committed report back (week 6) ----------- #

# The checkout is found by its pyproject.toml, not a fixed number of parents: from a
# wrong folder the glob below would find no report and the test would only be skipped.
CHECKOUT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
DOCS = CHECKOUT / "docs"
FAILED_NO_CRITERION = GateItem("Pilot", "Dominick Dupuy", "", "a matrix", False, "not run")


@pytest.mark.behaviour
def test_the_sidecar_has_exactly_the_dashboard_schema() -> None:
    report = build_report("M2", [PASSING, FAILING, NO_CRITERION], date(2026, 10, 29))
    doc = json.loads(report.render_json())
    assert set(doc) == {"milestone", "report_date", "verdict", "items"}
    assert (doc["milestone"], doc["report_date"], doc["verdict"]) == (
        "M2",
        "2026-10-29",
        "NOT CLEARED",
    )
    keys = {"name", "owner", "verdict", "criterion", "evidence", "failed_first"}
    assert all(set(item) == keys for item in doc["items"])
    assert [i["verdict"] for i in doc["items"]] == ["PASS", "FAIL", "NO CRITERION"]
    assert doc["items"][0]["failed_first"] == ""
    assert doc["items"][1]["failed_first"] == FAILING.failed_first
    cleared = build_report("M1", [PASSING], date(2026, 10, 1))
    assert json.loads(cleared.render_json())["verdict"] == "PASS"


@pytest.mark.behaviour
def test_write_puts_both_halves_of_one_build_side_by_side(tmp_path: Path) -> None:
    report = build_report("M2", [PASSING, FAILING], date(2026, 10, 29))
    md, sidecar = report.write(tmp_path / "nested" / "M2_GATE_REPORT.md")
    assert sidecar == tmp_path / "nested" / "M2_GATE_REPORT.json"
    assert md.read_text() == report.render()
    assert json.loads(sidecar.read_text()) == report.to_dict()


@pytest.mark.behaviour
def test_a_written_sidecar_reads_back_through_the_model_to_the_report_it_came_from(
    tmp_path: Path,
) -> None:
    report = build_report("M2", [PASSING, FAILING, NO_CRITERION], date(2026, 10, 29))
    _, path = report.write(tmp_path / "M2_GATE_REPORT.md")
    text = path.read_text(encoding="utf-8")
    parsed = GateReportSidecar.parse(text)
    assert parsed == report.sidecar() and parsed.render() == text
    # Field by field against the report itself, not only against the model that wrote it.
    assert (parsed.milestone, parsed.report_date, parsed.verdict) == (
        "M2",
        "2026-10-29",
        "NOT CLEARED",
    )
    assert [
        (i.name, i.owner, i.verdict, i.criterion, i.evidence, i.failed_first) for i in parsed.items
    ] == [
        (i.name, i.owner, i.verdict, i.criterion, i.evidence, i.failed_first) for i in report.items
    ]


@pytest.mark.behaviour
@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",  # JSON, but not an object
        '{"report_date": "2026-10-29", "verdict": "PASS", "items": []}',  # no milestone
        '{"milestone": "M2", "verdict": "PASS", "items": [{"owner": "x"}]}',  # an item unnamed
        '{"milestone": "M2", "verdict": "PASS", "items": 3}',
    ],
)
def test_text_that_is_not_a_sidecar_is_refused_as_a_value_error(text: str) -> None:
    # The board reads a sidecar and treats a ValueError as "unreadable", so a refusal
    # must stay one (pydantic's ValidationError and json's JSONDecodeError both are).
    with pytest.raises(ValueError):
        GateReportSidecar.parse(text)


@pytest.mark.behaviour
@pytest.mark.parametrize(
    "items",
    [[PASSING], [PASSING, FAILING], [PASSING, FAILING, NO_CRITERION, FAILED_NO_CRITERION]],
)
def test_a_rendered_report_reads_back_to_itself(items: list[GateItem]) -> None:
    report = build_report("M9", items, date(2026, 12, 2))
    assert GateReport.from_markdown(report.render()) == report


@pytest.mark.behaviour
def test_context_lines_sit_under_the_date_and_read_back() -> None:
    context = ("Runs root: /data/runs", "Pilot root: /data/runs/pilot")
    report = build_report("M2", [PASSING], date(2026, 10, 29), context)
    lines = report.render().splitlines()
    assert lines[2:7] == ["Report date: 2026-10-29", "", context[0], "", context[1]]
    assert lines[8].startswith("**Verdict: PASS**")
    assert GateReport.from_markdown(report.render()) == report
    assert report.to_dict() == build_report("M2", [PASSING], date(2026, 10, 29)).to_dict()


@pytest.mark.invariant
@pytest.mark.parametrize("md", sorted(DOCS.glob("*_GATE_REPORT.md")), ids=lambda p: p.name)
def test_every_committed_report_has_a_sidecar_from_the_same_build(md: Path) -> None:
    recorded = GateReport.from_markdown(md.read_text())
    assert recorded.render() == md.read_text()
    sidecar = md.with_suffix(".json")
    assert sidecar.exists(), f"{md.name} has no JSON sidecar"
    assert json.loads(sidecar.read_text()) == recorded.to_dict()


@pytest.mark.behaviour
def test_a_hand_edited_report_is_refused() -> None:
    text = build_report("M1", [PASSING, FAILING], date(2026, 10, 1)).render()
    with pytest.raises(ValueError, match="re-render"):
        GateReport.from_markdown(text.replace("| FAIL |", "| PASS |"))
    with pytest.raises(ValueError, match="six cells"):
        GateReport.from_markdown(text.replace("Hemdutt Rao |", "Hemdutt | Rao |", 1))
    with pytest.raises(ValueError, match="not a gate report"):
        GateReport.from_markdown("# Notes\n")
    with pytest.raises(ValueError, match="Report date"):
        GateReport.from_markdown(text.replace("Report date: ", "Dated "))
    with pytest.raises(ValueError, match="item table"):
        GateReport.from_markdown("# M1 gate report\n\nReport date: 2026-10-01\n")
