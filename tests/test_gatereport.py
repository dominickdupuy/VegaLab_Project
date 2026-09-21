"""Dominick Dupuy: the gate report format (week 2).

A gate report exists to stop a milestone claiming more than the evidence
supports, so the tests here are all about what it refuses to render as a pass.
"""

from __future__ import annotations

from datetime import date

import pytest

from optspread.gatereport import GateItem, build_report

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


def test_a_report_of_passes_clears() -> None:
    report = build_report("M1", [PASSING], date(2026, 10, 1))
    assert report.passed
    assert "PASS" in report.render()


def test_a_failure_blocks_the_gate_and_is_named() -> None:
    report = build_report("M1", [PASSING, FAILING], date(2026, 10, 1))
    assert not report.passed
    assert report.failures == (FAILING,)
    rendered = report.render()
    assert "NOT CLEARED" in rendered
    assert "What failed first" in rendered
    assert "added twice" in rendered


def test_evidence_without_a_stated_criterion_is_not_a_pass() -> None:
    report = build_report("M1", [PASSING, NO_CRITERION], date(2026, 10, 1))
    assert not report.passed, "a check with no stated rule must not clear a gate"
    assert report.without_criterion == (NO_CRITERION,)
    assert NO_CRITERION.verdict == "NO CRITERION"
    assert "Items without a stated criterion" in report.render()


def test_an_empty_report_does_not_clear() -> None:
    assert not build_report("M1", [], date(2026, 10, 1)).passed


def test_a_pass_cannot_also_record_a_failure() -> None:
    with pytest.raises(ValueError, match="cannot record a first failure"):
        GateItem("x", "Dominick Dupuy", "c", "e", passed=True, failed_first="something")


def test_a_failure_must_say_what_failed() -> None:
    with pytest.raises(ValueError, match="must record what failed first"):
        GateItem("x", "Dominick Dupuy", "c", "e", passed=False)


def test_every_item_reaches_the_table() -> None:
    report = build_report("M1", [PASSING, FAILING, NO_CRITERION], date(2026, 10, 1))
    rendered = report.render()
    for item in report.items:
        assert item.name in rendered
        assert item.owner in rendered
