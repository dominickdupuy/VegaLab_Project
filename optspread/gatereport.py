"""The gate report: the artifact a milestone actually closes on.

Lane: Dominick Dupuy (statistical spine).

Working code does not clear a gate. A gate closes on a report that states, for
each item, the criterion it had to meet, the evidence that it did, and -- when it
did not -- what failed first. The third field is the one that makes the report
worth writing: a milestone review that only lists passes tells the advisor
nothing about where the build is fragile.

The report is rendered from live results rather than typed, so it cannot drift
away from the tests. ``scripts/m1_gate_report.py`` runs the gates and writes
``docs/M1_GATE_REPORT.md``; the Presentation 1 slides are built from that file.

Three states, and only three:

    PASS        the criterion is stated and the evidence meets it
    FAIL        the criterion is stated and something failed; ``failed_first`` says what
    NO CRITERION  there is evidence but no stated rule, which is not a pass

The last state exists because it is the common failure. A check whose criterion
lives only in its author's head reads as a pass every time it runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class GateItem:
    """One deliverable and its verdict.

    ``criterion`` is the rule someone else could check. ``evidence`` is what was
    observed. ``failed_first`` is required on a failure and must be empty on a
    pass, so a report cannot claim a pass while recording a failure.
    """

    name: str
    owner: str
    criterion: str
    evidence: str
    passed: bool
    failed_first: str = ""

    def __post_init__(self) -> None:
        if self.passed and self.failed_first:
            raise ValueError(f"{self.name}: a passing item cannot record a first failure")
        if not self.passed and not self.failed_first:
            raise ValueError(f"{self.name}: a failing item must record what failed first")

    @property
    def has_criterion(self) -> bool:
        return bool(self.criterion.strip())

    @property
    def verdict(self) -> str:
        if not self.has_criterion:
            return "NO CRITERION"
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True, slots=True)
class GateReport:
    milestone: str
    report_date: date
    items: tuple[GateItem, ...]

    @property
    def passed(self) -> bool:
        """Every item states a criterion and meets it. Silence is not a pass."""
        return bool(self.items) and all(i.has_criterion and i.passed for i in self.items)

    @property
    def without_criterion(self) -> tuple[GateItem, ...]:
        return tuple(i for i in self.items if not i.has_criterion)

    @property
    def failures(self) -> tuple[GateItem, ...]:
        return tuple(i for i in self.items if i.has_criterion and not i.passed)

    def render(self) -> str:
        lines = [
            f"# {self.milestone} gate report",
            "",
            f"Report date: {self.report_date.isoformat()}",
            "",
            f"**Verdict: {'PASS' if self.passed else 'NOT CLEARED'}** "
            f"({len(self.items)} items, {len(self.failures)} failed, "
            f"{len(self.without_criterion)} without a stated criterion)",
            "",
            "| Item | Owner | Verdict | Criterion | Evidence | What failed first |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.items:
            lines.append(
                f"| {item.name} | {item.owner} | {item.verdict} | {item.criterion} "
                f"| {item.evidence} | {item.failed_first or '--'} |"
            )
        if self.without_criterion:
            lines += [
                "",
                "## Items without a stated criterion",
                "",
                "These have evidence but no rule another person could check, so they do "
                "not clear the gate whatever the evidence says.",
                "",
            ]
            lines += [f"- {i.name} ({i.owner})" for i in self.without_criterion]
        if self.failures:
            lines += ["", "## What failed first", ""]
            lines += [f"- **{i.name}**: {i.failed_first}" for i in self.failures]
        return "\n".join(lines) + "\n"


def build_report(milestone: str, items: Sequence[GateItem], on: date | None = None) -> GateReport:
    return GateReport(milestone, on or date.today(), tuple(items))
