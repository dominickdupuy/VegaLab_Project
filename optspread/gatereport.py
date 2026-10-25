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

Every report is written twice from one build: the Markdown the slides are made
from, and a JSON sidecar beside it (``M1_GATE_REPORT.json`` next to
``M1_GATE_REPORT.md``) that the dashboard's gate board reads, so the board never
parses prose. The sidecar's shape is fixed by ``to_dict``:

    {"milestone": "M1", "report_date": "YYYY-MM-DD", "verdict": "PASS" | "NOT CLEARED",
     "items": [{"name", "owner", "verdict", "criterion", "evidence", "failed_first"}]}

with each item's verdict exactly as ``GateItem.verdict`` gives it and an empty
``failed_first`` on a pass. ``from_markdown`` reads a committed report back, which
is how a sidecar is produced for a report already on record without re-running
(and so re-dating) it: the read is accepted only if it re-renders the file byte
for byte.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_HEADER = "| Item | Owner | Verdict | Criterion | Evidence | What failed first |"
_TITLE = " gate report"
_DATE = "Report date: "


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
        if self.passed and not self.evidence.strip():
            raise ValueError(f"{self.name}: a passing item must show its evidence")
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
    """The items, and ``context``: lines under the date saying what the evidence was read
    from (the roots a report scanned, say). Context is in the Markdown, not the sidecar."""

    milestone: str
    report_date: date
    items: tuple[GateItem, ...]
    context: tuple[str, ...] = ()

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

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "NOT CLEARED"

    def to_dict(self) -> dict[str, Any]:
        """The JSON sidecar's schema. The dashboard reads it; change the two together."""
        return {
            "milestone": self.milestone,
            "report_date": self.report_date.isoformat(),
            "verdict": self.verdict,
            "items": [
                {
                    "name": i.name,
                    "owner": i.owner,
                    "verdict": i.verdict,
                    "criterion": i.criterion,
                    "evidence": i.evidence,
                    "failed_first": i.failed_first,
                }
                for i in self.items
            ],
        }

    def render_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def write(self, path: Path) -> tuple[Path, Path]:
        """The Markdown at ``path`` and the JSON sidecar beside it, from this one build."""
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = path.with_suffix(".json")
        path.write_text(self.render(), newline="\n")
        sidecar.write_text(self.render_json(), newline="\n")
        return path, sidecar

    @classmethod
    def from_markdown(cls, text: str) -> GateReport:
        """Read back a report ``render`` wrote, refusing anything it would not reproduce."""
        lines = text.splitlines()
        if not lines or not lines[0].startswith("# ") or not lines[0].endswith(_TITLE):
            raise ValueError(
                "not a gate report: the first line must be '# <milestone> gate report'"
            )
        dated = [ln for ln in lines if ln.startswith(_DATE)]
        if len(dated) != 1:
            raise ValueError("a gate report carries exactly one 'Report date:' line")
        if _HEADER not in lines:
            raise ValueError("no item table in the report")
        at = lines.index(dated[0]) + 1
        verdict_at = next((i for i, ln in enumerate(lines) if ln.startswith("**Verdict: ")), at)
        context = tuple(ln for ln in lines[at:verdict_at] if ln)
        items: list[GateItem] = []
        for row in lines[lines.index(_HEADER) + 2 :]:
            if not row.startswith("| "):
                break
            cells = row[2:-2].split(" | ") if row.endswith(" |") else []
            if len(cells) != 6:
                raise ValueError(f"cannot split this row into six cells: {row!r}")
            name, owner, _, criterion, evidence, failed = cells
            failed_first = "" if failed == "--" else failed
            items.append(GateItem(name, owner, criterion, evidence, not failed_first, failed_first))
        report = cls(
            lines[0][2 : -len(_TITLE)],
            date.fromisoformat(dated[0].removeprefix(_DATE)),
            tuple(items),
            context,
        )
        if report.render() != text:
            # Catches a hand-edited verdict, a cell holding ' | ', or a format change.
            raise ValueError("the report does not re-render to the same text; refusing to guess")
        return report

    def render(self) -> str:
        lines = [
            f"# {self.milestone} gate report",
            "",
            f"Report date: {self.report_date.isoformat()}",
            "",
            *(line for c in self.context for line in (c, "")),
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


def build_report(
    milestone: str,
    items: Sequence[GateItem],
    on: date | None = None,
    context: Sequence[str] = (),
) -> GateReport:
    return GateReport(milestone, on or date.today(), tuple(items), tuple(context))
