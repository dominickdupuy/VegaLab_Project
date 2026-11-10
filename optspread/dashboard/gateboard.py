"""The gate board: each gate report under ``docs/``, its verdict re-derived from its items.

Lane: Ilan Danial (machinery).

A report is read from the JSON sidecar ``optspread.gatereport`` writes when there
is one, else from its rendered markdown table. A report this reader cannot parse
raises ``GateReportFormatError`` rather than being guessed around;
``load_gate_boards`` leaves it out and ``unreadable_gate_reports`` names it.
``scan_gate_reports`` reads every report once and gives both, each unreadable one
with the reason it did not parse.

Moved from ``optspread.dashboard_data``, which re-exports it; ``optspread.m2gate``
looks ``load_gate_boards`` up there by name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from optspread.gatereport import GateReportSidecar

# -- the gate board ----------------------------------------------------------- #

GATE_STEM = "_GATE_REPORT"
VERDICTS = ("PASS", "FAIL", "NO CRITERION", "NOT CLEARED")
RUN_ID_PATTERN = re.compile(r"\b[0-9a-f]{16}-s\d+\b")


class GateReportFormatError(ValueError):
    """A gate report this reader cannot parse. Raised, never guessed around."""


@dataclass(frozen=True, slots=True)
class GateRow:
    name: str
    owner: str
    verdict: str
    criterion: str
    evidence: str
    failed_first: str = ""

    @property
    def cited_runs(self) -> tuple[str, ...]:
        """Run ids quoted in the evidence, so the board can link to the runs behind it."""
        return tuple(RUN_ID_PATTERN.findall(self.evidence))


@dataclass(frozen=True, slots=True)
class GateBoard:
    """One gate report as the board shows it, with its verdict re-derived from its items.

    ``stated_verdict`` is whatever the file says. ``verdict`` is what the items support:
    PASS only when there are items and every one is a PASS with a criterion and
    evidence written down, NOT CLEARED otherwise, which is ``GateReport.passed`` read
    back. The board shows the derived one, so a hand-edited verdict line or an empty
    item list cannot put a PASS on the page; ``verdict_mismatch`` says the two differ.
    """

    milestone: str
    report_date: str
    stated_verdict: str
    items: tuple[GateRow, ...]
    source: Path
    from_json: bool

    @property
    def verdict(self) -> str:
        cleared = bool(self.items) and all(
            i.verdict == "PASS" and i.criterion.strip() and i.evidence.strip() for i in self.items
        )
        return "PASS" if cleared else "NOT CLEARED"

    @property
    def verdict_mismatch(self) -> bool:
        return self.stated_verdict != self.verdict

    @property
    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for item in self.items:
            out[item.verdict] = out.get(item.verdict, 0) + 1
        return out

    @property
    def not_passing(self) -> tuple[GateRow, ...]:
        return tuple(i for i in self.items if i.verdict != "PASS")


def gate_milestones(docs: Path) -> tuple[str, ...]:
    """Milestones with a report on disk, JSON or markdown, in milestone order."""
    if not docs.is_dir():
        return ()
    names = {
        p.name.split(GATE_STEM)[0]
        for p in docs.glob(f"M*{GATE_STEM}.*")
        if p.suffix in (".json", ".md")
    }

    def order(name: str) -> tuple[int, str]:
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 0, name)

    return tuple(sorted(names, key=order))


def _reason(exc: Exception) -> str:
    """One line saying why: pydantic's first error with where it is, else the message."""
    if isinstance(exc, ValidationError) and exc.errors():
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return f"{where}: {first['msg']}" if where else str(first["msg"])
    return str(exc)


def _board(sidecar: GateReportSidecar, source: Path, *, from_json: bool) -> GateBoard:
    return GateBoard(
        milestone=sidecar.milestone,
        report_date=sidecar.report_date,
        stated_verdict=sidecar.verdict,
        items=tuple(
            GateRow(i.name, i.owner, i.verdict, i.criterion, i.evidence, i.failed_first)
            for i in sidecar.items
        ),
        source=source,
        from_json=from_json,
    )


def parse_gate_json(text: str, source: Path) -> GateBoard:
    """The JSON sidecar ``optspread.gatereport`` writes beside the markdown, read through
    the same model that writes it (``GateReportSidecar``)."""
    try:
        sidecar = GateReportSidecar.parse(text)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise GateReportFormatError(f"{source.name}: {_reason(exc)}") from exc
    return _board(sidecar, source, from_json=True)


_DATE_LINE = re.compile(r"^Report date:\s*(\S+)")
_VERDICT_LINE = re.compile(r"^\*\*Verdict:\s*([A-Z ]+?)\*\*")
_TITLE_LINE = re.compile(r"^#\s+(\S+)\s+gate report", re.IGNORECASE)


def _split_row(line: str) -> list[str]:
    # Cells are separated by " | " as ``GateReport.render`` writes them. A bare pipe
    # inside a cell, such as "|t| <= 3", is not a separator.
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [c.strip() for c in inner.split(" | ")]


def parse_gate_markdown(text: str, source: Path) -> GateBoard:
    """The rendered markdown table, for reports written before the JSON sidecar."""
    milestone, report_date, verdict = source.name.split(GATE_STEM)[0], "", ""
    items: list[GateRow] = []
    header_seen = False
    for line in text.splitlines():
        if m := _TITLE_LINE.match(line):
            milestone = m.group(1)
        elif m := _DATE_LINE.match(line):
            report_date = m.group(1)
        elif m := _VERDICT_LINE.match(line):
            verdict = m.group(1).strip()
        elif line.startswith("| Item |"):
            header_seen = True
        elif header_seen and line.startswith("|") and not line.startswith("| ---"):
            cells = _split_row(line)
            if len(cells) < 6:
                raise GateReportFormatError(f"{source.name}: a table row has {len(cells)} cells")
            # name, owner and verdict lead; evidence and failed-first close the row; a
            # stray separator can only have come from the criterion, so it is rejoined.
            name, owner, item_verdict = cells[0], cells[1], cells[2]
            criterion = " | ".join(cells[3:-2])
            failed = "" if cells[-1] in ("--", "") else cells[-1]
            items.append(GateRow(name, owner, item_verdict, criterion, cells[-2], failed))
        elif header_seen and not line.startswith("|"):
            header_seen = False
    if not items or not verdict:
        raise GateReportFormatError(f"{source.name}: no verdict line or no item table")
    return GateBoard(milestone, report_date, verdict, tuple(items), source, from_json=False)


def load_gate_board(docs: Path, milestone: str) -> GateBoard | None:
    """The JSON sidecar when present, else the markdown table; None when neither exists."""
    sidecar = docs / f"{milestone}{GATE_STEM}.json"
    if sidecar.is_file():
        return parse_gate_json(sidecar.read_text(), sidecar)
    markdown = docs / f"{milestone}{GATE_STEM}.md"
    if markdown.is_file():
        return parse_gate_markdown(markdown.read_text(), markdown)
    return None


@dataclass(frozen=True, slots=True)
class GateScan:
    """Every gate report under a docs folder: the boards that read, and the ones that
    do not, each as ``(milestone, reason)``, both in milestone order."""

    boards: tuple[GateBoard, ...]
    unreadable: tuple[tuple[str, str], ...]


def scan_gate_reports(docs: Path) -> GateScan:
    """Read every milestone's report once; one that does not parse is listed with why."""
    boards: list[GateBoard] = []
    bad: list[tuple[str, str]] = []
    for m in gate_milestones(docs):
        try:
            board = load_gate_board(docs, m)
        except ValueError as exc:  # GateReportFormatError is a ValueError
            bad.append((m, str(exc)))
            continue
        if board is not None:
            boards.append(board)
    return GateScan(tuple(boards), tuple(bad))


def load_gate_boards(docs: Path) -> tuple[GateBoard, ...]:
    """Every board that reads; an unreadable one is named by ``unreadable_gate_reports``.

    One malformed sidecar must not take down every page that shows a board, so it
    is left out here and reported there, rather than raised through the API.
    """
    return scan_gate_reports(docs).boards


def unreadable_gate_reports(docs: Path) -> tuple[str, ...]:
    """The milestones whose report exists but does not parse."""
    return tuple(m for m, _ in scan_gate_reports(docs).unreadable)
