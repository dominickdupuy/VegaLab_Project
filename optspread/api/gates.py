"""API payloads for the gate items: ``/api/checks`` (every item, one row each) and the docs
the items cite (``/api/docs/{name}``).

Lane: Ilan Danial (machinery).

Also the builders of the old ``/api/gates`` board (``gates``, ``board_out`` and their
models), which no route calls since ``/api/checks`` replaced it; one test still does.
Moved from ``optspread.webviews`` (the checks) and ``optspread.webapi`` (the board).
"""

from __future__ import annotations

import re
from pathlib import Path

from optspread.api._base import Out, _from_attrs
from optspread.dashboard.gateboard import GateBoard, load_gate_boards

# -- gates ------------------------------------------------------------------------ #


class GateItemOut(Out):
    name: str
    owner: str
    verdict: str
    criterion: str
    evidence: str
    failed_first: str
    cited_runs: list[str]


class GateBoardOut(Out):
    milestone: str
    report_date: str
    verdict: str  # derived from the items, never taken from the file
    stated_verdict: str  # what the file says
    verdict_mismatch: bool
    counts: dict[str, int]
    items: list[GateItemOut]
    source: str
    from_json: bool


class GatesOut(Out):
    docs: str
    boards: list[GateBoardOut]


def board_out(board: GateBoard) -> GateBoardOut:
    # the items are read from the board's rows by name too
    return _from_attrs(GateBoardOut, board, source=str(board.source))


def gates(docs: Path) -> GatesOut:
    return GatesOut(docs=str(docs), boards=[board_out(b) for b in load_gate_boards(docs)])


# -- checks: every gate item, one row each ----------------------------------------- #


class CheckOut(Out):
    name: str  # as the report names the item
    label: str  # as the page names it: what is checked, in plain words
    verdict: str
    failed_first: str
    criterion: str
    evidence: str
    owner: str
    report: str  # the readable report (its .md when there is one), at /api/docs/<report>
    sidecar: str  # the file the row was read from
    report_date: str


# Report items named after the process that produced them, named on the page for what
# they check. Every other item's name already says that.
CHECK_LABELS: dict[str, str] = {
    "Wave 0 no-edge gate": "No edge in the fair world",
    "Dashboard run browser and gate board": "Dashboard run browser and check board",
}


def check_label(name: str) -> str:
    if name in CHECK_LABELS:
        return CHECK_LABELS[name]
    return re.sub(r"\bgate\b", "check", name)


def checks(docs: Path) -> list[CheckOut]:
    """One row per check across the gate reports, the latest report's row where two share a name."""
    boards = sorted(
        load_gate_boards(docs), key=lambda b: (b.report_date, b.milestone), reverse=True
    )
    seen: set[str] = set()
    out: list[CheckOut] = []
    for b in boards:
        for i in b.items:
            if i.name in seen:
                continue
            seen.add(i.name)
            readable = b.source.with_suffix(".md")
            out.append(
                _from_attrs(
                    CheckOut,
                    i,
                    label=check_label(i.name),
                    report=readable.name if readable.is_file() else b.source.name,
                    sidecar=b.source.name,
                    report_date=b.report_date,
                )
            )
    return out


DOC_NAME = re.compile(r"^[A-Za-z0-9_]+\.(md|json)$")


def doc_file(docs: Path, name: str) -> Path:
    """A report or record in ``docs/`` by its file name; nothing else is served."""
    target = docs / name
    if not DOC_NAME.match(name) or not target.is_file():
        raise KeyError(name)
    return target
