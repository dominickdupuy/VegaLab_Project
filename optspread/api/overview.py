"""API payloads for the landing page (the study at a glance) and the health check.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi``, which re-exports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from optspread.api._base import Out
from optspread.api.panel import DataStatusOut, PanelReader, SpanOut
from optspread.api.runs import RunRowOut, row_out
from optspread.api.sweeps import _grid_files, _sweep_dirs
from optspread.dashboard.gateboard import load_gate_boards
from optspread.dashboard.jobs.spec import available_arms
from optspread.dashboard.landing import readme_summary
from optspread.dashboard.runindex import RunRow

# -- the overview ------------------------------------------------------------------ #


class HealthOut(Out):
    ok: bool
    version: str
    git_sha: str  # HEAD of the checkout the server runs from, "unknown" without git


class GateTileOut(Out):
    milestone: str
    # The two verdicts GateBoard.verdict derives from a board's items; what the report
    # file states is not shown here.
    verdict: Literal["PASS", "NOT CLEARED"]
    verdict_mismatch: bool
    report_date: str
    n_items: int
    n_pass: int
    source: str


class OverviewOut(Out):
    summary: list[str]
    gates: list[GateTileOut]
    data: DataStatusOut
    coverage: list[SpanOut]
    overlap: tuple[str, str, int] | None
    n_runs: int
    n_ok: int
    latest: list[RunRowOut]
    sweeps: int
    ppo_available: bool


def overview(
    readme: Path, docs: Path, root: Path, reader: PanelReader, rows: Sequence[RunRow]
) -> OverviewOut:
    boards = load_gate_boards(docs)
    spans: list[SpanOut] = []
    overlap = None
    if reader.layout.coverage.is_file():
        cov = reader.coverage_out()
        spans, overlap = cov.spans, cov.overlap
    return OverviewOut(
        summary=list(readme_summary(readme)),
        gates=[
            GateTileOut(
                milestone=b.milestone,
                verdict=b.verdict,
                verdict_mismatch=b.verdict_mismatch,
                report_date=b.report_date,
                n_items=len(b.items),
                n_pass=b.counts.get("PASS", 0),
                source=str(b.source),
            )
            for b in boards
        ],
        data=reader.status(),
        coverage=spans,
        overlap=overlap,
        n_runs=len(rows),
        n_ok=sum(r.status == "OK" for r in rows),
        latest=[row_out(r, root) for r in rows[:8]],
        sweeps=len(_sweep_dirs(root)) + len(_grid_files(root)),
        ppo_available="ppo" in available_arms(),
    )
