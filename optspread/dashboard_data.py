"""What the dashboard reads: the run index, the gate board and the live leak demo.

Lane: Ilan Danial (machinery).

The code lives in ``optspread.dashboard``: the run index in ``runindex``, the gate
board in ``gateboard``, the leak demo in ``leakdemo`` and the README summary in
``landing``. Nothing here needs the web stack, so it is tested and covered on the
plain ``dev`` install the CI runs.

This module is the permanent entry point, kept when other old paths go, because
code outside the dashboard finds it by its module path. ``optspread.m2gate`` loads
``scan_runs`` and ``load_gate_boards`` from ``optspread.dashboard_data`` through
importlib, and prints that path in its gate criterion, so in the committed M2
report; a name missing here would fail that item quietly as NOT CLEARED.
``optspread.reproduction`` imports ``run_leak_demo`` from here when its step runs
and records the call under this path. So every public name of the four modules is
re-exported here, by name.

Only functions and constants are re-exported, never the row cache or the grid
index, which stay in ``runindex``: a caller on either path shares the one cache. A
patch on a name here reaches only code that looks the name up here when it runs,
as ``reproduction`` does for ``run_leak_demo``.
"""

from optspread.dashboard.gateboard import GATE_STEM as GATE_STEM
from optspread.dashboard.gateboard import RUN_ID_PATTERN as RUN_ID_PATTERN
from optspread.dashboard.gateboard import VERDICTS as VERDICTS
from optspread.dashboard.gateboard import GateBoard as GateBoard
from optspread.dashboard.gateboard import GateReportFormatError as GateReportFormatError
from optspread.dashboard.gateboard import GateRow as GateRow
from optspread.dashboard.gateboard import gate_milestones as gate_milestones
from optspread.dashboard.gateboard import load_gate_board as load_gate_board
from optspread.dashboard.gateboard import load_gate_boards as load_gate_boards
from optspread.dashboard.gateboard import parse_gate_json as parse_gate_json
from optspread.dashboard.gateboard import parse_gate_markdown as parse_gate_markdown
from optspread.dashboard.gateboard import unreadable_gate_reports as unreadable_gate_reports
from optspread.dashboard.landing import readme_summary as readme_summary
from optspread.dashboard.leakdemo import LeakDemo as LeakDemo
from optspread.dashboard.leakdemo import run_leak_demo as run_leak_demo
from optspread.dashboard.runindex import DOCS_ENV as DOCS_ENV
from optspread.dashboard.runindex import GRID_PATTERN as GRID_PATTERN
from optspread.dashboard.runindex import REPO_ROOT as REPO_ROOT
from optspread.dashboard.runindex import RUNS_ENV as RUNS_ENV
from optspread.dashboard.runindex import ArtifactCheck as ArtifactCheck
from optspread.dashboard.runindex import CurvePoint as CurvePoint
from optspread.dashboard.runindex import RiskStats as RiskStats
from optspread.dashboard.runindex import RunComparison as RunComparison
from optspread.dashboard.runindex import RunDetail as RunDetail
from optspread.dashboard.runindex import RunFacets as RunFacets
from optspread.dashboard.runindex import RunRow as RunRow
from optspread.dashboard.runindex import check_artifacts as check_artifacts
from optspread.dashboard.runindex import compare_runs as compare_runs
from optspread.dashboard.runindex import docs_dir as docs_dir
from optspread.dashboard.runindex import filter_runs as filter_runs
from optspread.dashboard.runindex import flatten_config as flatten_config
from optspread.dashboard.runindex import grid_of as grid_of
from optspread.dashboard.runindex import is_matrix_manifest as is_matrix_manifest
from optspread.dashboard.runindex import load_run_detail as load_run_detail
from optspread.dashboard.runindex import load_run_row as load_run_row
from optspread.dashboard.runindex import premium_curve as premium_curve
from optspread.dashboard.runindex import risk_stats as risk_stats
from optspread.dashboard.runindex import run_facets as run_facets
from optspread.dashboard.runindex import runs_root as runs_root
from optspread.dashboard.runindex import scan_runs as scan_runs
