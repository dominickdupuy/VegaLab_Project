"""Build the M4 gate report from live evidence and write docs/M4_GATE_REPORT.md and .json.

Lane: Dominick Dupuy (statistical spine).

The M4 row of the schedule, item by item, plus the attribution residual (section
10.9). The evidence functions live in ``optspread.m4gate``; this script only runs
them and writes the report. Until the confirmatory sweep is analysed (#34) the
analysis items read NOT CLEARED with what failed first, which is the report doing
its job rather than a build failure to hide.

    python -m scripts.m4_gate_report
    python -m scripts.m4_gate_report --out -
    python -m scripts.m4_gate_report --runs-root $VEGALAB_DATA/runs --web web/out

It reads ``docs/M4_ANALYSIS.json`` (``make m4``), ``docs/VRP_MAPPING.json`` (``make
vrp``) and ``docs/DATA_MANIFEST.json`` from ``--docs``; the A7 grid record
(``grid-7d9d53d340006f02.json``) under ``--runs-root``, which defaults to
``$VEGALAB_DATA/runs`` when ``VEGALAB_DATA`` is set, else ``runs`` (where the sweep
writes); the attribution of its cells under ``--attribution``
(``optspread.attribution.default_root()``); and the built site under ``--web``
(``$OPTSPREAD_WEB``, else ``web/out``, as the server serves it). The report's header
names every root it read, as ``$VEGALAB_DATA`` or ``~`` rather than one machine's
home, and the commit it was built at. Exit status 1 until every item passes.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from optspread.attribution import default_root
from optspread.dashboard_data import REPO_ROOT
from optspread.experiment.grid import grid_filename
from optspread.gatereport import build_report
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.m4gate import (
    bootstrap_item,
    collaboration_item,
    dashboard_item,
    homogeneity_item,
    power_curve_item,
    residual_item,
    vrp_item,
)
from optspread.manifest import git_dirty, git_sha
from scripts.m2_gate_report import default_roots, portable, portable_items

DOC = Path("docs/M4_GATE_REPORT.md")
WEB_ENV = "OPTSPREAD_WEB"


def default_web(env: dict[str, str] | None = None) -> Path:
    """``$OPTSPREAD_WEB`` if set, else ``web/out`` at the repository root: what ``make
    serve`` serves."""
    value = (os.environ if env is None else env).get(WEB_ENV, "")
    return Path(value).expanduser() if value else REPO_ROOT / "web" / "out"


def main(argv: Sequence[str] | None = None, on: date | None = None) -> int:
    runs_root, _ = default_roots()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--out", default=str(DOC), help="output path, or - for stdout")
    parser.add_argument("--runs-root", type=Path, default=runs_root, help="where the sweep is")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--attribution", type=Path, default=default_root())
    parser.add_argument("--web", type=Path, default=default_web(), help="the built site")
    parser.add_argument("--repo", type=Path, default=Path("."), help="for the code SHA")
    args = parser.parse_args(argv)

    sha = git_sha(args.repo)
    report = build_report(
        "M4",
        portable_items(
            [
                power_curve_item(args.docs),
                bootstrap_item(args.docs),
                homogeneity_item(args.docs),
                vrp_item(args.docs),
                dashboard_item(args.runs_root, args.docs, args.web),
                collaboration_item(args.docs),
                residual_item(args.runs_root, args.attribution),
            ]
        ),
        on or date.today(),
        (
            portable(
                f"Runs root: `{args.runs_root}` (grid `{grid_filename(CONFIRMATORY_GRID)}`); "
                f"docs: `{args.docs}`; attribution root: `{args.attribution}`; built site: "
                f"`{args.web}`"
            ),
            f"Built by `python -m scripts.m4_gate_report` at {sha[:8]}"
            + (" (dirty)" if git_dirty(args.repo) else ""),
        ),
    )
    if args.out == "-":
        print(report.render())
        return 0 if report.passed else 1
    target, sidecar = report.write(Path(args.out))
    print(f"wrote {target} and {sidecar} ({report.verdict})")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
