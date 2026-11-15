"""Build the M5 gate report from live evidence and write docs/M5_GATE_REPORT.md and .json.

Lane: Hemdutt Rao (market layer).

The M5 row of the schedule, item by item (#18). The evidence functions live in
``optspread.m5gate``; this script only runs them and writes the report. Until the
usability sessions are recorded in ``docs/USABILITY.md`` that item reads NOT CLEARED
with what failed first, which is the report doing its job rather than a build failure
to hide.

    python -m scripts.m5_gate_report --out -
    python -m scripts.m5_gate_report --runs-root $VEGALAB_DATA/runs --web web/out
    make m5gate ARGS="--runs-root $VEGALAB_DATA/runs --attribution attribution"

It reads the A7 grid record (``grid-7d9d53d340006f02.json``) under ``--runs-root``
(``$VEGALAB_DATA/runs`` when ``VEGALAB_DATA`` is set, else ``runs``); the sweep's
attribution under ``--attribution`` (``optspread.attribution.default_root()``) and the
action mix under ``--actionmix`` (``optspread.actionmix.default_root()``), both through
the dashboard's app; ``docs/USABILITY.md``, ``docs/M4_ANALYSIS.json`` and the documents
under ``--docs``, the README and the commits findings cite in ``--repo``; and the built
site under ``--web`` (``$OPTSPREAD_WEB``, else ``web/out``). The header names every root,
as ``$VEGALAB_DATA`` or ``~``, and the commit it was built at. Exit status 1 until every
item passes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from optspread.actionmix import default_root as default_actionmix
from optspread.attribution import default_root
from optspread.experiment.grid import grid_filename
from optspread.gatereport import build_report
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.m5gate import adapter_item, dashboard_item, documentation_item, usability_item
from optspread.manifest import git_dirty, git_sha
from scripts.m2_gate_report import default_roots, portable, portable_items
from scripts.m4_gate_report import default_web

DOC = Path("docs/M5_GATE_REPORT.md")


def main(argv: Sequence[str] | None = None, on: date | None = None) -> int:
    runs_root, _ = default_roots()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--out", default=str(DOC), help="output path, or - for stdout")
    parser.add_argument("--runs-root", type=Path, default=runs_root, help="where the sweep is")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--attribution", type=Path, default=default_root())
    parser.add_argument("--actionmix", type=Path, default=default_actionmix())
    parser.add_argument("--web", type=Path, default=default_web(), help="the built site")
    parser.add_argument("--repo", type=Path, default=Path("."), help="README, cited commits")
    args = parser.parse_args(argv)

    sha = git_sha(args.repo)
    report = build_report(
        "M5",
        portable_items(
            [
                dashboard_item(
                    args.runs_root, args.docs, args.web, args.attribution, args.actionmix
                ),
                usability_item(args.docs, args.repo),
                documentation_item(args.repo, args.docs),
                adapter_item(),
            ]
        ),
        on or date.today(),
        (
            portable(
                f"Runs root: `{args.runs_root}` (grid `{grid_filename(CONFIRMATORY_GRID)}`); "
                f"docs: `{args.docs}`; attribution root: `{args.attribution}`; action mix "
                f"root: `{args.actionmix}`; built site: `{args.web}`"
            ),
            f"Built by `python -m scripts.m5_gate_report` at {sha[:8]}"
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
