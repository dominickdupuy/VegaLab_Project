"""Build the M3 gate report from live evidence and write docs/M3_GATE_REPORT.md and .json.

Lane: Dominick Dupuy (statistical spine).

The M3 row of the schedule, item by item (#11). The evidence functions live in
``optspread.m3gate``; this script only runs them and writes the report. Until the
partner team's record is in ``docs/`` (#10) the partner item reads NOT CLEARED with
what failed first, which is the report doing its job rather than a build failure to hide.

    python -m scripts.m3_gate_report --out -
    python -m scripts.m3_gate_report --runs-root $VEGALAB_DATA/runs --replay 2
    make m3gate ARGS="--runs-root $VEGALAB_DATA/runs --web web/out"

It reads the A7 grid record (``grid-7d9d53d340006f02.json``), its cells and ``replay.log``
under ``--runs-root`` (``$VEGALAB_DATA/runs`` when ``VEGALAB_DATA`` is set, else ``runs``);
``docs/M1_GATE_REPORT.json`` (the floor M1 recorded), ``docs/M3_PARTNER_REPRODUCTION.json``
and the git history of ``--repo`` for ``docs/SWEEP_EXPECTED.json``; and the built site under
``--web`` (``$OPTSPREAD_WEB``, else ``web/out``). ``--replay N`` also replays N sweep cells
here, the arms in turn (a PPO cell retrains: about 20 minutes). The floor is measured
again, which takes seconds; the demo is called through the app in-process. The header
names every root, as ``$VEGALAB_DATA`` or ``~``, and the commit it was built at. Exit
status 1 until every item passes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from optspread.gatereport import build_report
from optspread.m3gate import FLOOR_SEEDS, demo_item, floor_item, partner_item, sweep_item
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.manifest import git_dirty, git_sha
from optspread.orchestrate import grid_filename
from scripts.m2_gate_report import default_roots, portable, portable_items
from scripts.m4_gate_report import default_web

DOC = Path("docs/M3_GATE_REPORT.md")


def main(argv: Sequence[str] | None = None, on: date | None = None) -> int:
    runs_root, _ = default_roots()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--out", default=str(DOC), help="output path, or - for stdout")
    parser.add_argument("--runs-root", type=Path, default=runs_root, help="where the sweep is")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--web", type=Path, default=default_web(), help="the built site")
    parser.add_argument("--repo", type=Path, default=Path("."), help="for tags and history")
    parser.add_argument("--replay", type=int, default=0, help="sweep cells to replay here")
    parser.add_argument("--seeds", type=int, default=FLOOR_SEEDS, help="the floor's seeds")
    args = parser.parse_args(argv)
    if args.replay < 0:
        parser.error(f"--replay must be at least 0, got {args.replay}")

    sha = git_sha(args.repo)
    report = build_report(
        "M3",
        portable_items(
            [
                sweep_item(args.runs_root, args.repo, replays=args.replay),
                floor_item(args.docs, seeds=args.seeds),
                demo_item(args.runs_root, args.docs, args.web),
                partner_item(args.docs, args.runs_root, args.repo),
            ]
        ),
        on or date.today(),
        (
            portable(
                f"Runs root: `{args.runs_root}` (grid `{grid_filename(CONFIRMATORY_GRID)}`); "
                f"docs: `{args.docs}`; built site: `{args.web}`"
            ),
            f"Built by `python -m scripts.m3_gate_report` at {sha[:8]}"
            + (" (dirty)" if git_dirty(args.repo) else "")
            + (f", replaying {args.replay} cell(s)" if args.replay else ""),
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
