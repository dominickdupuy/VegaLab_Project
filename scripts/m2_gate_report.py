"""Build the M2 gate report from live evidence and write docs/M2_GATE_REPORT.md and .json.

Lane: Dominick Dupuy (statistical spine).

The M2 row of the schedule, item by item, plus the real data adapter (issue #3,
W6-4). The evidence functions live in ``optspread.m2gate``; this script only runs
them and writes the report. Items whose code is still being written read NOT
CLEARED with what failed first, which is the report doing its job rather than a
build failure to hide.

    python -m scripts.m2_gate_report
    python -m scripts.m2_gate_report --quick --out -
    python -m scripts.m2_gate_report --pilot-root runs/pilot --delta 1500

``--delta`` is optional. The effect to plan for is not pre-registered in outcome
units, so without it the pilot item reports the detection floor at 5 and 50 seeds,
which needs no delta; with it, the cheapest design that reaches 80% power.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from optspread.gatereport import build_report
from optspread.m2gate import (
    adapter_item,
    analysis_item,
    dashboard_item,
    pilot_item,
    ppo_harness_item,
    prereg_item,
    replay_item,
)

DOC = Path("docs/M2_GATE_REPORT.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer replications, for a smoke run")
    parser.add_argument("--out", default=str(DOC), help="output path, or - for stdout")
    parser.add_argument("--pilot-root", type=Path, default=Path("runs/pilot"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--delta", type=float, help="effect to plan the pilot design for")
    args = parser.parse_args()
    replications = 4_000 if args.quick else 10_000

    report = build_report(
        "M2",
        [
            ppo_harness_item(args.runs_root),
            replay_item(args.runs_root),
            pilot_item(args.pilot_root, args.delta),
            analysis_item(replications),
            prereg_item(args.repo),
            dashboard_item(args.runs_root, args.docs),
            adapter_item(),
        ],
        date.today(),
    )
    if args.out == "-":
        print(report.render())
        return 0 if report.passed else 1
    target, sidecar = report.write(Path(args.out))
    print(f"wrote {target} and {sidecar} ({report.verdict})")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
