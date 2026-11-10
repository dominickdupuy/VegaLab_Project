"""Build the M2 gate report from live evidence and write docs/M2_GATE_REPORT.md and .json.

Lane: Dominick Dupuy (statistical spine).

The M2 row of the schedule, item by item, plus the real data adapter (issue #3,
W6-4). The evidence functions live in ``optspread.m2gate``; this script only runs
them and writes the report. An item whose evidence is missing reads NOT CLEARED
with what failed first, which is the report doing its job rather than a build
failure to hide.

    python -m scripts.m2_gate_report
    python -m scripts.m2_gate_report --quick --out -
    python -m scripts.m2_gate_report --pilot-root runs/pilot --delta 1500

The runs root defaults to ``$VEGALAB_DATA/runs`` and the pilot root to
``$VEGALAB_DATA/runs/pilot`` (where the README's pilot and sweep commands write) when
``VEGALAB_DATA`` is set, else to ``runs`` and ``runs/pilot`` under the current
directory. The report's header names the roots it read.

``--delta`` is optional. The effect to plan for is not pre-registered in outcome
units, so without it the pilot item reports the detection floor at 5 and 50 seeds,
which needs no delta; with it, the cheapest design that reaches 80% power.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path

from optspread.gatereport import GateItem, build_report
from optspread.m2gate import (
    adapter_item,
    analysis_item,
    dashboard_item,
    pilot_item,
    ppo_harness_item,
    prereg_item,
    replay_item,
)
from optspread.runs.rundir import default_runs_root

DOC = Path("docs/M2_GATE_REPORT.md")


def default_roots(env: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    """``(runs root, pilot root)``: under ``$VEGALAB_DATA`` when set, else cwd-relative.

    The same rule as ``scripts/pilot.py`` and ``scripts/sweep.py``
    (``orchestrate.default_runs_root``), so the report reads where they write."""
    runs = default_runs_root(env)
    return runs, runs / "pilot"


def portable(text: str) -> str:
    """A committed report names roots as $VEGALAB_DATA or ~, never one machine's home.

    The path after the placeholder is written with forward slashes, so a report
    written on Windows reads the same as one written anywhere else."""
    data = os.environ.get("VEGALAB_DATA")
    named = [(Path(data).expanduser(), "$VEGALAB_DATA")] if data else []
    # Native and forward-slash spellings of each root; on POSIX the two are one string.
    roots = {
        (form, name)
        for path, name in [*named, (Path.home(), "~")]
        for form in (str(path), path.as_posix())
    }
    for root, name in sorted(roots, key=lambda r: (r[1] == "~", -len(r[0]))):
        text = re.sub(re.escape(root) + r"[^\s`;,()]*", _renamer(root, name), text)
    return text


def _renamer(root: str, name: str) -> Callable[[re.Match[str]], str]:
    def rename(match: re.Match[str]) -> str:
        return name + match[0][len(root) :].replace("\\", "/")

    return rename


def portable_items(items: Sequence[GateItem]) -> list[GateItem]:
    return [
        dataclasses.replace(i, evidence=portable(i.evidence), failed_first=portable(i.failed_first))
        for i in items
    ]


def main(argv: Sequence[str] | None = None) -> int:
    runs_root, pilot_root = default_roots()
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer replications, for a smoke run")
    parser.add_argument("--out", default=str(DOC), help="output path, or - for stdout")
    parser.add_argument("--pilot-root", type=Path, default=pilot_root)
    parser.add_argument("--runs-root", type=Path, default=runs_root)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--delta", type=float, help="effect to plan the pilot design for")
    args = parser.parse_args(argv)
    replications = 4_000 if args.quick else 10_000

    report = build_report(
        "M2",
        portable_items(
            [
                ppo_harness_item(args.runs_root),
                replay_item(args.runs_root),
                pilot_item(args.pilot_root, args.delta),
                analysis_item(replications),
                prereg_item(args.repo),
                dashboard_item(args.runs_root, args.docs),
                adapter_item(),
            ]
        ),
        date.today(),
        (portable(f"Runs root: `{args.runs_root}`; pilot root: `{args.pilot_root}`"),),
    )
    if args.out == "-":
        print(report.render())
        return 0 if report.passed else 1
    target, sidecar = report.write(Path(args.out))
    print(f"wrote {target} and {sidecar} ({report.verdict})")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
