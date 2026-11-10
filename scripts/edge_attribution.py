"""Attribute the detected edge, not a run's P&L: per component and structure, per arm and premium.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.edge_attribution --grid <sweep root> --first 50 --workers 2
    python -m scripts.edge_attribution --grid <root>/grid-<hash>.json --arms delta_seller

Every cell of the grid's arms (``--arms``, default all) is re-simulated, its episodes
attributed into delta, gamma, vega, theta, carry, cost and a residual and split by the
structure each step was about; each arm at premium p is then contrasted with itself at
zero on the same paths (A1) and tested on the S block means (A2), component by
component. ``--first N`` reads only each cell's first N episodes (a quick look; at
least 2). Writes ``<out>/edge-<hash>.json`` and ``.md`` (``-first<N>`` before the
suffix for a partial read), ``--out`` defaulting to ``optspread.edge.default_root()``
(``attribution/edge/``); the schema is the module docstring of ``optspread.edge``. A ppo
cell needs its ``policy.zip`` and the ``train`` extra. Exit status 1 when a cell
neither reproduces nor agrees with its ``run.json``, or the grid cannot be read.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from optspread.actionmix import LearnerUnavailableError
from optspread.attribution import COMPONENTS
from optspread.config import ArmName
from optspread.edge import GridEdge, carriers, default_root, edge_grid, write_edge
from optspread.experiment.grid import GridCell


def render(result: GridEdge) -> str:
    """Per arm and premium: the P&L edge and the components carrying it."""
    lines = [
        f"grid {result.grid_hash}  {result.source}  S = {result.n_seeds}  "
        f"N = {result.n_episodes}  ({result.reproduction}; max |dP&L| "
        f"{result.max_abs_pnl_diff:.2g})"
    ]
    for e in result.edges:
        parts = "  ".join(f"{c} {e.components[c].estimate:,.0f}" for c in COMPONENTS)
        lines.append(
            f"  p={e.premium:<5g} {e.arm:<13} edge {e.pnl.estimate:,.0f} (se {e.pnl.se:,.0f}, "
            f"lb {e.pnl.lower_bound:,.0f}) = {parts}"
        )
        lines.append(
            f"  {'':<20}carried by {', '.join(carriers(e)) or 'none'}; residual "
            f"{e.residual_edge_share:.1%} of the edge, {e.residual_share:.2%} of gross"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--grid", type=Path, required=True, help="a sweep root or its grid file")
    parser.add_argument("--arms", nargs="+", choices=get_args(ArmName), help="grid arms kept")
    parser.add_argument("--first", type=int, help="attribute only each cell's first N episodes")
    parser.add_argument("--workers", type=int, default=1, help="processes")
    parser.add_argument("--out", type=Path, help="output folder (default: default_root())")
    args = parser.parse_args(argv)
    if args.first is not None and args.first < 2:
        parser.error("--first takes a count of at least 2")
    if args.workers < 1:
        parser.error("--workers takes a positive count")
    out: Path = args.out if args.out is not None else default_root()
    start = time.monotonic()

    def report(done: int, total: int, cell: GridCell) -> None:
        elapsed = time.monotonic() - start
        print(
            f"[{done}/{total}] {cell.arm} p={cell.premium:g} block {cell.block} ({elapsed:,.0f}s)",
            file=sys.stderr,
            flush=True,
        )

    try:
        result = edge_grid(
            args.grid, arms=args.arms, first=args.first, workers=args.workers, report=report
        )
        json_path, md_path = write_edge(result, out)
    except LearnerUnavailableError as exc:
        print(f"cannot load the policy: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"cannot attribute the edge: {exc}", file=sys.stderr)
        return 1
    print(render(result))
    print(f"    -> {json_path}\n    -> {md_path}", flush=True)
    return 0 if result.agrees or result.reproduces else 1


if __name__ == "__main__":
    sys.exit(main())
