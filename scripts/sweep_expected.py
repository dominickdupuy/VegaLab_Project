"""Write docs/SWEEP_EXPECTED.json from the confirmatory sweep, for a clone with no sweep.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.sweep_expected --root $VEGALAB_DATA/runs

Reads the A7 grid record (``grid-7d9d53d340006f02.json``) and its cells under ``--root``
(read only) and writes, for every cell, its run id, place in the grid, the sha256 of its
``run.json`` and its headline numbers, and for the reference cells
(``reproduction.reference_cells``: one seller cell per premium and one PPO cell) every
episode. ``scripts/reproduce.py`` regenerates cells on a partner's clone and compares
them with this file. Refuses a grid with a missing or invalid cell.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.orchestrate import default_runs_root, grid_filename, read_grid
from optspread.reproduction import EXPECTED_FILE, build_expected, render_expected

DOC = Path("docs") / EXPECTED_FILE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--root", type=Path, default=default_runs_root(), help="the sweep")
    parser.add_argument("--out", type=Path, default=DOC)
    args = parser.parse_args(argv)

    grid_path = args.root / grid_filename(CONFIRMATORY_GRID)
    if not grid_path.is_file():
        print(f"no grid record at {grid_path}", file=sys.stderr)
        return 1
    try:
        expected = build_expected(read_grid(grid_path), args.root)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    text = render_expected(expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, newline="\n")
    refs = sum(c.episodes is not None for c in expected.cells)
    print(
        f"wrote {args.out} ({len(text.encode()):,} bytes): {len(expected.cells)} cells of grid "
        f"{expected.grid_hash}, {refs} with every episode"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
