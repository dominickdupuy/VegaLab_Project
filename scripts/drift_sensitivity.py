"""The drift sensitivity of the confirmatory sweep: docs/DRIFT_SENSITIVITY.md and .json.

Lane: Hemdutt Rao (market layer).

    python -m scripts.drift_sensitivity --root $VEGALAB_DATA/runs --workers 10
    python -m scripts.drift_sensitivity --grid runs/test/grid-<hash>.json --out runs/test/DRIFT

Post hoc, not pre-registered (amendment A11). Every cell of the grid (default the A7
one, ``grid-7d9d53d340006f02.json`` under ``--root``) is re-simulated on its own seeds in
the world it ran in, checked against its ``run.json``, and again in the fair world
(``market.r = 0``, nothing else changed): a scripted cell with its policy, a PPO cell with
the ``policy.zip`` it saved, scored and never retrained, so the ``train`` extra is needed.
The M4 analysis is then run on each world; ``optspread.drift`` says what is compared.
Nothing is written under ``--root``. ``--root`` defaults to ``$VEGALAB_DATA/runs`` when
``VEGALAB_DATA`` is set, else ``runs``. Exit status 1 when the grid cannot be read or a
policy cannot be loaded; a cell whose original world differs from its ``run.json`` is
reported, not refused.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from optspread.drift import Provenance, drift_grid, render, render_json, sweep_shas
from optspread.experiment.grid import GridCell, grid_filename
from optspread.m4analysis import CONFIRMATORY_GRID
from optspread.manifest import git_dirty, git_sha, library_versions
from scripts.m2_gate_report import default_roots, portable

OUT = Path("docs/DRIFT_SENSITIVITY")


def main(argv: Sequence[str] | None = None, on: date | None = None) -> int:
    runs_root, _ = default_roots()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--root", type=Path, default=runs_root, help="the runs root")
    parser.add_argument("--grid", type=Path, help="grid-<hash>.json (default: the A7 grid)")
    parser.add_argument("--workers", type=int, default=1, help="processes")
    parser.add_argument("--out", help=f"path stem for .md and .json (default {OUT}), or -")
    parser.add_argument("--n-boot", type=int, help="bootstrap resamples (default M4's)")
    parser.add_argument("--replications", type=int, help="reference sets (default M4's)")
    parser.add_argument("--repo", type=Path, default=Path("."), help="for the code SHA")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers takes a positive count")
    grid_path: Path = args.grid or args.root / grid_filename(CONFIRMATORY_GRID)
    if not grid_path.exists():
        print(portable(f"no grid record at {grid_path}"), file=sys.stderr)
        return 1
    if args.out not in (None, "-") and Path(args.out).resolve().is_relative_to(
        grid_path.parent.resolve()
    ):
        print(portable(f"refused: --out {args.out} lies inside the sweep's root"), file=sys.stderr)
        return 1
    start = time.monotonic()

    def report(done: int, total: int, cell: GridCell) -> None:
        elapsed = time.monotonic() - start
        print(
            f"[{done}/{total}] {cell.arm} p={cell.premium:g} block {cell.block} ({elapsed:,.0f}s)",
            file=sys.stderr,
            flush=True,
        )

    try:
        result = drift_grid(
            grid_path,
            workers=args.workers,
            report=report,
            n_boot=args.n_boot,
            replications=args.replications,
        )
    except ImportError as exc:
        print(f"cannot load a saved policy (install the train extra): {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(portable(f"cannot re-simulate {grid_path}: {exc}"), file=sys.stderr)
        return 1
    provenance = Provenance(
        report_date=on or date.today(),
        root=portable(str(grid_path.parent)),
        analysis_git_sha=git_sha(args.repo),
        analysis_git_dirty=git_dirty(args.repo),
        versions=library_versions(),
        sweep_git_shas=sweep_shas(result),
    )
    text = render(result, provenance)
    if args.out == "-":
        print(text)
        return 0
    stem = Path(args.out or OUT)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, newline="\n")
    stem.with_suffix(".json").write_text(render_json(result, provenance), newline="\n")
    print(
        f"wrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')} "
        f"({len(result.cells)} cells in {result.seconds:,.0f}s on {args.workers} workers)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
