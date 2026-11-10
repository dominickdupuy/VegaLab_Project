"""The M4 analysis of the confirmatory sweep: docs/M4_ANALYSIS.md and its JSON sidecar.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.m4_analysis --root $VEGALAB_DATA/runs
    python -m scripts.m4_analysis --root runs/test --grid runs/test/grid-<hash>.json \
        --exploratory --out runs/test/M4_ANALYSIS

Issues #12 (power curve, floor, gap), #13 (tau^2 across premiums) and #14 (bootstrap),
as ``optspread.m4analysis`` defines them. The grid defaults to the A7 confirmatory one,
``grid-7d9d53d340006f02.json`` under ``--root``. A grid that is not that design, or has
a cell missing or invalid under the root, is refused unless ``--exploratory`` is given;
an exploratory report says so at the top and needs an explicit ``--out``, so it never
lands in docs/ by default. ``--root`` defaults to ``$VEGALAB_DATA/runs`` when
``VEGALAB_DATA`` is set, else ``runs``. Exit status 2 on a refused grid.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from optspread.experiment.grid import grid_filename, read_grid
from optspread.m4analysis import (
    AGREEMENT_ALPHA,
    BOOT_SEED,
    CONFIRMATORY_GRID,
    N_BOOT,
    REFERENCE_REPLICATIONS,
    Provenance,
    RefusedGridError,
    analyse,
)
from optspread.manifest import git_dirty, git_sha, library_versions
from scripts.m2_gate_report import default_roots, portable

OUT = Path("docs/M4_ANALYSIS")
# Premiums above zero in the A7 grid: the Bonferroni split of the agreement tolerance.
CONFIRMATORY_LEVELS = sum(p > 0 for p in CONFIRMATORY_GRID.premiums)


def main(argv: Sequence[str] | None = None, on: date | None = None) -> int:
    runs_root, _ = default_roots()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--root", type=Path, default=runs_root, help="the runs root")
    parser.add_argument("--grid", type=Path, help="grid-<hash>.json (default: the A7 grid)")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="analyse a grid that is not the confirmatory design or is incomplete",
    )
    parser.add_argument("--out", help=f"path stem for .md and .json (default {OUT}), or -")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--replications", type=int, default=REFERENCE_REPLICATIONS)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    parser.add_argument("--repo", type=Path, default=Path("."), help="for the code SHA")
    args = parser.parse_args(argv)

    # The smallest reference p-value is 2 / (R + 1); below this R no level of a
    # five-level arm could ever diverge at its Bonferroni threshold.
    if 2.0 / (args.replications + 1) > AGREEMENT_ALPHA / CONFIRMATORY_LEVELS:
        parser.error(
            f"--replications {args.replications} cannot reach p < "
            f"{AGREEMENT_ALPHA / CONFIRMATORY_LEVELS:g}; use at least "
            f"{math.ceil(2 * CONFIRMATORY_LEVELS / AGREEMENT_ALPHA) - 1}"
        )
    if args.exploratory and args.out is None:
        parser.error("an exploratory analysis needs an explicit --out; docs/ is for the real one")
    grid_path = args.grid or args.root / grid_filename(CONFIRMATORY_GRID)
    if not grid_path.exists():
        print(portable(f"no grid record at {grid_path}"), file=sys.stderr)
        return 2
    record = read_grid(grid_path)
    try:
        result = analyse(
            record,
            args.root,
            exploratory=args.exploratory,
            n_boot=args.n_boot,
            replications=args.replications,
            seed=args.seed,
        )
    except RefusedGridError as exc:
        hint = "" if args.exploratory else " (--exploratory analyses it anyway, labelled)"
        print(portable(f"refused {grid_path}{hint}:"), file=sys.stderr)
        for reason in exc.reasons:
            print(portable(f"  {reason}"), file=sys.stderr)
        return 2
    provenance = Provenance(
        report_date=on or date.today(),
        root=portable(str(args.root)),
        grid_file=grid_path.name,
        analysis_git_sha=git_sha(args.repo),
        analysis_git_dirty=git_dirty(args.repo),
        versions=library_versions(),
    )
    text = result.render(provenance)
    if args.out == "-":
        print(text)
        return 0
    stem = Path(args.out or OUT)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, newline="\n")
    stem.with_suffix(".json").write_text(result.render_json(provenance), newline="\n")
    kind = "confirmatory" if result.confirmatory else "EXPLORATORY"
    print(f"wrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')} ({kind})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
