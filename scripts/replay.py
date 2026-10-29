"""Replay finished runs from their manifests and check them byte for byte.

Lane: Ilan Danial (machinery). The reproducibility gate of project document 10.7.

    python -m scripts.replay runs/pilot/<run id>
    python -m scripts.replay runs/pilot --max 4
    python -m scripts.replay runs/sweep --rtol 1e-9     # a run made on another platform

Each argument is a run directory, or a folder of run directories to replay in turn
(at most ``--max`` of them per folder when given). A PPO run is retrained from its
recorded config, so it needs the ``train`` extra and costs one training run. Exit
status 0 only when at least one run was replayed and every replayed run is
identical (or, with ``--rtol``, within it): a path that does not exist, or nothing
replayed, is evidence of nothing.

``--rtol`` is for a scripted run written on another platform, whose last bits libm may
round otherwise: every number of ``run.json`` within the relative tolerance, every count
and identifier exact. It cannot make a retrained PPO run pass across platforms: training
compounds a last-bit difference into another policy, so such a run is DIFFERS anywhere
but on the platform it was trained on, whatever the tolerance.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from optspread.replay import find_runs, replay_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("paths", type=Path, nargs="+", help="run directories or their folder")
    parser.add_argument("--max", type=int, help="replay at most this many runs per folder")
    parser.add_argument(
        "--rtol",
        type=float,
        help="accept numbers within this relative tolerance, for a scripted run made on "
        "another platform (default: bytes only; a retrained PPO run is not comparable "
        "across platforms at any tolerance)",
    )
    args = parser.parse_args(argv)
    if args.max is not None and args.max < 1:
        parser.error(f"--max must be at least 1, got {args.max}")
    if args.rtol is not None and not args.rtol >= 0.0:
        parser.error(f"--rtol must be a tolerance of zero or more, got {args.rtol}")

    outcomes = []
    for path in args.paths:
        if not Path(path).is_dir():
            print(f"no such run directory or folder: {path}", file=sys.stderr)
            return 1
        runs = find_runs(Path(path))
        if not runs:
            print(f"no run directories at {path}", file=sys.stderr)
            return 1
        for run_dir in runs[: args.max]:
            outcome = replay_run(run_dir, rtol=args.rtol)
            print(outcome.render(), flush=True)
            outcomes.append(outcome)
    same = sum(o.matches for o in outcomes)
    line = f"{same}/{len(outcomes)} runs reproduce bit for bit"
    if args.rtol is not None:
        line += f", {sum(o.within for o in outcomes)} more within rtol {args.rtol:g}"
    print(line)
    return 0 if all(o.reproduced for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
