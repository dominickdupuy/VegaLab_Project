"""Replay finished runs from their manifests and check them byte for byte.

Lane: Ilan Danial (machinery). The reproducibility gate of project document 10.7.

    python -m scripts.replay runs/pilot/<run id>
    python -m scripts.replay runs/pilot --max 4

Each argument is a run directory, or a folder of run directories to replay in turn
(at most ``--max`` of them per folder when given). A PPO run is retrained from its
recorded config, so it needs the ``train`` extra and costs one training run. Exit
status 0 only when every replayed run is identical.
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
    args = parser.parse_args(argv)

    outcomes = []
    for path in args.paths:
        runs = find_runs(Path(path))
        if not runs:
            print(f"no run directories at {path}", file=sys.stderr)
            return 1
        for run_dir in runs[: args.max]:
            outcome = replay_run(run_dir)
            print(outcome.render(), flush=True)
            outcomes.append(outcome)
    same = sum(o.matches for o in outcomes)
    print(f"{same}/{len(outcomes)} runs reproduce bit for bit")
    return 0 if same == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
