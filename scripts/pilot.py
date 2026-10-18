"""Run the variance-components pilot and write the S x N matrix the budget reads.

Lane: Ilan Danial (machinery).

    python -m scripts.pilot --seeds 8 --episodes 200 --timesteps 1000000 --workers 8
    python -m scripts.pilot --seeds 8 --episodes 200 --timesteps 1000000 --dry-run
    python scripts/pilot_budget.py --delta 1500 --matrix runs/pilot/matrix.npy

S policies trained at the frozen D3 hyperparameters, seed i on its own training
paths, each evaluated over N episodes on its own block of path seeds; the
fixed-delta seller runs the same blocks as the negative control, whose
between-seed component is zero by construction. Every cell is a run directory
under ``--root``; ``matrix.npy`` (the learner) and ``matrix_oracle.npy`` (the
control) land beside them. A second run with the same arguments reads finished
cells back, so a killed pilot resumes where it stopped.

``--timesteps`` has no default: D3 fixes the hyperparameters, not the training
budget, and the pilot does not get to pick one silently. Costs are on. Every
number is synthetic.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from optspread.config import CostConfig, RunConfig, TrainConfig
from optspread.orchestrate import (
    IncompleteGridError,
    grid_cells,
    pilot_grid,
    plan_cells,
    run_grid,
    write_pilot_matrices,
)
from optspread.power import variance_components
from optspread.train import Trainer, train_ppo


def main(argv: Sequence[str] | None = None, trainer: Trainer = train_ppo) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--root", type=Path, default=Path("runs") / "pilot")
    parser.add_argument("--seeds", type=int, default=8, help="S: independently trained policies")
    parser.add_argument("--episodes", type=int, default=100, help="N: evaluation episodes each")
    parser.add_argument(
        "--timesteps", type=int, required=True, help="PPO steps per seed (not pre-registered)"
    )
    parser.add_argument("--premium", type=float, default=0.0, help="planted premium, vol points")
    parser.add_argument("--seed", type=int, default=4000, help="first evaluation path seed")
    parser.add_argument("--train-seed", type=int, default=0, help="training seed of block 0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args(argv)

    grid = pilot_grid(
        RunConfig(seed=int(args.seed), cost=CostConfig(enabled=True)),
        n_seeds=int(args.seeds),
        n_episodes=int(args.episodes),
        premium=float(args.premium),
        train=TrainConfig(
            total_timesteps=int(args.timesteps), seed=int(args.train_seed), device=args.device
        ),
    )
    root = Path(args.root)
    print(plan_cells(grid_cells(grid), root, int(args.workers)).render(), flush=True)
    if args.dry_run:
        return 0
    try:
        record = run_grid(
            grid,
            root,
            workers=int(args.workers),
            trainer=trainer,
            report=lambda p: print(p.render(), flush=True),
        )
    except IncompleteGridError as exc:
        print(exc, file=sys.stderr)
        return 1
    print()
    print(record.render())
    paths = write_pilot_matrices(record, root)
    for arm, path in paths.items():
        print()
        print(f"{arm}: {path}")
        if grid.n_seeds >= 2 and grid.base.n_episodes >= 2:
            print(variance_components(record.matrix(root, grid.premiums[0], arm)).render())
    print()
    print(f"next: python scripts/pilot_budget.py --delta <effect> --matrix {paths['ppo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
