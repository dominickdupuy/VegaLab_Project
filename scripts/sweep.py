"""Run the full sweep: planted premium x arm x paired seed blocks, parallel and resumable.

Lane: Ilan Danial (machinery).

    python -m scripts.sweep --seeds 50 --episodes 200 --timesteps 1000000 --workers 32
    python -m scripts.sweep --seeds 50 --episodes 200 --timesteps 1000000 --dry-run
    python -m scripts.sweep --premiums 0 1 --arms delta_seller --seeds 4 --episodes 50

The grid is decision D2's axis (default 0, 0.25, 0.5, 1, 2 and 4 vol points) by
the primary contrast of D1 (the fixed-delta seller and PPO) by S seed blocks of N
episodes. Within a block every arm at every premium runs the same paths, and
block i's policy is training seed ``--train-seed + i`` at every premium. Every
cell is a run directory under ``--root``; ``grid-<hash>.json`` lists them. A
second run with the same arguments reads finished cells back, so a killed sweep
resumes where it stopped. One progress line per cell, with an ETA.

``--timesteps`` has no default and is required whenever the ppo arm is in the
grid. Costs are on. Every number is synthetic (D2: the null world by design).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

from optspread.config import ArmName, CostConfig, RunConfig, TrainConfig
from optspread.orchestrate import (
    PRIMARY_ARMS,
    SWEEP_PREMIUMS,
    GridConfig,
    IncompleteGridError,
    grid_cells,
    grid_filename,
    plan_cells,
    run_grid,
)
from optspread.train import Trainer, train_ppo

ARM_NAMES: tuple[str, ...] = get_args(ArmName)


def main(argv: Sequence[str] | None = None, trainer: Trainer = train_ppo) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--premiums", type=float, nargs="+", default=list(SWEEP_PREMIUMS))
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(PRIMARY_ARMS))
    parser.add_argument("--seeds", type=int, default=50, help="S: seed blocks")
    parser.add_argument("--episodes", type=int, default=200, help="N: episodes per block")
    parser.add_argument("--timesteps", type=int, help="PPO steps per seed (not pre-registered)")
    parser.add_argument("--seed", type=int, default=100_000, help="first evaluation path seed")
    parser.add_argument("--train-seed", type=int, default=1000, help="training seed of block 0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args(argv)

    arms = cast(tuple[ArmName, ...], tuple(str(a) for a in args.arms))
    train = None
    if "ppo" in arms:
        if args.timesteps is None:
            parser.error("--timesteps is required with the ppo arm; it is not pre-registered")
        train = TrainConfig(
            total_timesteps=int(args.timesteps), seed=int(args.train_seed), device=args.device
        )
    grid = GridConfig(
        base=RunConfig(
            seed=int(args.seed), n_episodes=int(args.episodes), cost=CostConfig(enabled=True)
        ),
        premiums=tuple(float(p) for p in args.premiums),
        arms=arms,
        n_seeds=int(args.seeds),
        train=train,
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
    print(f"cells under {root}, listed in {root / grid_filename(grid)} (synthetic; costs on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
