"""Run the full sweep: planted premium x arm x paired seed blocks, parallel and resumable.

Lane: Ilan Danial (machinery).

    python -m scripts.sweep --seeds 50 --episodes 200 --timesteps 1000000 --workers 32
    python -m scripts.sweep --seeds 50 --episodes 200 --timesteps 1000000 --dry-run
    python -m scripts.sweep --premiums 0 1 --arms delta_seller --seeds 4 --episodes 50

The grid is decision D2's axis (default 0, 0.25, 0.5, 1, 2 and 4 vol points) by
the arms of the primary analysis (the fixed-delta seller and PPO, each tested
against itself at zero premium, #26) by S seed blocks of N episodes. Within a block
every arm at every premium runs the same paths, and block i's policy is training
seed ``--train-seed + i`` at every premium. Every cell is a run directory under
``--root``; ``grid-<hash>.json`` lists them. A second run with the same arguments
reads finished cells back, so a killed sweep resumes where it stopped. One progress
line per cell, with an ETA. ``--root`` defaults to ``$VEGALAB_DATA/runs`` when
``VEGALAB_DATA`` is set, else ``runs``: the runs root ``make m2`` and ``make m4`` read.

``--timesteps`` has no default and is required whenever the ppo arm is in the
grid. Costs are on. On the generator (the default) every number is synthetic (D2: the
null world by design).

``--source optionmetrics`` runs the grid on the real SPX panel: every cell is evaluated
on the ``eval`` split and every PPO cell trained on ``train`` (the dates of
pre-registration amendment A8), under
``$VEGALAB_DATA/runs/optionmetrics`` by default. The planted premium is not applied to
real chains, so there ``--premiums 0`` is the one level that means anything.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

from optspread.config import ArmName, CostConfig, RunConfig, TrainConfig
from optspread.experiment.cli import (
    add_source_arguments,
    chosen_source,
    describe,
    premium_note,
    runs_root,
)
from optspread.experiment.grid import (
    PRIMARY_ARMS,
    SWEEP_PREMIUMS,
    GridConfig,
    grid_cells,
    grid_filename,
)
from optspread.experiment.schedule import IncompleteGridError, plan_cells, run_grid
from optspread.train import Trainer, train_ppo

ARM_NAMES: tuple[str, ...] = get_args(ArmName)


def main(argv: Sequence[str] | None = None, trainer: Trainer = train_ppo) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument(
        "--root",
        type=Path,
        help="runs root (default: $VEGALAB_DATA/runs when VEGALAB_DATA is set, else runs; "
        "where make m2 and make m4 look; .../runs/optionmetrics on the real source)",
    )
    parser.add_argument("--premiums", type=float, nargs="+", default=list(SWEEP_PREMIUMS))
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(PRIMARY_ARMS))
    parser.add_argument("--seeds", type=int, default=50, help="S: seed blocks")
    parser.add_argument("--episodes", type=int, default=200, help="N: episodes per block")
    parser.add_argument("--timesteps", type=int, help="PPO steps per seed (A4 fixes 1000000)")
    parser.add_argument("--seed", type=int, default=100_000, help="first evaluation path seed")
    parser.add_argument("--train-seed", type=int, default=1000, help="training seed of block 0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    source = chosen_source(parser, args)

    arms = cast(tuple[ArmName, ...], tuple(str(a) for a in args.arms))
    train = None
    if "ppo" in arms:
        if args.timesteps is None:
            parser.error(
                "--timesteps is required with the ppo arm (A4: 1000000); it has no default"
            )
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
    root = runs_root(args.root, source)
    if len(grid.premiums) > 1 and (note := premium_note(source)):
        print(note, file=sys.stderr)
    plan = plan_cells(grid_cells(grid), root, int(args.workers), trainer=trainer, source=source)
    print(f"{plan.render()}; on {describe(source)}", flush=True)
    if args.dry_run:
        return 0
    try:
        record = run_grid(
            grid,
            root,
            workers=int(args.workers),
            trainer=trainer,
            report=lambda p: print(p.render(), flush=True),
            source=source,
        )
    except IncompleteGridError as exc:
        print(exc, file=sys.stderr)
        return 1
    print()
    print(record.render())
    print(
        f"cells under {root}, listed in {root / grid_filename(grid)} ({describe(source)}; costs on)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
