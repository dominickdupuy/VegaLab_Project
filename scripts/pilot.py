"""Run the variance-components pilot and write the S x N matrix the budget reads.

Lane: Ilan Danial (machinery).

    python -m scripts.pilot --seeds 8 --episodes 200 --timesteps 1000000 --workers 8
    python -m scripts.pilot --seeds 8 --episodes 200 --timesteps 1000000 --dry-run
    python scripts/pilot_budget.py --delta 1500 --matrix runs/pilot/matrix.npy

S policies trained at the frozen D3 hyperparameters, seed i on its own training
paths, each evaluated over N episodes on its own block of path seeds, at zero
premium and at ``--premium`` (default 1 vol point, #5): the detection event
compares each arm with itself at zero (#26), so the pilot measures that contrast.
The fixed-delta seller runs the same blocks as the negative control, whose
between-seed component is zero by construction. Every cell is a run directory
under ``--root``; ``matrix.npy`` (the learner's contrast) and ``matrix_oracle.npy``
(the control's) land beside them with each arm's raw matrices. A second run with
the same arguments reads finished cells back, so a killed pilot resumes where it
stopped. ``--root`` defaults to ``$VEGALAB_DATA/runs/pilot`` when ``VEGALAB_DATA`` is
set, else ``runs/pilot``: the pilot root ``make m2`` reads.

``--timesteps`` has no default: D3 fixes the hyperparameters, not the training
budget, and the pilot does not get to pick one silently. Costs are on.

``--source optionmetrics`` runs it on the real SPX panel instead of the generator:
every policy trains on the ``train`` split and is evaluated on ``eval`` (the dates of
pre-registration amendment A8), under
``$VEGALAB_DATA/runs/optionmetrics/pilot`` by default. The planted premium is not
applied to real chains, so there the contrast against zero is zero by construction.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from optspread.config import CostConfig, RunConfig

# The ``name as name`` imports: what this script imported before its flow moved to
# optspread.experiment.cli, still importable from here.
from optspread.config import TrainConfig as TrainConfig
from optspread.experiment.cli import (
    add_run_options,
    add_source_arguments,
    chosen_source,
    plan_and_run,
    premium_note,
    runs_root,
    train_config,
)
from optspread.experiment.cli import describe as describe
from optspread.experiment.grid import GridRecord
from optspread.experiment.grid import grid_cells as grid_cells
from optspread.experiment.pilot import pilot_grid, write_pilot_matrices
from optspread.experiment.schedule import IncompleteGridError as IncompleteGridError
from optspread.experiment.schedule import plan_cells as plan_cells
from optspread.experiment.schedule import run_grid as run_grid
from optspread.power import variance_components
from optspread.train import Trainer, train_ppo


def main(argv: Sequence[str] | None = None, trainer: Trainer = train_ppo) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument(
        "--root",
        type=Path,
        help="runs root (default: $VEGALAB_DATA/runs/pilot when VEGALAB_DATA is set, else "
        "runs/pilot; where make m2 looks; .../runs/optionmetrics/pilot on the real source)",
    )
    parser.add_argument("--seeds", type=int, default=8, help="S: independently trained policies")
    parser.add_argument("--episodes", type=int, default=100, help="N: evaluation episodes each")
    parser.add_argument(
        "--timesteps", type=int, required=True, help="PPO steps per seed (A4 fixes 1000000)"
    )
    parser.add_argument(
        "--premium", type=float, default=1.0, help="planted premium against zero, vol points"
    )
    add_run_options(parser, seed=4000, train_seed=0)
    add_source_arguments(parser)
    args = parser.parse_args(argv)
    source = chosen_source(parser, args)

    grid = pilot_grid(
        RunConfig(seed=int(args.seed), cost=CostConfig(enabled=True)),
        n_seeds=int(args.seeds),
        n_episodes=int(args.episodes),
        premium=float(args.premium),
        train=train_config(args),
    )
    root = runs_root(args.root, source, "pilot")
    if note := premium_note(source):
        print(note, file=sys.stderr)

    def finish(record: GridRecord) -> None:
        paths = write_pilot_matrices(record, root, source=source)
        for arm, path in paths.items():
            print()
            print(f"{arm}: {path}")
            if grid.n_seeds >= 2 and grid.base.n_episodes >= 2:
                print(variance_components(record.contrast(root, grid.premiums[1], arm)).render())
        print()
        print(f"next: python scripts/pilot_budget.py --delta <effect> --matrix {paths['ppo']}")

    return plan_and_run(grid, root, args, trainer, source, finish)


if __name__ == "__main__":
    sys.exit(main())
