"""The data-source flags every command that launches runs takes, read the same way.

Lane: Ilan Danial (machinery).

    --source {synthetic,optionmetrics}   the generator (default), or the real SPX panel

On the real source a run's dates are not a flag: it is evaluated on the ``eval`` split
and a learned arm trained on ``train`` (``optspread.realdata.SPLITS``), the dates
pre-registration amendment A8 fixed, and nothing from ``HOLDOUT_START`` on is read. A
real-data run lands under ``<runs root>/optionmetrics`` unless ``--root`` says otherwise
(``optspread.runs.rundir.source_root``), because the source is not in the run id.

What ``scripts/sweep.py`` and ``scripts/pilot.py`` share is here too: the options both
end with (``add_run_options``), the ``TrainConfig`` those options name (``train_config``),
and their flow (``plan_and_run``): print the plan, stop there on ``--dry-run``, run the
grid with one progress line per cell, report failed cells on stderr with exit code 1, and
otherwise print the grid record before each script's own closing lines.

Moved from ``optspread.cli``, which re-exports it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast, get_args

from optspread.config import TrainConfig
from optspread.data import DataUnavailableError, SourceName, make_source
from optspread.experiment.grid import GridConfig, GridRecord, grid_cells
from optspread.experiment.schedule import IncompleteGridError, plan_cells, run_grid
from optspread.realdata import SPLITS, Split
from optspread.runs.rundir import JobSource, default_runs_root, source_root
from optspread.train import Trainer

SOURCES: tuple[str, ...] = get_args(JobSource)


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data source")
    group.add_argument(
        "--source",
        choices=SOURCES,
        default="synthetic",
        help="synthetic: the generator (default). optionmetrics: the real SPX panel under "
        f"$VEGALAB_DATA/panel; evaluation on {_dates('eval')}, PPO trained on "
        f"{_dates('train')} (pre-registration A8), runs under <runs root>/optionmetrics "
        "unless --root is given",
    )


def _dates(split: Split) -> str:
    first, last = SPLITS[split]
    return f"{first}..{last}"


def chosen_source(parser: argparse.ArgumentParser, args: argparse.Namespace) -> SourceName:
    """The source the flags ask for; exits when it is the real one and the panel is absent."""
    source = cast(SourceName, str(args.source))
    if source == "synthetic":
        return source
    data = make_source(source)
    if not data.available:
        try:
            data.provider()
        except DataUnavailableError as exc:
            parser.error(str(exc))
    return source


def runs_root(given: Path | None, source: SourceName, *parts: str) -> Path:
    """``--root`` when given, else ``parts`` under the source's folder of the default runs
    root: ``$VEGALAB_DATA/runs`` (or ``runs``) for the generator, its ``optionmetrics``
    folder for the real source."""
    if given is not None:
        return Path(given)
    return source_root(default_runs_root(), source).joinpath(*parts)


def describe(source: SourceName) -> str:
    if source == "synthetic":
        return source
    return f"{source}: trains on {_dates('train')}, evaluates on {_dates('eval')} (A8)"


def premium_note(source: SourceName) -> str:
    """Why the premium axis means nothing on the real source, for the commands to print."""
    if source == "synthetic":
        return ""
    return (
        "note: the planted premium is not applied on the real source (the surface carries "
        "the premium the market priced), so cells that differ only in premium see the same "
        "chains"
    )


# -- the grid commands' shared options and flow ------------------------------------ #


def add_run_options(parser: argparse.ArgumentParser, *, seed: int, train_seed: int) -> None:
    """The options ``scripts/sweep.py`` and ``scripts/pilot.py`` both end with, in this
    order, with each script's own seed defaults."""
    parser.add_argument("--seed", type=int, default=seed, help="first evaluation path seed")
    parser.add_argument(
        "--train-seed", type=int, default=train_seed, help="training seed of block 0"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")


def train_config(args: argparse.Namespace) -> TrainConfig:
    """The ``TrainConfig`` of ``--timesteps``, ``--train-seed`` and ``--device``."""
    return TrainConfig(
        total_timesteps=int(args.timesteps), seed=int(args.train_seed), device=args.device
    )


def plan_and_run(
    grid: GridConfig,
    root: Path,
    args: argparse.Namespace,
    trainer: Trainer,
    source: SourceName,
    finish: Callable[[GridRecord], None],
) -> int:
    """Print the plan on ``source``; unless ``--dry-run``, run the grid with ``--workers``
    and one progress line per cell, print its record, then ``finish(record)``.

    The exit code: 1 when a cell failed (listed on stderr), else 0."""
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
    finish(record)
    return 0
