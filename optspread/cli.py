"""The data-source flags every command that launches runs takes, read the same way.

Lane: Ilan Danial (machinery).

    --source {synthetic,optionmetrics}   the generator (default), or the real SPX panel

On the real source a run's dates are not a flag: it is evaluated on the ``eval`` split
and a learned arm trained on ``train`` (``optspread.realdata.SPLITS``), the dates
pre-registration amendment A8 fixed, and nothing from ``HOLDOUT_START`` on is read. A
real-data run lands under ``<runs root>/optionmetrics`` unless ``--root`` says otherwise
(``optspread.orchestrate.source_root``), because the source is not in the run id.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast, get_args

from optspread.data import DataUnavailableError, SourceName, make_source
from optspread.jobs import JobSource
from optspread.orchestrate import default_runs_root, source_root
from optspread.realdata import SPLITS, Split

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
