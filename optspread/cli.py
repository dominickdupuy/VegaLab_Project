"""The data-source flags every command that launches runs takes, read the same way.

Lane: Ilan Danial (machinery).

The code now lives in ``optspread.experiment.cli``. This module re-exports it so existing
imports keep working.
"""

from optspread.experiment.cli import SOURCES as SOURCES
from optspread.experiment.cli import add_source_arguments as add_source_arguments
from optspread.experiment.cli import chosen_source as chosen_source
from optspread.experiment.cli import describe as describe
from optspread.experiment.cli import premium_note as premium_note
from optspread.experiment.cli import runs_root as runs_root
