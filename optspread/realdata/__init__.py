"""Real SPX option data: extraction, the stitched panel, and the loader behind the adapter.

Lane: Hemdutt Rao (data adapter).

Two vendors, one schema. WRDS OptionMetrics (``opprcd``, ``secprd``) runs from
2010-01-04 to its vendor ceiling, 2025-08-29. Databento OPRA.PILLAR end-of-day
consolidated quotes run from 2025-07-01 on. The two overlap for July and August
2025, which is where the seam is checked before the cut-over.

Nothing in here is imported by the simulator. The sweep stays on the synthetic
null world (decision D2); real data enters through ``optspread.data`` as a
``ChainSource`` like any other, and through the VRP series the dashboard reads.

Market data never enters version control. The repository versions this code, the
schema in ``docs/DATA.md`` and a checksummed manifest; the files themselves live in
the team's shared data folder, named by ``VEGALAB_DATA``.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

DEFAULT_ROOT = (
    Path.home()
    / "Library/CloudStorage/OneDrive-UniversityofFlorida"
    / "Dupuy, Dominick's files - wrds_optionmetrics"
)

ROOTS = ("SPX", "SPXW")
OPTIONMETRICS_END = "2025-08-29"  # vendor ceiling at extraction, see docs/DATA.md

# The dates a strategy run may read (#28). Training and evaluation are disjoint
# ranges, so a policy is never scored on a day it trained on, and nothing after
# HOLDOUT_START - 1 is reachable through a provider at all: there is no split
# for it to ask for. Market measurement (the VRP series) is not a strategy run
# and reads every date.
Split = Literal["train", "eval"]
SPLITS: dict[Split, tuple[date, date]] = {
    "train": (date(2010, 1, 1), date(2018, 12, 31)),
    "eval": (date(2019, 1, 1), date(2022, 8, 24)),
}
HOLDOUT_START = date(2022, 8, 25)


def data_root() -> Path:
    """The shared data folder: ``VEGALAB_DATA`` if set, else the team's synced folder."""
    env = os.environ.get("VEGALAB_DATA")
    return Path(env).expanduser() if env else DEFAULT_ROOT
