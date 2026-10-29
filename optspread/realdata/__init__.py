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

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

DEFAULT_ROOT = (
    Path.home()
    / "Library/CloudStorage/OneDrive-UniversityofFlorida"
    / "Dupuy, Dominick's files - wrds_optionmetrics"
)

ROOTS = ("SPX", "SPXW")
OPTIONMETRICS_END = "2025-08-29"  # vendor ceiling at extraction, see docs/DATA.md

# The dates a strategy run may read (#28, pre-registration A8). Training and
# evaluation are disjoint ranges, so a policy is never scored on a day it trained
# on, and nothing from HOLDOUT_START on is reachable through a provider at all:
# there is no split for it to ask for. Market measurement (the VRP series, the
# dashboard's views) is not a strategy run and reads every date, through an API of
# its own (``provider.market_surface``) that no provider takes.
Split = Literal["train", "eval"]
SPLITS: dict[Split, tuple[date, date]] = {
    "train": (date(2010, 1, 1), date(2018, 12, 31)),
    "eval": (date(2019, 1, 1), date(2022, 8, 24)),
}
HOLDOUT_START = date(2022, 8, 25)

# The two files a provider reads from ``<root>/panel``.
PANEL_FILES = ("surface_grid.parquet", "spx_daily.parquet")
# The repository's copy of the team panel's manifest (scripts/build_spx_panel.py manifest).
REPO_MANIFEST = Path(__file__).resolve().parents[2] / "docs" / "DATA_MANIFEST.json"


def check_split(split: object) -> Split:
    """``split`` if it names one of ``SPLITS``; a ValueError for anything else.

    ``Split`` is a type hint and nothing checks a hint at runtime, so every entry to a
    provider calls this: ``None`` in particular is not "every date" (#28, A8).
    """
    if not isinstance(split, str) or split not in SPLITS:
        raise ValueError(
            f"split must be one of {sorted(SPLITS)} (pre-registration A8), got {split!r}; "
            "every date is for market measurement only (provider.market_surface)"
        )
    return split


@dataclass(frozen=True, slots=True)
class SplitRanges:
    """The dates each split covers on a surface of your own, in place of ``SPLITS``.

    Inclusive ``(first, last)`` per split, training wholly before evaluation. Only a
    source with a ``root`` of its own takes one (``optspread.data.OptionMetricsSource``,
    docs/EXTENDING.md): the team's SPX panel is read on ``SPLITS``, the dates A8 fixed,
    and a source over it refuses other ranges, so nothing from ``HOLDOUT_START`` on
    can be reached that way.
    """

    train: tuple[date, date]
    eval: tuple[date, date]

    def __post_init__(self) -> None:
        (a, b), (c, d) = self.train, self.eval
        if not a <= b < c <= d:
            raise ValueError(f"need train {a}..{b} wholly before eval {c}..{d}, each first <= last")

    def __getitem__(self, split: Split) -> tuple[date, date]:
        return self.train if split == "train" else self.eval

    def __str__(self) -> str:
        (a, b), (c, d) = self.train, self.eval
        return f"train {a}..{b}, eval {c}..{d}"


def data_root() -> Path:
    """The shared data folder: ``VEGALAB_DATA`` if set, else the team's synced folder."""
    env = os.environ.get("VEGALAB_DATA")
    return Path(env).expanduser() if env else DEFAULT_ROOT


# -- the team's panel, known by its content (#28) ------------------------------ #


@lru_cache(maxsize=64)
def _sha256(path: Path, size: int, mtime_ns: int) -> str:  # size and mtime key the cache
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def panel_digests(panel_dir: Path) -> dict[str, str]:
    """sha256 of each of ``PANEL_FILES`` present under ``panel_dir``, symlinks followed,
    keyed ``panel/<file>`` as the manifests key them."""
    out: dict[str, str] = {}
    for name in PANEL_FILES:
        f = (panel_dir / name).resolve()
        if f.is_file():
            st = f.stat()
            out[f"panel/{name}"] = _sha256(f, st.st_size, st.st_mtime_ns)
    return out


def team_digests() -> frozenset[str]:
    """Every panel file digest a team build recorded: ``docs/DATA_MANIFEST.json`` and the
    data root's ``MANIFEST.json`` (the build writes both; between a rebuild and its commit
    only the second has the new digests). A manifest that is absent or unreadable adds
    nothing."""
    found: set[str] = set()
    for manifest in (REPO_MANIFEST, data_root() / "MANIFEST.json"):
        try:
            files = json.loads(manifest.read_text())["files"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        found |= {
            str(v["sha256"])
            for k, v in files.items()
            if k.startswith("panel/") and isinstance(v, dict) and v.get("sha256")
        }
    return frozenset(found)


def is_team_panel(panel_dir: Path) -> bool:
    """Whether ``panel_dir`` holds the team's SPX panel: by path, or by content.

    By path: it is (or resolves to) ``panel`` under ``data_root()`` or ``DEFAULT_ROOT``.
    By content: either panel file, symlinks followed, has a digest a team manifest
    recorded (``team_digests``), so a copy, a symlinked folder or a symlinked file is the
    team's panel too, wherever it sits.
    """
    if panel_dir in (data_root() / "panel", DEFAULT_ROOT / "panel"):
        return True
    resolved = panel_dir.resolve()
    if resolved == (data_root() / "panel").resolve():
        return True
    digests = panel_digests(panel_dir)
    return bool(digests) and not set(digests.values()).isdisjoint(team_digests())


def refuse_team_ranges(panel_dir: Path, ranges: SplitRanges | None) -> None:
    """A ValueError when ``ranges`` would cut the team's panel on dates other than A8's."""
    if ranges is not None and is_team_panel(panel_dir):
        raise ValueError(
            f"date ranges of your own ({ranges}) are for a surface under a root of your own; "
            f"{panel_dir} is the team's SPX panel (by path or by its files' sha256), which "
            "serves only the splits pre-registration A8 fixed (optspread.realdata.SPLITS)"
        )
