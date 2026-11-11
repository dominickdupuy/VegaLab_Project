"""API payloads for the action mix: read from the files ``make actionmix`` writes, and one
run's mix view, which the run page starts and polls.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi`` (the action mix) and
``optspread.webviews`` (the run's mix view), which re-export it.
"""

from __future__ import annotations

import contextlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from optspread import actionmix as am
from optspread.api._base import Out, _f, _from_attrs, _source_suffix
from optspread.api.runwork import _ppo_reason, real_data_reason
from optspread.dashboard.runindex import RunRow
from optspread.dashboard.tasks import TaskState

# -- the action mix ---------------------------------------------------------------- #

_RUN_ID = re.compile(r"[0-9a-f]+-s\d+")
_GRID_HASH = re.compile(r"[0-9a-f]+")


class ActionMixOut(Out):
    """Which structures a policy trades (``optspread.actionmix``), read from the files
    ``scripts/action_mix.py`` wrote: one run, its episodes without their day-by-day
    lists, or one grid, per cell and pooled per arm and premium. Never computed here:
    a ppo run's re-simulation needs its learner, and a cell of 500 episodes takes
    half a minute."""

    kind: Literal["run", "grid"]  # the two branches of action_mix
    path: str
    run: am.RunActionMix | None = None
    grid: am.GridActionMix | None = None


def action_mix(
    root: Path | None = None,
    *,
    run: str | None = None,
    grid: str | None = None,
    source: str = "synthetic",
) -> ActionMixOut:
    """The action mix of ``run`` (a run id, or a run key ending in one) or of ``grid`` (a
    grid hash) under ``root`` (``actionmix.default_root()`` when None). ``KeyError`` when
    it has not been written; ``ValueError`` for a request that names neither or both, or
    a name that is not a run id or grid hash."""
    if (run is None) == (grid is None):
        raise ValueError("ask for one run or one grid")
    if source not in ("synthetic", "optionmetrics_stub", "optionmetrics"):
        raise ValueError(f"unknown source {source!r}")
    folder = root if root is not None else am.default_root()
    suffix = _source_suffix(source)
    if run is not None:
        rid = run.rstrip("/").rsplit("/", 1)[-1]
        if not _RUN_ID.fullmatch(rid):
            raise ValueError(f"{run!r} is not a run id")
        path = folder / f"{rid}{suffix}.json"
        if not path.is_file():
            raise KeyError(rid)
        return ActionMixOut(kind="run", path=str(path), run=am.read_mix(path).lean())
    assert grid is not None
    gh = grid.removeprefix("grid-").removesuffix(".json")
    if not _GRID_HASH.fullmatch(gh):
        raise ValueError(f"{grid!r} is not a grid hash")
    path = folder / f"grid-{gh}{suffix}.json"
    if not path.is_file():
        raise KeyError(gh)
    return ActionMixOut(kind="grid", path=str(path), grid=am.read_grid_mix(path))


# the action mix of one run, re-simulated on demand (optspread.actionmix)

MIX_DIR = "actionmix"  # under the attribution root: full mixes, each holding kept
MAX_HOLDINGS_SENT = 400


class HoldingOut(Out):
    """One structure from its open to the day the book was flat again."""

    seed: int
    structure: str
    open_day: int
    exit_day: int
    days: int
    exit: am.Exit  # how the structure left the book, as actionmix.Holding.exit records it
    tenor_days: int
    short_delta: float | None  # mean |delta| of the short legs at the open
    long_delta: float | None  # of the long wings; None without any
    credit: float
    margin: float
    strikes: list[float]


class EpisodeTimelineOut(Out):
    """One episode day by day: what was held over each move and what the trade did."""

    seed: int
    pnl: float
    held: list[str]
    events: list[str]
    holdings: list[HoldingOut]


class MixDetailOut(Out):
    n_holdings: int
    days_held: dict[str, list[int]]  # per structure: holdings by days held (index = days)
    holdings: list[HoldingOut]  # the first MAX_HOLDINGS_SENT, in episode order
    example: EpisodeTimelineOut | None  # the episode of median P&L


class MixViewOut(Out):
    # "cached" (full) | "lean" (summary only) | "computing" | "absent" | "failed": the
    # branches of mix_view, which alone builds one
    state: Literal["cached", "lean", "computing", "absent", "failed"]
    can_compute: bool
    reason: str
    file: str
    mix: am.RunActionMix | None  # lean: the summary and each episode's counts
    detail: MixDetailOut | None
    seconds: float | None = None


def mix_file(root: Path, row: RunRow) -> Path:
    """``<attribution root>/actionmix/<run_id>[.<source>].json``."""
    return root / MIX_DIR / f"{row.run_id}{_source_suffix(row.source)}.json"


def mix_reason(row: RunRow) -> str:
    """Why this run's action mix cannot be re-simulated here; empty when it can."""
    if row.status != "OK":
        return f"the run is {row.status}; the mix is re-simulated from files that match"
    return _ppo_reason(row) or real_data_reason(row)


def compute_mix(run_dir: Path, out: Path) -> Path:
    """``actionmix.mix_run`` on the run, every holding kept, written to ``out``."""
    return am.write_mix(am.mix_run(run_dir), out, episodes=True)


def _holding(seed: int, h: am.Holding) -> HoldingOut:
    shorts, longs = h.abs_deltas(short=True), h.abs_deltas(short=False)
    return _from_attrs(
        HoldingOut,
        h,
        seed=seed,
        short_delta=_f(sum(shorts) / len(shorts)) if shorts else None,
        long_delta=_f(sum(longs) / len(longs)) if longs else None,
        strikes=sorted(leg.strike for leg in h.legs),
    )


def mix_detail(mix: am.RunActionMix) -> MixDetailOut:
    """What the summary pools away: holding periods, each holding, and one episode."""
    names = mix.action_names
    held = [_holding(e.seed, h) for e in mix.episodes for h in e.holdings]
    longest = max((h.days for h in held), default=0)
    days_held = {s: [0] * (longest + 1) for s in am.STRUCTURES}
    for h in held:
        days_held.setdefault(h.structure, [0] * (longest + 1))[h.days] += 1
    example = None
    if mix.episodes:
        ranked = sorted(mix.episodes, key=lambda e: (e.pnl, e.seed))
        e = ranked[(len(ranked) - 1) // 2]
        example = EpisodeTimelineOut(
            seed=e.seed,
            pnl=e.pnl,
            held=[names[i] if 0 <= i < len(names) else str(i) for i in e.held],
            events=list(e.events),
            holdings=[_holding(e.seed, h) for h in e.holdings],
        )
    return MixDetailOut(
        n_holdings=len(held),
        days_held=days_held,
        holdings=held[:MAX_HOLDINGS_SENT],
        example=example,
    )


@lru_cache(maxsize=16)
def _read_full_mix(path: str, stamp: int) -> tuple[am.RunActionMix, MixDetailOut]:
    mix = am.read_mix(Path(path))
    return mix.lean(), mix_detail(mix)


def mix_view(root: Path, row: RunRow, task: TaskState, lean_root: Path | None = None) -> MixViewOut:
    """The run's mix: the full one cached under ``root`` (the attribution root); else the
    lean one ``scripts/action_mix.py`` wrote under ``lean_root``, its summary without the
    holdings, while the full one is computed or when it cannot be."""
    reason = mix_reason(row)
    full = mix_file(root, row)
    base = {"can_compute": not reason, "reason": reason, "file": str(full)}
    if full.is_file() and task.state != "running":
        try:
            mix, detail = _read_full_mix(str(full), full.stat().st_mtime_ns)
        except (OSError, ValueError) as exc:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            return MixViewOut(state="failed", **{**base, "reason": text}, mix=None, detail=None)
        return MixViewOut(state="cached", **base, mix=mix, detail=detail)
    lean = None
    folder = lean_root if lean_root is not None else am.default_root()
    lean_path = folder / f"{row.run_id}{_source_suffix(row.source)}.json"
    if lean_path.is_file():
        with contextlib.suppress(OSError, ValueError):
            lean = am.read_mix(lean_path).lean()
    if task.state == "running":
        return MixViewOut(state="computing", **base, mix=lean, detail=None, seconds=task.seconds)
    if task.state == "failed":
        return MixViewOut(state="failed", **{**base, "reason": task.error}, mix=lean, detail=None)
    if lean is not None:
        return MixViewOut(state="lean", **{**base, "file": str(lean_path)}, mix=lean, detail=None)
    return MixViewOut(state="absent", **base, mix=None, detail=None)
