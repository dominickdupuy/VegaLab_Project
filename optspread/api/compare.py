"""API payload for two runs side by side: their difference, seed by seed, and its verdict.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi``, which re-exports it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from optspread.api._base import Out
from optspread.api.catalog import CatalogOut
from optspread.api.premiums import _interval
from optspread.api.runs import (
    HistogramOut,
    RunDetailOut,
    histogram,
    row_out,
    run_detail,
    run_source,
)
from optspread.dashboard.runindex import RunDetail, RunRow, compare_runs


class DiffOut(Out):
    key: str
    a: Any
    b: Any


class SeedDiffOut(Out):
    seed: int
    a: float
    b: float
    diff: float | None  # b - a; null when it is not finite (two P&Ls that overflow apart)


class CompareOut(Out):
    a: RunDetailOut
    b: RunDetailOut
    config_diff: list[DiffOut]
    shared_seeds: int
    diff_mean: float | None
    diff_se: float | None
    diff_t: float | None
    diff_histogram: HistogramOut
    # the paired difference's two-sided 95% interval (t on the shared seeds) and each seed
    diff_df: int | None = None
    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    b_better_share: float | None = None  # share of shared seeds where B made more than A
    diffs: list[SeedDiffOut] = Field(default_factory=list)
    verdict: str = ""  # one plain sentence
    # from the runs root's catalog.json, when it lists the runs: what each one is, and the
    # question the two answer when they are in one group
    a_label: str = ""
    b_label: str = ""
    group: str = ""
    question: str = ""


def _label(row: RunRow) -> str:
    names = {
        "delta_seller": "the fixed-delta seller",
        "ppo": "PPO",
        "always_on": "the always-on condor",
        "random": "the random policy",
        "flat": "flat",
    }
    arm = names.get(row.policy, row.policy or "a run")
    where = (
        "real SPX history"
        if run_source(row) not in ("", "synthetic")
        else (
            f"{row.premium:g} vol pt{'' if row.premium == 1 else 's'} planted"
            if row.premium is not None
            else "the generator"
        )
    )
    return f"{arm} ({where})"


MINUS = "\u2212"  # the typographic minus the pages print


def _usd(x: float) -> str:
    """Dollars as the pages print them, with a true minus: "-$169" as U+2212, "$1,051"."""
    return f"{MINUS if x < 0 and round(x) != 0 else ''}${abs(x):,.0f}"


def compare_verdict(
    a: RunRow,
    b: RunRow,
    n: int,
    mean: float | None,
    lo: float | None,
    hi: float | None,
    names: tuple[str, str] = ("", ""),
) -> str:
    """One sentence a reader can act on: which run made more on the same paths, and whether
    the interval says the gap is real. ``names`` are the catalog's labels, when it has them."""
    na, nb = names[0] or _label(a), names[1] or _label(b)
    if n == 0 or mean is None:
        return "The two runs share no seeds, so they cannot be compared episode by episode."
    paths = f"the same {n} path{'s' if n != 1 else ''}"
    if lo is None or hi is None:
        return f"On {paths} {nb} less {na} is {_usd(mean)} per episode; too few to say more."
    if lo == hi == mean:
        return f"{nb} and {na} made exactly the same on all {n} shared paths."
    learned = "ppo" in (a.policy, b.policy)
    caveat = " (one trained policy each: a retrain could move this)" if learned else ""
    half = (hi - lo) / 2.0
    gap = (
        f"{nb} made {_usd(abs(mean))} {'more' if mean >= 0 else 'less'} per episode than "
        f"{na} on {paths} (± {_usd(half)}, 95%)"
    )
    if lo > 0 or hi < 0:
        return f"{gap}: a clear difference{caveat}."
    return f"{gap}: not distinguishable from noise at this sample size; run more episodes{caveat}."


def compare(
    a: RunDetail,
    b: RunDetail,
    root: Path,
    index: Sequence[RunRow],
    listed: CatalogOut | None = None,
    group: str = "",
) -> CompareOut:
    """``a`` and ``b`` paired on their shared seeds. With the runs root's catalog, the
    labels and the question come from ``group`` when both runs are in it, else from the
    first group that lists both (a baseline run can sit in several)."""
    cmp = compare_runs(a, b)
    ka, kb = row_out(a.row, root).key, row_out(b.row, root).key
    both = [
        g
        for g in (listed.groups if listed is not None else [])
        if {ka, kb} <= {r.key for r in g.runs}
    ]
    chosen = next((g for g in both if g.group == group), both[0] if both else None)
    labels = (
        {r.key: r.label for r in chosen.runs}
        if chosen is not None
        else (listed.labels if listed is not None else {})
    )
    names = (labels.get(ka, ""), labels.get(kb, ""))
    pa = {e.seed: e.pnl for e in a.record.episodes} if a.record else {}
    pb = {e.seed: e.pnl for e in b.record.episodes} if b.record else {}
    diffs = np.asarray([pb[s] - pa[s] for s in cmp.shared_seeds], dtype=np.float64)
    n = len(diffs)
    lo, hi = (
        _interval(float(cmp.diff_mean), float(cmp.diff_se), n)
        if cmp.diff_mean is not None and cmp.diff_se is not None
        else (None, None)
    )
    mean = cmp.diff_mean
    # the verdict reads a mean difference that is not finite (two P&Ls that overflow
    # apart) as no mean at all
    verdict_mean = mean if mean is not None and math.isfinite(mean) else None
    return CompareOut(
        a=run_detail(a, root, index),
        b=run_detail(b, root, index),
        config_diff=[DiffOut(key=k, a=va, b=vb) for k, va, vb in cmp.config_diff],
        shared_seeds=len(cmp.shared_seeds),
        diff_mean=cmp.diff_mean,
        diff_se=cmp.diff_se,
        diff_t=cmp.diff_t,
        diff_histogram=histogram(diffs),
        diff_df=n - 1 if n > 1 else None,
        diff_ci_low=lo,
        diff_ci_high=hi,
        b_better_share=float((diffs > 0).mean()) if n else None,
        diffs=[SeedDiffOut(seed=s, a=pa[s], b=pb[s], diff=pb[s] - pa[s]) for s in cmp.shared_seeds],
        verdict=compare_verdict(a.row, b.row, n, verdict_mean, lo, hi, names),
        a_label=names[0],
        b_label=names[1],
        group=chosen.group if chosen is not None else "",
        question=chosen.question if chosen is not None else "",
    )
