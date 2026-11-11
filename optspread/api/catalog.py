"""API payload for a designed matrix of runs: ``<runs root>/catalog.json``, optional.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from optspread.api._base import Out
from optspread.api.premiums import _interval
from optspread.api.runs import RunStatus, row_out, run_source
from optspread.artifacts import SchemaVersionError, read_run
from optspread.dashboard.jobs.spec import JobSpec
from optspread.dashboard.runindex import RunRow

# -- a designed matrix of runs: <runs root>/catalog.json, optional ------------------- #

CATALOG_FILENAME = "catalog.json"
_NOT_A_VARIANT = ("iv_premium_vol_pts", "seed")


class CatalogRunOut(Out):
    key: str
    label: str
    settings: dict[str, Any]
    present: bool  # the run is in the index under this key
    status: RunStatus | Literal["MISSING"]  # the run's status, or MISSING when not present
    policy: str
    premium: float | None
    source: str
    n_episodes: int | None
    mean_pnl: float | None
    se_pnl: float | None


class CatalogPairOut(Out):
    """One click to compare: a run against the group's baseline at the same premium, with
    the paired difference b - a over the seeds both ran and its two-sided 95% interval."""

    a: str  # the baseline's key
    b: str
    title: str  # "<b's label> against <a's label>"
    varies: list[str]  # the settings that differ
    premium: float | None
    n_shared: int = 0
    diff_mean: float | None = None
    diff_ci_low: float | None = None
    diff_ci_high: float | None = None
    clear: bool = False  # the interval excludes zero


class CatalogGroupOut(Out):
    group: str
    question: str
    runs: list[CatalogRunOut]
    pairs: list[CatalogPairOut]


class CatalogOut(Out):
    available: bool
    file: str
    seed: int | None = None
    groups: list[CatalogGroupOut] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)  # run key -> label
    questions: dict[str, str] = Field(default_factory=dict)  # run key -> its group's question
    groups_of: dict[str, str] = Field(default_factory=dict)  # run key -> group
    problem: str = ""  # why a catalog that is there did not read


def _distance(settings: dict[str, Any]) -> int:
    """How many settings sit away from the launch defaults (the premium and seed aside)."""
    defaults = JobSpec().model_dump(mode="json")
    return sum(
        1
        for k, v in settings.items()
        if k in defaults and k not in _NOT_A_VARIANT and defaults[k] != v
    )


def _variant(settings: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in settings.items() if k not in _NOT_A_VARIANT}


def _pairs(runs: list[CatalogRunOut]) -> list[CatalogPairOut]:
    """Within a group: at each premium, every variant against the one nearest the
    defaults; a group whose runs differ in the premium only, each premium against zero."""
    present = [r for r in runs if r.present]
    variants = {json.dumps(_variant(r.settings), sort_keys=True, default=str) for r in present}
    out: list[CatalogPairOut] = []
    if len(variants) <= 1:
        nulls = [r for r in present if (r.premium or 0.0) == 0.0]
        for r in present:
            if nulls and r is not nulls[0] and (r.premium or 0.0) != 0.0:
                out.append(
                    CatalogPairOut(
                        a=nulls[0].key,
                        b=r.key,
                        title=f"{r.label} against {nulls[0].label}",
                        varies=["iv_premium_vol_pts"],
                        premium=r.premium,
                    )
                )
        return out
    levels: dict[float | None, list[CatalogRunOut]] = {}
    for r in present:
        levels.setdefault(r.premium, []).append(r)
    for premium in sorted(levels, key=lambda p: (p is None, p or 0.0)):
        at = levels[premium]
        base = min(at, key=lambda r: (_distance(r.settings), r.key))
        for r in at:
            if r is base:
                continue
            keys = sorted(set(r.settings) | set(base.settings))
            varies = [
                k
                for k in keys
                if k not in _NOT_A_VARIANT and r.settings.get(k) != base.settings.get(k)
            ]
            out.append(
                CatalogPairOut(
                    a=base.key,
                    b=r.key,
                    title=f"{r.label} against {base.label}",
                    varies=varies,
                    premium=premium,
                )
            )
    return out


def _paired(pair: CatalogPairOut, a: dict[int, float], b: dict[int, float]) -> CatalogPairOut:
    shared = sorted(set(a) & set(b))
    d = np.asarray([b[s] - a[s] for s in shared], dtype=np.float64)
    if len(d) < 2:
        return pair.model_copy(update={"n_shared": len(d)})
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    lo, hi = _interval(mean, se, len(d))
    return pair.model_copy(
        update={
            "n_shared": len(d),
            "diff_mean": mean,
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "clear": lo is not None and hi is not None and (lo > 0 or hi < 0),
        }
    )


def catalog(root: Path, rows: Sequence[RunRow]) -> CatalogOut:
    """The runs root's ``catalog.json``: the runs grouped by the question each answers,
    with their labels and the comparisons that answer it. ``available`` is False when the
    root has none, and every page then reads as it would without it."""
    path = root / CATALOG_FILENAME
    if not path.is_file():
        return CatalogOut(available=False, file=str(path))
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        entries = list(raw["runs"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return CatalogOut(available=False, file=str(path), problem=str(exc).splitlines()[0])
    by_key = {row_out(r, root).key: r for r in rows}
    groups: dict[str, CatalogGroupOut] = {}
    labels: dict[str, str] = {}
    questions: dict[str, str] = {}
    groups_of: dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict) or "key" not in e:
            continue
        key = str(e["key"]).strip("/")
        group = str(e.get("group", "other"))
        question = str(e.get("question", ""))
        label = str(e.get("label", key))
        given = e.get("settings")
        settings: dict[str, Any] = given if isinstance(given, dict) else {}
        row = by_key.get(key)
        run = CatalogRunOut(
            key=key,
            label=label,
            settings=settings,
            present=row is not None,
            status=row.status if row else "MISSING",
            policy=row.policy if row else str(settings.get("policy", "")),
            premium=row.premium if row else settings.get("iv_premium_vol_pts"),
            source=run_source(row) if row else str(settings.get("source", "synthetic")),
            n_episodes=row.n_episodes if row else settings.get("n_episodes"),
            mean_pnl=row.mean_pnl if row else None,
            se_pnl=row.se_pnl if row else None,
        )
        held = groups.setdefault(
            group, CatalogGroupOut(group=group, question=question, runs=[], pairs=[])
        )
        held.runs.append(run)
        labels.setdefault(key, label)  # a run in several groups keeps its first label
        questions.setdefault(key, question or held.question)
        groups_of.setdefault(key, group)
    episodes: dict[str, dict[int, float]] = {}

    def pnl(key: str) -> dict[int, float]:
        if key not in episodes:
            try:
                episodes[key] = {e.seed: e.pnl for e in read_run(by_key[key].path).episodes}
            except (KeyError, OSError, ValueError, SchemaVersionError):
                episodes[key] = {}
        return episodes[key]

    out = [
        g.model_copy(update={"pairs": [_paired(p, pnl(p.a), pnl(p.b)) for p in _pairs(g.runs)]})
        for g in groups.values()
    ]
    seed = raw.get("seed") if isinstance(raw, dict) else None
    return CatalogOut(
        available=True,
        file=str(path),
        seed=int(seed) if isinstance(seed, int) else None,
        groups=out,
        labels=labels,
        questions=questions,
        groups_of=groups_of,
    )
