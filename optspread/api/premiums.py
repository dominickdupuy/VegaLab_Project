"""API payload for the runs list read by planted premium: P&L per arm, and each arm against
itself at zero.

Lane: Ilan Danial (machinery). Moved from ``optspread.webapi``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from optspread import power as pw
from optspread.analysis import ALPHA, NULL_PREMIUM, decide_contrast
from optspread.api._base import Out
from optspread.api.runs import RunFilters, _filtered, row_out, run_source
from optspread.artifacts import SchemaVersionError, read_run
from optspread.dashboard.runindex import RunRow, flatten_config
from optspread.manifest import RunManifest

# -- the runs list: P&L by planted premium, per arm, and each arm against itself at zero -- #

# A world is a run's config less what varies inside a comparison: the arm, the seeds, the
# episode count and the premium. Runs of one arm in one world at two premiums differ only
# in the premium, so their episodes on the same seeds are paired.
_WORLD_DROP = ("policy", "seed", "n_episodes")
_PLAIN_KEYS: dict[str, Callable[[Any], str]] = {
    "cost.enabled": lambda v: "costs on" if v else "costs off",
    "market.n_days": lambda v: f"{v}-day path",
    "env.episode_days": lambda v: f"{v}-day episodes",
    "market.sigma": lambda v: f"path vol {float(v):g}",
    "env.short_delta": lambda v: f"short delta {float(v):g}",
    "env.wing_delta": lambda v: f"wing delta {float(v):g}",
    "env.roll_dte": lambda v: f"roll at {v} days",
    "market.tenor_days": lambda v: f"tenors {', '.join(str(x) for x in v)}",
    "env.initial_cash": lambda v: f"cash ${float(v):,.0f}",
    "env.expiry_idx": lambda v: "opens the nearest tenor" if v == 0 else f"opens tenor {v}",
    "cost.commission_per_contract": lambda v: f"commission ${float(v):g}",
    "cost.half_spread_frac": lambda v: f"half-spread {float(v):.1%} of mid",
    "cost.min_half_spread": lambda v: f"half-spread floor {float(v):g}",
}
P_LEVELS = (5, 25, 50, 75, 95)


class WorldOut(Out):
    id: str
    label: str  # what sets it apart from the most common world here; "" for that one
    source: str
    n_runs: int
    arms: list[str]


class PremiumGroupOut(Out):
    """Every episode of one arm in one world at one premium, pooled over its runs."""

    arm: str
    variant: str  # a learned arm's training budget; "" for a scripted arm
    world: str
    source: str
    premium: float
    n_runs: int
    n_episodes: int
    mean: float | None
    se: float | None
    ci_low: float | None  # two-sided 95%
    ci_high: float | None
    # "runs": t on the run means; "episodes": one run, t on its episodes (_group's two)
    interval: Literal["runs", "episodes"]
    p5: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    run_keys: list[str]


class PremiumContrastOut(Out):
    """One arm at premium p against itself at zero, on the seeds both ran: the detection
    event of ``analysis.decide_premium`` when two or more runs pair up."""

    arm: str
    variant: str
    world: str
    source: str
    premium: float
    n_blocks: int  # run pairs on the same first seed
    n_pairs: int  # episodes paired, over every block
    estimate: float | None
    se: float | None
    df: int | None
    ci_low: float | None  # two-sided 95%
    ci_high: float | None
    lower_bound: float | None  # one-sided 95%, the pre-registered bound (D1)
    detected: bool
    # "runs" (the rule), "episodes" (one pair: no between-run term), or "" with a reason
    interval: Literal["", "runs", "episodes"]
    reason: str = ""


class PremiumsOut(Out):
    n_runs: int  # OK runs read
    n_skipped: int  # OK runs whose files did not read
    worlds: list[WorldOut]
    groups: list[PremiumGroupOut]
    contrasts: list[PremiumContrastOut]


def _world(config: dict[str, Any], source: str) -> tuple[str, dict[str, Any]]:
    flat = flatten_config({k: v for k, v in config.items() if k not in _WORLD_DROP})
    flat.pop("market.iv_premium_vol_pts", None)
    flat = {k: v for k, v in flat.items() if not k.startswith("train.")}  # a variant, not a world
    flat["source"] = source
    blob = json.dumps(flat, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10], flat


def _world_label(flat: dict[str, Any], main: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in sorted(set(flat) | set(main)):
        if k == "source" or k.startswith("train.") or flat.get(k) == main.get(k):
            continue
        v = flat.get(k)
        parts.append(_PLAIN_KEYS[k](v) if k in _PLAIN_KEYS and v is not None else f"{k} {v}")
    return ", ".join(parts)


def _arm_variant(config: dict[str, Any]) -> str:
    """What sets a learned arm apart within a world: its training budget."""
    train = config.get("train")
    if isinstance(train, dict) and train.get("total_timesteps"):
        return f"{int(train['total_timesteps']):,} training steps"
    return ""


def _t95(n: int) -> float:
    return pw.t_critical(n, 0.025)


def _interval(mean: float, se: float, n: int) -> tuple[float | None, float | None]:
    if n < 2 or not math.isfinite(se):
        return None, None
    half = _t95(n) * se
    lo, hi = float(mean - half), float(mean + half)
    # an end that is not finite (it overflowed) is no end: the compare verdict and a
    # catalog pair's "clear" read None, and the page gets null
    return (lo if math.isfinite(lo) else None), (hi if math.isfinite(hi) else None)


def _group(
    arm: str,
    variant: str,
    world: str,
    source: str,
    premium: float,
    runs: list[tuple[str, int, dict[int, float]]],
) -> PremiumGroupOut:
    pooled = np.asarray([p for _, _, eps in runs for p in eps.values()], dtype=np.float64)
    means = np.asarray([np.mean(list(eps.values())) for _, _, eps in runs if eps])
    n = len(pooled)
    mean = float(pooled.mean()) if n else math.nan
    if len(means) >= 2:
        se, k, kind = float(means.std(ddof=1) / math.sqrt(len(means))), len(means), "runs"
    else:
        se = float(pooled.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        k, kind = n, "episodes"
    lo, hi = _interval(mean, se, k)
    q = np.percentile(pooled, P_LEVELS) if n else [math.nan] * len(P_LEVELS)
    return PremiumGroupOut(
        arm=arm,
        variant=variant,
        world=world,
        source=source,
        premium=premium,
        n_runs=len(runs),
        n_episodes=n,
        mean=mean,
        se=se,
        ci_low=lo,
        ci_high=hi,
        interval=kind,
        p5=q[0],
        p25=q[1],
        p50=q[2],
        p75=q[3],
        p95=q[4],
        run_keys=[key for key, _, _ in runs],
    )


def _premium_contrast(
    arm: str,
    variant: str,
    world: str,
    source: str,
    premium: float,
    treated: list[tuple[str, int, dict[int, float]]],
    null: list[tuple[str, int, dict[int, float]]],
) -> PremiumContrastOut:
    base = {"arm": arm, "variant": variant, "world": world, "source": source, "premium": premium}
    at_null = {first: eps for _, first, eps in null}
    blocks = []
    for _, first, eps in sorted(treated, key=lambda r: r[1]):
        other = at_null.get(first)
        if other is None:
            continue
        shared = sorted(set(eps) & set(other))
        if shared:
            blocks.append([eps[s] - other[s] for s in shared])
    empty = {
        "n_blocks": len(blocks),
        "n_pairs": sum(len(b) for b in blocks),
        "estimate": None,
        "se": None,
        "df": None,
        "ci_low": None,
        "ci_high": None,
        "lower_bound": None,
        "detected": False,
    }
    if not blocks:
        reason = "no run at zero premium on the same seeds" if null else "no run at zero premium"
        return PremiumContrastOut(**base, **empty, interval="", reason=reason)
    if len(blocks) >= 2:
        n = min(len(b) for b in blocks)  # a balanced matrix: the first n shared seeds of each
        decision = decide_contrast(np.asarray([b[:n] for b in blocks], dtype=np.float64))
        lo, hi = _interval(decision.estimate, decision.se, len(blocks))
        return PremiumContrastOut(
            **base,
            n_blocks=len(blocks),
            n_pairs=n * len(blocks),
            estimate=decision.estimate,
            se=decision.se,
            df=decision.df,
            ci_low=lo,
            ci_high=hi,
            lower_bound=decision.lower_bound,
            detected=decision.rejected,
            interval="runs",
        )
    d = np.asarray(blocks[0], dtype=np.float64)
    if len(d) < 2:
        return PremiumContrastOut(**base, **empty, interval="", reason="one paired episode")
    est, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    lo, hi = _interval(est, se, len(d))
    bound = est - pw.t_critical(len(d), ALPHA) * se
    return PremiumContrastOut(
        **base,
        n_blocks=1,
        n_pairs=len(d),
        estimate=est,
        se=se,
        df=len(d) - 1,
        ci_low=lo,
        ci_high=hi,
        lower_bound=bound,
        detected=bool(bound > 0),
        interval="episodes",
    )


def premium_view(rows: Sequence[RunRow], root: Path, filters: RunFilters) -> PremiumsOut:
    """The OK runs ``filters`` shows, as P&L distributions per arm and premium in each
    world, and each arm against itself at zero premium on shared seeds (``#26``)."""
    read: list[tuple[str, str, str, str, float, str, int, dict[int, float], dict[str, Any]]] = []
    skipped = 0
    for row in _filtered(rows, filters):
        if row.status != "OK" or row.premium is None:
            continue
        try:
            config = RunManifest.read(row.path).config
            record = read_run(row.path)
        except (OSError, ValueError, SchemaVersionError):
            skipped += 1
            continue
        dumped = config.model_dump(mode="json")
        world, flat = _world(dumped, run_source(row))
        read.append(
            (
                row_out(row, root).key,
                str(config.policy),
                _arm_variant(dumped),
                world,
                float(config.market.iv_premium_vol_pts),
                run_source(row),
                int(config.seed),
                {e.seed: e.pnl for e in record.episodes},
                flat,
            )
        )
    counts: dict[str, int] = {}
    flats: dict[str, dict[str, Any]] = {}
    arms: dict[str, set[str]] = {}
    for _, arm, _, world, _, _, _, _, flat in read:
        counts[world] = counts.get(world, 0) + 1
        flats[world] = flat
        arms.setdefault(world, set()).add(arm)
    main = flats[max(counts, key=lambda w_: (counts[w_], w_))] if counts else {}
    cells: dict[tuple[str, str, str, str, float], list[tuple[str, int, dict[int, float]]]] = {}
    for key, arm, variant, world, premium, source, first, eps, _ in read:
        cells.setdefault((arm, variant, world, source, premium), []).append((key, first, eps))
    groups = [_group(*k, v) for k, v in sorted(cells.items())]
    contrasts = [
        _premium_contrast(
            arm,
            variant,
            world,
            source,
            premium,
            runs,
            cells.get((arm, variant, world, source, 0.0), []),
        )
        for (arm, variant, world, source, premium), runs in sorted(cells.items())
        if premium != NULL_PREMIUM and source == "synthetic"
    ]
    worlds = [
        WorldOut(
            id=w_,
            label=_world_label(flats[w_], main),
            source=str(flats[w_]["source"]),
            n_runs=counts[w_],
            arms=sorted(arms[w_]),
        )
        for w_ in sorted(counts, key=lambda w_: -counts[w_])
    ]
    return PremiumsOut(
        n_runs=len(read), n_skipped=skipped, worlds=worlds, groups=groups, contrasts=contrasts
    )
