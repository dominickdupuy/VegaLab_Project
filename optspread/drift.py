"""The drift sensitivity: the confirmatory sweep re-simulated in the fair world, post hoc.

Lane: Hemdutt Rao (market layer).

**The bias.** The study's synthetic path is a zero-drift GBM (spot is a martingale) while
its chain was priced by Black-Scholes at ``r = 0.04`` and the book earns nothing on cash,
so a contract was not priced at its expected payoff under the path: a put was priced
off a forward above the spot the path keeps, cheaper than it pays out, a call dearer.
A seller of puts lost the carry, before costs, in the null world. The fair world is
``r = 0`` on the same path (forward = spot, no discounting): every contract is priced at
its expected payoff, which ``tests/core/test_drift.py`` checks by Monte Carlo.

**What this measures** (pre-registration amendment A11; post hoc, not pre-registered).
Every cell of a grid is re-simulated twice from its manifest, on the same paths (the
same seeds), with the same policy: the scripted arm as the harness builds it, a learned
arm with the ``policy.zip`` it saved, loaded and never retrained (``load_policy``, as
``optspread.attribution`` and ``optspread.actionmix`` load it). First in the world it
ran in, each episode checked against its ``run.json`` row (``agrees`` within
``actionmix.PNL_RTOL`` of the P&L and the cost with the same trades and steps, as the
action mix and the edge attribution check it; ``reproduces`` bit for bit), then in the
fair world, its config with ``market.r = 0`` and nothing else changed. A cell that
neither reproduces nor agrees is reported with its differing episodes, not refused:
a PPO cell trained on another platform can take one discrete branch the other way
(docs/DEFECTS.md, "One edge episode differs").

The M4 analysis (``optspread.m4analysis.analyse_arm`` and ``gap_between``, the functions
``scripts/m4_analysis.py`` runs) is then run on three sets of per-episode P&L: the saved
``run.json`` rows (the headline), the original world re-simulated here, and the fair
world. Original against fair is the comparison, both re-simulated on one machine, so
any platform difference is in both. The bootstrap check runs as M4 runs it; the curve,
the floor and the gap are t-based whichever rule is in force, as in M4.

Nothing is written into the sweep's root. The JSON (``render_json``)::

    {"schema_version": 1, "kind": "drift_sensitivity", "post_hoc": true,
     "pre_registered": false, "statement": str, "report_date": str,
     "provenance": {"root", "grid_file", "grid_hash", "analysis_git_sha",
                    "analysis_git_dirty", "sweep_git_shas", "versions", "workers",
                    "seconds"},
     "worlds": {"<world>": {"r": float | null, "what": str}},   saved, original, fair
     "reproduction": {"rtol", "cells", "reproduce", "agree", "differ",
                      "differing": [{"run_id", "arm", "premium", "block", "seed",
                                     "saved_pnl", "pnl", "saved_cost", "cost"}]},
     "arms": {"<arm>": {"means": [{"premium", "saved", "original", "fair",
                                  "trades_original", "trades_fair", "trades_changed"}],
                        "worlds": {"<world>": {"levels": [{"premium", "estimate", "se",
                                     "t", "lower_bound", "detected"}],
                                   "curve": [{"premium", "measured", "noncentrality",
                                     "noncentrality_lo", "noncentrality_hi", "power",
                                     "power_lo", "power_hi"}],
                                   "floor": {"premium", "lo", "hi", "status",
                                     "bracket", "max_power"},
                                   "rule_in_force": "t" | "bootstrap",
                                   "diverged_at": [float]}}}},
     "gap": {"<world>": {"value", "kind", "band_range", "bracket"}},
     "changes": {"verdicts": [str], "floors": [str], "gap": str, "rule": [str]},
     "cells": [{"premium", "arm", "block", "run_id", "fair_run_id", "reproduces",
                "agrees", "max_abs_pnl_diff", "mean_saved", "mean_original",
                "mean_fair", "trades_changed", "seconds"}]}
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from optspread.actionmix import PNL_RTOL, find_grid
from optspread.analysis import NULL_PREMIUM
from optspread.artifacts import read_run
from optspread.attribution import PolicyLoader, load_sb3_policy
from optspread.config import ArmName, RunConfig
from optspread.experiment.grid import GridCell, GridRecord, read_grid
from optspread.harness import EpisodeResult, run_episodes
from optspread.m4analysis import (
    ArmAnalysis,
    Gap,
    analyse_arm,
    gap_between,
)
from optspread.manifest import MANIFEST_FILENAME, RunManifest
from optspread.policies import Policy
from optspread.runs.rundir import changed_artifacts, run_id
from optspread.train import POLICY_FILENAME

SCHEMA_VERSION = 1
FAIR_R = 0.0  # forward = spot under the zero-drift path: every contract at its expected payoff
WORLDS = ("saved", "original", "fair")
WORLD_TEXT = {
    "saved": "the run.json rows, as the sweep wrote them (the headline, docs/M4_ANALYSIS.md)",
    "original": "the recorded config re-simulated here (the study's world, r = 0.04)",
    "fair": "the recorded config with market.r = 0, re-simulated here (the fair world)",
}
STATEMENT = (
    "Post hoc and not pre-registered. The bias was found by an outside review of the code "
    "after the confirmatory sweep ran and after its M4 analysis was written; this analysis "
    "was written and run on 2026-11-14 and is recorded as amendment A11. It changes "
    "nothing in A1, A2 or A7 and reruns or retrains nothing: the confirmatory study, in "
    "the world it ran in, stays the headline. Read what follows as a sensitivity of that "
    "result to the world's rate, not as a second test."
)


# Episodes of the A7 sweep known not to re-simulate as saved, and why (docs/DEFECTS.md).
KNOWN_EPISODES: dict[tuple[str, int], str] = {
    ("ea5be29e60901c1a-s100000", 100148): (
        'a cross-platform path split (docs/DEFECTS.md, "One edge episode differs")'
    ),
    ("4a48a04bb0099da1-s101500", 101623): (
        "the defined-risk margin check, fixed after the sweep (4642b1a, GH-39; "
        'docs/DEFECTS.md, "An open defect in the frozen margin check")'
    ),
}


MOVES_TOO = (
    "Two things change with the rate other than what a contract is worth against its "
    "payoff. A strike picked by delta moves, since delta depends on r: the seller opens a "
    "slightly different structure in the fair world (a 16-delta put about one strike lower "
    "on the day-0 chain). And a learned policy was trained in the study's world: scored, "
    "not retrained, in the fair one it reads prices and greeks it did not train on, so its "
    "fair-world figures mix the bias's removal with a policy evaluated outside the world "
    "it learned in. Retraining at r = 0, which A11 leaves out, would separate the two."
)


def fair_config(config: RunConfig) -> RunConfig:
    """``config`` in the fair world: ``market.r = 0`` and nothing else changed."""
    return config.model_copy(update={"market": config.market.model_copy(update={"r": FAIR_R})})


# -- one cell ------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DifferingEpisode:
    seed: int
    saved_pnl: float
    pnl: float
    saved_cost: float
    cost: float


@dataclass(frozen=True, slots=True)
class CellWorlds:
    """One cell's episodes in each world, in seed order, and how the original reproduced."""

    run_id: str
    fair_run_id: str
    saved: tuple[float, ...]
    original: tuple[float, ...]
    fair: tuple[float, ...]
    reproduces: bool  # every episode bit for bit
    agrees: bool  # every episode within PNL_RTOL, same trades and steps
    differing: tuple[DifferingEpisode, ...]  # the episodes that do not agree
    max_abs_pnl_diff: float
    seconds: float
    trades_original: tuple[int, ...] = ()
    trades_fair: tuple[int, ...] = ()

    @property
    def trades_changed(self) -> int:
        """Episodes whose trade count differs between the original world and the fair one."""
        return sum(a != b for a, b in zip(self.trades_original, self.trades_fair, strict=True))

    def world(self, name: str) -> tuple[float, ...]:
        return {"saved": self.saved, "original": self.original, "fair": self.fair}[name]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= PNL_RTOL * max(1.0, abs(b))


def _policy(run_dir: Path, config: RunConfig, load_policy: PolicyLoader) -> Policy | None:
    if config.train is None:
        return None
    path = run_dir / POLICY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"a {config.policy} cell is scored with its policy; {path} is missing"
        )
    return load_policy(path, config)


def resimulate_cell(run_dir: Path, load_policy: PolicyLoader = load_sb3_policy) -> CellWorlds:
    """``run_dir`` re-simulated in its own world and in the fair one; never writes into it.

    A run whose files no longer match their manifest is refused, and so is one not made on
    the generator (a real source reads its own rate, so ``r`` changes nothing there)."""
    start = time.perf_counter()
    manifest = RunManifest.read(run_dir)
    changed = changed_artifacts(run_dir, manifest)
    if changed:
        raise ValueError(f"{', '.join(changed)} no longer match the manifest in {run_dir}")
    if manifest.data_record.source != "synthetic":
        raise ValueError(
            f"{manifest.run_id} ran on {manifest.data_record.source}; the rate moves only "
            "the generator's chain"
        )
    record = read_run(run_dir)
    config = manifest.config
    # A fresh policy object for each world, as a fresh harness run would build one.
    original = run_episodes(config, _policy(run_dir, config, load_policy))
    fair_cfg = fair_config(config)
    fair = run_episodes(fair_cfg, _policy(run_dir, config, load_policy))

    rows = {row.seed: row for row in record.episodes}
    exact, close = True, True
    differing: list[DifferingEpisode] = []
    worst = 0.0
    for ep in original:
        row = rows.get(ep.seed)
        if row is None:
            raise ValueError(f"{manifest.run_id}: episode seed {ep.seed} has no run.json row")
        same = ep.n_trades == row.n_trades and ep.steps == row.steps
        exact = exact and same and ep.pnl == row.pnl and ep.cost == row.cost
        agrees = same and _close(ep.pnl, row.pnl) and _close(ep.cost, row.cost)
        close = close and agrees
        worst = max(worst, abs(ep.pnl - row.pnl))
        if not agrees:
            differing.append(DifferingEpisode(ep.seed, row.pnl, ep.pnl, row.cost, ep.cost))
    if len(rows) != len(original):
        raise ValueError(f"{manifest.run_id}: {len(rows)} rows saved, {len(original)} rerun")

    def pnls(eps: Sequence[EpisodeResult]) -> tuple[float, ...]:
        return tuple(e.pnl for e in eps)

    return CellWorlds(
        run_id=manifest.run_id,
        fair_run_id=run_id(fair_cfg),
        saved=tuple(r.pnl for r in record.episodes),
        original=pnls(original),
        fair=pnls(fair),
        reproduces=exact,
        agrees=close,
        differing=tuple(differing),
        max_abs_pnl_diff=worst,
        seconds=time.perf_counter() - start,
        trades_original=tuple(e.n_trades for e in original),
        trades_fair=tuple(e.n_trades for e in fair),
    )


def _cell_task(task: tuple[Path, PolicyLoader]) -> CellWorlds:
    run_dir, load_policy = task
    return resimulate_cell(run_dir, load_policy)


# -- the grid ------------------------------------------------------------------- #

Reporter = Callable[[int, int, GridCell], None]


@dataclass(frozen=True, slots=True)
class DriftSensitivity:
    record: GridRecord
    grid_file: Path
    cells: tuple[tuple[GridCell, CellWorlds], ...]
    arms: dict[str, dict[str, ArmAnalysis]]  # world -> arm -> analysis
    gaps: dict[str, Gap | None]  # world -> gap
    workers: int
    seconds: float

    @property
    def premiums(self) -> list[float]:
        return sorted(self.record.grid.premiums)

    @property
    def arm_names(self) -> tuple[ArmName, ...]:
        return self.record.grid.arms

    def mean(self, world: str, premium: float, arm: str) -> float:
        vals = [
            v for c, w in self.cells if (c.premium, c.arm) == (premium, arm) for v in w.world(world)
        ]
        return math.fsum(vals) / len(vals)

    def trades(self, world: str, premium: float, arm: str) -> float:
        """Mean trades an episode in the original or the fair world."""
        vals = [
            t
            for c, w in self.cells
            if (c.premium, c.arm) == (premium, arm)
            for t in (w.trades_original if world == "original" else w.trades_fair)
        ]
        return sum(vals) / len(vals)

    def trades_changed(self, premium: float, arm: str) -> float:
        """The share of episodes whose trade count the fair world changes."""
        cells = [w for c, w in self.cells if (c.premium, c.arm) == (premium, arm)]
        return sum(w.trades_changed for w in cells) / sum(len(w.fair) for w in cells)

    @property
    def n_reproduce(self) -> int:
        return sum(w.reproduces for _, w in self.cells)

    @property
    def n_agree(self) -> int:
        return sum(w.agrees and not w.reproduces for _, w in self.cells)

    @property
    def n_differ(self) -> int:
        return sum(not w.agrees for _, w in self.cells)


def _matrix(
    cells: Sequence[tuple[GridCell, CellWorlds]], world: str, premium: float, arm: str, n_seeds: int
) -> NDArray[np.float64]:
    by_block = {c.block: w.world(world) for c, w in cells if (c.premium, c.arm) == (premium, arm)}
    return np.asarray([by_block[b] for b in range(n_seeds)], dtype=np.float64)


def drift_grid(
    path: Path,
    *,
    workers: int = 1,
    load_policy: PolicyLoader = load_sb3_policy,
    report: Reporter | None = None,
    n_boot: int | None = None,
    replications: int | None = None,
) -> DriftSensitivity:
    """Every cell of the grid at ``path`` (``grid-<hash>.json`` or its root) in both worlds,
    on ``workers`` spawned processes (the learned arm's cells first), then the M4 analysis
    of each world. ``n_boot`` and ``replications`` default to M4's."""
    start = time.perf_counter()
    grid_file = find_grid(path)
    record = read_grid(grid_file)
    root = grid_file.parent
    grid = record.grid
    if NULL_PREMIUM not in grid.premiums or grid.n_seeds < 2:
        raise ValueError("the analysis needs a zero premium and at least 2 seed blocks")
    cells = list(record.cells)
    for c in cells:
        if not (root / c.run_id / MANIFEST_FILENAME).is_file():
            raise FileNotFoundError(
                f"cell {c.arm} at {c.premium:g} ({c.run_id}) is not under {root}"
            )
    tasks = [(root / c.run_id, load_policy) for c in cells]
    order = sorted(range(len(cells)), key=lambda i: scripted_first(cells[i]))
    results: dict[int, CellWorlds] = {}
    if workers > 1 and len(tasks) > 1:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context) as pool:
            futures = {pool.submit(_cell_task, tasks[i]): i for i in order}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                if report is not None:
                    report(len(results), len(tasks), cells[i])
    else:
        for i in order:
            results[i] = _cell_task(tasks[i])
            if report is not None:
                report(len(results), len(tasks), cells[i])
    pairs = tuple((c, results[i]) for i, c in enumerate(cells))

    premiums = sorted(grid.premiums)
    kw: dict[str, Any] = {}
    if n_boot is not None:
        kw["n_boot"] = n_boot
    if replications is not None:
        kw["replications"] = replications
    arms: dict[str, dict[str, ArmAnalysis]] = {}
    gaps: dict[str, Gap | None] = {}
    for world in WORLDS:
        arms[world] = {
            arm: analyse_arm(
                arm,
                premiums,
                {p: _matrix(pairs, world, p, arm, grid.n_seeds) for p in premiums},
                **kw,
            )
            for arm in grid.arms
        }
        names = list(grid.arms)
        if "delta_seller" in names and "ppo" in names:
            o, lf = arms[world]["delta_seller"], arms[world]["ppo"]
            gaps[world] = gap_between(o.arm, o.floor, lf.arm, lf.floor, premiums[-1])
        else:
            gaps[world] = None
    return DriftSensitivity(
        record=record,
        grid_file=grid_file,
        cells=pairs,
        arms=arms,
        gaps=gaps,
        workers=workers,
        seconds=time.perf_counter() - start,
    )


def scripted_first(cell: GridCell) -> bool:
    """Sort key: the learned arm's cells (slower to load) go first."""
    return cell.arm != "ppo"


# -- what changed --------------------------------------------------------------- #


def _yes(flag: bool) -> str:
    return "detected" if flag else "not detected"


def _vp(x: float | None) -> str:
    return "--" if x is None else f"{x:.3g}"


def changes(result: DriftSensitivity, a: str = "original", b: str = "fair") -> dict[str, Any]:
    """Every verdict, floor status, rule and gap that differs between worlds ``a`` and ``b``."""
    verdicts, floors, rules = [], [], []
    for arm in result.arm_names:
        x, y = result.arms[a][arm], result.arms[b][arm]
        for rx, ry in zip(x.results, y.results, strict=True):
            if rx.decision.rejected != ry.decision.rejected:
                verdicts.append(
                    f"{arm} at {rx.premium:g}: {_yes(rx.decision.rejected)} -> "
                    f"{_yes(ry.decision.rejected)}"
                )
        if x.floor.status != y.floor.status:
            floors.append(f"{arm}: floor status {x.floor.status} -> {y.floor.status}")
        if x.rule_in_force != y.rule_in_force:
            rules.append(f"{arm}: rule {x.rule_in_force} -> {y.rule_in_force}")
    ga, gb = result.gaps.get(a), result.gaps.get(b)
    gap = ""
    if ga is not None and gb is not None and ga.kind != gb.kind:
        gap = f"gap kind {ga.kind} -> {gb.kind}"
    return {"verdicts": verdicts, "floors": floors, "gap": gap, "rule": rules}


# -- the outputs ---------------------------------------------------------------- #


def _f(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _arm_world(a: ArmAnalysis) -> dict[str, Any]:
    return {
        "levels": [
            {
                "premium": r.premium,
                "estimate": r.decision.estimate,
                "se": r.decision.se,
                "t": _f(r.decision.t_stat),
                "lower_bound": r.decision.lower_bound,
                "detected": r.decision.rejected,
            }
            for r in a.results
        ],
        "curve": [
            {
                "premium": c.premium,
                "measured": c.measured,
                "noncentrality": _f(c.noncentrality),
                "noncentrality_lo": _f(c.noncentrality_lo),
                "noncentrality_hi": _f(c.noncentrality_hi),
                "power": c.power,
                "power_lo": c.power_lo,
                "power_hi": c.power_hi,
            }
            for c in a.curve
        ],
        "floor": {
            "premium": a.floor.premium,
            "lo": a.floor.lo,
            "hi": a.floor.hi,
            "status": a.floor.status,
            "bracket": list(a.floor.bracket),
            "max_power": a.floor.max_power,
        },
        "rule_in_force": a.rule_in_force,
        "diverged_at": list(a.diverged_at),
    }


def _gap(g: Gap | None) -> dict[str, Any] | None:
    if g is None:
        return None
    return {
        "value": g.value,
        "kind": g.kind,
        "band_range": list(g.band_range),
        "bracket": list(g.bracket),
    }


@dataclass(frozen=True, slots=True)
class Provenance:
    report_date: date
    root: str  # portable
    analysis_git_sha: str
    analysis_git_dirty: bool
    versions: dict[str, str]
    sweep_git_shas: tuple[str, ...]


def sweep_shas(result: DriftSensitivity) -> tuple[str, ...]:
    root = result.grid_file.parent
    return tuple(sorted({RunManifest.read(root / c.run_id).git_sha for c, _ in result.cells}))


def to_dict(result: DriftSensitivity, prov: Provenance) -> dict[str, Any]:
    differing = [
        {
            "run_id": w.run_id,
            "arm": c.arm,
            "premium": c.premium,
            "block": c.block,
            "seed": d.seed,
            "saved_pnl": d.saved_pnl,
            "pnl": d.pnl,
            "saved_cost": d.saved_cost,
            "cost": d.cost,
        }
        for c, w in result.cells
        for d in w.differing
    ]
    arms: dict[str, Any] = {}
    for arm in result.arm_names:
        arms[arm] = {
            "means": [
                {
                    "premium": p,
                    **{world: result.mean(world, p, arm) for world in WORLDS},
                    "trades_original": result.trades("original", p, arm),
                    "trades_fair": result.trades("fair", p, arm),
                    "trades_changed": result.trades_changed(p, arm),
                }
                for p in result.premiums
            ],
            "worlds": {world: _arm_world(result.arms[world][arm]) for world in WORLDS},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "drift_sensitivity",
        "post_hoc": True,
        "pre_registered": False,
        "statement": STATEMENT,
        "report_date": prov.report_date.isoformat(),
        "provenance": {
            "root": prov.root,
            "grid_file": result.grid_file.name,
            "grid_hash": result.record.grid_hash,
            "analysis_git_sha": prov.analysis_git_sha,
            "analysis_git_dirty": prov.analysis_git_dirty,
            "sweep_git_shas": list(prov.sweep_git_shas),
            "versions": dict(prov.versions),
            "workers": result.workers,
            "seconds": round(result.seconds, 1),
        },
        "worlds": {
            "saved": {"r": None, "what": WORLD_TEXT["saved"]},
            "original": {"r": result.record.grid.base.market.r, "what": WORLD_TEXT["original"]},
            "fair": {"r": FAIR_R, "what": WORLD_TEXT["fair"]},
        },
        "reproduction": {
            "rtol": PNL_RTOL,
            "cells": len(result.cells),
            "reproduce": result.n_reproduce,
            "agree": result.n_agree,
            "differ": result.n_differ,
            "differing": differing,
            "known": {f"{rid} seed {seed}": why for (rid, seed), why in KNOWN_EPISODES.items()},
        },
        "arms": arms,
        "gap": {world: _gap(result.gaps[world]) for world in WORLDS},
        "changes": {
            "original_to_fair": changes(result, "original", "fair"),
            "saved_to_original": changes(result, "saved", "original"),
        },
        "cells": [
            {
                "premium": c.premium,
                "arm": c.arm,
                "block": c.block,
                "run_id": w.run_id,
                "fair_run_id": w.fair_run_id,
                "reproduces": w.reproduces,
                "agrees": w.agrees,
                "max_abs_pnl_diff": w.max_abs_pnl_diff,
                "mean_saved": math.fsum(w.saved) / len(w.saved),
                "mean_original": math.fsum(w.original) / len(w.original),
                "mean_fair": math.fsum(w.fair) / len(w.fair),
                "trades_changed": w.trades_changed,
                "seconds": round(w.seconds, 2),
            }
            for c, w in result.cells
        ],
    }


def render_json(result: DriftSensitivity, prov: Provenance) -> str:
    return json.dumps(to_dict(result, prov), indent=2, allow_nan=False) + "\n"


def _usd(x: float) -> str:
    return f"{x:,.0f}"


def _signed(x: float) -> str:
    return f"{x:+,.0f}"


def _floor_text(a: ArmAnalysis) -> str:
    f = a.floor
    if f.premium is None:
        return f"not reached (max power {f.max_power:.3f})"
    hi = _vp(f.hi) if f.hi is not None else "above the grid"
    return f"{f.premium:.3g} ({f.status}; band {_vp(f.lo)} to {hi})"


def _gap_text(g: Gap | None) -> str:
    if g is None or g.value is None:
        return "undefined"
    kind = "" if g.kind == "point" else f" ({g.kind.replace('_', ' ')})"
    return f"{g.value:.3g}{kind}"


def _pct(change: float, base: float) -> str:
    return f"{change / base:+.0%}" if base else "--"


def reading(result: DriftSensitivity) -> str:
    """The plain one-paragraph reading, from the numbers: does any verdict, floor or gap
    change, and by how much."""
    ch = changes(result)
    parts = []
    moved = ch["verdicts"] + ch["floors"] + ch["rule"] + ([ch["gap"]] if ch["gap"] else [])
    if moved:
        parts.append("In the fair world these change: " + "; ".join(moved) + ".")
    else:
        parts.append(
            "No verdict changes: every premium detected in the study's world is detected in "
            "the fair one and every premium not detected is not, and no floor status or rule "
            "in force changes."
        )
    for arm in result.arm_names:
        o, f = result.arms["original"][arm], result.arms["fair"][arm]
        null_shift = result.mean("fair", NULL_PREMIUM, arm) - result.mean(
            "original", NULL_PREMIUM, arm
        )
        shifts = [
            (rx.premium, ry.decision.estimate - rx.decision.estimate, rx.decision.estimate)
            for rx, ry in zip(o.results, f.results, strict=True)
        ]
        edges = ", ".join(f"{_signed(d)} ({_pct(d, e)}) at {p:g}" for p, d, e in shifts)
        if o.floor.premium is not None and f.floor.premium is not None:
            floor = (
                f"its floor moves from {_vp(o.floor.premium)} to {_vp(f.floor.premium)} vol "
                f"points ({f.floor.premium - o.floor.premium:+.3g})"
            )
        else:
            floor = f"its floor is {_vp(o.floor.premium)} against {_vp(f.floor.premium)}"
        changed = sum(w.trades_changed for c, w in result.cells if c.arm == arm)
        total = sum(len(w.fair) for c, w in result.cells if c.arm == arm)
        parts.append(
            f"{arm}: its mean at zero premium moves by {_signed(null_shift)} dollars an episode "
            f"in the fair world, its edge by {edges} dollars, and {floor}; "
            f"{changed / total:.0%} of its episodes trade a different number of times."
        )
    go, gf = result.gaps.get("original"), result.gaps.get("fair")
    if go is not None and gf is not None and go.value is not None and gf.value is not None:
        parts.append(
            f"The gap moves from {_gap_text(go)} to {_gap_text(gf)} vol points "
            f"({gf.value - go.value:+.3g})."
        )
    return " ".join(parts)


def render(result: DriftSensitivity, prov: Provenance) -> str:
    g = result.record.grid
    lines = [
        "# Drift sensitivity: the confirmatory sweep in the fair world (A11)",
        "",
        f"**{STATEMENT}**",
        "",
        f"Run {prov.report_date.isoformat()} by `python -m scripts.drift_sensitivity` on "
        f"{result.workers} workers in {result.seconds / 60:.1f} minutes; `optspread/drift.py` "
        f"defines every number. Synthetic, costs on. Analysis code {prov.analysis_git_sha[:8]}"
        f"{' (dirty)' if prov.analysis_git_dirty else ''}; sweep cells written at "
        f"{', '.join(s[:8] for s in prov.sweep_git_shas)}.",
        "",
        f"Grid `{result.grid_file.name}` under `{prov.root}`: S = {g.n_seeds} blocks x N = "
        f"{g.base.n_episodes} episodes, premiums "
        f"{', '.join(f'{p:g}' for p in result.premiums)} vol points, arms "
        f"{', '.join(g.arms)}.",
        "",
        "## The worlds",
        "",
        "The study's path is a zero-drift GBM, but its chain was priced at r = "
        f"{g.base.market.r:g} and the book earns nothing on cash, so its null world was not "
        "fair: a put was priced below its expected payoff under the path and a call above it. "
        "The fair world is r = 0 on the same path. Every cell is re-simulated in both, on its "
        "own seeds, with the scripted policy or the PPO policy it saved (scored, never "
        "retrained):",
        "",
        *[f"- **{w}**: {WORLD_TEXT[w]}." for w in WORLDS],
        "",
        f"**What moves besides the carry.** {MOVES_TOO}",
        "",
        f"**The original world reproduces the run.** {result.n_reproduce} of "
        f"{len(result.cells)} cells reproduce bit for bit here, {result.n_agree} agree within "
        f"{PNL_RTOL:g} (P&L and cost, same trades and steps), {result.n_differ} differ.",
    ]
    for c, w in result.cells:
        for d in w.differing:
            why = KNOWN_EPISODES.get((w.run_id, d.seed), "not a known episode")
            lines.append(
                f"- {c.arm} at {c.premium:g}, block {c.block} (`{w.run_id}`), seed {d.seed}: "
                f"saved P&L {d.saved_pnl:,.2f}, re-simulated {d.pnl:,.2f} (cost "
                f"{d.saved_cost:,.2f} against {d.cost:,.2f}); the cell's other episodes agree. "
                f"Known: {why}."
                if why != "not a known episode"
                else f"- {c.arm} at {c.premium:g}, block {c.block} (`{w.run_id}`), seed "
                f"{d.seed}: saved P&L {d.saved_pnl:,.2f}, re-simulated {d.pnl:,.2f} (cost "
                f"{d.saved_cost:,.2f} against {d.cost:,.2f}); not a known episode."
            )
    so = changes(result, "saved", "original")
    same = not (so["verdicts"] or so["floors"] or so["rule"] or so["gap"])
    lines += [
        "",
        "The comparison below is original against fair, both re-simulated here. "
        + (
            "The original world gives the headline's verdicts, floor statuses and rules."
            if same
            else "The original world differs from the headline in: "
            + "; ".join(so["verdicts"] + so["floors"] + so["rule"] + [so["gap"]])
            + "."
        ),
        "",
        "## Mean P&L per premium",
        "",
        "Dollars per episode, pooled over the blocks; the fair world minus the original is the "
        "bias the arm carried at that premium.",
        "",
        "| arm | premium | saved (run.json) | original | fair | fair - original | trades "
        "(original, fair) | episodes trading differently |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in result.arm_names:
        for p in result.premiums:
            ms, mo, mf = (result.mean(w, p, arm) for w in WORLDS)
            lines.append(
                f"| {arm} | {p:g} | {_usd(ms)} | {_usd(mo)} | {_usd(mf)} | {_signed(mf - mo)} | "
                f"{result.trades('original', p, arm):.2f}, {result.trades('fair', p, arm):.2f} | "
                f"{result.trades_changed(p, arm):.0%} |"
            )
    lines += [
        "",
        "## The decisions (A1, A2), original against fair",
        "",
        "Each arm at p against itself at zero, one-sided t on the seed means. Dollars per episode.",
        "",
        "| arm | premium | edge | edge (fair) | change | t | t (fair) | detected "
        "| detected (fair) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in result.arm_names:
        o, f = result.arms["original"][arm], result.arms["fair"][arm]
        for rx, ry in zip(o.results, f.results, strict=True):
            dx, dy = rx.decision, ry.decision
            lines.append(
                f"| {arm} | {rx.premium:g} | {_usd(dx.estimate)} | {_usd(dy.estimate)} | "
                f"{_signed(dy.estimate - dx.estimate)} | {dx.t_stat:.2f} | {dy.t_stat:.2f} | "
                f"{'yes' if dx.rejected else 'no'} | {'yes' if dy.rejected else 'no'} |"
            )
    lines += [
        "",
        "## The power curve, the floor and the gap",
        "",
        "Power of a fresh experiment of this design at each premium (M4's median-unbiased "
        "noncentrality, its band in brackets), original against fair.",
        "",
        "| arm | premium | power | power (fair) | noncentrality | noncentrality (fair) |",
        "|---|---|---|---|---|---|",
    ]
    for arm in result.arm_names:
        o, f = result.arms["original"][arm], result.arms["fair"][arm]
        for cx, cy in zip(o.curve, f.curve, strict=True):
            if not cx.measured:
                continue
            lines.append(
                f"| {arm} | {cx.premium:g} | {cx.power:.3f} ({cx.power_lo:.3f} to "
                f"{cx.power_hi:.3f}) | {cy.power:.3f} ({cy.power_lo:.3f} to {cy.power_hi:.3f}) | "
                f"{cx.noncentrality:.2f} | {cy.noncentrality:.2f} |"
            )
    lines += [
        "",
        "| arm | floor (vol points) | floor (fair) | rule | rule (fair) |",
        "|---|---|---|---|---|",
    ]
    for arm in result.arm_names:
        o, f = result.arms["original"][arm], result.arms["fair"][arm]
        lines.append(
            f"| {arm} | {_floor_text(o)} | {_floor_text(f)} | {o.rule_in_force} "
            f"| {f.rule_in_force} |"
        )
    go, gf = result.gaps.get("original"), result.gaps.get("fair")
    if go is not None and gf is not None:
        lines += [
            "",
            f"**Gap** (ppo floor minus delta_seller floor): {_gap_text(go)} vol points in the "
            f"original world, {_gap_text(gf)} in the fair world.",
        ]
    lines += ["", "## Reading", "", reading(result), ""]
    return "\n".join(lines)
