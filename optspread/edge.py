"""Edge attribution: which components, and which structures, carry the detected edge.

Lane: Dominick Dupuy (statistical spine). Project document 10.9, on the event of
pre-registration A1 and the test of A2.

``optspread.attribution`` splits one run's P&L into delta, gamma, vega, theta, carry,
cost and a residual. The detection event is not a run's P&L but a paired contrast:
each arm at premium p against the same arm at zero, on the same evaluation paths
(seed block i, path j) and, for PPO, the same training seed (A1). This module
attributes that contrast. Every episode of every cell of a grid is re-simulated by
``attribution.attribute_run`` (the env, the policy object and the reset order the
harness used; a ppo cell with the ``policy.zip`` it saved) and by
``actionmix.mix_run``, whose per-step record says which structure each step was
about. Nothing on the simulation path is touched and nothing is written into a run.

**Per episode and component.** ``c[a, p, i, j]`` is component c of episode j of block
i, arm a, premium p: the attribution's episode total, so the seven components sum to
the harness's episode P&L to one rounding (the residual closes it). The structure
split sums the same step terms over the steps each structure was about::

    step structure  the structure held over the move (``actionmix`` ``held[d]``); on
                    a step that ends flat because the policy closed (a close, or an
                    open refused after closing what was held), the structure closed
                    that day (its holding's ``exit == "policy"``), so a close's cost
                    is charged to what it closed; FLAT only for a step that held and
                    closed nothing. A switch's cost (closing one, opening the other)
                    is the new structure's, which is the one held over the move.

so ``sum over structures of c_s == c`` per episode, to rounding (the steps' P&L
against the episode's final-less-initial equity). The attribution's ``action_id`` of
each step is checked against the action mix's ``held`` step for step; a disagreement
is an error, not a silent re-labelling.

**The edge (A1, A2).** For arm a, premium p > 0 and component c (or component c of
structure s), the contrast matrix is ``D[i, j] = c[a, p, i, j] - c[a, 0, i, j]``,
``(S, N)``, paired cell by cell (``analysis.paired_contrast``): the two cells of block
i must have the same seeds and, for ppo, the same training seed, or the grid is
refused. The test is ``analysis.decide_contrast`` on D, exactly as the P&L's: the
estimate is the grand mean, the block mean ``m_i = mean_j D[i, j]``, ``se^2 =
sum_i (m_i - mean m)^2 / (S - 1) / S`` (the sample variance of the S block means over
S, A2), ``t = estimate / se`` on ``S - 1`` df against ``t_{1 - alpha, S - 1}``, and the
one-sided ``1 - alpha`` interval ``[lower_bound, +inf)`` with ``lower_bound = estimate -
critical * se``, positive exactly when the component's edge is detected. ``upper_bound
= estimate + critical * se`` is reported beside it, so a component that carries the
edge downward (a gamma loss, a larger cost) reads as clearly as one that carries it
up; it is not part of the pre-registered test. Means are linear, so the components'
estimates sum to the P&L edge's (``identity_gap``, of the order of 1e-10 dollars) and
the structures' to the P&L edge too (``structure_gap``); the ses do not add. A
component equal at p and at 0 on every path has a contrast of exact zeros and an
edge of exactly 0 with se 0 (``carry``, which ``Book`` never books, is one).

``pnl`` is the edge of the re-simulated P&L, the sum the components decompose;
``pnl_saved`` is the same contrast on the P&L each cell's ``run.json`` recorded
(``GridRecord.contrast``'s, restricted to the episodes attributed): the detection
event as A2 reads it. They are equal when every cell reproduces bit for bit and
within ``PNL_RTOL`` when it only agrees.

**Reproduction.** Per cell, ``reproduces`` is the attribution's own test (every
episode's P&L, cost, trades and steps equal to its ``run.json`` row, bit for bit) and
``agrees`` the action mix's tolerance (``actionmix.PNL_RTOL`` of the P&L and the cost,
the same trades and steps, both re-simulations); ``reproduction`` is ``"reproduces"``,
``"agrees"`` (a run made on another platform, whose libm rounds ``exp`` and ``erf``
its own way) or ``"differs"``, and the grid's is the weakest of its cells'.

**Residual share.** Per cell, the attribution's ``residual_share``: the sum over steps
of |residual| over the sum of |market + carry| (gross mark-to-market turnover), as
``optspread.attribution`` defines it. Per arm and premium, ``residual_edge_share`` is
|residual edge| over the sum of the seven components' |edge|: how much of the
contrast the second-order expansion leaves unexplained.

**JSON** (``dump_edge``; ``SCHEMA_VERSION`` 1; money in dollars per episode)::

    GridEdge = {
      "schema_version": 1,
      "grid_hash": str, "grid_file": str, "source": "synthetic" | ...,
      "arms": [str], "premiums": [float], "n_seeds": int,
      "n_episodes": int,                  episodes attributed per cell
      "first": int | null,                null when every episode was attributed
      "alpha": 0.05, "critical": float,   t_{1 - alpha, S - 1}
      "components": ["delta", "gamma", "vega", "theta", "carry", "cost", "residual"],
      "structures": ["FLAT", "IRON_CONDOR", "SHORT_STRANGLE", "PUT_CREDIT_SPREAD"],
      "reproduction": "reproduces" | "agrees" | "differs",   the weakest cell's
      "reproduces": bool, "agrees": bool, "max_abs_pnl_diff": float,
      "cells": [EdgeCell],                every cell read, in the grid's order
      "edges": [ArmEdge]                  per premium > 0, then arm, in the grid's order
    }

    EdgeCell = {
      "premium": float, "arm": str, "block": int, "run_id": str,
      "seed": int, "train_seed": int | null, "n_episodes": int,
      "reproduces": bool, "agrees": bool, "reproduction": str,
      "max_abs_pnl_diff": float,          largest |re-simulated - saved| episode P&L
      "mean": Terms,                      per-episode mean of each component
      "by_structure": {structure: Terms}, the same, split by step structure
      "held_share": {structure: float},   share of steps each was held (FLAT included)
      "gross": float, "abs_residual": float, "residual_share": float,
      "max_abs_step_residual": float
    }

    ArmEdge = {
      "arm": str, "premium": float,
      "blocks": [{"block": int, "run_id": str, "null_run_id": str,
                  "seed": int, "train_seed": int | null}],
      "pnl": Edge, "pnl_saved": Edge,
      "components": {component: Edge},
      "by_structure": {structure: {component: Edge, ..., "pnl": Edge}},
      "identity_gap": float,              sum of component estimates - pnl estimate
      "structure_gap": float,             sum of structure pnl estimates - pnl estimate
      "residual_edge_share": float,
      "residual_share": float             the cells at p, pooled (|residual| / gross)
    }

    Edge = {"estimate": float, "se": float, "t_stat": float, "lower_bound": float,
            "upper_bound": float, "rejected": bool, "block_means": [float]}

    Terms = optspread.attribution.Terms (delta ... residual, pnl, market)

Output never goes into a run directory or a folder of runs: ``scripts/edge_attribution.py``
writes ``edge-<grid hash>.json`` and ``edge-<grid hash>.md`` (``-first<N>`` before the
suffix for a partial read) under ``default_root()``: ``attribution/edge/`` at the
repository root (gitignored with ``attribution/``), or ``$OPTSPREAD_ATTRIBUTION/edge``.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from optspread.actionmix import PNL_RTOL, find_grid, mix_run
from optspread.actions import ACTION_NAMES, FLAT
from optspread.analysis import ALPHA, NULL_PREMIUM, decide_contrast, paired_contrast
from optspread.artifacts import read_run
from optspread.attribution import (
    COMPONENTS,
    PolicyLoader,
    Terms,
    attribute_run,
    load_sb3_policy,
)
from optspread.attribution import default_root as attribution_root
from optspread.experiment.grid import GridCell, read_grid
from optspread.manifest import MANIFEST_FILENAME
from optspread.power import t_critical

SCHEMA_VERSION = 1
STRUCTURES = ACTION_NAMES  # FLAT included: a step that held nothing
FIELDS = (*COMPONENTS, "pnl", "market")  # the Terms fields, in array order
_PNL, _MARKET = FIELDS.index("pnl"), FIELDS.index("market")

Reproduction = Literal["reproduces", "agrees", "differs"]
_STRENGTH: dict[Reproduction, int] = {"reproduces": 2, "agrees": 1, "differs": 0}


def default_root(env: Mapping[str, str] | None = None) -> Path:
    """``edge/`` under the attribution root: ``attribution/edge/`` at the repository root."""
    return attribution_root(env) / "edge"


# -- the typed result, which is also the JSON schema --------------------------- #


class Edge(BaseModel):
    """One contrast's A2 test: the estimate, its se and the one-sided interval."""

    model_config = ConfigDict(frozen=True)

    estimate: float
    se: float
    t_stat: float
    lower_bound: float  # the one-sided 1 - alpha interval is [lower_bound, +inf)
    upper_bound: float  # estimate + critical * se, for a component that carries it down
    rejected: bool
    block_means: tuple[float, ...]


class EdgeCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    premium: float
    arm: str
    block: int
    run_id: str
    seed: int
    train_seed: int | None
    n_episodes: int
    reproduces: bool
    agrees: bool
    reproduction: Reproduction
    max_abs_pnl_diff: float
    mean: Terms
    by_structure: dict[str, Terms]
    held_share: dict[str, float]
    gross: float
    abs_residual: float
    residual_share: float
    max_abs_step_residual: float


class EdgeBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    block: int
    run_id: str
    null_run_id: str
    seed: int
    train_seed: int | None


class ArmEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    arm: str
    premium: float
    blocks: tuple[EdgeBlock, ...]
    pnl: Edge
    pnl_saved: Edge
    components: dict[str, Edge]
    by_structure: dict[str, dict[str, Edge]]
    identity_gap: float
    structure_gap: float
    residual_edge_share: float
    residual_share: float


class GridEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    grid_hash: str
    grid_file: str
    source: str
    arms: tuple[str, ...]
    premiums: tuple[float, ...]
    n_seeds: int
    n_episodes: int
    first: int | None
    alpha: float = ALPHA
    critical: float
    components: tuple[str, ...] = COMPONENTS
    structures: tuple[str, ...] = STRUCTURES
    reproduction: Reproduction
    reproduces: bool
    agrees: bool
    max_abs_pnl_diff: float
    cells: tuple[EdgeCell, ...]
    edges: tuple[ArmEdge, ...]

    def edge(self, arm: str, premium: float) -> ArmEdge:
        for e in self.edges:
            if (e.arm, e.premium) == (arm, premium):
                return e
        raise KeyError(f"no edge for arm {arm!r} at premium {premium}")


def dump_edge(result: GridEdge) -> str:
    return result.model_dump_json()


def _refuse_run_folder(folder: Path) -> None:
    if (folder / MANIFEST_FILENAME).exists():
        raise ValueError(f"{folder} is a run directory; the edge is written elsewhere")
    if folder.is_dir() and (
        any((d / MANIFEST_FILENAME).is_file() for d in folder.iterdir() if d.is_dir())
        or any(folder.glob("grid-*.json"))
    ):
        raise ValueError(f"{folder} holds a sweep's runs; the edge is written elsewhere")


def output_stem(result: GridEdge) -> str:
    partial = "" if result.first is None else f"-first{result.first}"
    return f"edge-{result.grid_hash}{partial}"


def write_edge(result: GridEdge, out: Path) -> tuple[Path, Path]:
    """Write the JSON and the markdown summary under ``out``, refusing a run or a sweep root."""
    _refuse_run_folder(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = output_stem(result)
    json_path, md_path = out / f"{stem}.json", out / f"{stem}.md"
    json_path.write_text(dump_edge(result), newline="\n")
    md_path.write_text(render_markdown(result), newline="\n")
    return json_path, md_path


def read_edge(path: Path) -> GridEdge:
    result = GridEdge.model_validate_json(path.read_text())
    if result.schema_version != SCHEMA_VERSION:
        raise ValueError(f"edge is schema v{result.schema_version}, not v{SCHEMA_VERSION}")
    return result


# -- the test on one contrast ---------------------------------------------------- #


def edge_of(at_premium: ArrayLike, at_null: ArrayLike, alpha: float = ALPHA) -> Edge:
    """A2 on ``at_premium - at_null``, two ``(S, N)`` matrices paired cell by cell."""
    contrast = paired_contrast(at_premium, at_null)
    decision = decide_contrast(contrast, alpha)
    return Edge(
        estimate=decision.estimate,
        se=decision.se,
        t_stat=decision.t_stat,
        lower_bound=decision.lower_bound,
        upper_bound=decision.estimate + decision.critical * decision.se,
        rejected=decision.rejected,
        block_means=tuple(float(m) for m in contrast.mean(axis=1)),
    )


# -- one cell, re-simulated --------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CellParts:
    """One cell's episodes as arrays: what the edge needs, small enough to pickle."""

    run_id: str
    source: str
    seeds: tuple[int, ...]
    totals: NDArray[np.float64]  # (N, len(FIELDS)): the attribution's episode totals
    by_structure: NDArray[np.float64]  # (N, len(STRUCTURES), len(FIELDS))
    saved_pnl: NDArray[np.float64]  # (N,): run.json's episode P&L
    held_share: dict[str, float]
    reproduces: bool
    agrees: bool
    max_abs_pnl_diff: float
    gross: float
    abs_residual: float
    residual_share: float
    max_abs_step_residual: float

    @property
    def reproduction(self) -> Reproduction:
        return "reproduces" if self.reproduces else "agrees" if self.agrees else "differs"


def _row(terms: Terms) -> list[float]:
    return [getattr(terms, f) for f in FIELDS]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= PNL_RTOL * max(1.0, abs(b))


def cell_parts(
    run_dir: Path,
    seeds: Sequence[int] | None = None,
    *,
    load_policy: PolicyLoader = load_sb3_policy,
) -> CellParts:
    """Attribute ``run_dir``'s episodes (all, or ``seeds``) and split them by structure."""
    att = attribute_run(run_dir, seeds, load_policy=load_policy)
    mix = mix_run(run_dir, seeds, load_policy=load_policy)
    rows = {row.seed: row for row in read_run(run_dir).episodes}
    totals, split, saved = [], [], []
    agrees = mix.agrees is True
    max_diff = 0.0
    for ea, em in zip(att.episodes, mix.episodes, strict=True):
        held = tuple(s.action_id for s in ea.steps)
        if ea.seed != em.seed or held != em.held:
            raise ValueError(
                f"{att.run_id} seed {ea.seed}: the attribution's held structures are not "
                "the action mix's"
            )
        closed = {h.exit_day: h.action_id for h in em.holdings if h.exit == "policy"}
        acc: list[list[list[float]]] = [[[] for _ in FIELDS] for _ in STRUCTURES]
        for step in ea.steps:
            s = step.action_id if step.action_id != FLAT else closed.get(step.day, FLAT)
            for k, value in enumerate(_row(step.terms)):
                acc[s][k].append(value)
        split.append([[math.fsum(v) for v in per] for per in acc])
        totals.append(_row(ea.totals))
        row = rows[ea.seed]
        saved.append(row.pnl)
        max_diff = max(max_diff, abs(ea.totals.pnl - row.pnl))
        agrees = (
            agrees
            and ea.n_trades == row.n_trades
            and ea.n_steps == row.steps
            and _close(ea.totals.pnl, row.pnl)
            and _close(-ea.totals.cost, row.cost)
        )
    return CellParts(
        run_id=att.run_id,
        source=mix.source,
        seeds=tuple(e.seed for e in att.episodes),
        totals=np.asarray(totals, dtype=np.float64),
        by_structure=np.asarray(split, dtype=np.float64),
        saved_pnl=np.asarray(saved, dtype=np.float64),
        held_share=dict(mix.summary.held_share),
        reproduces=att.reproduces is True,
        agrees=agrees,
        max_abs_pnl_diff=max_diff,
        gross=att.gross,
        abs_residual=att.abs_residual,
        residual_share=att.residual_share,
        max_abs_step_residual=att.max_abs_step_residual,
    )


def _cell_task(task: tuple[Path, tuple[int, ...] | None, PolicyLoader]) -> CellParts:
    run_dir, seeds, load_policy = task
    return cell_parts(run_dir, seeds, load_policy=load_policy)


# -- the grid --------------------------------------------------------------------- #

Reporter = Callable[[int, int, GridCell], None]


def _terms(values: NDArray[np.float64]) -> Terms:
    """Per-episode means of an ``(N, len(FIELDS))`` block as ``Terms``."""
    n = values.shape[0]
    return Terms(**{f: math.fsum(values[:, k]) / n for k, f in enumerate(FIELDS)})


def _edge_cell(c: GridCell, parts: CellParts) -> EdgeCell:
    return EdgeCell(
        premium=c.premium,
        arm=c.arm,
        block=c.block,
        run_id=c.run_id,
        seed=c.seed,
        train_seed=c.train_seed,
        n_episodes=len(parts.seeds),
        reproduces=parts.reproduces,
        agrees=parts.agrees,
        reproduction=parts.reproduction,
        max_abs_pnl_diff=parts.max_abs_pnl_diff,
        mean=_terms(parts.totals),
        by_structure={
            name: _terms(parts.by_structure[:, s, :]) for s, name in enumerate(STRUCTURES)
        },
        held_share=parts.held_share,
        gross=parts.gross,
        abs_residual=parts.abs_residual,
        residual_share=parts.residual_share,
        max_abs_step_residual=parts.max_abs_step_residual,
    )


def arm_edge(
    arm: str,
    premium: float,
    treated: Sequence[tuple[GridCell, CellParts]],
    null: Sequence[tuple[GridCell, CellParts]],
    alpha: float = ALPHA,
) -> ArmEdge:
    """The edge of ``arm`` at ``premium``: block i of ``treated`` against block i of
    ``null``, refused unless the two share the seeds and the training seed (A1)."""
    if len(treated) != len(null):
        raise ValueError(f"{arm} at {premium:g} has {len(treated)} blocks against {len(null)}")
    blocks = []
    for (ct, pt), (c0, p0) in zip(treated, null, strict=True):
        if ct.block != c0.block or pt.seeds != p0.seeds or ct.train_seed != c0.train_seed:
            raise ValueError(
                f"{arm} at {premium:g} block {ct.block} is not paired with block {c0.block} "
                "at zero: the same paths and training seed are required (A1)"
            )
        blocks.append(
            EdgeBlock(
                block=ct.block,
                run_id=ct.run_id,
                null_run_id=c0.run_id,
                seed=ct.seed,
                train_seed=ct.train_seed,
            )
        )

    # (S, N, ...) arrays of each side; slicing the trailing axes gives each (S, N) matrix.
    tot1, tot0 = (np.stack([p.totals for _, p in side]) for side in (treated, null))
    str1, str0 = (np.stack([p.by_structure for _, p in side]) for side in (treated, null))
    saved1, saved0 = (np.stack([p.saved_pnl for _, p in side]) for side in (treated, null))

    components = {c: edge_of(tot1[..., k], tot0[..., k], alpha) for k, c in enumerate(COMPONENTS)}
    pnl = edge_of(tot1[..., _PNL], tot0[..., _PNL], alpha)
    by_structure = {
        name: {
            f: edge_of(str1[..., s, k], str0[..., s, k], alpha)
            for k, f in enumerate(FIELDS)
            if k != _MARKET
        }
        for s, name in enumerate(STRUCTURES)
    }
    explained = math.fsum(abs(e.estimate) for e in components.values())
    gross = math.fsum(p.gross for _, p in treated)
    return ArmEdge(
        arm=arm,
        premium=premium,
        blocks=tuple(blocks),
        pnl=pnl,
        pnl_saved=edge_of(saved1, saved0, alpha),
        components=components,
        by_structure=by_structure,
        identity_gap=math.fsum(e.estimate for e in components.values()) - pnl.estimate,
        structure_gap=math.fsum(by_structure[s]["pnl"].estimate for s in STRUCTURES) - pnl.estimate,
        residual_edge_share=abs(components["residual"].estimate) / explained
        if explained > 0.0
        else 0.0,
        residual_share=math.fsum(p.abs_residual for _, p in treated) / gross
        if gross > 0.0
        else 0.0,
    )


def edge_grid(
    path: Path,
    *,
    arms: Sequence[str] | None = None,
    first: int | None = None,
    workers: int = 1,
    load_policy: PolicyLoader = load_sb3_policy,
    report: Reporter | None = None,
    alpha: float = ALPHA,
) -> GridEdge:
    """The edge attribution of a grid (``grid-<hash>.json``, or its sweep root).

    ``arms`` keeps only those arms; ``first`` attributes only each cell's first
    ``first`` episodes (at least 2, which the variance components need). Cells are
    re-simulated on ``workers`` spawned processes, the learned arm's first; ``report``
    hears of each as it finishes.
    """
    grid_file = find_grid(path)
    record = read_grid(grid_file)
    root = grid_file.parent
    keep = tuple(a for a in record.grid.arms if arms is None or a in arms)
    unknown = sorted(set(arms or ()) - set(record.grid.arms))
    if unknown or not keep:
        raise ValueError(
            f"arms {unknown or list(arms or ())} are not in this grid {record.grid.arms}"
        )
    if NULL_PREMIUM not in record.grid.premiums:
        raise ValueError(f"the grid has no {NULL_PREMIUM:g} premium to contrast against")
    if record.grid.n_seeds < 2:
        raise ValueError(
            f"the test needs at least 2 seed blocks, the grid has {record.grid.n_seeds}"
        )
    if first is not None and first < 2:
        raise ValueError(f"--first must be at least 2 episodes per cell, got {first}")
    cells = [c for c in record.cells if c.arm in keep]
    tasks: list[tuple[Path, tuple[int, ...] | None, PolicyLoader]] = []
    for c in cells:
        run_dir = root / c.run_id
        if not (run_dir / MANIFEST_FILENAME).is_file():
            raise FileNotFoundError(
                f"cell {c.arm} at {c.premium:g} ({c.run_id}) is not under {root}"
            )
        seeds = None if first is None else tuple(range(c.seed, c.seed + min(first, c.n_episodes)))
        tasks.append((run_dir, seeds, load_policy))

    order = sorted(range(len(cells)), key=lambda i: cells[i].arm != "ppo")
    results: dict[int, CellParts] = {}
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
    parts = [results[i] for i in range(len(cells))]

    sources = {p.source for p in parts}
    if len(sources) != 1:
        raise ValueError(f"the grid's cells were run on more than one source: {sorted(sources)}")
    by_key = {(c.premium, c.arm, c.block): (c, p) for c, p in zip(cells, parts, strict=True)}
    blocks = range(record.grid.n_seeds)
    edges = [
        arm_edge(
            arm,
            premium,
            [by_key[(premium, arm, b)] for b in blocks],
            [by_key[(NULL_PREMIUM, arm, b)] for b in blocks],
            alpha,
        )
        for premium in record.grid.premiums
        if premium != NULL_PREMIUM
        for arm in keep
    ]
    reproduction = min((p.reproduction for p in parts), key=_STRENGTH.__getitem__)
    return GridEdge(
        grid_hash=record.grid_hash,
        grid_file=str(grid_file),
        source=sources.pop(),
        arms=keep,
        premiums=record.grid.premiums,
        n_seeds=record.grid.n_seeds,
        n_episodes=min(len(p.seeds) for p in parts),
        first=first,
        alpha=alpha,
        critical=t_critical(record.grid.n_seeds, alpha),
        reproduction=reproduction,
        reproduces=all(p.reproduces for p in parts),
        agrees=all(p.agrees for p in parts),
        max_abs_pnl_diff=max(p.max_abs_pnl_diff for p in parts),
        cells=tuple(_edge_cell(c, p) for c, p in zip(cells, parts, strict=True)),
        edges=tuple(edges),
    )


# -- the markdown summary ------------------------------------------------------------ #

LABELS = {
    "FLAT": "flat",
    "IRON_CONDOR": "iron condor",
    "SHORT_STRANGLE": "strangle",
    "PUT_CREDIT_SPREAD": "put spread",
}


def _cell(e: Edge) -> str:
    mark = "**" if e.rejected or e.upper_bound < 0.0 else ""
    return f"{mark}{e.estimate:,.0f}{mark} ({e.se:,.0f})"


def carriers(e: ArmEdge) -> list[str]:
    """The components whose one-sided interval excludes zero, signed, largest first."""
    named = [(c, x) for c, x in e.components.items() if x.lower_bound > 0.0 or x.upper_bound < 0.0]
    named.sort(key=lambda cx: -abs(cx[1].estimate))
    return [f"{c} {'+' if x.estimate > 0 else '-'}" for c, x in named]


def render_markdown(result: GridEdge) -> str:
    """The summary: edge per component, per structure, residual shares, reproduction."""
    episodes = "every episode" if result.first is None else f"the first {result.first} episodes"
    lines = [
        f"# Edge attribution, grid {result.grid_hash}",
        "",
        f"{result.source}; arms {', '.join(result.arms)}; S = {result.n_seeds} blocks x "
        f"N = {result.n_episodes} ({episodes} of each cell). Each arm at premium p minus "
        f"itself at 0 on the same paths and training seed (A1), tested on the S block "
        f"means one-sided at {result.alpha:g}, critical t = {result.critical:.3f} (A2). "
        "Dollars per episode; estimate (se); bold where the one-sided 95% interval "
        "excludes zero (lower bound > 0, or the upper bound < 0 for a component that "
        "carries the edge down).",
        "",
        f"Reproduction: **{result.reproduction}** (bit for bit: {result.reproduces}; within "
        f"{PNL_RTOL:g}: {result.agrees}; largest |re-simulated - saved| episode P&L "
        f"{result.max_abs_pnl_diff:.2g}).",
        "",
        "## Edge by component",
        "",
        "| arm | p | P&L edge (lower bound) | " + " | ".join(COMPONENTS) + " | carried by |",
        "|---|---:|---:|" + "---:|" * len(COMPONENTS) + "---|",
    ]
    for e in result.edges:
        lines.append(
            f"| {e.arm} | {e.premium:g} | {_cell(e.pnl)} [{e.pnl.lower_bound:,.0f}] | "
            + " | ".join(_cell(e.components[c]) for c in COMPONENTS)
            + f" | {', '.join(carriers(e)) or 'none'} |"
        )
    lines += [
        "",
        "## Edge by structure (P&L edge of the steps each structure was about)",
        "",
        "| arm | p | "
        + " | ".join(LABELS[s] for s in STRUCTURES)
        + " | largest component per structure |",
        "|---|---:|" + "---:|" * len(STRUCTURES) + "---|",
    ]
    for e in result.edges:
        tops = []
        for s in STRUCTURES:
            comps = {c: x for c, x in e.by_structure[s].items() if c != "pnl"}
            top, x = max(comps.items(), key=lambda cx: abs(cx[1].estimate))
            if x.estimate != 0.0:
                tops.append(f"{LABELS[s]}: {top} {x.estimate:,.0f}")
        lines.append(
            f"| {e.arm} | {e.premium:g} | "
            + " | ".join(_cell(e.by_structure[s]["pnl"]) for s in STRUCTURES)
            + f" | {'; '.join(tops) or '-'} |"
        )
    lines += [
        "",
        "## Residual and identity",
        "",
        "| arm | p | residual share (cells at p) | residual share of the edge | "
        "identity gap | structure gap | P&L edge, run.json |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for e in result.edges:
        lines.append(
            f"| {e.arm} | {e.premium:g} | {e.residual_share:.2%} | "
            f"{e.residual_edge_share:.2%} | {e.identity_gap:.2g} | {e.structure_gap:.2g} | "
            f"{_cell(e.pnl_saved)} |"
        )
    lines += [
        "",
        "## Cells",
        "",
        "| arm | p | block | run | N | reproduction | mean P&L | residual share | held "
        + " / ".join(LABELS[s] for s in STRUCTURES)
        + " |",
        "|---|---:|---:|---|---:|---|---:|---:|---|",
    ]
    for c in result.cells:
        held = " / ".join(f"{c.held_share.get(s, 0.0):.0%}" for s in STRUCTURES)
        lines.append(
            f"| {c.arm} | {c.premium:g} | {c.block} | {c.run_id} | {c.n_episodes} | "
            f"{c.reproduction} | {c.mean.pnl:,.0f} | {c.residual_share:.2%} | {held} |"
        )
    return "\n".join(lines) + "\n"
