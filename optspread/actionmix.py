"""Action mix: which structures a policy actually trades, day by day and across a grid.

Lane: Dominick Dupuy (statistical spine). The SDP's question is whether the agent
learns *structure selection* from market conditions; a P&L figure cannot say which
structures it picked, so this reads the choices themselves.

An episode is re-simulated from its config exactly as ``optspread.harness`` runs it
(the env ``build_env`` would build on the source the manifest records, the same
policy object across seeds, the same reset order), and every step records what the
policy chose and what the book did with it. Nothing in the environment changes: the
provider is wrapped by ``_Tap``, which delegates every call and, at the moment the
env asks the path to advance (after the trade, before the move), copies what the
book holds, its cash and its count of refused opens. A run directory is read only if
its files still match their manifest, and each re-simulated episode is compared with
its row in ``run.json``, so the choices counted are the ones that made the P&L the
run reported. ``reproduces`` is the attribution's test, bit for bit; ``agrees`` allows
the last digits a different platform rounds differently (``PNL_RTOL`` of the P&L and
the cost, the same trade count and steps). A run made on another machine, whose libm
and numpy build round ``exp`` and ``erf`` in their own way, agrees without reproducing;
the same choices then made the same trades, which is what the mix is about, and
``max_abs_pnl_diff`` says by how much the dollars moved.

A ppo run is re-simulated with the ``policy.zip`` it saved, which needs
Stable-Baselines3 (the ``train`` extra); without it a ppo run is refused with
``LearnerUnavailableError``, and the scripted arms never need it.

**What a step records.** ``actions[d]`` is the action id chosen on day ``d`` (the
library's ids, ``optspread.actions``), ``held[d]`` the structure held over the move
from ``d`` to ``d + 1`` (0 when flat), and ``events[d]`` what the trade did::

    idle     flat, and the policy asked for flat
    hold     the policy asked for the structure it already holds: no trade
    open     flat before, a structure opened
    close    a structure closed, flat after (the policy asked for FLAT)
    switch   a structure closed and another opened in the same trade
    refused  an open the book could not margin (``Book.rebalance``); the book is flat
             after it, having closed what it held if it held something

Each structure opened is a ``holding``: its legs at the open (right, strike, signed
quantity, per-share Black-Scholes delta of a long contract, implied vol, strike over
spot), the tenor in days to its first expiry, the credit taken in and the margin it
consumed, and how it left the book::

    policy    closed by the policy's own trade (a close or a switch)
    expiry    cash-settled at its first expiry (``Book.settle_expired``)
    margin    liquidated by the env's breach check (equity below margin after the move)
    horizon   still live at the episode's last day, closed there (``Book.liquidate``)

``days`` is the exit day less the open day, the exit day being the day the book is
flat again (the trade's day for ``policy``, the day after the move otherwise).

**Summaries** (``MixSummary``) pool any set of episodes: one run, or every block of a
grid cell. Shares are of steps (``action_share``: how often each id was chosen;
``held_share``: how much of the time each structure, FLAT included, was carried);
``structures`` gives, per structure of the library other than FLAT, the opens and
their mean tenor, short and long |delta|, credit, margin, holding days and exits;
``by_day`` counts, for each day of the episode, the actions chosen and structures
held (lists indexed by action id), the opens and the refusals. For a grid
(``mix_grid``) the summary is given per cell and pooled per (arm, premium), so the
learner's mix can be read against the fixed-delta seller's as the planted premium
grows.

**JSON** (``dump_mix``; ``SCHEMA_VERSION`` 1; money in dollars, tenors in days)::

    RunActionMix = {
      "schema_version": 1,
      "run_id": "<config hash>-s<seed>",       the run index's key, to join on
      "config_hash": str, "policy": str, "premium_vol_pts": float,
      "source": "synthetic" | "optionmetrics_stub" | "optionmetrics",
      "run_dir": str | null,                     the run read, or null for a config
      "reproduces": bool | null,                 every episode equals its run.json row;
                                                 null when there is no run to compare
      "agrees": bool | null,                     equal to within PNL_RTOL, same trades
      "max_abs_pnl_diff": float | null,          largest |re-simulated - saved| P&L
      "action_names": ["FLAT", "IRON_CONDOR", ...],   index = action id
      "summary": MixSummary,
      "episodes": [
        {"seed": int, "n_steps": int, "n_trades": int, "n_opens": int,
         "n_refused": int, "pnl": float, "cost": float, "reproduces": bool | null,
         "agrees": bool | null, "pnl_diff": float | null,
         "actions": [int], "held": [int], "events": [str],   empty when lean
         "holdings": [Holding]}                               empty when lean
      ]
    }

    Holding = {"action_id": int, "structure": str, "open_day": int, "exit_day": int,
               "days": int, "exit": "policy" | "expiry" | "margin" | "horizon",
               "spot": float, "expiry_day": int, "tenor_days": int,
               "credit": float, "margin": float, "net_delta": float,
               "legs": [{"right": "C" | "P", "strike": float, "qty": int,
                         "delta": float, "iv": float, "moneyness": float,
                         "entry_price": float}]}

    MixSummary = {
      "n_episodes": int, "n_steps": int,
      "actions": {name: int}, "action_share": {name: float},
      "held": {name: int}, "held_share": {name: float},
      "events": {event: int},
      "n_opens": int, "n_refused": int,
      "opens_per_episode": float, "refused_per_episode": float,
      "refused_share": float,                     refused / (opens + refused)
      "structures": {name: {"n_opens": int, "open_share": float,
                            "mean_tenor_days", "mean_short_abs_delta",
                            "mean_long_abs_delta", "mean_credit", "mean_margin",
                            "mean_holding_days": float | null,
                            "exits": {exit: int}}},   every name but FLAT
      "holding_days": {"mean", "p10", "p50", "p90", "max": float} | null,
      "exits": {exit: int},
      "by_day": [{"day": int, "actions": [int], "held": [int],
                  "opens": int, "refused": int}]
    }

    GridActionMix = {
      "schema_version": 1, "grid_hash": str, "grid_file": str, "source": str,
      "arms": [str], "premiums": [float], "n_seeds": int,
      "reproduces": bool, "agrees": bool,        over every cell read
      "cells":  [{"premium", "arm", "block", "run_id", "reproduces", "agrees",
                  "summary"}],
      "pooled": [{"premium", "arm", "run_ids": [str], "reproduces", "agrees",
                  "summary"}]
    }

Output never goes into a run directory or a folder of runs (a sweep root):
``scripts/action_mix.py`` writes ``<run_id>.json`` per run (``<run_id>.<source>.json``
off the synthetic world) and ``grid-<hash>.json`` per grid under ``default_root()``:
``$OPTSPREAD_ACTIONMIX`` if set, else ``actionmix/`` at the repository root
(gitignored, as ``attribution/`` is). That folder holds no ``manifest.json``, so the
run index and the gate scanners never mistake a file there for a run. The dashboard
reads it there through ``GET /api/runs/actionmix`` (``optspread.webapi.action_mix``).
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict

from optspread.actions import ACTION_NAMES, FLAT, N_ACTIONS
from optspread.artifacts import EpisodeRow, read_run
from optspread.attribution import PolicyLoader, load_sb3_policy, recorded_provider
from optspread.config import RunConfig, config_hash
from optspread.costs import make_cost_model
from optspread.data import SourceName, make_source
from optspread.env import SpreadEnv
from optspread.experiment.grid import read_grid
from optspread.manifest import MANIFEST_FILENAME, RunManifest
from optspread.margin import margin_requirement
from optspread.policies import Policy, make_policy
from optspread.rng import StepRNG
from optspread.runs.rundir import changed_artifacts
from optspread.train import POLICY_FILENAME
from optspread.types import Chain, ChainProvider, Position, Right

SCHEMA_VERSION = 1
# Agreement with the saved row, relative to the larger of 1 dollar and the saved figure:
# far above the 1e-16 rounding a platform's libm moves a $1M book by, far below a cent.
PNL_RTOL = 1e-9
ACTIONMIX_ENV = "OPTSPREAD_ACTIONMIX"
REPO_ROOT = Path(__file__).resolve().parent.parent

Event = Literal["idle", "hold", "open", "close", "switch", "refused"]
Exit = Literal["policy", "expiry", "margin", "horizon"]
EVENTS: tuple[Event, ...] = get_args(Event)
EXITS: tuple[Exit, ...] = get_args(Exit)
STRUCTURES = ACTION_NAMES[1:]  # what can be opened: every name but FLAT


class LearnerUnavailableError(RuntimeError):
    """A ppo run whose saved policy cannot be loaded here (no Stable-Baselines3)."""


def default_root(env: Mapping[str, str] | None = None) -> Path:
    """``$OPTSPREAD_ACTIONMIX`` if set, else ``actionmix/`` at the repository root."""
    value = (os.environ if env is None else env).get(ACTIONMIX_ENV, "")
    return Path(value).expanduser() if value else REPO_ROOT / "actionmix"


# -- the typed result, which is also the JSON schema --------------------------- #


class LegOpen(BaseModel):
    """One leg as it was opened, read off the chain it was traded on."""

    model_config = ConfigDict(frozen=True)

    right: Right
    strike: float
    qty: int  # signed: short legs negative
    delta: float  # per-share Black-Scholes delta of one long contract
    iv: float
    moneyness: float  # strike / spot
    entry_price: float  # per share


class Holding(BaseModel):
    """One structure from its open to the day the book was flat again."""

    model_config = ConfigDict(frozen=True)

    action_id: int
    structure: str
    open_day: int
    exit_day: int
    days: int
    exit: Exit
    spot: float
    expiry_day: int
    tenor_days: int  # first expiry less the open, in the chain's days
    credit: float  # premium taken in, dollars (negative for a debit)
    margin: float  # requirement at the open, dollars
    net_delta: float  # sum of qty * delta, per share
    legs: tuple[LegOpen, ...]

    def abs_deltas(self, short: bool) -> list[float]:
        return [abs(leg.delta) for leg in self.legs if (leg.qty < 0) == short]


class EpisodeMix(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    n_steps: int
    n_trades: int
    n_opens: int
    n_refused: int
    pnl: float
    cost: float
    reproduces: bool | None  # equal to the run.json row; None with no run to compare
    agrees: bool | None = None  # equal to within PNL_RTOL, same trades and steps
    pnl_diff: float | None = None  # re-simulated less saved P&L
    actions: tuple[int, ...] = ()  # chosen on each day
    held: tuple[int, ...] = ()  # held over each day's move
    events: tuple[Event, ...] = ()
    holdings: tuple[Holding, ...] = ()

    def lean(self) -> EpisodeMix:
        return self.model_copy(update={"actions": (), "held": (), "events": (), "holdings": ()})


class StructureMix(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_opens: int
    open_share: float  # of every open in the summary
    mean_tenor_days: float | None
    mean_short_abs_delta: float | None
    mean_long_abs_delta: float | None  # None for a structure with no long leg
    mean_credit: float | None
    mean_margin: float | None
    mean_holding_days: float | None
    exits: dict[str, int]


class Quantiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: float
    p10: float
    p50: float
    p90: float
    max: float


class DayMix(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: int
    actions: tuple[int, ...]  # chosen, indexed by action id
    held: tuple[int, ...]  # held over the move, indexed by action id
    opens: int
    refused: int


class MixSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_episodes: int
    n_steps: int
    actions: dict[str, int]
    action_share: dict[str, float]
    held: dict[str, int]
    held_share: dict[str, float]
    events: dict[str, int]
    n_opens: int
    n_refused: int
    opens_per_episode: float
    refused_per_episode: float
    refused_share: float
    structures: dict[str, StructureMix]
    holding_days: Quantiles | None
    exits: dict[str, int]
    by_day: tuple[DayMix, ...]


class RunActionMix(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    run_id: str
    config_hash: str
    policy: str
    premium_vol_pts: float
    source: SourceName
    run_dir: str | None
    reproduces: bool | None
    agrees: bool | None = None
    max_abs_pnl_diff: float | None = None
    action_names: tuple[str, ...] = ACTION_NAMES
    summary: MixSummary
    episodes: tuple[EpisodeMix, ...]

    def lean(self) -> RunActionMix:
        """The same result with each episode's day-by-day lists and holdings left out."""
        return self.model_copy(update={"episodes": tuple(e.lean() for e in self.episodes)})


class GridMixCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    premium: float
    arm: str
    block: int
    run_id: str
    reproduces: bool | None
    agrees: bool | None
    summary: MixSummary


class GridMixPooled(BaseModel):
    model_config = ConfigDict(frozen=True)

    premium: float
    arm: str
    run_ids: tuple[str, ...]
    reproduces: bool | None
    agrees: bool | None
    summary: MixSummary


class GridActionMix(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    grid_hash: str
    grid_file: str
    source: str
    arms: tuple[str, ...]
    premiums: tuple[float, ...]
    n_seeds: int
    reproduces: bool
    agrees: bool
    action_names: tuple[str, ...] = ACTION_NAMES
    cells: tuple[GridMixCell, ...]
    pooled: tuple[GridMixPooled, ...]

    def pooled_cell(self, premium: float, arm: str) -> GridMixPooled:
        for p in self.pooled:
            if (p.premium, p.arm) == (premium, arm):
                return p
        raise KeyError(f"no pooled cell for premium {premium}, arm {arm!r}")


# -- writing and reading ------------------------------------------------------- #


def dump_mix(result: RunActionMix | GridActionMix, *, episodes: bool = True) -> str:
    """The JSON the dashboard reads (schema in the module docstring)."""
    if isinstance(result, RunActionMix) and not episodes:
        result = result.lean()
    return result.model_dump_json()


def _refuse_run_folder(folder: Path) -> None:
    if (folder / MANIFEST_FILENAME).exists():
        raise ValueError(f"{folder} is a run directory; the action mix is written elsewhere")
    if folder.is_dir() and any((d / MANIFEST_FILENAME).is_file() for d in folder.iterdir()):
        raise ValueError(f"{folder} holds run directories; the action mix is written elsewhere")


def write_mix(result: RunActionMix | GridActionMix, path: Path, *, episodes: bool = True) -> Path:
    """Write ``result`` to ``path``, refusing a run directory or a folder of runs."""
    _refuse_run_folder(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_mix(result, episodes=episodes), newline="\n")
    return path


def _check_version(version: int) -> None:
    if version != SCHEMA_VERSION:
        raise ValueError(f"action mix is schema v{version}, not v{SCHEMA_VERSION}")


def read_mix(path: Path) -> RunActionMix:
    result = RunActionMix.model_validate_json(path.read_text())
    _check_version(result.schema_version)
    return result


def read_grid_mix(path: Path) -> GridActionMix:
    result = GridActionMix.model_validate_json(path.read_text())
    _check_version(result.schema_version)
    return result


def _suffix(source: str) -> str:
    return "" if source == "synthetic" else f".{source}"


def output_name(result: RunActionMix | GridActionMix) -> str:
    if isinstance(result, GridActionMix):
        return f"grid-{result.grid_hash}{_suffix(result.source)}.json"
    return f"{result.run_id}{_suffix(result.source)}.json"


# -- re-simulation ------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Trade:
    """The book between the trade and the move."""

    chain: Chain  # the chain the trade was made on
    position: Position | None
    cash: float
    refused_opens: int


class _Tap:
    """A ``ChainProvider`` that delegates everything and remembers the book at each move.

    ``SpreadEnv.step`` trades at today's chain and then calls ``provider.step()``, so
    at that call the book holds exactly what the trade left. The tap copies it there;
    nothing it does reaches the env or the path.
    """

    def __init__(self, inner: ChainProvider) -> None:
        self.inner = inner
        self.env: SpreadEnv | None = None
        self.chain: Chain | None = None  # the last chain handed out
        self.trade: _Trade | None = None

    @property
    def n_days(self) -> int:
        return self.inner.n_days

    @property
    def done(self) -> bool:
        return self.inner.done

    def reset(self, rng: StepRNG) -> Chain:
        self.chain = self.inner.reset(rng)
        return self.chain

    def step(self) -> Chain:
        if self.env is None or self.chain is None:
            raise RuntimeError("tap stepped before it was bound to its env and reset")
        book = self.env._book
        self.trade = _Trade(self.chain, book.position, book.cash, book.refused_opens)
        self.chain = self.inner.step()
        return self.chain


def _event(action: int, before: Position | None, after: Position | None, refused: bool) -> Event:
    held_before = FLAT if before is None else before.action_id
    if action == held_before:
        return "idle" if before is None else "hold"
    if refused:
        return "refused"
    if after is None:
        return "close"
    return "open" if before is None else "switch"


def _opened(position: Position, chain: Chain, day: int) -> Holding:
    """The ``Holding`` as known at the open on ``day``; its exit is filled in when it leaves."""
    legs = []
    for leg in position.legs:
        delta = chain.greeks(leg.right, leg.strike, leg.expiry_day).delta
        legs.append(
            LegOpen(
                right=leg.right,
                strike=leg.strike,
                qty=leg.qty,
                delta=delta,
                iv=chain.iv_at(leg.expiry_day, leg.strike),
                moneyness=leg.strike / chain.spot,
                entry_price=leg.entry_price,
            )
        )
    return Holding(
        action_id=position.action_id,
        structure=ACTION_NAMES[position.action_id],
        open_day=day,
        exit_day=day,
        days=0,
        exit="horizon",  # a placeholder until the structure leaves the book
        spot=chain.spot,
        expiry_day=position.first_expiry,
        tenor_days=position.first_expiry - chain.t,
        credit=-position.entry_value(),
        margin=margin_requirement(position.legs, chain),
        net_delta=math.fsum(leg.qty * leg.delta for leg in legs),
        legs=tuple(legs),
    )


def _exit_after_move(held: Position, trade: _Trade, chain: Chain) -> Exit:
    """Why a structure carried over the move is gone after it: the env's own order
    (settle, then the breach check, then the horizon) and its own inequalities."""
    if chain.t >= held.first_expiry:
        return "expiry"
    if trade.cash + held.value(chain) < margin_requirement(held.legs, chain):
        return "margin"
    return "horizon"


def mix_episode(env: SpreadEnv, tap: _Tap, policy: Policy, seed: int) -> EpisodeMix:
    """``harness.run_episode`` step for step, with each step's choice recorded."""
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    equity0, total_cost, done = info["equity"], 0.0, False
    actions: list[int] = []
    held: list[int] = []
    events: list[Event] = []
    holdings: list[Holding] = []
    live: Holding | None = None  # the structure on the book, as it was opened

    def finish(exit_day: int, how: Exit) -> None:
        nonlocal live
        assert live is not None
        update = {"exit_day": exit_day, "days": exit_day - live.open_day, "exit": how}
        holdings.append(live.model_copy(update=update))
        live = None

    while not done:
        day = int(info["day"])
        before: Position | None = info["position"]
        refused0 = int(info["refused_opens"])
        action = int(policy.act(obs, info))
        obs, _, done, _, info = env.step(action)
        total_cost += info["cost"]
        trade = tap.trade
        assert trade is not None and tap.chain is not None
        after = trade.position
        event = _event(action, before, after, trade.refused_opens > refused0)
        if event in ("close", "switch") or (event == "refused" and before is not None):
            finish(day, "policy")
        if event in ("open", "switch"):
            assert after is not None
            live = _opened(after, trade.chain, day)
        if after is not None and info["position"] is None:
            finish(int(info["day"]), _exit_after_move(after, trade, tap.chain))
        actions.append(action)
        held.append(FLAT if after is None else after.action_id)
        events.append(event)

    return EpisodeMix(
        seed=seed,
        n_steps=len(actions),
        n_trades=int(info["n_trades"]),
        n_opens=sum(e in ("open", "switch") for e in events),
        n_refused=sum(e == "refused" for e in events),
        pnl=info["equity"] - equity0,
        cost=total_cost,
        reproduces=None,
        actions=tuple(actions),
        held=tuple(held),
        events=tuple(events),
        holdings=tuple(holdings),
    )


def _compared(episode: EpisodeMix, row: EpisodeRow | None) -> EpisodeMix:
    """``episode`` with its verdicts against the saved ``row`` (False when there is none)."""
    if row is None:
        return episode.model_copy(update={"reproduces": False, "agrees": False})
    same_counts = episode.n_trades == row.n_trades and episode.n_steps == row.steps
    exact = same_counts and episode.pnl == row.pnl and episode.cost == row.cost
    close = same_counts and all(
        abs(a - b) <= PNL_RTOL * max(1.0, abs(b))
        for a, b in ((episode.pnl, row.pnl), (episode.cost, row.cost))
    )
    return episode.model_copy(
        update={"reproduces": exact, "agrees": close, "pnl_diff": episode.pnl - row.pnl}
    )


def _seeds(config: RunConfig, seeds: Sequence[int] | None) -> tuple[int, ...]:
    every = range(config.seed, config.seed + config.n_episodes)
    if seeds is None:
        return tuple(every)
    outside = sorted(set(seeds) - set(every))
    if outside:
        raise ValueError(
            f"seeds {outside} are not episodes of this run ({every.start}..{every.stop - 1})"
        )
    return tuple(seeds)


# -- summaries ------------------------------------------------------------------ #


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _by_name(counts: Counter[int]) -> dict[str, int]:
    return {name: counts[i] for i, name in enumerate(ACTION_NAMES)}


def _shares(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def _exits(holdings: Iterable[Holding]) -> dict[str, int]:
    counts = Counter(h.exit for h in holdings)
    return {name: counts[name] for name in EXITS}


def _structure(holdings: Sequence[Holding], n_opens: int) -> StructureMix:
    shorts = [d for h in holdings for d in h.abs_deltas(short=True)]
    longs = [d for h in holdings for d in h.abs_deltas(short=False)]
    return StructureMix(
        n_opens=len(holdings),
        open_share=len(holdings) / n_opens if n_opens else 0.0,
        mean_tenor_days=_mean([h.tenor_days for h in holdings]),
        mean_short_abs_delta=_mean(shorts),
        mean_long_abs_delta=_mean(longs),
        mean_credit=_mean([h.credit for h in holdings]),
        mean_margin=_mean([h.margin for h in holdings]),
        mean_holding_days=_mean([h.days for h in holdings]),
        exits=_exits(holdings),
    )


def _quantiles(days: Sequence[int]) -> Quantiles | None:
    if not days:
        return None
    arr = np.asarray(days, dtype=np.float64)
    p10, p50, p90 = (float(q) for q in np.percentile(arr, (10, 50, 90)))
    return Quantiles(mean=float(arr.mean()), p10=p10, p50=p50, p90=p90, max=float(arr.max()))


def summarize_mix(episodes: Sequence[EpisodeMix]) -> MixSummary:
    """Pool ``episodes`` (full, not lean) into one ``MixSummary``."""
    if any(e.n_steps and not e.actions for e in episodes):
        raise ValueError("a lean episode has no day-by-day record to summarize")
    actions: Counter[int] = Counter(a for e in episodes for a in e.actions)
    held: Counter[int] = Counter(h for e in episodes for h in e.held)
    events = Counter(ev for e in episodes for ev in e.events)
    holdings = [h for e in episodes for h in e.holdings]
    n_opens = sum(e.n_opens for e in episodes)
    n_refused = sum(e.n_refused for e in episodes)
    n_days = max((e.n_steps for e in episodes), default=0)
    by_day = []
    for d in range(n_days):
        chosen = Counter(e.actions[d] for e in episodes if d < e.n_steps)
        carried = Counter(e.held[d] for e in episodes if d < e.n_steps)
        today = Counter(e.events[d] for e in episodes if d < e.n_steps)
        by_day.append(
            DayMix(
                day=d,
                actions=tuple(chosen[i] for i in range(N_ACTIONS)),
                held=tuple(carried[i] for i in range(N_ACTIONS)),
                opens=today["open"] + today["switch"],
                refused=today["refused"],
            )
        )
    n = len(episodes)
    by_action = _by_name(actions)
    by_held = _by_name(held)
    return MixSummary(
        n_episodes=n,
        n_steps=sum(e.n_steps for e in episodes),
        actions=by_action,
        action_share=_shares(by_action),
        held=by_held,
        held_share=_shares(by_held),
        events={name: events[name] for name in EVENTS},
        n_opens=n_opens,
        n_refused=n_refused,
        opens_per_episode=n_opens / n if n else 0.0,
        refused_per_episode=n_refused / n if n else 0.0,
        refused_share=n_refused / (n_opens + n_refused) if n_opens + n_refused else 0.0,
        structures={
            name: _structure([h for h in holdings if h.structure == name], n_opens)
            for name in STRUCTURES
        },
        holding_days=_quantiles([h.days for h in holdings]),
        exits=_exits(holdings),
        by_day=tuple(by_day),
    )


# -- a config, a run, a grid ---------------------------------------------------- #


def mix_config(
    config: RunConfig,
    seeds: Sequence[int] | None = None,
    *,
    policy: Policy | None = None,
    provider: ChainProvider | None = None,
    source: SourceName = "synthetic",
    rows: Mapping[int, EpisodeRow] | None = None,
    run_dir: Path | None = None,
) -> RunActionMix:
    """The action mix of the episodes of ``config`` (all, or ``seeds``).

    ``provider`` defaults to the one ``source`` serves on its evaluation range (the
    ``eval`` split on the real panel, as ``harness.build_env`` builds it). A ppo config needs
    its trained ``policy``. ``rows`` are the saved episodes to check each one against.
    """
    if policy is not None:
        pol = policy
    elif config.policy == "ppo":
        raise ValueError("a ppo arm is re-simulated with its trained policy; pass it in")
    else:
        pol = make_policy(config.policy, roll_dte=config.env.roll_dte)
    inner = (
        provider
        if provider is not None
        else make_source(source, config.market, split="eval").provider()
    )
    tap = _Tap(inner)
    env = SpreadEnv(tap, make_cost_model(config.cost), config.env)
    tap.env = env

    episodes = []
    for seed in _seeds(config, seeds):
        ep = mix_episode(env, tap, pol, seed)
        if rows is not None:
            ep = _compared(ep, rows.get(seed))
        episodes.append(ep)
    h = config_hash(config)
    return RunActionMix(
        run_id=f"{h}-s{config.seed}",
        config_hash=h,
        policy=config.policy,
        premium_vol_pts=config.market.iv_premium_vol_pts,
        source=source,
        run_dir=None if run_dir is None else str(run_dir),
        reproduces=None if rows is None else all(e.reproduces for e in episodes),
        agrees=None if rows is None else all(e.agrees for e in episodes),
        max_abs_pnl_diff=None
        if rows is None
        else max((abs(e.pnl_diff or 0.0) for e in episodes), default=0.0),
        summary=summarize_mix(episodes),
        episodes=tuple(episodes),
    )


def mix_run(
    run_dir: Path,
    seeds: Sequence[int] | None = None,
    *,
    load_policy: PolicyLoader = load_sb3_policy,
) -> RunActionMix:
    """The action mix of a completed run, from its manifest; never writes into ``run_dir``.

    A run whose files no longer match their manifest is refused, as replay refuses it;
    a real-data run is re-simulated on the panel it read and refused on another. A ppo
    run is re-simulated with the ``policy.zip`` it saved (``load_policy``,
    Stable-Baselines3 by default), not retrained; ``LearnerUnavailableError`` when that
    cannot be loaded here.
    """
    manifest = RunManifest.read(run_dir)
    changed = changed_artifacts(run_dir, manifest)
    if changed:
        raise ValueError(f"{', '.join(changed)} no longer match the manifest in {run_dir}")
    record = read_run(run_dir)
    config = manifest.config
    policy = None
    if config.policy == "ppo":
        path = run_dir / POLICY_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"a ppo run is re-simulated with its policy; {path} is missing")
        try:
            policy = load_policy(path, config)
        except ImportError as exc:
            raise LearnerUnavailableError(
                f"{manifest.run_id} is a ppo run and its {POLICY_FILENAME} needs "
                f"stable-baselines3 and torch, which this interpreter cannot import ({exc}); "
                "install the train extra, or leave the ppo arm out (--arms delta_seller)"
            ) from exc
    source, provider = recorded_provider(manifest)
    rows = {row.seed: row for row in record.episodes}
    return mix_config(
        config, seeds, policy=policy, provider=provider, source=source, rows=rows, run_dir=run_dir
    )


def _mix_task(task: tuple[Path, tuple[int, ...] | None, PolicyLoader]) -> RunActionMix:
    run_dir, seeds, load_policy = task
    return mix_run(run_dir, seeds, load_policy=load_policy)


def find_grid(path: Path) -> Path:
    """``path`` if it is a grid file, else the one ``grid-*.json`` directly under it."""
    if path.is_file():
        return path
    found = sorted(path.glob("grid-*.json")) if path.is_dir() else []
    if len(found) != 1:
        names = ", ".join(p.name for p in found) or "none"
        raise FileNotFoundError(f"expected one grid-<hash>.json under {path}, found {names}")
    return found[0]


def mix_grid(
    path: Path,
    *,
    arms: Sequence[str] | None = None,
    first: int | None = None,
    workers: int = 1,
    load_policy: PolicyLoader = load_sb3_policy,
) -> tuple[GridActionMix, tuple[RunActionMix, ...]]:
    """The action mix of every cell of a grid (``grid-<hash>.json``, or its sweep root).

    ``arms`` keeps only those arms (the scripted ones need no learner); ``first`` reads
    only each cell's first ``first`` episodes. Cells are re-simulated on ``workers``
    processes. Returns the grid summary and each cell's own result, in cell order.
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
    cells = [c for c in record.cells if c.arm in keep]
    tasks = []
    for c in cells:
        run_dir = root / c.run_id
        if not (run_dir / MANIFEST_FILENAME).is_file():
            raise FileNotFoundError(
                f"cell {c.arm} at {c.premium:g} ({c.run_id}) is not under {root}"
            )
        seeds = None if first is None else tuple(range(c.seed, c.seed + min(first, c.n_episodes)))
        tasks.append((run_dir, seeds, load_policy))
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            results = tuple(pool.map(_mix_task, tasks))
    else:
        results = tuple(_mix_task(t) for t in tasks)

    sources = {r.source for r in results}
    if len(sources) != 1:
        raise ValueError(f"the grid's cells were run on more than one source: {sorted(sources)}")
    grid_cells = tuple(
        GridMixCell(
            premium=c.premium,
            arm=c.arm,
            block=c.block,
            run_id=c.run_id,
            reproduces=r.reproduces,
            agrees=r.agrees,
            summary=r.summary,
        )
        for c, r in zip(cells, results, strict=True)
    )
    pooled = []
    for premium in record.grid.premiums:
        for arm in keep:
            mine = [
                r
                for c, r in zip(cells, results, strict=True)
                if (c.premium, c.arm) == (premium, arm)
            ]
            pooled.append(
                GridMixPooled(
                    premium=premium,
                    arm=arm,
                    run_ids=tuple(r.run_id for r in mine),
                    reproduces=all(r.reproduces for r in mine),
                    agrees=all(r.agrees for r in mine),
                    summary=summarize_mix([e for r in mine for e in r.episodes]),
                )
            )
    grid = GridActionMix(
        grid_hash=record.grid_hash,
        grid_file=str(grid_file),
        source=sources.pop(),
        arms=keep,
        premiums=record.grid.premiums,
        n_seeds=record.grid.n_seeds,
        reproduces=all(r.reproduces for r in results),
        agrees=all(r.agrees for r in results),
        cells=grid_cells,
        pooled=tuple(pooled),
    )
    return grid, results
