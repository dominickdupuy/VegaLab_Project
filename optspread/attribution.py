"""P&L attribution: each step's equity change split into greeks, carry, cost and a residual.

Lane: Dominick Dupuy (statistical spine). Project document 10.9.

An episode is re-simulated from its config exactly as ``optspread.harness`` runs it
(same env, same policy object across seeds, same reset order), and at every step
the structure held over the move is read off the book between the trade and the
move. Nothing in the environment changes: the provider is wrapped by ``_Tap``, which
delegates every call and only remembers the chain it handed out and what the book
held when the path was asked to advance. A run directory is attributed only if its
files still match their manifest, and each re-simulated episode is compared with
the row in ``run.json``, so the numbers decomposed are the numbers the run reported.
``reproduces`` is that comparison bit for bit; ``agrees`` allows the last digits a
different platform rounds differently (``PNL_RTOL`` of the P&L and the cost, the same
trade count and steps), as ``optspread.actionmix`` does. A run made on another
machine, whose libm and numpy build round ``exp`` and ``erf`` in their own way, agrees
without reproducing: the same trades made the same P&L to within a billionth, which
is all a decomposition of it needs. ``agrees`` is the verdict the M4 gate reads.

**The decomposition.** Over one step from chain ``c0`` (day t, after the trade) to
``c1`` (day t+1), for each held leg with greeks read from ``c0``::

    delta  = q * M * Delta * dS
    gamma  = q * M * 0.5 * Gamma * dS**2
    vega   = q * M * Vega * dsigma
    theta  = q * M * Theta * dt
    market = q * M * (mark(c1) - mark(c0))          the leg's actual move

with ``q`` the signed contract count (short legs negative, so their greeks enter
with the sign the book carries), ``M = MULTIPLIER`` (100 dollars per point),
``dS = c1.spot - c0.spot``, ``dsigma`` the change in the vol the leg is marked at
(``Chain.iv_at`` on each chain, a unit vol: 0.01 is one vol point, because
``Greeks.vega`` is per unit vol), and ``dt = (c1.t - c0.t) / trading_days_per_year``
in years, because ``Greeks.theta`` is per year. Black-Scholes theta here includes
the drift of the discounted strike (the ``r K e^{-rT}`` term), so it is the option's
own time-and-financing carry; ``carry`` below is the book's.

Per step, summed over legs, with ``pnl`` the env's equity change for the step::

    carry    = interest the book credits on cash: 0.0, because ``Book`` books none
               (the column is kept so the schema holds if the book ever finances)
    cost     = minus the dollars charged at the step (the trade at t, and a
               forced or horizon close at t+1), the env's ``info["cost"]``
    residual = pnl - (delta + gamma + vega + theta + carry + cost)

so ``delta + gamma + vega + theta + carry + cost + residual == pnl`` by definition,
to the one rounding of the subtraction. The residual is everything the second-order
expansion misses: speed, charm, vanna, volga, the expiry step (where a leg's value
stops being smooth in time), a change in ``r`` (rho; nonzero on real chains, where
``r`` is the day's carry), and accounting rounding (``pnl - market - cost``, of the
order of 1e-10 dollars on a million-dollar book). ``market`` is reported beside it,
so ``market - (delta + gamma + vega + theta)`` is the pricing part on its own.

An episode's ``pnl`` is the harness's figure (final equity less initial), and its
residual is again ``pnl`` less the summed components, so the identity holds at
every level. ``gross`` is the sum over steps of ``|market + carry|``, the
mark-to-market turnover, and ``residual_share`` is the sum of ``|residual|`` over
steps divided by it. Gross is the denominator because net P&L before costs is near
zero by construction in the null world, so a share of it would mean nothing.

**What the residual can and cannot see.** A sign or unit error in delta, gamma or theta
moves the residual share far past 5% on the synthetic sweep's own cells (theta's sign
to 43-47%, theta per day instead of per year to 22-23%, gamma's sign to 33-39%, delta's
sign to 194%). Vega's cannot: the synthetic surface is constant (``optspread.market``),
so ``dsigma`` and the vega term are exactly zero on every sweep cell, and a vega with
the wrong sign or off by 100 leaves the share where correct code puts it (1.3% to
1.9%). On a moving surface the same errors give 37% to 3,880%. So the residual of a
synthetic cell checks delta, gamma and theta only; vega's sign and units are checked by
``tests/resim/test_attribution.py`` on a moving surface, and on any attribution of real
chains, where the surface moves.

**JSON** (``dump_attribution``; ``SCHEMA_VERSION`` 1; all money in dollars)::

    {
      "schema_version": 1,
      "run_id": "<config hash>-s<seed>",     the run index's key, to join on
      "config_hash": str, "policy": str, "premium_vol_pts": float,
      "source": "synthetic" | "optionmetrics",
      "run_dir": str | null,                   the run attributed, or null for a config
      "reproduces": bool | null,               every episode equals its run.json row;
                                               null when there is no run to compare
      "agrees": bool | null,                   every episode within PNL_RTOL, same trades
                                               and steps (absent in files written before)
      "components": ["delta", "gamma", "vega", "theta", "carry", "cost", "residual"],
      "totals":   Terms,                       summed over the episodes attributed
      "gross": float, "abs_residual": float, "residual_share": float,
      "max_abs_step_residual": float,
      "episodes": [
        {
          "seed": int, "n_steps": int, "n_trades": int,
          "reproduces": bool | null, "agrees": bool | null,
          "totals": Terms, "gross": float, "abs_residual": float,
          "residual_share": float, "max_abs_step_residual": float,
          "steps": [                           empty when written with steps=False
            {"day": int, "spot": float, "spot_next": float, "action_id": int,
             "n_legs": int, "terms": Terms}
          ]
        }
      ]
    }

    Terms = {"delta", "gamma", "vega", "theta", "carry", "cost", "residual",
             "pnl", "market": float}

``day`` is the day the step starts on (the trade's day); ``action_id`` and
``n_legs`` describe the structure held over the move (0 and 0 when flat).

Output never goes into a run directory: ``scripts/attribution.py`` writes
``<run_id>.json`` (``<run_id>.<source>.json`` off the synthetic world) under
``default_root()``: ``$OPTSPREAD_ATTRIBUTION`` if set, else ``attribution/`` at the
repository root. That folder holds no ``manifest.json``, so the run index and the
gate scanners never mistake a file there for a run. The dashboard's attribution
view reads it there.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from optspread.artifacts import EpisodeRow, read_run
from optspread.config import RunConfig, config_hash
from optspread.costs import make_cost_model
from optspread.data import ChainSource, SourceName, make_source
from optspread.env import SpreadEnv
from optspread.manifest import MANIFEST_FILENAME, DataRecord, RunManifest
from optspread.market import SyntheticChainProvider
from optspread.policies import Policy, make_policy
from optspread.rng import StepRNG
from optspread.runs.rundir import changed_artifacts
from optspread.train import POLICY_FILENAME, TORCH_THREADS, SB3Policy
from optspread.types import MULTIPLIER, Chain, ChainProvider, OptionLeg, Position

SCHEMA_VERSION = 1
ATTRIBUTION_ENV = "OPTSPREAD_ATTRIBUTION"
REPO_ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ("delta", "gamma", "vega", "theta", "carry", "cost", "residual")
# Agreement with the saved row, relative to the larger of 1 dollar and the saved figure:
# actionmix.PNL_RTOL's, far above the rounding a platform moves a $1M book by, far
# below a cent.
PNL_RTOL = 1e-9

Source = SourceName
PolicyLoader = Callable[[Path, RunConfig], Policy]


def default_root(env: Mapping[str, str] | None = None) -> Path:
    """``$OPTSPREAD_ATTRIBUTION`` if set, else ``attribution/`` at the repository root."""
    value = (os.environ if env is None else env).get(ATTRIBUTION_ENV, "")
    return Path(value).expanduser() if value else REPO_ROOT / "attribution"


# -- the typed result, which is also the JSON schema --------------------------- #


class Terms(BaseModel):
    """One decomposition. ``explained + residual == pnl``; every field in dollars."""

    model_config = ConfigDict(frozen=True)

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    carry: float = 0.0
    cost: float = 0.0  # minus what was charged
    residual: float = 0.0
    pnl: float = 0.0  # the equity change being decomposed
    market: float = 0.0  # the held legs' actual mark-to-market change

    @property
    def explained(self) -> float:
        return self.delta + self.gamma + self.vega + self.theta + self.carry + self.cost

    @property
    def greeks(self) -> float:
        return self.delta + self.gamma + self.vega + self.theta


class StepAttribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: int  # the step starts here: the trade's day
    spot: float
    spot_next: float
    action_id: int  # the structure held over the move; 0 when flat
    n_legs: int
    terms: Terms


class EpisodeAttribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    n_steps: int
    n_trades: int
    reproduces: bool | None  # equal to the run.json row; None with no run to compare
    totals: Terms
    gross: float
    abs_residual: float
    residual_share: float
    max_abs_step_residual: float
    steps: tuple[StepAttribution, ...] = ()
    agrees: bool | None = None  # within PNL_RTOL of the row, same trades and steps


class RunAttribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    run_id: str
    config_hash: str
    policy: str
    premium_vol_pts: float
    source: Source
    run_dir: str | None
    reproduces: bool | None
    components: tuple[str, ...] = COMPONENTS
    totals: Terms
    gross: float
    abs_residual: float
    residual_share: float
    max_abs_step_residual: float
    episodes: tuple[EpisodeAttribution, ...]
    agrees: bool | None = None  # every episode agrees; None with no run, or written before

    def without_steps(self) -> RunAttribution:
        return self.model_copy(
            update={"episodes": tuple(e.model_copy(update={"steps": ()}) for e in self.episodes)}
        )


def dump_attribution(result: RunAttribution, *, steps: bool = True) -> str:
    """The JSON the dashboard reads (schema in the module docstring)."""
    return (result if steps else result.without_steps()).model_dump_json()


def write_attribution(result: RunAttribution, path: Path, *, steps: bool = True) -> Path:
    """Write ``result`` to ``path``, refusing a directory that is a run."""
    if (path.parent / MANIFEST_FILENAME).exists():
        raise ValueError(f"{path.parent} is a run directory; attribution is written elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_attribution(result, steps=steps), newline="\n")
    return path


def read_attribution(path: Path) -> RunAttribution:
    result = RunAttribution.model_validate_json(path.read_text())
    if result.schema_version != SCHEMA_VERSION:
        raise ValueError(f"attribution is schema v{result.schema_version}, not v{SCHEMA_VERSION}")
    return result


def output_name(result: RunAttribution) -> str:
    suffix = "" if result.source == "synthetic" else f".{result.source}"
    return f"{result.run_id}{suffix}.json"


# -- the arithmetic ---------------------------------------------------------- #


def leg_terms(leg: OptionLeg, before: Chain, after: Chain) -> Terms:
    """One leg over one step: the greeks of ``before`` against the actual move.

    ``pnl`` is the leg's own market move here, so ``residual`` is the leg's Taylor
    remainder; carry and cost belong to the book and are zero.
    """
    g = before.greeks(leg.right, leg.strike, leg.expiry_day)
    scale = leg.qty * MULTIPLIER
    dS = after.spot - before.spot
    dsigma = after.iv_at(leg.expiry_day, leg.strike) - before.iv_at(leg.expiry_day, leg.strike)
    dt = (after.t - before.t) / before.trading_days_per_year
    market = scale * (after.mark(leg) - before.mark(leg))
    delta = scale * g.delta * dS
    gamma = scale * 0.5 * g.gamma * dS * dS
    vega = scale * g.vega * dsigma
    theta = scale * g.theta * dt
    return Terms(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        residual=market - (delta + gamma + vega + theta),
        pnl=market,
        market=market,
    )


def attribute_step(
    day: int,
    held: Position | None,
    before: Chain,
    after: Chain,
    pnl: float,
    charged: float,
    carry: float = 0.0,
) -> StepAttribution:
    """Decompose one step's equity change ``pnl``, of which ``charged`` was cost."""
    legs = held.legs if held is not None else ()
    per_leg = [leg_terms(leg, before, after) for leg in legs]
    terms = _sum_terms(per_leg, pnl, -charged, carry)
    return StepAttribution(
        day=day,
        spot=before.spot,
        spot_next=after.spot,
        action_id=0 if held is None else held.action_id,
        n_legs=len(legs),
        terms=terms,
    )


def _sum_terms(parts: Sequence[Terms], pnl: float, cost: float, carry: float = 0.0) -> Terms:
    """Componentwise sums of ``parts`` plus ``carry``, with ``pnl`` and ``cost`` given and
    the residual closed against ``pnl``: the one place the identity is made."""
    delta, gamma, vega, theta, part_carry, market = (
        math.fsum(getattr(t, n) for t in parts)
        for n in ("delta", "gamma", "vega", "theta", "carry", "market")
    )
    carry += part_carry
    explained = delta + gamma + vega + theta + carry + cost
    return Terms(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        carry=carry,
        cost=cost,
        residual=pnl - explained,
        pnl=pnl,
        market=market,
    )


def _share(abs_residual: float, gross: float) -> float:
    return abs_residual / gross if gross > 0.0 else 0.0


# -- re-simulation ----------------------------------------------------------- #


class _Tap:
    """A ``ChainProvider`` that delegates everything and remembers what it saw.

    ``SpreadEnv.step`` trades at today's chain and then calls ``provider.step()``, so
    at that call the book holds exactly the structure carried over the move. The tap
    reads it there; nothing it does reaches the env or the path.
    """

    def __init__(self, inner: ChainProvider) -> None:
        self.inner = inner
        self.env: SpreadEnv | None = None
        self.chain: Chain | None = None  # the last chain handed out
        self.held: Position | None = None
        self.before: Chain | None = None

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
        if self.env is None:
            raise RuntimeError("tap stepped before it was bound to its env")
        # The env's book between the trade and the move: the one moment the held
        # structure exists separately from what settlement or a close leaves.
        self.held = self.env._book.position
        self.before = self.chain
        self.chain = self.inner.step()
        return self.chain


def attribute_episode(env: SpreadEnv, tap: _Tap, policy: Policy, seed: int) -> EpisodeAttribution:
    """``harness.run_episode`` step for step, with each step decomposed."""
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    equity0, total_cost, done = info["equity"], 0.0, False
    steps: list[StepAttribution] = []
    while not done:
        day = int(info["day"])
        obs, _, done, _, info = env.step(policy.act(obs, info))
        total_cost += info["cost"]
        assert tap.before is not None and tap.chain is not None
        steps.append(
            attribute_step(day, tap.held, tap.before, tap.chain, info["pnl"], info["cost"])
        )
    pnl = info["equity"] - equity0
    residuals = [abs(s.terms.residual) for s in steps]
    gross = math.fsum(abs(s.terms.market + s.terms.carry) for s in steps)
    abs_residual = math.fsum(residuals)
    return EpisodeAttribution(
        seed=seed,
        n_steps=len(steps),
        n_trades=int(info["n_trades"]),
        reproduces=None,
        totals=_sum_terms([s.terms for s in steps], pnl, -total_cost),
        gross=gross,
        abs_residual=abs_residual,
        residual_share=_share(abs_residual, gross),
        max_abs_step_residual=max(residuals, default=0.0),
        steps=tuple(steps),
    )


def _compared(episode: EpisodeAttribution, row: EpisodeRow | None) -> EpisodeAttribution:
    """``episode`` with its verdicts against the saved ``row`` (False when there is none)."""
    if row is None:
        return episode.model_copy(update={"reproduces": False, "agrees": False})
    pnl, cost = episode.totals.pnl, -episode.totals.cost
    same_counts = episode.n_trades == row.n_trades and episode.n_steps == row.steps
    exact = same_counts and pnl == row.pnl and cost == row.cost
    close = same_counts and all(
        abs(a - b) <= PNL_RTOL * max(1.0, abs(b)) for a, b in ((pnl, row.pnl), (cost, row.cost))
    )
    return episode.model_copy(update={"reproduces": exact, "agrees": close})


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


def attribute_config(
    config: RunConfig,
    seeds: Sequence[int] | None = None,
    *,
    policy: Policy | None = None,
    provider: ChainProvider | None = None,
    source: Source = "synthetic",
    rows: Mapping[int, EpisodeRow] | None = None,
    run_dir: Path | None = None,
) -> RunAttribution:
    """Attribute the episodes of ``config`` (all, or ``seeds``).

    ``provider`` defaults to the synthetic world ``harness.build_env`` builds; pass a
    real one (``OptionMetricsSource(...).provider()``) with ``source`` to say so. A
    ppo config needs its trained ``policy``. ``rows`` are the saved episodes to check
    each re-simulated one against.
    """
    if policy is not None:
        pol = policy
    elif config.policy == "ppo":
        raise ValueError("a ppo arm is attributed with its trained policy; pass it in")
    else:
        pol = make_policy(config.policy, roll_dte=config.env.roll_dte)
    tap = _Tap(provider if provider is not None else SyntheticChainProvider(config.market))
    env = SpreadEnv(tap, make_cost_model(config.cost), config.env)
    tap.env = env

    episodes = []
    for seed in _seeds(config, seeds):
        ep = attribute_episode(env, tap, pol, seed)
        if rows is not None:
            ep = _compared(ep, rows.get(seed))
        episodes.append(ep)

    totals = _sum_terms(
        [e.totals for e in episodes],
        math.fsum(e.totals.pnl for e in episodes),
        math.fsum(e.totals.cost for e in episodes),
    )
    gross = math.fsum(e.gross for e in episodes)
    abs_residual = math.fsum(e.abs_residual for e in episodes)
    h = config_hash(config)
    return RunAttribution(
        run_id=f"{h}-s{config.seed}",
        config_hash=h,
        policy=config.policy,
        premium_vol_pts=config.market.iv_premium_vol_pts,
        source=source,
        run_dir=None if run_dir is None else str(run_dir),
        reproduces=None if rows is None else all(e.reproduces for e in episodes),
        agrees=None if rows is None else all(e.agrees for e in episodes),
        totals=totals,
        gross=gross,
        abs_residual=abs_residual,
        residual_share=_share(abs_residual, gross),
        max_abs_step_residual=max((e.max_abs_step_residual for e in episodes), default=0.0),
        episodes=tuple(episodes),
    )


def load_sb3_policy(path: Path, config: RunConfig) -> Policy:  # pragma: no cover - train extra
    """The saved PPO policy of a run, greedy, on one torch thread as it was evaluated."""
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(TORCH_THREADS)
    device = config.train.device if config.train is not None else "cpu"
    model: Any = PPO.load(path, device=device)
    return SB3Policy(model)


def recorded_source(manifest: RunManifest) -> ChainSource:
    """The source a run's episodes came from, refusing a panel rebuilt since.

    As replay does: the harness writes real-data runs on A8's eval split only, so a run
    recorded on other dates (``DataRecord.ranges``, another split) is refused, and so is
    one whose panel's files no longer hash to the panel it recorded. The one rule for
    every re-simulation of a run (attribution, capital, the action mix)."""
    data = manifest.data_record
    if data.ranges is not None or data.effective_split not in (None, "eval"):
        dates = data.ranges if data.ranges is not None else f"the {data.effective_split} split"
        raise ValueError(f"{manifest.run_id} ran on {dates}; only A8's eval split is re-simulated")
    source = make_source(data.source, manifest.config.market, split="eval")
    now = DataRecord.of(source)
    if now.panel != data.panel:
        raise ValueError(
            f"the {data.source} panel is not the one {manifest.run_id} read "
            f"(recorded {data.panel_id}, found {now.panel_id})"
        )
    return source


def recorded_provider(manifest: RunManifest) -> tuple[SourceName, ChainProvider]:
    """``recorded_source``'s name and the provider it serves the eval split with."""
    return manifest.data_record.source, recorded_source(manifest).provider()


def attribute_run(
    run_dir: Path,
    seeds: Sequence[int] | None = None,
    *,
    load_policy: PolicyLoader = load_sb3_policy,
) -> RunAttribution:
    """Attribute a completed run from its manifest; never writes into ``run_dir``.

    A run whose files no longer match their manifest is refused, as replay refuses
    it. A real-data run is re-simulated on the panel and split it recorded
    (``recorded_provider``) and refused on another. A ppo run is attributed with the
    ``policy.zip`` it saved (``load_policy``, Stable-Baselines3 by default), not retrained.
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
            raise FileNotFoundError(f"a ppo run is attributed with its policy; {path} is missing")
        policy = load_policy(path, config)
    rows = {row.seed: row for row in record.episodes}
    if manifest.data_record.source == "synthetic":  # the generator, as it always was
        return attribute_config(config, seeds, policy=policy, rows=rows, run_dir=run_dir)
    source, provider = recorded_provider(manifest)
    return attribute_config(
        config, seeds, policy=policy, provider=provider, source=source, rows=rows, run_dir=run_dir
    )
