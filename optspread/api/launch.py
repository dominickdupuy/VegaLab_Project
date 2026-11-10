"""API payloads for launch: a PPO preset, the wall-time estimate at the rates A7 measured,
and a live preview of the run a spec stands for.

Lane: Ilan Danial (machinery). Moved from ``optspread.webviews``, which re-exports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from optspread.api._base import Out, _f
from optspread.api.jobs import JobOptionsOut, PresetOut, job_options
from optspread.api.study import A7_EVAL_RATE, A7_TRAIN_RATE
from optspread.dashboard.jobs.spec import JobSpec
from optspread.dashboard.runindex import RunRow
from optspread.manifest import MANIFEST_FILENAME, RunManifest

# -- launch: a PPO preset, and the estimate with training in it ---------------------- #

PPO_PRESET = PresetOut(
    name="ppo_short",
    label="PPO, short",
    description=(
        "PPO at 4 vol pts, 100 episodes, 100,000 training steps: a short illustration; "
        "the sweep trains 1,000,000."
    ),
    spec={
        **JobSpec().model_dump(mode="json"),
        "policy": "ppo",
        "iv_premium_vol_pts": 4.0,
        "n_episodes": 100,
        "total_timesteps": 100_000,
    },
)


class EstimateOut(Out):
    eval_steps: int
    train_steps: int
    eval_hours: float
    train_hours: float
    total_hours: float
    train_rate: float
    eval_rate: float
    source: str


def launch_options() -> JobOptionsOut:
    """``webapi.job_options`` with the PPO preset beside the scripted ones."""
    return job_options().model_copy(update={"ppo_preset": PPO_PRESET})


def launch_estimate(n_episodes: int, n_days: int, total_timesteps: int | None) -> EstimateOut:
    """Wall time of one run on one core at the rates A7 measured, training included."""
    steps = n_episodes * n_days
    train = int(total_timesteps or 0)
    eval_h = steps / A7_EVAL_RATE / 3600.0
    train_h = train / A7_TRAIN_RATE / 3600.0
    return EstimateOut(
        eval_steps=steps,
        train_steps=train,
        eval_hours=eval_h,
        train_hours=train_h,
        total_hours=eval_h + train_h,
        train_rate=A7_TRAIN_RATE,
        eval_rate=A7_EVAL_RATE,
        source="docs/PRE_REGISTRATION.md A7: 804 train and 1,538 eval steps/s/core",
    )


# -- launch: a live preview of the run a spec stands for ---------------------------- #

STRUCTURE_PLAIN = {
    "IRON_CONDOR": (
        "an iron condor (a short put and a short call, each with a long wing further out)"
    ),
    "SHORT_STRANGLE": "a short strangle (a short put and a short call, no wings)",
    "PUT_CREDIT_SPREAD": "a put credit spread (a short put with a long put further out)",
}


class PreviewLegOut(Out):
    right: str
    strike: float
    qty: int
    price: float  # per share, the chain's mid
    iv: float
    delta: float  # per-share delta of one long contract
    moneyness: float  # K / spot


class DayZeroOut(Out):
    """What the policy opens on day 0 of the first episode: the chain it sees, its legs."""

    policy: str
    reference: bool  # True for PPO: the fixed-delta seller's choice, shown for scale
    structure: str  # "FLAT" when nothing is opened
    plain: str  # one sentence
    spot: float
    tenor_days: int | None
    legs: list[PreviewLegOut]
    credit: float | None  # $ received for one structure (negative: paid)
    margin: float | None
    max_loss: float | None  # $, defined risk only
    breakevens: list[float]


class PeerOut(Out):
    key: str
    policy: str
    premium: float | None
    source: str
    n_episodes: int | None
    mean_pnl: float | None
    relation: str  # "same run" | "control" | "same arm" | "other arm"
    paired: bool  # starts on the same first seed: its episodes pair with this run's
    note: str


class PreviewOut(Out):
    ok: bool
    errors: list[str]
    run_id: str = ""
    config_hash: str = ""
    exists: bool = False  # this exact run is on disk: launching it reuses it
    run_key: str = ""
    world: str = ""  # the market the run sees, in one sentence
    implied_vol: float | None = None  # sigma plus the premium, decimal; None on real data
    day0: DayZeroOut | None = None
    day0_note: str = ""  # why there is no day 0, when there is none
    estimate: EstimateOut | None = None
    peers: list[PeerOut] = Field(default_factory=list)


def _clean(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _day_zero(config: Any, source: str) -> DayZeroOut:
    from optspread.actions import ACTION_NAMES, build_position
    from optspread.harness import build_env
    from optspread.margin import is_defined_risk, margin_requirement, max_loss
    from optspread.policies import DeltaSellerPolicy, make_policy
    from optspread.types import MULTIPLIER

    env = build_env(config, source)  # type: ignore[arg-type]
    obs, info = env.reset(seed=config.seed)
    learned = config.policy == "ppo"
    policy: Any = (
        DeltaSellerPolicy(roll_dte=config.env.roll_dte)
        if learned
        else make_policy(config.policy, roll_dte=config.env.roll_dte)
    )
    policy.reset(config.seed)
    action = int(policy.act(obs, info))
    chain = info["chain"]
    pos = build_position(action, chain, config.env)
    who = (
        "The fixed-delta seller"
        if learned
        else {
            "flat": "Flat",
            "always_on": "The always-on condor",
            "random": "The random policy",
            "delta_seller": "The fixed-delta seller",
        }.get(config.policy, config.policy)
    )
    if pos is None:
        return DayZeroOut(
            policy=config.policy,
            reference=learned,
            structure=ACTION_NAMES[action],
            plain=f"{who} stays flat on day 0: it opens nothing.",
            spot=float(chain.spot),
            tenor_days=None,
            legs=[],
            credit=None,
            margin=None,
            max_loss=None,
            breakevens=[],
        )
    legs = pos.legs
    exp = legs[0].expiry_day
    credit = -sum(leg.qty * leg.entry_price for leg in legs) * MULTIPLIER
    shorts = {leg.right: leg.strike for leg in legs if leg.qty < 0}
    per_share = credit / MULTIPLIER
    breakevens = [shorts["P"] - per_share] if "P" in shorts else []
    breakevens += [shorts["C"] + per_share] if "C" in shorts else []
    name = ACTION_NAMES[action]
    tenor = int(exp - chain.t)
    plain = (
        f"{who} opens {STRUCTURE_PLAIN.get(name, name)} {tenor} trading days to expiry, "
        f"short strikes at {config.env.short_delta:g} delta, taking in ${credit:,.0f}."
    )
    return DayZeroOut(
        policy=config.policy,
        reference=learned,
        structure=name,
        plain=plain,
        spot=float(chain.spot),
        tenor_days=tenor,
        legs=[
            PreviewLegOut(
                right=leg.right,
                strike=leg.strike,
                qty=leg.qty,
                price=leg.entry_price,
                iv=chain.iv_at(exp, leg.strike),
                delta=chain.greeks(leg.right, leg.strike, exp).delta,
                moneyness=leg.strike / float(chain.spot),
            )
            for leg in sorted(legs, key=lambda x: x.strike)
        ],
        credit=float(credit),
        margin=float(margin_requirement(legs, chain)),
        max_loss=float(max_loss(legs)) if is_defined_risk(legs) else None,
        breakevens=[float(b) for b in breakevens],
    )


def _peers(spec: JobSpec, config: Any, runs_root: Path, rows: Sequence[RunRow]) -> list[PeerOut]:
    from optspread.api.premiums import _world
    from optspread.api.runs import row_out, run_source

    world, _ = _world(config.model_dump(mode="json"), spec.source)
    run_id = f"{config_hash_of(config)}-s{config.seed}"
    found: list[tuple[int, PeerOut]] = []
    for row in rows:
        if row.status != "OK":
            continue
        try:
            other = RunManifest.read(row.path).config
        except (OSError, ValueError):
            continue
        if _world(other.model_dump(mode="json"), run_source(row))[0] != world:
            continue
        paired = other.seed == config.seed
        same_arm = other.policy == config.policy
        premium = float(other.market.iv_premium_vol_pts)
        mine = float(config.market.iv_premium_vol_pts)
        if row.run_id == run_id and same_arm:
            rank, relation, note = 0, "same run", "this exact run is on disk: launching reuses it"
        elif same_arm and premium == 0.0 and mine != 0.0:
            rank, relation = 1, "control"
            note = "the same arm at zero premium: the paired control" + (
                "" if paired else " (other seeds: not paired)"
            )
        elif same_arm:
            rank, relation, note = 2, "same arm", f"the same arm at {premium:g} vol pts"
        elif premium == mine:
            rank, relation, note = 3, "other arm", "another arm in the same world"
        else:
            continue
        out = row_out(row, runs_root)
        found.append(
            (
                rank * 2 + (0 if paired else 1),
                PeerOut(
                    key=out.key,
                    policy=row.policy,
                    premium=row.premium,
                    source=run_source(row),
                    n_episodes=row.n_episodes,
                    mean_pnl=_f(row.mean_pnl),
                    relation=relation,
                    paired=paired,
                    note=note,
                ),
            )
        )
    found.sort(key=lambda x: (x[0], x[1].premium or 0.0, x[1].key))
    # a relation met on the same seeds needs no unpaired runs beside it
    paired = {p.relation for _, p in found if p.paired}
    kept = [p for _, p in found if p.paired or p.relation not in paired]
    return kept[:MAX_PEERS]


MAX_PEERS = 12


def config_hash_of(config: Any) -> str:
    from optspread.config import config_hash

    return config_hash(config)


def launch_preview(raw: dict[str, Any], runs_root: Path, rows: Sequence[RunRow]) -> PreviewOut:
    """What the launch form's spec would run: refused with every reason, or its run id, the
    world in a sentence, what it opens on day 0, its wall time and the runs it meets."""
    from pydantic import ValidationError

    from optspread.dashboard.jobs.spec import LAUNCH_DIR, to_run_config
    from optspread.runs.rundir import source_root

    try:
        spec = JobSpec.model_validate(raw)
    except ValidationError as exc:
        return PreviewOut(ok=False, errors=[_clean(str(e["msg"])) for e in exc.errors()])
    config = to_run_config(spec)
    h = config_hash_of(config)
    run_id = f"{h}-s{config.seed}"
    run_dir = source_root(runs_root, spec.source) / LAUNCH_DIR / run_id
    real = spec.source != "synthetic"
    if real:
        world = (
            "Real SPX closes on A8's evaluation dates: each episode starts on a date drawn "
            "from the seed and trades the day's real surface; no premium is planted."
        )
        iv = None
    else:
        iv = spec.sigma + spec.iv_premium_vol_pts / 100.0
        world = (
            f"A simulated index whose path moves at {spec.sigma:.0%} vol while options are "
            f"priced at {iv:.0%}: a planted premium of {spec.iv_premium_vol_pts:g} vol "
            f"point{'' if spec.iv_premium_vol_pts == 1 else 's'} for a seller to earn."
            if spec.iv_premium_vol_pts
            else f"A simulated index at {spec.sigma:.0%} vol with options priced at the same "
            "vol: the null world, where selling earns nothing before costs."
        )
    day0, note = None, ""
    try:
        day0 = _day_zero(config, spec.source)
    except Exception as exc:  # a missing panel, a chain the deltas cannot reach
        note = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    steps = config.train.total_timesteps if config.train is not None else None
    return PreviewOut(
        ok=True,
        errors=[],
        run_id=run_id,
        config_hash=h,
        exists=(run_dir / MANIFEST_FILENAME).is_file(),
        run_key=run_dir.relative_to(runs_root).as_posix(),
        world=world,
        implied_vol=iv,
        day0=day0,
        day0_note=note,
        estimate=launch_estimate(spec.n_episodes, spec.n_days, steps),
        peers=_peers(spec, config, runs_root, rows),
    )
