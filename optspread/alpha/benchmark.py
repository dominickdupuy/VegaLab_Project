"""Score an arm trained on a frozen synthetic dataset against what it should have earned.

Lane: Dominick Dupuy.

Four comparisons, each on paths the arm was not trained on unless stated:

* **attainable**: the oracle on the signal world minus the oracle on the twin, paired by
  path. The oracle is the best of a fixed scripted menu, chosen on a *fresh* sample of
  the same generator (never the frozen data), so the ceiling cannot itself be overfit.
  A menu policy qualifies only if it beats flat by |z| > 2 there; otherwise flat is the
  oracle and the attainable edge is zero. The choice depends on the generator and the
  market, not on the dataset's sizes, so it is cached (``ORACLE_DIR``): a sweep over the
  number of train paths chooses once.
* **captured**: the arm on the signal world minus the arm on the twin, paired. The twin
  has the same spot shocks and nothing planted, so cost drag and path luck cancel and
  what is left came from the signal.
* **overfit gap**: the arm on its own training paths minus the held-out paths.
* **noise fit** (learned arms only): the same learner trained on the twin's training
  paths, in sample minus out of sample. There is nothing to learn on the twin, so any
  in-sample edge is memorised noise.

A learned arm is trained on ``n_seeds`` training seeds (and so is its noise control), each
on the same frozen paths. A figure's standard error then has two parts, the split the
power lab uses: the spread of the per-seed figures (training luck, tau^2 / S) and the
path noise of the seed-averaged figure (sigma^2 / (S N), estimated from the per-path
averages). With one seed the first part is absent and the se is the path se alone, which
says nothing about training luck; the page says so. Every verdict is at |z| > 2.

An arm is a built-in (``ARMS``) or a plug-in (``optspread.alpha.plugins``): a plug-in rule
is scored like the scripted arms and a plug-in learner trained like the learned ones, on
the same frozen paths, seeds, twin and oracle. Each result also keeps the captured edge on
``BLOCKS`` disjoint blocks of held-out paths, the per-"seed" figures a deterministic arm's
detection test reads (``optspread.alpha.floor``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from optspread.actions import FLAT, IRON_CONDOR, PUT_CREDIT_SPREAD, SHORT_STRANGLE
from optspread.alpha import plugins as plug
from optspread.alpha.dataset import Split, SynthDataset, SynthSpec, World, generate_arrays
from optspread.alpha.generators import FloatArray, MrVrpParams
from optspread.alpha.provider import DatasetChainProvider, dataset_env
from optspread.config import (
    LEARNED_ARMS,
    EnvConfig,
    LearnedArm,
    PolicyName,
    RunConfig,
    TrainConfig,
    config_hash,
)
from optspread.env import OBS_NAMES
from optspread.fileio import atomic_write_text
from optspread.harness import run_episode
from optspread.policies import FlatPolicy, Info, Observation, Policy, make_policy
from optspread.scripted import make_seller
from optspread.types import ChainProvider, Position

BuiltinArm = Literal["ppo", "a2c", "dqn", "delta_seller", "always_on", "flat"]
ARMS: tuple[BuiltinArm, ...] = ("ppo", "a2c", "dqn", "delta_seller", "always_on", "flat")
# A built-in arm's name or a registered plug-in's (``optspread.alpha.plugins``).
Arm = str
# Config and the chains to train on in, trained policy out. ``learned_trainer`` is the
# production one (``optspread.train.train``); the tests pass a stand-in without torch.
SynthTrainer = Callable[[RunConfig, ChainProvider], Policy]

FRESH_N = 200  # paths in the fresh sample the oracle is chosen on
FRESH_OFFSET = 2**30  # fresh path seeds start here past the dataset's seed
MAX_SEEDS = 10
Z = 2.0
ROLL_DTE = 5
BENCH_DIR = Path("alpha") / "bench"
BENCH_FILE = "benchmark.json"
ORACLE_DIR = Path("alpha") / "oracle"
BLOCKS = 5  # disjoint held-out path blocks, the S of a deterministic arm's test
_IV = OBS_NAMES.index("atm_iv")


# -- arms ---------------------------------------------------------------------- #


def arm_names() -> tuple[str, ...]:
    """Every arm a benchmark can run: the built-ins, then the registered plug-ins."""
    return (*ARMS, *plug.plugins())


def check_arm(arm: str) -> str:
    """``arm`` if it is a built-in or a registered plug-in; a ValueError naming both otherwise."""
    if arm in ARMS or plug.find(arm) is not None:
        return arm
    extra = ", ".join(plug.plugins()) or f"none (set {plug.PLUGINS_ENV})"
    raise ValueError(
        f"unknown arm {arm!r}: the built-in arms are {', '.join(ARMS)}; plug-ins: {extra}"
    )


def is_learned(arm: str) -> bool:
    """Trained per seed (a built-in learner or a plug-in with a trainer), with a budget."""
    if arm in LEARNED_ARMS:
        return True
    found = plug.find(arm)
    return found is not None and found.plugin.trainable


def arm_key(arm: str) -> str:
    """What a benchmark directory is named by: the arm, or a plug-in's name and code digest."""
    found = plug.find(arm)
    return found.key if found is not None else arm


# -- policies ------------------------------------------------------------------ #


class TimingSeller:
    """Holds ``action_id`` (rolled at ``roll_dte``) only while ATM implied vol is above
    ``iv_floor``; flat otherwise. The oracle for a premium that comes and goes."""

    def __init__(self, action_id: int, roll_dte: int, iv_floor: float) -> None:
        self.action_id, self.roll_dte, self.iv_floor = action_id, roll_dte, iv_floor

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: Observation, info: Info) -> int:
        if float(obs[_IV]) <= self.iv_floor:
            return FLAT
        pos: Position | None = info["position"]
        if pos is not None and pos.first_expiry - int(info["day"]) <= self.roll_dte:
            return FLAT
        return self.action_id


def menu(spec: SynthSpec) -> dict[str, Policy]:
    """The scripted policies the oracle is chosen from."""
    out: dict[str, Policy] = {"flat": FlatPolicy()}
    for name, action in (
        ("condor", IRON_CONDOR),
        ("strangle", SHORT_STRANGLE),
        ("put_spread", PUT_CREDIT_SPREAD),
    ):
        out[f"hold_{name}"] = make_seller(action, None)
        out[f"roll_{name}"] = make_seller(action, ROLL_DTE)
    if isinstance(spec.generator, MrVrpParams):
        for k in (0, 1, 2):
            out[f"timing_{k}"] = TimingSeller(SHORT_STRANGLE, ROLL_DTE, spec.sigma + k / 100.0)
    return out


def learned_trainer(config: RunConfig, provider: ChainProvider) -> Policy:
    """The learner ``config.train.algorithm`` names, trained on ``provider``'s chains."""
    from optspread.train import train

    return train(config, provider=provider)


# -- statistics ---------------------------------------------------------------- #


def _se(x: FloatArray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def _var_of_mean(x: FloatArray) -> float:
    return float(x.var(ddof=1) / len(x)) if len(x) > 1 else 0.0


def level(m: FloatArray) -> tuple[float, float]:
    """Mean and se of ``m`` (seeds x paths): seed spread plus path noise of the seed average."""
    m = np.atleast_2d(m)
    se2 = _var_of_mean(m.mean(axis=1)) + _var_of_mean(m.mean(axis=0))
    return float(m.mean()), float(np.sqrt(se2))


def paired(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    """``a - b`` on the same paths (and seeds), as ``level`` reads it."""
    return level(np.atleast_2d(a) - np.atleast_2d(b))


def unpaired(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    """mean(a) - mean(b) on different paths, the same seeds: seed spread of the per-seed
    difference plus the path noise of each seed-averaged mean."""
    a2, b2 = np.atleast_2d(a), np.atleast_2d(b)
    seed = _var_of_mean(a2.mean(axis=1) - b2.mean(axis=1))
    path = _var_of_mean(a2.mean(axis=0)) + _var_of_mean(b2.mean(axis=0))
    return float(a2.mean() - b2.mean()), float(np.sqrt(seed + path))


def ratio(num: tuple[float, float], den: tuple[float, float]) -> tuple[float, float]:
    """num / den and its delta-method se, the two taken as independent."""
    (n, sn), (d, sd) = num, den
    r = n / d
    return r, float(np.sqrt(sn * sn + r * r * sd * sd) / abs(d))


def _z(value: float, se: float) -> float:
    if se > 0.0:
        return value / se
    return 0.0 if value == 0.0 else float(np.sign(value)) * np.inf


# -- results ------------------------------------------------------------------- #


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class BenchRow(_Frozen):
    policy: str  # arm | noise_control | oracle | flat
    world: World
    split: Split
    mean: float
    se: float
    n: int


class BenchMenuRow(_Frozen):
    policy: str
    mean: float
    se: float


class BenchStat(_Frozen):
    value: float | None
    se: float | None
    verdict: str


class SeedResult(_Frozen):
    """One training seed's figures, on the same paths as every other seed."""

    seed: int
    captured: float
    capture_ratio: float | None
    overfit_gap: float
    noise_fit: float | None


class Benchmark(_Frozen):
    dataset: str
    generator: str
    arm: str
    total_timesteps: int | None
    n_seeds: int = 1
    n_train: int = 0
    oracle: str
    noise_control: bool
    menu: tuple[BenchMenuRow, ...]
    rows: tuple[BenchRow, ...]
    attainable: BenchStat
    captured: BenchStat
    capture_ratio: BenchStat
    overfit_gap: BenchStat
    noise_fit: BenchStat | None
    seeds: tuple[SeedResult, ...] = Field(default_factory=tuple)
    # The seed-averaged captured edge on BLOCKS disjoint blocks of held-out paths, in order;
    # empty on results written before it was kept, or with under 2 paths a block.
    blocks: tuple[float, ...] = Field(default_factory=tuple)
    # A plug-in arm's import path and code digest, ``pkg.mod:attr@digest``.
    plugin: str | None = None


def bench_path(
    root: Path, dataset: str, arm: Arm, total_timesteps: int | None, n_seeds: int = 1
) -> Path:
    name = f"{dataset}-{arm_key(arm)}-{total_timesteps or 0}-s{n_seeds}"
    return root / BENCH_DIR / name / BENCH_FILE


def load_benchmark(path: Path) -> Benchmark:
    return Benchmark.model_validate_json(path.read_text())


def list_benchmarks(root: Path, dataset: str) -> list[Benchmark]:
    """Every benchmark written for ``dataset``, newest file first; unreadable ones skipped."""
    files = sorted(
        (root / BENCH_DIR).glob(f"{dataset}-*/{BENCH_FILE}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out = []
    for f in files:
        try:
            out.append(load_benchmark(f))
        except (OSError, ValueError):
            continue
    return out


# -- the oracle ---------------------------------------------------------------- #


class OracleChoice(_Frozen):
    policy: str
    menu: tuple[BenchMenuRow, ...]


def oracle_key(spec: SynthSpec) -> str:
    """What the oracle's choice depends on: the generator, the market, the seed and the
    fresh sample, not the dataset's sizes (one sweep over train paths chooses once)."""
    base = spec.model_copy(update={"n_train": 1, "n_heldout": 2})
    return f"{config_hash(base)}-f{FRESH_N}-r{ROLL_DTE}"


def fresh_dataset(ds: SynthDataset) -> SynthDataset:
    """An in-memory held-out sample of the same generator on seeds no dataset uses."""
    spec = ds.spec.model_copy(
        update={"seed": ds.spec.seed + FRESH_OFFSET, "n_train": 1, "n_heldout": FRESH_N}
    )
    arrays = generate_arrays(spec, {"heldout": spec.seeds("heldout")})
    return SynthDataset(spec, f"{ds.hash}-fresh", ds.dir, arrays)


def choose_oracle(ds: SynthDataset, root: Path, say: Callable[[str], None]) -> OracleChoice:
    path = root / ORACLE_DIR / f"{oracle_key(ds.spec)}.json"
    say(f"choosing the oracle on {FRESH_N} fresh paths")
    try:
        choice = OracleChoice.model_validate_json(path.read_text())
        say(f"oracle: {choice.policy} (chosen before for this generator; reused)")
        return choice
    except (OSError, ValueError):
        pass
    fresh = fresh_dataset(ds)
    scored = []
    for name, pol in menu(ds.spec).items():
        pnls = evaluate_on(fresh, "signal", "heldout", pol)
        scored.append(BenchMenuRow(policy=name, mean=float(pnls.mean()), se=_se(pnls)))
    # The oracle must show its edge, not just top a noisy ranking: a policy that does not
    # beat flat by Z standard errors on the fresh sample is not an edge, and flat stands.
    shown = [m for m in scored if _z(m.mean, m.se) > Z]
    best = max(shown, key=lambda m: m.mean) if shown else scored[0]
    choice = OracleChoice(policy=best.policy, menu=tuple(scored))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, choice.model_dump_json(indent=2), newline="\n")
    say(f"oracle: {best.policy} ({best.mean:,.0f} a path on the fresh sample)")
    return choice


# -- running ------------------------------------------------------------------- #


def evaluate_on(ds: SynthDataset, world: World, split: Split, policy: Policy) -> FloatArray:
    """Per-path net P&L of ``policy`` over every path of one world and split, in order."""
    env = dataset_env(ds, world, split)
    return np.array(
        [run_episode(env, policy, i).pnl for i in range(ds.n_paths(split))], dtype=np.float64
    )


def train_config(ds: SynthDataset, arm: LearnedArm, total_timesteps: int, seed: int) -> RunConfig:
    """What a learner is trained with; a plug-in learner gets the ``ppo`` one, a placeholder
    for the arm name ``RunConfig`` cannot hold, with its own seed and budget."""
    return RunConfig(
        market=ds.spec.market(),
        env=EnvConfig(episode_days=ds.spec.n_days),
        policy=arm,
        train=TrainConfig(algorithm=arm, total_timesteps=total_timesteps, seed=seed),
    )


def stages(arm: Arm, n_seeds: int) -> int:
    """Lines ``run_benchmark`` logs when it computes, for a progress bar."""
    return 2 * n_seeds + 4 if is_learned(arm) else 4


def captured_blocks(
    held: FloatArray, twin: FloatArray, n_blocks: int = BLOCKS
) -> tuple[float, ...]:
    """The seed-averaged captured edge on ``n_blocks`` equal, disjoint blocks of held-out
    paths (a remainder past ``n_blocks`` times the block size is left out); empty when a
    block would hold fewer than two paths."""
    per_path = (np.atleast_2d(held) - np.atleast_2d(twin)).mean(axis=0)
    size = len(per_path) // n_blocks
    if size < 2:
        return ()
    return tuple(float(b.mean()) for b in per_path[: size * n_blocks].reshape(n_blocks, size))


EDGE_SHOWN = "the planted edge is tradeable: the oracle earned it"
NO_EDGE = "no attainable edge: the best scripted policy earned nothing measurable from it"


def edge_shown(b: Benchmark) -> bool:
    """The benchmark's own verdict on the attainable edge: the oracle beat the twin by
    ``Z`` standard errors (what the capture ratio needs), not just a nonzero figure."""
    return b.attainable.verdict == EDGE_SHOWN


def _verdicts(
    attainable: tuple[float, float],
    captured: tuple[float, float],
    ratio_: tuple[float, float] | None,
    gap: tuple[float, float],
    noise: tuple[float, float] | None,
) -> tuple[BenchStat, BenchStat, BenchStat, BenchStat, BenchStat | None]:
    za, zc, zg = _z(*attainable), _z(*captured), _z(*gap)
    a = BenchStat(
        value=attainable[0],
        se=attainable[1],
        verdict=EDGE_SHOWN if za > Z else NO_EDGE,
    )
    if zc > Z:
        cv = "the arm earned money from the planted signal"
    elif zc < -Z:
        cv = "the arm did worse on the signal world than on its noise twin"
    else:
        cv = "the arm's edge from the signal is not distinguishable from zero"
    c = BenchStat(value=captured[0], se=captured[1], verdict=cv)
    if ratio_ is None:
        r = BenchStat(value=None, se=None, verdict="no attainable edge to capture")
    else:
        r = BenchStat(
            value=ratio_[0], se=ratio_[1], verdict="the share of the attainable edge the arm took"
        )
    g = BenchStat(
        value=gap[0],
        se=gap[1],
        verdict="overfit: the arm did better on its training paths than on new ones"
        if zg > Z
        else "no detectable overfitting",
    )
    nf = None
    if noise is not None:
        nf = BenchStat(
            value=noise[0],
            se=noise[1],
            verdict="noise memorised: the noise-trained arm earned in sample only"
            if _z(*noise) > Z
            else "noise not memorised",
        )
    return a, c, r, g, nf


def run_benchmark(
    ds: SynthDataset,
    arm: Arm,
    total_timesteps: int | None,
    root: Path,
    trainer: SynthTrainer = learned_trainer,
    log: Callable[[str], None] | None = None,
    n_seeds: int = 1,
) -> Benchmark:
    """Train (if learned) on ``n_seeds`` seeds, evaluate, and write ``benchmark.json``.

    ``trainer`` trains the built-in learners; a plug-in learner is trained by its own.
    """
    check_arm(arm)
    found = plug.find(arm)
    learned = is_learned(arm)
    if learned != (total_timesteps is not None):
        raise ValueError(
            "a training budget (total_timesteps) is given exactly when the arm is learned"
        )
    if not 1 <= n_seeds <= MAX_SEEDS or (not learned and n_seeds != 1):
        raise ValueError(f"n_seeds must be 1 to {MAX_SEEDS} for a learned arm and 1 otherwise")
    path = bench_path(root, ds.hash, arm, total_timesteps, n_seeds)
    try:
        cached = load_benchmark(path) if path.is_file() else None
    except (OSError, ValueError):
        cached = None  # unreadable (cut short, or an older shape): computed again below
    # A deterministic result written before blocks were kept is quick to redo; a
    # learner's is kept as it is (its test reads the per-seed figures).
    if cached is not None and (learned or cached.blocks or ds.n_paths("heldout") < 2 * BLOCKS):
        return cached
    say = log if log is not None else (lambda _: None)
    train_with = found.plugin.trainer if found is not None and found.plugin.trainer else trainer

    def row(policy: str, world: World, split: Split, pnls: FloatArray) -> BenchRow:
        mean, se = level(pnls)
        return BenchRow(policy=policy, world=world, split=split, mean=mean, se=se, n=pnls.shape[-1])

    # seeds x paths for the arm on each split it is scored on, and for its noise control
    held: list[FloatArray] = []
    twin: list[FloatArray] = []
    train: list[FloatArray] = []
    nc_in: list[FloatArray] = []
    nc_out: list[FloatArray] = []
    if learned:
        assert total_timesteps is not None
        for seed in range(n_seeds):
            learner = cast(LearnedArm, arm if found is None else "ppo")
            config = train_config(ds, learner, total_timesteps, seed)
            say(f"seed {seed}: training {arm} on {ds.n_paths('train')} signal paths")
            policy = train_with(config, DatasetChainProvider(ds, "signal", "train"))
            train.append(evaluate_on(ds, "signal", "train", policy))
            held.append(evaluate_on(ds, "signal", "heldout", policy))
            twin.append(evaluate_on(ds, "twin", "heldout", policy))
            say(f"seed {seed}: training the noise control on the twin's train paths")
            control = train_with(config, DatasetChainProvider(ds, "twin", "train"))
            nc_in.append(evaluate_on(ds, "twin", "train", control))
            nc_out.append(evaluate_on(ds, "twin", "heldout", control))
        say(f"evaluated {n_seeds} seed(s) on train, held-out and the twin")
    else:
        if found is not None and found.plugin.policy is not None:
            policy = found.plugin.policy()
        else:
            policy = make_policy(cast(PolicyName, arm), roll_dte=ROLL_DTE)
        train.append(evaluate_on(ds, "signal", "train", policy))
        held.append(evaluate_on(ds, "signal", "heldout", policy))
        twin.append(evaluate_on(ds, "twin", "heldout", policy))
        say(f"evaluated {arm} on train, held-out and the twin")
    a_train, a_held, a_twin = np.array(train), np.array(held), np.array(twin)

    choice = choose_oracle(ds, root, say)
    oracle = menu(ds.spec)[choice.policy]
    or_train = evaluate_on(ds, "signal", "train", oracle)
    or_held = evaluate_on(ds, "signal", "heldout", oracle)
    or_twin = evaluate_on(ds, "twin", "heldout", oracle)

    rows = [
        row("arm", "signal", "train", a_train),
        row("arm", "signal", "heldout", a_held),
        row("arm", "twin", "heldout", a_twin),
    ]
    noise_fit = None
    if learned:
        n_in, n_out = np.array(nc_in), np.array(nc_out)
        rows += [row("noise_control", "twin", "train", n_in)]
        rows += [row("noise_control", "twin", "heldout", n_out)]
        noise_fit = unpaired(n_in, n_out)
    rows += [
        row("oracle", "signal", "train", or_train),
        row("oracle", "signal", "heldout", or_held),
        row("oracle", "twin", "heldout", or_twin),
        row("flat", "signal", "heldout", np.zeros(ds.n_paths("heldout"))),
    ]

    attainable = paired(or_held, or_twin)
    captured = paired(a_held, a_twin)
    has_edge = _z(*attainable) > Z
    ratio_ = ratio(captured, attainable) if has_edge else None
    stats = _verdicts(attainable, captured, ratio_, unpaired(a_train, a_held), noise_fit)

    per_seed = []
    for s in range(a_held.shape[0]):
        cap = float((a_held[s] - a_twin[s]).mean())
        per_seed.append(
            SeedResult(
                seed=s,
                captured=cap,
                capture_ratio=cap / attainable[0] if has_edge else None,
                overfit_gap=float(a_train[s].mean() - a_held[s].mean()),
                noise_fit=float(nc_in[s].mean() - nc_out[s].mean()) if learned else None,
            )
        )
    bench = Benchmark(
        dataset=ds.hash,
        generator=ds.spec.generator.kind,
        arm=arm,
        total_timesteps=total_timesteps,
        n_seeds=n_seeds,
        n_train=ds.n_paths("train"),
        oracle=choice.policy,
        noise_control=learned,
        menu=choice.menu,
        rows=tuple(rows),
        attainable=stats[0],
        captured=stats[1],
        capture_ratio=stats[2],
        overfit_gap=stats[3],
        noise_fit=stats[4],
        seeds=tuple(per_seed),
        blocks=captured_blocks(a_held, a_twin),
        plugin=f"{found.source}@{found.digest}" if found is not None else None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, bench.model_dump_json(indent=2), newline="\n")
    say(f"{bench.captured.verdict}; {bench.overfit_gap.verdict}")
    return bench
