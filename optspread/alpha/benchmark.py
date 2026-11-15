"""Score an arm trained on a frozen synthetic dataset against what it should have earned.

Four comparisons, each on paths the arm was not trained on unless stated:

* **attainable**: the oracle on the signal world minus the oracle on the twin, paired by
  path. The oracle is the best of a fixed scripted menu, chosen on a *fresh* sample of
  the same generator (never the frozen data), so the ceiling cannot itself be overfit.
  A menu policy qualifies only if it beats flat by |z| > 2 there; otherwise flat is the
  oracle and the attainable edge is zero.
* **captured**: the arm on the signal world minus the arm on the twin, paired. The twin
  has the same spot shocks and nothing planted, so cost drag and path luck cancel and
  what is left came from the signal.
* **overfit gap**: the arm on its own training paths minus the held-out paths.
* **noise fit** (learned arms only): the same learner trained on the twin's training
  paths, in sample minus out of sample. There is nothing to learn on the twin, so any
  in-sample edge is memorised noise.

Every number carries a standard error and a verdict at |z| > 2.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from optspread.actions import FLAT, IRON_CONDOR, PUT_CREDIT_SPREAD, SHORT_STRANGLE
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
)
from optspread.env import OBS_NAMES
from optspread.harness import run_episode
from optspread.policies import FlatPolicy, Info, Observation, Policy, make_policy
from optspread.scripted import make_seller
from optspread.types import ChainProvider, Position

Arm = Literal["ppo", "a2c", "dqn", "delta_seller", "always_on", "flat"]
ARMS: tuple[Arm, ...] = ("ppo", "a2c", "dqn", "delta_seller", "always_on", "flat")
# Config and the chains to train on in, trained policy out. ``learned_trainer`` is the
# production one (``optspread.train.train``); the tests pass a stand-in without torch.
SynthTrainer = Callable[[RunConfig, ChainProvider], Policy]

FRESH_N = 200  # paths in the fresh sample the oracle is chosen on
FRESH_OFFSET = 2**30  # fresh path seeds start here past the dataset's seed
BOOTSTRAP = 2_000
Z = 2.0
ROLL_DTE = 5
BENCH_DIR = Path("alpha") / "bench"
BENCH_FILE = "benchmark.json"
_IV = OBS_NAMES.index("atm_iv")


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


def paired(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    d = a - b
    return float(d.mean()), _se(d)


def unpaired(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    return float(a.mean() - b.mean()), float(np.hypot(_se(a), _se(b)))


def ratio(num: FloatArray, den: FloatArray) -> tuple[float, float]:
    """mean(num) / mean(den) and its paired-bootstrap standard error (fixed seed)."""
    idx = np.random.default_rng(0).integers(0, len(num), size=(BOOTSTRAP, len(num)))
    dens = den[idx].mean(axis=1)
    ok = dens != 0.0
    boot = num[idx].mean(axis=1)[ok] / dens[ok]
    return float(num.mean() / den.mean()), float(boot.std(ddof=1)) if len(boot) > 1 else 0.0


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


class Benchmark(_Frozen):
    dataset: str
    generator: str
    arm: Arm
    total_timesteps: int | None
    oracle: str
    noise_control: bool
    menu: tuple[BenchMenuRow, ...]
    rows: tuple[BenchRow, ...]
    attainable: BenchStat
    captured: BenchStat
    capture_ratio: BenchStat
    overfit_gap: BenchStat
    noise_fit: BenchStat | None


def bench_path(root: Path, dataset: str, arm: Arm, total_timesteps: int | None) -> Path:
    return root / BENCH_DIR / f"{dataset}-{arm}-{total_timesteps or 0}" / BENCH_FILE


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


# -- running ------------------------------------------------------------------- #


def evaluate_on(ds: SynthDataset, world: World, split: Split, policy: Policy) -> FloatArray:
    """Per-path net P&L of ``policy`` over every path of one world and split, in order."""
    env = dataset_env(ds, world, split)
    return np.array(
        [run_episode(env, policy, i).pnl for i in range(ds.n_paths(split))], dtype=np.float64
    )


def fresh_dataset(ds: SynthDataset) -> SynthDataset:
    """An in-memory held-out sample of the same generator on seeds no dataset uses."""
    spec = ds.spec.model_copy(
        update={"seed": ds.spec.seed + FRESH_OFFSET, "n_train": 1, "n_heldout": FRESH_N}
    )
    arrays = generate_arrays(spec, {"heldout": spec.seeds("heldout")})
    return SynthDataset(spec, f"{ds.hash}-fresh", ds.dir, arrays)


def _train_config(ds: SynthDataset, arm: LearnedArm, total_timesteps: int) -> RunConfig:
    return RunConfig(
        market=ds.spec.market(),
        env=EnvConfig(episode_days=ds.spec.n_days),
        policy=arm,
        train=TrainConfig(algorithm=arm, total_timesteps=total_timesteps, seed=0),
    )


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
        verdict="the planted edge is tradeable: the oracle earned it"
        if za > Z
        else "no attainable edge: the best scripted policy earned nothing measurable from it",
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
            value=ratio_[0],
            se=ratio_[1],
            verdict="the share of the attainable edge the arm took",
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
) -> Benchmark:
    """Train (if learned), evaluate, choose the oracle, and write ``benchmark.json``."""
    if (arm in LEARNED_ARMS) != (total_timesteps is not None):
        raise ValueError(
            "a training budget (total_timesteps) is given exactly when the arm is learned"
        )
    path = bench_path(root, ds.hash, arm, total_timesteps)
    if path.is_file():
        return load_benchmark(path)
    say = log if log is not None else (lambda _: None)

    def summary(policy: str, world: World, split: Split, pnls: FloatArray) -> BenchRow:
        return BenchRow(
            policy=policy,
            world=world,
            split=split,
            mean=float(pnls.mean()),
            se=_se(pnls),
            n=len(pnls),
        )

    rows: list[BenchRow] = []
    noise_fit: tuple[float, float] | None = None
    if arm in LEARNED_ARMS:
        assert total_timesteps is not None
        config = _train_config(ds, arm, total_timesteps)
        say(f"training {arm} on {ds.n_paths('train')} signal paths, {total_timesteps:,} steps")
        policy = trainer(config, DatasetChainProvider(ds, "signal", "train"))
        say(f"training the noise control on the {ds.n_paths('train')} twin paths")
        control = trainer(config, DatasetChainProvider(ds, "twin", "train"))
        say("evaluating the noise control")
        nc_in = evaluate_on(ds, "twin", "train", control)
        nc_out = evaluate_on(ds, "twin", "heldout", control)
        rows += [summary("noise_control", "twin", "train", nc_in)]
        rows += [summary("noise_control", "twin", "heldout", nc_out)]
        noise_fit = unpaired(nc_in, nc_out)
    else:
        policy = make_policy(cast(PolicyName, arm), roll_dte=ROLL_DTE)

    say(f"evaluating {arm} on train, held-out and the twin")
    arm_train = evaluate_on(ds, "signal", "train", policy)
    arm_held = evaluate_on(ds, "signal", "heldout", policy)
    arm_twin = evaluate_on(ds, "twin", "heldout", policy)
    rows += [
        summary("arm", "signal", "train", arm_train),
        summary("arm", "signal", "heldout", arm_held),
        summary("arm", "twin", "heldout", arm_twin),
    ]

    fresh = fresh_dataset(ds)
    say(f"choosing the oracle on {FRESH_N} fresh paths")
    scored: list[BenchMenuRow] = []
    policies = menu(ds.spec)
    for name, pol in policies.items():
        pnls = evaluate_on(fresh, "signal", "heldout", pol)
        scored.append(BenchMenuRow(policy=name, mean=float(pnls.mean()), se=_se(pnls)))
    # The oracle must show its edge, not just top a noisy ranking: a policy that does not
    # beat flat by Z standard errors on the fresh sample is not an edge, and flat stands.
    shown = [m for m in scored if _z(m.mean, m.se) > Z]
    best = max(shown, key=lambda m: m.mean) if shown else scored[0]
    say(f"oracle: {best.policy} ({best.mean:,.0f} a path on the fresh sample)")
    oracle = policies[best.policy]
    or_train = evaluate_on(ds, "signal", "train", oracle)
    or_held = evaluate_on(ds, "signal", "heldout", oracle)
    or_twin = evaluate_on(ds, "twin", "heldout", oracle)
    rows += [
        summary("oracle", "signal", "train", or_train),
        summary("oracle", "signal", "heldout", or_held),
        summary("oracle", "twin", "heldout", or_twin),
        summary("flat", "signal", "heldout", np.zeros(ds.n_paths("heldout"))),
    ]

    attainable = paired(or_held, or_twin)
    captured = paired(arm_held, arm_twin)
    ratio_ = ratio(arm_held - arm_twin, or_held - or_twin) if _z(*attainable) > Z else None
    stats = _verdicts(attainable, captured, ratio_, unpaired(arm_train, arm_held), noise_fit)
    bench = Benchmark(
        dataset=ds.hash,
        generator=ds.spec.generator.kind,
        arm=arm,
        total_timesteps=total_timesteps,
        oracle=best.policy,
        noise_control=noise_fit is not None,
        menu=tuple(scored),
        rows=tuple(rows),
        attainable=stats[0],
        captured=stats[1],
        capture_ratio=stats[2],
        overfit_gap=stats[3],
        noise_fit=stats[4],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(bench.model_dump_json(indent=2), newline="\n")
    tmp.replace(path)
    say(f"{bench.captured.verdict}; {bench.overfit_gap.verdict}")
    return bench
