"""Score a Synthetic alpha benchmark's policies on real SPX chains (#46, item 5).

Lane: Dominick Dupuy.

A benchmark keeps its numbers, not its weights, so a learned arm is rebuilt the way the
benchmark built it: the same config (``benchmark.train_config``) on the same frozen train
paths, one policy per training seed. Training is deterministic on CPU (``optspread.train``),
so the rebuilt policies are the benchmark's; each is first scored again on the held-out
synthetic paths, and the result says whether that reproduces the stored mean. A scripted
arm needs no rebuilding.

A plug-in arm (``optspread.alpha.plugins``) is tested the same way: a rule plug-in is
scored as it stands, and a learner plug-in is rebuilt per seed through its own registered
trainer, with the config and seeds its benchmark used and the same reproduce check. The
benchmark records the plug-in's import path and source digest, so a plug-in that is no
longer registered, or whose source has changed since (its policies could not be the
benchmark's), is refused with the reason rather than tested.

Each policy then plays every window of the real panel's ``eval`` split, the dates
pre-registration A8 fixed for scoring (2019-01-02 to 2022-08-24), with the dataset's
episode length, the same environment and the same cost model as on the synthetic paths.
Nothing from A8's holdout (2022-08-25 on) is read: no split serves it. The windows are
walked in date order, one episode per start, so consecutive episodes share all but one of
their days and their P&Ls are strongly correlated. The standard error is therefore the
one the real-data VRP mapping uses for the same reason (``optspread.vrp``): a stationary
bootstrap of the mean, Politis-White mean block, never shorter than the overlap (the
episode length), on the seed-averaged per-window P&L. With several training seeds the
spread of the per-seed means is added, the split ``benchmark.level`` makes. The flat
baseline and the benchmark's oracle play the same windows, and the arm's edge over flat
is taken window by window.

Real-data results are exploratory: the pre-registered analysis is the synthetic sweep
(D2), and nothing here feeds it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np

from optspread.alpha import plugins as plug
from optspread.alpha.benchmark import (
    BENCH_DIR,
    BENCH_FILE,
    ROLL_DTE,
    Benchmark,
    BenchStat,
    SynthTrainer,
    _Frozen,
    _z,
    evaluate_on,
    is_learned,
    learned_trainer,
    menu,
    train_config,
)
from optspread.alpha.dataset import SynthDataset
from optspread.alpha.generators import FloatArray
from optspread.alpha.provider import DatasetChainProvider
from optspread.config import (
    LEARNED_ARMS,
    CostConfig,
    EnvConfig,
    LearnedArm,
    MarketConfig,
    PolicyName,
)
from optspread.costs import make_cost_model
from optspread.data import OptionMetricsSource
from optspread.env import SpreadEnv
from optspread.harness import run_episode
from optspread.policies import FlatPolicy, Policy, make_policy
from optspread.realdata import SPLITS, Split
from optspread.rng import StepRNG
from optspread.types import Chain, ChainProvider
from optspread.vrp import N_BOOT, SEED, block_length, bootstrap_mean_median

REAL_DIR = Path("alpha") / "real"
REAL_FILE = "realtest.json"
REAL_SPLIT: Split = "eval"  # A8's scoring dates; the holdout has no split at all
REPRODUCE_TOL = 1e-6  # dollars a path: a rebuilt policy matches the benchmark's
SCRIPTED: tuple[str, ...] = ("delta_seller", "always_on", "flat")
ARM_NAME = {
    "ppo": "PPO",
    "a2c": "A2C",
    "dqn": "DQN",
    "delta_seller": "the fixed-delta seller",
    "always_on": "the always-on condor",
    "flat": "flat",
}
EXPLORATORY = (
    "Exploratory: a real-data result is not part of the pre-registered analysis, which is "
    "the synthetic sweep (D2)."
)


def real_dates() -> tuple[date, date]:
    """The first and last date a real-data test may read (A8's evaluation range)."""
    return SPLITS[REAL_SPLIT]


def panel_dir(data_root: Path | None) -> Path:
    """Where the panel is read from: ``<data root>/panel`` (``VEGALAB_DATA`` when None)."""
    return OptionMetricsSource(root=data_root, split=REAL_SPLIT).panel_dir


def real_available(data_root: Path | None) -> bool:
    return OptionMetricsSource(root=data_root, split=REAL_SPLIT).available


def unavailable_message(data_root: Path | None) -> str:
    return (
        "Real data is not available on this machine: there is no SPX panel under "
        f"{panel_dir(data_root)}. The test runs where the panel is (VEGALAB_DATA)."
    )


class RealDataUnavailableError(RuntimeError):
    """No real panel here. Never a fallback to synthetic chains."""


# -- plug-in arms -------------------------------------------------------------- #


class PluginChangedError(ValueError):
    """A plug-in benchmark whose plug-in is gone or edited: its policies cannot be rebuilt."""


def stored_plugin(bench: Benchmark) -> tuple[str, str] | None:
    """The import path and source digest a plug-in benchmark was run with; None otherwise."""
    if bench.plugin is None:
        return None
    source, _, digest = bench.plugin.rpartition("@")
    return source, digest


def plugin_for(bench: Benchmark) -> plug.LoadedPlugin | None:
    """The registered plug-in ``bench`` scored, unchanged since; None for a built-in arm.

    A ``PluginChangedError`` when it is no longer registered here or its source digest
    differs from the one the benchmark recorded."""
    stored = stored_plugin(bench)
    if stored is None:
        return None
    source, digest = stored
    found = plug.find(bench.arm)
    if found is None:
        raise PluginChangedError(
            f"the plug-in {bench.arm!r} this benchmark scored ({source}) is no longer "
            f"registered on this server, so its policies cannot be rebuilt: register it "
            f"again ({plug.PLUGINS_ENV}) to test it on real days"
        )
    if found.digest != digest:
        raise PluginChangedError(
            f"the plug-in {bench.arm!r} has changed since this benchmark ran (source digest "
            f"{digest} then, {found.digest} now, {found.source}), so its policies would not be "
            "the benchmark's: benchmark the new code first, then test that on real days"
        )
    return found


def bench_name(bench: Benchmark) -> str:
    """The benchmark's directory name, a plug-in's by the digest it was run with (not the
    digest its code has now, which ``bench_path`` would use)."""
    stored = stored_plugin(bench)
    key = bench.arm if stored is None else f"{bench.arm}.{stored[1]}"
    return f"{bench.dataset}-{key}-{bench.total_timesteps or 0}-s{bench.n_seeds}"


def arm_title(bench: Benchmark, found: plug.LoadedPlugin | None) -> str:
    """The arm in a sentence: "PPO", "the fixed-delta seller", "IV timer (plug-in)"."""
    if found is not None:
        return f"{found.plugin.title} (plug-in)"
    return ARM_NAME.get(bench.arm, bench.arm)


# -- the windows --------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RealWindows:
    """Every window of the eval split at one episode length, served in date order:
    episode seed ``i`` plays the ``i``-th window (``i mod n``), where
    ``HistoricalChainProvider`` itself draws a start at random."""

    provider: ChainProvider
    starts: tuple[str, ...]  # each window's first day
    ends: tuple[str, ...]  # and its last
    panel: tuple[tuple[str, str], ...]  # the panel files' digests

    @property
    def n(self) -> int:
        return len(self.starts)


def real_windows(market: MarketConfig, data_root: Path | None) -> RealWindows:
    """The eval split's windows at ``market.n_days``; a ``RealDataUnavailableError``
    when there is no panel. The panel is only read."""
    if not real_available(data_root):
        raise RealDataUnavailableError(unavailable_message(data_root))
    from optspread.realdata.provider import HistoricalChainProvider, load_store

    class InDateOrder(HistoricalChainProvider):
        def reset(self, rng: StepRNG) -> Chain:
            self._start = int(self._starts[DatasetChainProvider.path_for(rng, self.n_starts)])
            self._day = 0
            self._reset = True
            return self._chain()

    store = load_store(panel_dir(data_root), REAL_SPLIT)
    provider = InDateOrder(market, store)
    starts = [int(i) for i in store.window_starts(market.n_days)]
    return RealWindows(
        provider,
        tuple(store.dates[i] for i in starts),
        tuple(store.dates[i + market.n_days] for i in starts),
        store.panel,
    )


def real_env(ds: SynthDataset, windows: RealWindows) -> SpreadEnv:
    """The environment ``dataset_env`` builds, on real chains: same costs, same episode."""
    return SpreadEnv(
        windows.provider, make_cost_model(CostConfig()), EnvConfig(episode_days=ds.spec.n_days)
    )


def evaluate_real(env: SpreadEnv, n: int, policy: Policy) -> FloatArray:
    """Per-window net P&L of ``policy`` over the first ``n`` windows, in date order."""
    return np.array([run_episode(env, policy, i).pnl for i in range(n)], dtype=np.float64)


# -- the standard error -------------------------------------------------------- #


def overlap_level(m: FloatArray, overlap: int) -> tuple[float, float, float]:
    """Mean, se and mean block of ``m`` (seeds x windows, windows in date order).

    The path part is the stationary-bootstrap variance of the mean of the seed-averaged
    series, its mean block the Politis-White choice floored at ``overlap`` (``vrp``'s rule);
    the seed part is the variance of the per-seed means over the seeds, as in
    ``benchmark.level``."""
    m = np.atleast_2d(m)
    series = m.mean(axis=0)
    block = block_length(series, window=overlap)
    if series.size > 1 and np.ptp(series) > 0.0:
        means, _ = bootstrap_mean_median(series, block, n_boot=N_BOOT, seed=SEED)
        path = float(np.var(means, ddof=1))
    else:
        path = 0.0
    per_seed = m.mean(axis=1)
    seed = float(per_seed.var(ddof=1) / per_seed.size) if per_seed.size > 1 else 0.0
    return float(m.mean()), float(np.sqrt(path + seed)), block


# -- the result ---------------------------------------------------------------- #


class RealRow(_Frozen):
    policy: str  # arm | oracle | flat
    synthetic_mean: float  # held-out signal paths, from the benchmark
    synthetic_se: float
    synthetic_n: int
    real_mean: float  # every window of the eval split
    real_se: float
    real_per_seed: tuple[float, ...]


class RealTest(_Frozen):
    dataset: str
    generator: str
    arm: str
    total_timesteps: int | None
    n_seeds: int
    oracle: str
    benchmark: str  # <runs root>-relative path of the benchmark scored
    split: str
    first_date: str  # first window's start
    last_date: str  # last window's end
    n_days: int
    n_windows: int
    block: float  # mean bootstrap block, in windows
    panel: tuple[tuple[str, str], ...]
    reproduced: bool | None  # rebuilt policies' held-out mean matches the benchmark's
    reproduced_mean: float | None
    rows: tuple[RealRow, ...]
    edge: BenchStat  # arm minus flat, window by window, on the real days
    verdict: str
    exploratory: bool = True
    note: str = EXPLORATORY


def real_path(root: Path, bench: Benchmark) -> Path:
    return root / REAL_DIR / bench_name(bench) / REAL_FILE


def load_real(path: Path) -> RealTest:
    return RealTest.model_validate_json(path.read_text())


def stages(arm: str, n_seeds: int) -> int:
    """Lines ``run_real_test`` logs, for a progress bar (a learner plug-in is rebuilt per
    seed, as a built-in learner is)."""
    return 2 * n_seeds + 4 if is_learned(arm) else 5


def usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _verdict(arm: str, synth: RealRow, edge: tuple[float, float], n: int, span: str) -> str:
    zs, zr = _z(synth.synthetic_mean, synth.synthetic_se), _z(*edge)
    head = (
        f"On {n:,} overlapping real SPX windows ({span}), {arm} made {usd(synth.real_mean)} a "
        f"window (se {usd(synth.real_se)}), against {usd(synth.synthetic_mean)} "
        f"(se {usd(synth.synthetic_se)}) on the held-out synthetic paths."
    )
    if zs > 2 and zr > 2:
        tail = "It beat flat by more than 2 se on both."
    elif zs > 2 and zr < -2:
        tail = "It earned on the synthetic paths but lost against flat on real days."
    elif zs > 2:
        tail = (
            "What it earned on the synthetic paths did not carry over: on real days it is "
            "not distinguishable from flat."
        )
    elif zr > 2:
        tail = "It beat flat on real days, though not on the synthetic paths."
    elif zr < -2:
        tail = "On real days it lost against flat by more than 2 se."
    else:
        tail = "Neither on the synthetic paths nor on real days is it distinguishable from flat."
    return f"{head} {tail}"


def _synthetic(bench: Benchmark, policy: str) -> tuple[float, float, int]:
    for r in bench.rows:
        if r.policy == policy and r.world == "signal" and r.split == "heldout":
            return r.mean, r.se, r.n
    return 0.0, 0.0, 0


def run_real_test(
    bench: Benchmark,
    ds: SynthDataset,
    root: Path,
    data_root: Path | None,
    trainer: SynthTrainer = learned_trainer,
    log: Callable[[str], None] | None = None,
) -> RealTest:
    """Score ``bench``'s policies on every real eval window; write and return the result.

    ``trainer`` rebuilds the built-in learners; a plug-in learner is rebuilt by its own
    registered trainer. A result already on disk for the same benchmark and the same panel
    (by digest) is returned as it is. Raises ``RealDataUnavailableError`` when there is no
    panel and ``PluginChangedError`` for a plug-in no longer registered or edited since."""
    if ds.hash != bench.dataset:
        raise ValueError(f"benchmark {bench.dataset} is not on dataset {ds.hash}")
    found = plugin_for(bench)
    if found is None and bench.arm not in (*LEARNED_ARMS, *SCRIPTED):
        raise ValueError(
            f"the real-data test rebuilds {', '.join((*LEARNED_ARMS, *SCRIPTED))} and "
            f"registered plug-ins, not {bench.arm}"
        )
    learned = bench.arm in LEARNED_ARMS or (found is not None and found.plugin.trainable)
    if learned != (bench.total_timesteps is not None):
        raise ValueError(f"the benchmark's training budget does not match its arm {bench.arm!r}")
    rebuild = found.plugin.trainer if found is not None and found.plugin.trainer else trainer
    say = log if log is not None else (lambda _: None)
    windows = real_windows(ds.spec.market(), data_root)
    path = real_path(root, bench)
    try:
        kept = load_real(path)
        if kept.panel == windows.panel:
            say("this real-data test is already on disk for this panel; not recomputed")
            return kept
    except (OSError, ValueError):
        pass
    env = real_env(ds, windows)
    n = windows.n
    span = f"{windows.starts[0]} to {windows.ends[-1]}"
    say(f"{n:,} real windows of {ds.spec.n_days} days, {span} (A8 eval)")

    arm_real: list[FloatArray] = []
    reproduced: bool | None = None
    reproduced_mean: float | None = None
    if learned:
        assert bench.total_timesteps is not None
        held: list[FloatArray] = []
        # the benchmark's own config: a plug-in learner gets the ppo placeholder it did
        learner = cast(LearnedArm, bench.arm if found is None else "ppo")
        for seed in range(bench.n_seeds):
            config = train_config(ds, learner, bench.total_timesteps, seed)
            say(f"seed {seed}: rebuilding {bench.arm} on {ds.n_paths('train')} signal paths")
            policy = rebuild(config, DatasetChainProvider(ds, "signal", "train"))
            held.append(evaluate_on(ds, "signal", "heldout", policy))
            arm_real.append(evaluate_real(env, n, policy))
            say(f"seed {seed}: scored on the held-out paths and the {n:,} real windows")
        reproduced_mean = float(np.mean(held))
        stored = _synthetic(bench, "arm")[0]
        reproduced = abs(reproduced_mean - stored) <= REPRODUCE_TOL * max(1.0, abs(stored))
    else:
        if found is not None and found.plugin.policy is not None:
            policy = found.plugin.policy()
        else:
            policy = make_policy(cast(PolicyName, bench.arm), roll_dte=ROLL_DTE)
        arm_real.append(evaluate_real(env, n, policy))
        say(f"{bench.arm} scored on the {n:,} real windows")
    a = np.array(arm_real)
    oracle = evaluate_real(env, n, menu(ds.spec)[bench.oracle])
    say(f"oracle ({bench.oracle}) scored on the real windows")
    flat = evaluate_real(env, n, FlatPolicy())
    say("flat scored on the real windows")

    overlap = ds.spec.n_days
    arm_mean, arm_se, block = overlap_level(a, overlap)

    def row(policy: str, m: FloatArray, mean: float, se: float) -> RealRow:
        s_mean, s_se, s_n = _synthetic(bench, policy)
        return RealRow(
            policy=policy,
            synthetic_mean=s_mean,
            synthetic_se=s_se,
            synthetic_n=s_n,
            real_mean=mean,
            real_se=se,
            real_per_seed=tuple(float(x) for x in np.atleast_2d(m).mean(axis=1)),
        )

    rows = (
        row("arm", a, arm_mean, arm_se),
        row("oracle", oracle, *overlap_level(oracle, overlap)[:2]),
        row("flat", flat, *overlap_level(flat, overlap)[:2]),
    )
    edge_value, edge_se, _ = overlap_level(a - flat[None, :], overlap)
    edge = (edge_value, edge_se)
    ze = _z(*edge)
    if ze > 2:
        ev = "the arm earned more than flat on the real days"
    elif ze < -2:
        ev = "the arm did worse than flat on the real days"
    else:
        ev = "the arm is not distinguishable from flat on the real days"
    result = RealTest(
        dataset=bench.dataset,
        generator=bench.generator,
        arm=bench.arm,
        total_timesteps=bench.total_timesteps,
        n_seeds=bench.n_seeds,
        oracle=bench.oracle,
        benchmark=(BENCH_DIR / bench_name(bench) / BENCH_FILE).as_posix(),
        split=REAL_SPLIT,
        first_date=windows.starts[0],
        last_date=windows.ends[-1],
        n_days=ds.spec.n_days,
        n_windows=n,
        block=block,
        panel=windows.panel,
        reproduced=reproduced,
        reproduced_mean=reproduced_mean,
        rows=rows,
        edge=BenchStat(value=edge_value, se=edge_se, verdict=ev),
        verdict=_verdict(arm_title(bench, found), rows[0], edge, n, span),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json(indent=2), newline="\n")
    tmp.replace(path)
    say(result.verdict)
    return result
