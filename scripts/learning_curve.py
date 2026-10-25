"""The PPO budget check (#4): mean P&L against training steps, and the budget it fixes.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.learning_curve --workers 3
    python -m scripts.learning_curve --timesteps 8192 --every 2048 --episodes 4 --out -

D3 freezes the hyperparameters but not ``total_timesteps``, and the pilot measures
tau^2 at whatever budget it is given, so the budget is fixed first. PPO trains at the
largest planted premium of the sweep (4 vol points, where learning is easiest to
see) on ``--seeds`` training seeds, and each one is evaluated every ``--every`` steps
up to ``--timesteps`` on the same ``--episodes`` evaluation paths. One training run
per seed gives the whole curve: ``optspread.train.train_ppo`` hands each checkpoint
over between rollouts, where the policy is the one a run of that budget returns. The
fixed-delta seller on the same paths is printed beside it as the oracle's level.

The rule, stated before the run: a checkpoint T has levelled off when, at every later
checkpoint T', the gain over T averaged over seeds is no more than ``K_SE`` standard
errors, where the standard error is the larger of the between-seed one (the spread of
the per-seed gains) and the within-seed one (the paired episode noise of each gain;
every checkpoint runs the same paths). The budget is ``CANDIDATES[0]`` (1M) when the
curve has levelled by it, else ``CANDIDATES[1]`` (2M), as the issue frames the choice.

Training seeds start at 2000 and evaluation paths at 300,000, clear of the pilot's
(training 0.., paths 4000..) and the sweep's (1000.., 100,000..), so no number here
enters either. Costs are on. Every number is synthetic.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

from optspread.config import CostConfig, MarketConfig, RunConfig, TrainConfig, config_hash
from optspread.harness import run_episodes
from optspread.train import SB3Policy, train_ppo

K_SE = 2.0
CANDIDATES = (1_000_000, 2_000_000)
OUT = Path("docs/LEARNING_CURVE")


def ppo_config(
    premium: float, seed: int, episodes: int, timesteps: int, train_seed: int
) -> RunConfig:
    return RunConfig(
        market=MarketConfig(iv_premium_vol_pts=premium),
        cost=CostConfig(enabled=True),
        policy="ppo",
        seed=seed,
        n_episodes=episodes,
        train=TrainConfig(total_timesteps=timesteps, seed=train_seed),
    )


@dataclass(frozen=True, slots=True)
class Point:
    steps: int
    pnls: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.pnls))

    @property
    def se(self) -> float:
        return float(np.std(self.pnls, ddof=1) / math.sqrt(len(self.pnls)))


def curve(task: tuple[RunConfig, int]) -> tuple[int, list[Point], float]:
    """Train one seed, evaluating at every checkpoint: ``(train seed, points, seconds)``."""
    import torch

    torch.set_num_threads(1)
    config, every = task
    points: list[Point] = []

    def evaluate(steps: int, policy: SB3Policy) -> None:
        pnls = tuple(e.pnl for e in run_episodes(config, policy))
        points.append(Point(steps, pnls))
        assert config.train is not None
        print(
            f"  seed {config.train.seed}  {steps:>9,} steps  mean {np.mean(pnls):>9,.0f}",
            flush=True,
        )

    start = time.perf_counter()
    train_ppo(config, every=every, on_checkpoint=evaluate)
    assert config.train is not None
    return config.train.seed, points, time.perf_counter() - start


@dataclass(frozen=True, slots=True)
class Gain:
    """From checkpoint ``base`` to a later one, averaged over seeds."""

    base: int
    later: int
    mean: float
    se_between: float
    se_within: float

    @property
    def se(self) -> float:
        return max(self.se_between, self.se_within)

    @property
    def flat(self) -> bool:
        return self.mean <= K_SE * self.se


def gains(curves: dict[int, list[Point]], base_index: int) -> list[Gain]:
    seeds = sorted(curves)
    n_points = min(len(curves[s]) for s in seeds)
    out = []
    for j in range(base_index + 1, n_points):
        diffs = [
            np.subtract(curves[s][j].pnls, curves[s][base_index].pnls, dtype=np.float64)
            for s in seeds
        ]
        per_seed = np.array([d.mean() for d in diffs])
        within = math.sqrt(sum(float(d.var(ddof=1)) / d.size for d in diffs)) / len(seeds)
        between = float(per_seed.std(ddof=1) / math.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
        out.append(
            Gain(
                curves[seeds[0]][base_index].steps,
                curves[seeds[0]][j].steps,
                float(per_seed.mean()),
                between,
                within,
            )
        )
    return out


def levelled_at(curves: dict[int, list[Point]]) -> int | None:
    """The first checkpoint whose every later gain is flat, or None if none is."""
    seeds = sorted(curves)
    n_points = min(len(curves[s]) for s in seeds)
    for i in range(n_points - 1):
        if all(g.flat for g in gains(curves, i)):
            return curves[seeds[0]][i].steps
    return None


def budget(levelled: int | None) -> int:
    if levelled is not None and levelled <= CANDIDATES[0]:
        return CANDIDATES[0]
    return CANDIDATES[1]


def render(
    args: argparse.Namespace,
    curves: dict[int, list[Point]],
    oracle: tuple[float, float],
    seconds: dict[int, float],
) -> str:
    seeds = sorted(curves)
    n_points = min(len(curves[s]) for s in seeds)
    steps = [curves[seeds[0]][j].steps for j in range(n_points)]
    levelled = levelled_at(curves)
    chosen = budget(levelled)
    head = "| steps | " + " | ".join(f"seed {s} mean (se)" for s in seeds) + " | mean over seeds |"
    rule = "|" + "---|" * (len(seeds) + 2)
    rows = []
    for j, t in enumerate(steps):
        cells = [f"{curves[s][j].mean:,.0f} ({curves[s][j].se:,.0f})" for s in seeds]
        across = float(np.mean([curves[s][j].mean for s in seeds]))
        rows.append(f"| {t:,} | " + " | ".join(cells) + f" | {across:,.0f} |")
    gain_rows = []
    for i in range(n_points - 1):
        for g in gains(curves, i):
            if g.later == steps[-1]:
                gain_rows.append(
                    f"| {g.base:,} | {g.mean:,.0f} | {g.se_between:,.0f} | {g.se_within:,.0f} | "
                    f"{'flat' if g.flat else 'rising'} |"
                )
    hours = sum(seconds.values()) / 3600
    return "\n".join(
        [
            "# PPO learning curve (#4)",
            "",
            f"Run {date.today().isoformat()} by `python -m scripts.learning_curve`; its "
            "docstring states the rule. Synthetic, costs on.",
            "",
            f"PPO at the frozen D3 hyperparameters, {args.premium:g} vol points planted, "
            f"{len(seeds)} training seeds ({', '.join(str(s) for s in seeds)}), each "
            f"evaluated every {args.every:,} steps on the same {args.episodes} paths "
            f"(seeds {args.seed:,} on). {hours:.2f} core-hours in all.",
            "",
            "Mean net P&L per 63-day episode, dollars, with its standard error over episodes:",
            "",
            head,
            rule,
            *rows,
            "",
            f"The fixed-delta seller on the same paths: {oracle[0]:,.0f} (se {oracle[1]:,.0f}).",
            "",
            f"Gain from each checkpoint to the last ({steps[-1]:,} steps), averaged over seeds; "
            f"flat when the gain is at most {K_SE:g} times the larger standard error:",
            "",
            "| from | gain | se between seeds | se within seeds | |",
            "|---|---|---|---|---|",
            *gain_rows,
            "",
            "**Levelled at:** "
            + (
                f"{levelled:,} steps (every later gain flat)."
                if levelled is not None
                else "no checkpoint; the curve is still rising at the last one."
            ),
            "",
            f"**Budget:** `total_timesteps = {chosen:,}` per seed "
            + (
                f"(levelled by {CANDIDATES[0]:,})."
                if chosen == CANDIDATES[0]
                else f"(not levelled by {CANDIDATES[0]:,})."
            ),
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--premium", type=float, default=4.0, help="planted, vol points")
    parser.add_argument("--seeds", type=int, default=3, help="training seeds")
    parser.add_argument("--train-seed", type=int, default=2000, help="the first training seed")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="steps per seed")
    parser.add_argument("--every", type=int, default=250_000, help="steps between checkpoints")
    parser.add_argument("--episodes", type=int, default=200, help="N per checkpoint")
    parser.add_argument("--seed", type=int, default=300_000, help="first evaluation path seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=str(OUT), help="path stem for .md and .json, or -")
    args = parser.parse_args(argv)

    configs = [
        ppo_config(args.premium, args.seed, args.episodes, args.timesteps, args.train_seed + i)
        for i in range(args.seeds)
    ]
    tasks = [(c, int(args.every)) for c in configs]
    if args.workers > 1 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
            results = pool.map(curve, tasks)
    else:
        results = [curve(t) for t in tasks]
    curves = {seed: points for seed, points, _ in results}
    seconds = {seed: s for seed, _, s in results}

    oracle_config = configs[0].model_copy(update={"policy": "delta_seller", "train": None})
    oracle = [e.pnl for e in run_episodes(oracle_config)]
    oracle_stats = (float(np.mean(oracle)), float(np.std(oracle, ddof=1) / math.sqrt(len(oracle))))

    text = render(args, curves, oracle_stats, seconds)
    if args.out == "-":
        print(text)
        return 0
    stem = Path(args.out)
    stem.with_suffix(".md").write_text(text, newline="\n")
    record = {
        "rule": {"k_se": K_SE, "candidates": list(CANDIDATES)},
        "levelled_at": levelled_at(curves),
        "total_timesteps": budget(levelled_at(curves)),
        "args": vars(args),
        "config_hashes": {c.train.seed: config_hash(c) for c in configs if c.train is not None},
        "oracle": {"mean": oracle_stats[0], "se": oracle_stats[1]},
        "seconds": seconds,
        "curves": {seed: [asdict(p) for p in points] for seed, points in curves.items()},
    }
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", newline="\n")
    print(text)
    print(f"wrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
