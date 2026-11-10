"""The confirmatory S and N from the pilot, by the rule pre-registered as A6 (#6).

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.confirmatory_design --pilot-root runs/pilot --timesteps 1000000 \
        --train-rate 1000 --eval-rate 1800

The rule, as ``docs/PRE_REGISTRATION.md`` A6 states it before the pilot ran:

    delta    the fixed-delta seller's contrast at the pilot premium (the grand mean of
             ``matrix_oracle.npy``): what a learner earns by finding the oracle's trade
    tau^2, sigma^2   the PPO contrast's components (``matrix.npy``), point estimates
    design   over N in ``EPISODES`` and S >= ``MIN_SEEDS``, the fewest sweep core-hours
             whose exact power ``power_t`` is at least 0.80; S = ``MAX_SEEDS``, N = 1000
             if none reaches it by then; S = 50, N = 200 if delta is not positive

The sweep's cost is every cell of the default grid (``SWEEP_PREMIUMS`` by
``PRIMARY_ARMS``) at the rates given, which should be the ones measured on the machine
that runs it. Prints the design, its power and floor, and the alternatives it beat.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optspread.config import EnvConfig
from optspread.experiment.grid import PRIMARY_ARMS, SWEEP_PREMIUMS
from optspread.experiment.pilot import PILOT_MATRIX, PILOT_ORACLE_MATRIX
from optspread.power import (
    VarianceComponents,
    detectable_effect_t,
    power_t,
    variance_components,
)

EPISODES = (200, 500, 1000)
MIN_SEEDS, MAX_SEEDS = 5, 100
TARGET = 0.80
FALLBACK_NO_EDGE = (50, 200)


@dataclass(frozen=True, slots=True)
class Candidate:
    n_seeds: int
    n_episodes: int
    power: float
    core_hours: float


def sweep_core_hours(
    n_seeds: int, n_episodes: int, timesteps: int, train_rate: float, eval_rate: float
) -> float:
    """Every cell of the default grid: PPO cells train and evaluate, scripted ones evaluate."""
    eval_seconds = n_episodes * EnvConfig().episode_days / eval_rate
    per_block = sum(
        eval_seconds + (timesteps / train_rate if arm == "ppo" else 0.0) for arm in PRIMARY_ARMS
    )
    return len(SWEEP_PREMIUMS) * n_seeds * per_block / 3600.0


def choose(
    delta: float,
    vc: VarianceComponents,
    timesteps: int,
    train_rate: float,
    eval_rate: float,
) -> tuple[Candidate, list[Candidate], str]:
    """``(design, the cheapest feasible design at each N, why)``, by A6."""

    def candidate(s: int, n: int) -> Candidate:
        return Candidate(
            s,
            n,
            power_t(delta, vc.tau2, vc.sigma2, s, n),
            sweep_core_hours(s, n, timesteps, train_rate, eval_rate),
        )

    if delta <= 0.0:
        return candidate(*FALLBACK_NO_EDGE), [], "delta is not positive: the A6 fallback"
    best: list[Candidate] = []
    for n in EPISODES:
        for s in range(MIN_SEEDS, MAX_SEEDS + 1):
            c = candidate(s, n)
            if c.power >= TARGET:
                best.append(c)
                break
    if not best:
        return candidate(MAX_SEEDS, 1000), [], f"no S up to {MAX_SEEDS} reaches {TARGET:.0%}"
    return min(best, key=lambda c: c.core_hours), best, "the cheapest design at 80% power"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--pilot-root", type=Path, default=Path("runs") / "pilot")
    parser.add_argument("--timesteps", type=int, required=True, help="the A4 budget")
    parser.add_argument("--train-rate", type=float, default=1_000.0, help="steps/s/core")
    parser.add_argument("--eval-rate", type=float, default=1_800.0, help="steps/s/core")
    args = parser.parse_args(argv)

    ppo = np.load(args.pilot_root / PILOT_MATRIX)
    oracle = np.load(args.pilot_root / PILOT_ORACLE_MATRIX)
    vc, oc = variance_components(ppo), variance_components(oracle)
    delta = oc.grand_mean
    design, alternatives, why = choose(delta, vc, args.timesteps, args.train_rate, args.eval_rate)
    print("PPO contrast:", vc.render(), sep="\n")
    print("seller's contrast:", oc.render(), sep="\n")
    print(f"delta (the seller's contrast at the pilot premium) = {delta:,.4g}")
    for c in alternatives:
        print(
            f"  N = {c.n_episodes:>5}: S = {c.n_seeds:>3}, exact power {c.power:.3f}, "
            f"{c.core_hours:,.1f} sweep core-hours"
        )
    floor = detectable_effect_t(vc.tau2, vc.sigma2, design.n_seeds, design.n_episodes)
    print(
        f"design: S = {design.n_seeds}, N = {design.n_episodes} ({why}); exact power "
        f"{design.power:.3f} at delta = {delta:,.4g}; exact-t floor {floor:,.4g}; "
        f"{design.core_hours:,.1f} sweep core-hours at {args.timesteps:,} steps, "
        f"{args.train_rate:,.0f}/{args.eval_rate:,.0f} steps/s/core"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
