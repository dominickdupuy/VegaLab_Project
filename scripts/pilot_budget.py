"""Sample size and compute budget for the M2 confirmatory run, from pilot variance components.

Lane: Dominick Dupuy (statistical spine).

    python scripts/pilot_budget.py --delta 1500 --tau2 250000 --sigma2 1.3e8
    python scripts/pilot_budget.py --delta 1500 --matrix runs/pilot/matrix.npy

``delta`` is the effect to detect in the outcome's own units (dollars of mean P&L per
episode for the primary contrast); ``tau2`` and ``sigma2`` are the pilot's variance
components in those units squared, or ``--matrix`` names an S x N ``.npy`` of outcomes
to estimate them from. The rates default to the M1-measured figures, rounded as the
README plans with them (about 1,000 steps/s/core with the training update, about 1,800
for scripted evaluation); re-measure with ``make bench`` on the machine that will run
the sweep. The per-seed training budget is pre-registration A4's 1,000,000 steps (from the
learning-curve check, docs/LEARNING_CURVE.md), which is the default; another value plans
a design the pre-registration does not cover.

Prints the components, the variance budget at each seed count, and the frontier: for
each seed count the fewest episodes at which the confirmatory t test has the target
power, by the exact noncentral t (``power_t``, pre-registration A3), and what the
design costs. Seed counts start at 2, the fewest the test can run on. Exit status 0
when at least one design is feasible, 1 when none is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from optspread.power import (
    frontier,
    render_frontier,
    variance_budget,
    variance_budget_t,
    variance_components,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument(
        "--delta", type=float, required=True, help="effect to detect, outcome units"
    )
    parser.add_argument("--tau2", type=float, help="between-seed variance from the pilot")
    parser.add_argument("--sigma2", type=float, help="within-seed variance from the pilot")
    parser.add_argument("--matrix", type=Path, help="S x N .npy of outcomes to estimate from")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1_000_000,
        help="PPO steps per seed (default: A4's 1,000,000)",
    )
    parser.add_argument("--episode-days", type=int, default=63)
    parser.add_argument("--train-rate", type=float, default=1_000.0, help="steps/s/core, training")
    parser.add_argument("--eval-rate", type=float, default=1_800.0, help="steps/s/core, scripted")
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()

    delta = float(args.delta)
    if args.matrix is not None:
        components = variance_components(np.load(args.matrix))
        print(components.render())
        print()
        tau2, sigma2 = components.tau2, components.sigma2
    elif args.tau2 is not None and args.sigma2 is not None:
        tau2, sigma2 = float(args.tau2), float(args.sigma2)
    else:
        parser.error("give --tau2 and --sigma2, or --matrix")

    seeds = [int(s) for s in args.seeds]
    if min(seeds) < 2:
        parser.error("every seed count must be at least 2: the t test runs on S seed means")
    alpha, target = float(args.alpha), float(args.power)
    budgets = ", ".join(
        f"{variance_budget_t(delta, s, alpha, target):,.4g} at S = {s}" for s in seeds
    )
    print(
        f"delta = {delta:,.4g}, tau^2 = {tau2:,.4g}, sigma^2 = {sigma2:,.4g}; one-sided t, "
        f"alpha = {args.alpha}, power = {args.power}: Var(grand mean) must not exceed "
        f"{budgets} (the normal approximation's {variance_budget(delta, alpha, target):,.4g} "
        f"at any S overstates it)"
    )
    designs = frontier(
        delta,
        tau2,
        sigma2,
        seeds=seeds,
        alpha=alpha,
        target_power=target,
        train_steps_per_seed=int(args.train_steps),
        episode_days=int(args.episode_days),
        train_rate=float(args.train_rate),
        eval_rate=float(args.eval_rate),
        cores=int(args.cores),
    )
    print(render_frontier(designs))
    print(
        f"rates: {args.train_rate:,.0f} steps/s/core training, {args.eval_rate:,.0f} scripted, "
        f"{args.cores} core(s); {args.train_steps:,} training steps per seed (not pre-registered)"
    )
    return 0 if any(d.feasible for d in designs) else 1


if __name__ == "__main__":
    sys.exit(main())
