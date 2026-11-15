"""A research sweep: each arm against a drift planted in the simulated market.

    python examples/sweep.py
    python examples/sweep.py --arms flat delta_seller ppo a2c --alphas 0 25 75 --episodes 200
    python examples/sweep.py --root runs/examples --vol-of-vol 0.3 --skew-slope 0.2

Not part of the study: an example of the research loop the market's knobs open. A drift
of ``alpha_bps`` basis points a day (``MarketConfig.alpha_bps``) gives a long position an
edge that is known in advance, so a learner can be checked for finding it; at zero the
market is the study's driftless world, where selling options earns only a planted
premium. ``--vol-of-vol`` and ``--skew-slope`` make the same market harder to read.

Each cell is an ordinary launched run, made as the dashboard's Launch page makes one:
a ``JobSpec`` handed to ``execute_job``, which writes a hashed run directory under the
runs root and returns its path relative to it. A cell already on disk under the same
root is read back, not recomputed, so ``--root`` makes a second sweep instant and puts
the runs on the dashboard's Runs page when it is the server's runs root. Without
``--root`` the runs go to a temporary folder that is removed afterwards.

The learned arms (ppo, a2c, dqn) need the ``train`` extra; without it they are left
out with a note. Every cell starts on the same first seed, so the cells of one drift
share their market paths and pair. The table is each run's mean P&L per episode in
dollars, net of costs, with its standard error.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

# Run as a file from anywhere: this checkout's optspread, not one an install points at.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optspread.artifacts import read_run
from optspread.config import LEARNED_ARMS
from optspread.dashboard.jobs.spec import JOBS_DIR, JobSpec, available_arms
from optspread.dashboard.jobs.worker import execute_job


def sweep(
    root: Path, args: argparse.Namespace, arms: Sequence[str]
) -> dict[tuple[str, float], str]:
    """Run every (arm, drift) cell under ``root``; each cell's line, by cell."""
    (root / JOBS_DIR).mkdir(parents=True, exist_ok=True)  # where execute_job writes progress
    cells = [(arm, float(alpha)) for arm in arms for alpha in args.alphas]
    out: dict[tuple[str, float], str] = {}
    for i, (arm, alpha) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] {arm} at {alpha:g} bps a day", flush=True)
        spec = JobSpec(
            policy=arm,
            alpha_bps=alpha,
            vol_of_vol=args.vol_of_vol,
            skew_slope=args.skew_slope,
            n_episodes=args.episodes,
            seed=args.seed,
            total_timesteps=args.timesteps if arm in LEARNED_ARMS else None,
        )
        rel = execute_job(f"sweep-{arm}-{alpha:g}", spec.model_dump_json(), str(root))
        record = read_run(root / rel)
        out[(arm, alpha)] = f"{record.mean_pnl:>+10,.0f} ± {record.se_pnl:>6,.0f}"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--arms", nargs="+", default=["flat", "delta_seller", "ppo"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 25.0, 75.0])
    parser.add_argument("--vol-of-vol", type=float, default=0.0)
    parser.add_argument("--skew-slope", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=50, help="per cell; 200+ for a reading")
    parser.add_argument("--seed", type=int, default=0, help="the first seed of every cell")
    parser.add_argument("--timesteps", type=int, default=10_000, help="training steps, learners")
    parser.add_argument("--root", type=Path, help="a runs root to keep the cells in")
    args = parser.parse_args(argv)

    here = available_arms()
    arms = [a for a in args.arms if a in here]
    for arm in sorted(set(args.arms) - set(arms)):
        why = "needs the train extra" if arm in LEARNED_ARMS else "is not an arm"
        print(f"left out: {arm} {why}", file=sys.stderr)
    if not arms:
        return 1
    if args.root is not None:
        lines = sweep(args.root, args, arms)
    else:
        with tempfile.TemporaryDirectory(prefix="optspread-sweep-") as tmp:
            lines = sweep(Path(tmp), args, arms)

    world = f"vol of vol {args.vol_of_vol:g}, skew slope {args.skew_slope:g}"
    print(f"\nMean P&L per episode, $ net of costs ± se: {args.episodes} episodes, {world}")
    print(f"{'arm':<14}" + "".join(f"{f'{a:g} bps/day':>20}" for a in args.alphas))
    for arm in arms:
        print(f"{arm:<14}" + "".join(f"{lines[(arm, float(a))]:>20}" for a in args.alphas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
