"""Run the planted-premium oracle sweep and print its table.

Lane: Ilan Danial (machinery).

    python scripts/oracle_sweep.py --episodes 100 --seed 3000
    python scripts/oracle_sweep.py --premiums 0 1 2 4 --arms flat always_on delta_seller

The scripted arms have no training variance, so their curve along the premium axis
(decision D2) is the oracle the M2 power curve is read against: how many dollars of
mean P&L one planted vol point is worth to a seller. Cells land under
``<root>/<sweep hash>/<run id>/`` as ordinary run directories, and a second run with
the same arguments reads them back instead of recomputing. Costs are on. Every number
is synthetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast, get_args

from optspread.config import CostConfig, PolicyName, RunConfig, config_hash
from optspread.sweep import SweepConfig, run_sweep

POLICY_NAMES: tuple[str, ...] = get_args(PolicyName)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--premiums", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--arms", nargs="+", choices=POLICY_NAMES, default=["flat", "always_on", "delta_seller"]
    )
    parser.add_argument("--root", type=Path, default=Path("runs") / "sweeps")
    args = parser.parse_args()

    arms = cast(tuple[PolicyName, ...], tuple(str(a) for a in args.arms))
    sweep = SweepConfig(
        base=RunConfig(
            seed=int(args.seed), n_episodes=int(args.episodes), cost=CostConfig(enabled=True)
        ),
        premiums=tuple(float(p) for p in args.premiums),
        arms=arms,
    )
    root = Path(args.root) / config_hash(sweep)
    record = run_sweep(sweep, root)
    print(record.render())
    print(f"cells under {root} (synthetic; costs on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
