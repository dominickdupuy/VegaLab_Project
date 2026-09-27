"""Print the leak suite's coverage table and its stated pass criterion.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.leak_report --seeds 20
"""

from __future__ import annotations

import argparse
import sys

from optspread.leaks import PASS_CRITERION, default_env, field_coverage, leak_floor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.25, help="control size, in field sigmas")
    args = parser.parse_args()
    seeds = range(args.seeds)

    print(PASS_CRITERION)
    print()
    coverage = field_coverage(default_env, seeds, eps=args.eps)
    print(coverage.clean.render())
    print()
    print(coverage.render())
    print()
    floor = leak_floor(default_env, eps_grid=(1.0, 0.5, 0.25, 0.1, 0.05), seeds=seeds)
    print(f"spot-leak detection floor: eps = {floor.floor} on grid {floor.eps_grid}")
    print(f"caught: {floor.flagged}")
    return 0 if coverage.passed else 1


if __name__ == "__main__":
    sys.exit(main())
