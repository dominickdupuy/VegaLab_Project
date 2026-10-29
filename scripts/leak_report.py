"""Print the leak suite's coverage table and its stated pass criterion.

Lane: Dominick Dupuy (statistical spine).

    python -m scripts.leak_report --seeds 20

Before the statistical detector, the structural invariants (``leaks.chain_violations``)
on one clean synthetic path per seed, and the calendar check's positive control: the
#29 surface, which bounds and parity cannot see, has to fire it.
"""

from __future__ import annotations

import argparse
import sys

from optspread.config import MarketConfig
from optspread.leaks import (
    PASS_CRITERION,
    calendar_control,
    calendar_violations,
    default_env,
    field_coverage,
    leak_floor,
    sequence_violations,
)
from optspread.market import SyntheticChainProvider
from optspread.rng import SeededStepRNG


def invariants(seeds: range) -> bool:
    """Clean paths have no violations; the calendar control has exactly its one."""
    provider = SyntheticChainProvider(MarketConfig())
    found: list[str] = []
    n_chains = 0
    for seed in seeds:
        chains = [provider.reset(SeededStepRNG(seed))]
        chains += [provider.step() for _ in range(provider.n_days)]
        found += sequence_violations(chains)
        n_chains += len(chains)
    control = calendar_violations(calendar_control(provider.reset(SeededStepRNG(0))))
    checks = (
        (f"clean world: no structural violation over {n_chains:,} chains", not found),
        ("calendar control (60% at 21d over 15% at 42d) fires", len(control) == 1),
    )
    print("structural invariants: spot, strikes, expiries, vols, price bounds, parity,")
    print("total variance non-decreasing in tenor; timestamps consecutive")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    for v in (*found[:5], *control):
        print(f"    - {v}")
    return all(ok for _, ok in checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.25, help="control size, in field sigmas")
    args = parser.parse_args()
    seeds = range(args.seeds)

    structural = invariants(seeds)
    print()
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
    return 0 if coverage.passed and structural else 1


if __name__ == "__main__":
    sys.exit(main())
