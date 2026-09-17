"""Throughput benchmark: environment steps per second under a random policy.

Lane: Ilan Danial (machinery). Project document section 11: the 500-3,000
steps/s/core estimate is unvalidated and is replaced by this measurement.
The PPO-inclusive figure is added once decision 7.4 lands.

    python scripts/bench.py --steps 10000
"""

from __future__ import annotations

import argparse
import time

from optspread.config import RunConfig
from optspread.harness import build_env, run_episode
from optspread.policies import make_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    args = parser.parse_args()

    env = build_env(RunConfig(policy="random"))
    policy = make_policy("random")
    steps, seed = 0, 0
    start = time.perf_counter()
    while steps < args.steps:
        steps += run_episode(env, policy, seed).steps
        seed += 1
    elapsed = time.perf_counter() - start
    print(f"{steps} steps in {elapsed:.2f}s -> {steps / elapsed:,.0f} steps/s (random policy)")


if __name__ == "__main__":
    main()
