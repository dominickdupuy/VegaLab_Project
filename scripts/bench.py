"""Throughput benchmark: measured environment steps per second, by workload.

Lane: Ilan Danial (machinery). Project document section 11: the 500-3,000
steps/s/core figure is an estimate nobody measured, and every scheduling number
downstream rests on it. This replaces it with a measurement.

Four arms, because a single number cannot be acted on and because the first
version of this benchmark was misleading. Timing the env with a FLAT action gave
37,000 steps/s, which looked like the environment was free -- but a flat book
never prices a position, so that arm measured the empty path rather than the
environment. The arms below hold the workload comparable:

    flat        empty book: the floor, and not a number to plan with
    always_on   one structure held and re-marked every step: the steady state
    random      churns, but is FLAT a quarter of the time
    ppo         always_on plus the Stable-Baselines3 update (decision D3)

The measurement says the cost is **per-step position marking**, not trading.
``flat`` runs about 20x ``always_on`` because a flat book never prices a leg, and
``random`` comes out *faster* than ``always_on`` despite 24x the trades, because
one action in four is FLAT and those steps mark nothing. Re-pricing greeks for
every held leg dominates; the policy's own decision is nanoseconds.

So plan the sweep off ``always_on``, which is what a trained policy will mostly
do, and ignore ``flat``: it is the arm that made the first version of this
benchmark misleading.

    python scripts/bench.py --steps 20000
    python scripts/bench.py --steps 20000 --ppo
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from optspread.config import PolicyName, RunConfig
from optspread.harness import build_env, run_episode
from optspread.policies import make_policy

ARMS: tuple[tuple[PolicyName, str], ...] = (
    ("flat", "empty book; the floor, not a planning number"),
    ("always_on", "one structure held and re-marked: the steady state"),
    ("random", "churns, but FLAT a quarter of the time, so it marks less"),
)


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    steps: int
    seconds: float
    trades: float
    note: str = ""

    @property
    def per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    def render(self) -> str:
        if not self.steps:
            return f"{self.name:<11}{'not measured':>14}   {self.note}"
        return (
            f"{self.name:<11}{self.per_second:>11,.0f}/s{self.steps:>9,} steps"
            f"{self.seconds:>8.2f}s{self.trades:>9.1f} trades/ep   {self.note}"
        )


def bench_policy(policy_name: PolicyName, target: int, note: str = "") -> Arm:
    env = build_env(RunConfig(policy=policy_name))
    policy = make_policy(policy_name)
    steps, seed, trades = 0, 0, 0
    start = time.perf_counter()
    while steps < target:
        result = run_episode(env, policy, seed)
        steps += result.steps
        trades += result.n_trades
        seed += 1
    elapsed = time.perf_counter() - start
    return Arm(policy_name, steps, elapsed, trades / max(seed, 1), note)


def bench_ppo(target: int) -> Arm:
    """always_on's workload plus the PPO update, at the fixed hyperparameters of D3."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        return Arm("ppo", 0, 0.0, 0.0, "stable-baselines3 not installed; install to measure")
    env = build_env(RunConfig(policy="always_on"))
    model = PPO("MlpPolicy", env, verbose=0, seed=0)
    start = time.perf_counter()
    model.learn(total_timesteps=target)
    return Arm("ppo", target, time.perf_counter() - start, 0.0, "includes the training update")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--ppo", action="store_true", help="also time the PPO learn loop")
    args = parser.parse_args()

    arms = [bench_policy(name, args.steps, note) for name, note in ARMS]
    if args.ppo:
        arms.append(bench_ppo(args.steps))
    for arm in arms:
        print(arm.render())

    by_name = {a.name: a for a in arms if a.steps}
    if {"flat", "always_on"} <= by_name.keys():
        ratio = by_name["flat"].per_second / by_name["always_on"].per_second
        print(
            f"{chr(10)}per-step marking is {ratio:.0f}x the empty path: the cost is"
            " pricing held legs, not trading. Plan the sweep off always_on."
        )


if __name__ == "__main__":
    main()
