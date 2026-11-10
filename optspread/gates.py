"""Wave 0 economic sanity gate (project document 3.1).

Lane: Hemdutt Rao (market layer).

Under fair implied volatility the null world pins three orderings. The arms
share seeds, so the with-cost and no-cost figures for one policy differ by
exactly the modelled cost drag (plus any margin-breach liquidation).

    1. FLAT sits at exactly zero, with or without costs.
    2. ALWAYS-ON reads ~0 without costs (|t| <= 3) and turns negative with them.
    3. RANDOM, which churns, bleeds faster than ALWAYS-ON.

Break any one and the build fails (``tests/gates/test_wave0_gate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from optspread.config import CostConfig, PolicyName, RunConfig
from optspread.harness import EvalSummary, evaluate


@dataclass(frozen=True, slots=True)
class Wave0Report:
    arms: dict[tuple[PolicyName, bool], EvalSummary]  # (policy, costs_enabled) -> summary
    flat_is_zero: bool
    always_on_null_ok: bool
    always_on_cost_negative: bool
    random_bleeds_faster: bool

    @property
    def passed(self) -> bool:
        return (
            self.flat_is_zero
            and self.always_on_null_ok
            and self.always_on_cost_negative
            and self.random_bleeds_faster
        )

    def render(self) -> str:
        lines = [f"{'arm':<14}{'costs':<8}{'mean P&L':>12}{'se':>10}{'trades':>8}"]
        for (name, costed), s in self.arms.items():
            lines.append(
                f"{name:<14}{costed!s:<8}{s.mean_pnl:>12,.0f}{s.se_pnl:>10,.0f}"
                f"{s.mean_trades:>8.1f}"
            )
        checks = (
            ("flat exactly zero", self.flat_is_zero),
            ("no-cost always-on ~ 0 (|t| <= 3)", self.always_on_null_ok),
            ("with-cost always-on < 0 and < no-cost", self.always_on_cost_negative),
            ("random bleeds faster", self.random_bleeds_faster),
        )
        lines += [f"  [{'PASS' if ok else 'FAIL'}] {label}" for label, ok in checks]
        lines.append(f"Wave 0 expectation {'HOLDS' if self.passed else 'FAILS'}.")
        return "\n".join(lines)


def wave0_gate(n_episodes: int = 100, seed: int = 1000) -> Wave0Report:
    arms: dict[tuple[PolicyName, bool], EvalSummary] = {}
    for policy in ("flat", "always_on", "random"):
        for costed in (False, True):
            cfg = RunConfig(
                policy=policy, seed=seed, n_episodes=n_episodes, cost=CostConfig(enabled=costed)
            )
            arms[(policy, costed)] = evaluate(cfg)

    flat = all(p == 0.0 for c in (False, True) for p in arms[("flat", c)].pnls)
    ao_free, ao_cost = arms[("always_on", False)], arms[("always_on", True)]
    rnd_cost = arms[("random", True)]
    return Wave0Report(
        arms=arms,
        flat_is_zero=flat,
        always_on_null_ok=abs(ao_free.t_stat) <= 3.0,
        always_on_cost_negative=ao_cost.mean_pnl < 0.0 and ao_cost.mean_pnl < ao_free.mean_pnl,
        random_bleeds_faster=rnd_cost.mean_pnl < ao_cost.mean_pnl,
    )
