"""A stand-in learner for the orchestration and replay tests: no torch, still a Trainer.

Not collected (no ``test_`` prefix). Module-level, so a spawned worker can import it
by name, which is what the parallel tests need. Deterministic in the training seed
and different across seeds, so the seed axis is real: block i's "policy" is a
fixed-delta seller with its own roll rule and structure.
"""

from __future__ import annotations

from pathlib import Path

from optspread.actions import IRON_CONDOR, PUT_CREDIT_SPREAD
from optspread.config import RunConfig
from optspread.policies import DeltaSellerPolicy, Info, Observation


class FakeLearned:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        action = IRON_CONDOR if seed % 2 == 0 else PUT_CREDIT_SPREAD
        self._inner = DeltaSellerPolicy(action_id=action, roll_dte=2 + 3 * (seed % 3))

    def reset(self, seed: int) -> None:
        self._inner.reset(seed)

    def act(self, obs: Observation, info: Info) -> int:
        return self._inner.act(obs, info)

    def save(self, path: Path) -> None:
        path.write_text(f"fake learner, training seed {self.seed}\n")


def fake_trainer(config: RunConfig) -> FakeLearned:
    assert config.train is not None
    return FakeLearned(config.train.seed)


def failing_trainer(config: RunConfig) -> FakeLearned:
    assert config.train is not None
    if config.train.seed % 2 == 1:
        raise RuntimeError(f"training seed {config.train.seed} diverged")
    return FakeLearned(config.train.seed)
