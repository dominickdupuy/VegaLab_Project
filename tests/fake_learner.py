"""A stand-in learner for the orchestration and replay tests: no torch, still a Trainer.

Not collected (no ``test_`` prefix). Module-level, so a spawned worker can import it
by name, which is what the parallel tests need. Deterministic in the training seed
and different across seeds, so the seed axis is real: block i's "policy" is a
fixed-delta seller with its own roll rule and structure.
"""

from __future__ import annotations

import os
from pathlib import Path

from optspread.actions import IRON_CONDOR, PUT_CREDIT_SPREAD
from optspread.artifacts import write_run
from optspread.config import RunConfig
from optspread.data import SourceName
from optspread.harness import EpisodeResult, build_env, run_episodes, summarize
from optspread.policies import DeltaSellerPolicy, Info, Observation
from optspread.sweep import expected_data, run_id
from optspread.train import TrainingPaths


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


# Episodes the real-data stand-in "trains" on: enough to touch several training windows.
FAKE_TRAINING_EPISODES = 3


def fake_real_trainer(config: RunConfig, *, source: SourceName = "synthetic") -> FakeLearned:
    """``fake_trainer`` that can train on a real source, as ``train_ppo`` does: it plays
    a few episodes in the training environment ``train_ppo`` would build (the source's
    ``train`` split, under ``TrainingPaths``), so the dates it reads are the training
    dates a real learner would read."""
    assert config.train is not None
    learned = FakeLearned(config.train.seed)
    env = TrainingPaths(build_env(config, source, split="train"), config.train.seed)
    for _ in range(FAKE_TRAINING_EPISODES):
        obs, info = env.reset()
        learned.reset(0)
        done = False
        while not done:
            obs, _, done, _, info = env.step(learned.act(obs, info))
    return learned


def failing_trainer(config: RunConfig) -> FakeLearned:
    assert config.train is not None
    if config.train.seed % 2 == 1:
        raise RuntimeError(f"training seed {config.train.seed} diverged")
    return FakeLearned(config.train.seed)


# The cell whose worker the out-of-memory killer takes in the dead-worker tests (#30):
# training seed 11 (block 1 of a grid trained from seed 10) at the 2 vol point premium,
# while $FAKE_LEARNER_DOOM is set (a resume unsets it and keeps the same trainer, whose
# cells alone it reuses).
DOOMED_TRAIN_SEED, DOOMED_PREMIUM = 11, 2.0
DOOM_ENV = "FAKE_LEARNER_DOOM"


def dying_trainer(config: RunConfig) -> FakeLearned:
    """``fake_trainer``, except that one cell's worker dies without raising, as an OOM kill does.

    ``os._exit`` skips every handler, so the pool hears of it only through the dead
    process. Only ever run in a worker: in the test process it would end the session.
    """
    assert config.train is not None
    doomed = (config.train.seed, config.market.iv_premium_vol_pts) == (
        DOOMED_TRAIN_SEED,
        DOOMED_PREMIUM,
    )
    if doomed and os.environ.get(DOOM_ENV):
        os._exit(137)
    return FakeLearned(config.train.seed)


def dead_trainer(config: RunConfig) -> FakeLearned:
    """Every worker that trains dies: a machine that cannot fit even one round."""
    os._exit(137)


def writes_then_dies(config: RunConfig, policy: object) -> tuple[EpisodeResult, ...]:
    """A runner whose worker dies after the doomed scripted cell is written in full.

    The race a dead pool leaves: a cell finished on disk whose result never came back.
    The cell's directory is under ``$FAKE_LEARNER_ROOT``, which the test sets before the
    pool spawns its workers, so they inherit it.
    """
    episodes = run_episodes(config, None)
    if (config.policy, config.market.iv_premium_vol_pts, config.seed) == (
        "delta_seller",
        2.0,
        7003,
    ):
        run_dir = Path(os.environ["FAKE_LEARNER_ROOT"]) / run_id(config)
        # As execute_run writes it: a run of this runner records it as its components.
        data = expected_data(writes_then_dies, config)
        write_run(run_dir, config, summarize(episodes), episodes, data=data)
        os._exit(137)
    return episodes
