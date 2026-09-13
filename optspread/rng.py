"""Per-step RNG stream contract (agreed at the week-one interface session).

Every random draw that shapes step ``t`` of a path comes from ``stream(t)``: a
``numpy.random.Generator`` seeded from ``(seed, channel, t)`` and nothing else.
Streams for different steps are independent, so the world at step ``t`` is a
pure function of the streams for steps ``<= t``.

That property is what the perturbation tests in ``optspread.leaks`` check:
resample every stream after ``t`` and the chain at ``t`` must not move. A single
generator drawn sequentially along the path cannot express "same past,
different future", which is why the contract is per-step rather than per-path.

Nothing in this package touches ``np.random.*`` global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

PATH_CHANNEL = 0  # the underlying price path
POLICY_CHANNEL = 1  # scripted policies that need randomness (RandomPolicy)


@runtime_checkable
class StepRNG(Protocol):
    def stream(self, step: int, channel: int = PATH_CHANNEL) -> np.random.Generator:
        """Fresh generator for ``(step, channel)``. Same inputs => identical stream."""
        ...


@dataclass(frozen=True, slots=True)
class SeededStepRNG:
    """The production implementation: one root seed, one independent stream per step."""

    seed: int

    def stream(self, step: int, channel: int = PATH_CHANNEL) -> np.random.Generator:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(channel, step)))


@dataclass(frozen=True, slots=True)
class SplicedStepRNG:
    """Steps ``<= split`` draw from ``before``; later steps draw from ``after``.

    A test fixture, not a production object: it constructs the "same past, resampled
    future" counterfactual that the no-look-ahead property test needs.
    """

    before: StepRNG
    after: StepRNG
    split: int

    def stream(self, step: int, channel: int = PATH_CHANNEL) -> np.random.Generator:
        source = self.before if step <= self.split else self.after
        return source.stream(step, channel)
