"""Scripted policies.

Lane: Hemdutt Rao (market layer).

Not competitors: their null-world P&L is analytically predictable, which makes
them a gate on the cost model, and with zero training variance they supply the
oracle curve for the power sweep.

A policy sees the observation vector and the env's ``info`` dict (chain,
position, day). Scripted policies read ``info``; a learned policy reads ``obs``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from optspread.actions import FLAT, IRON_CONDOR, N_ACTIONS
from optspread.config import PolicyName
from optspread.rng import POLICY_CHANNEL, SeededStepRNG
from optspread.types import Position

Observation = NDArray[np.float32]
Info = dict[str, Any]


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def act(self, obs: Observation, info: Info) -> int: ...


class FlatPolicy:
    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: Observation, info: Info) -> int:
        return FLAT


class AlwaysOnPolicy:
    """Holds one structure; re-enters the day after it settles."""

    def __init__(self, action_id: int = IRON_CONDOR) -> None:
        self.action_id = action_id

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: Observation, info: Info) -> int:
        return self.action_id


class RandomPolicy:
    """Uniform over the library each day: the churn control that bleeds costs fastest."""

    def __init__(self) -> None:
        self._rng = SeededStepRNG(0)

    def reset(self, seed: int) -> None:
        self._rng = SeededStepRNG(seed)

    def act(self, obs: Observation, info: Info) -> int:
        return int(self._rng.stream(int(info["day"]), POLICY_CHANNEL).integers(0, N_ACTIONS))


class DeltaSellerPolicy:
    """Fixed-delta seller rolled on schedule: closes at ``roll_dte`` days to expiry and
    re-enters the next day. The oracle for a planted vol premium."""

    def __init__(self, action_id: int = IRON_CONDOR, roll_dte: int = 5) -> None:
        self.action_id = action_id
        self.roll_dte = roll_dte

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: Observation, info: Info) -> int:
        pos: Position | None = info["position"]
        if pos is not None and pos.first_expiry - int(info["day"]) <= self.roll_dte:
            return FLAT
        return self.action_id


def make_policy(name: PolicyName, *, roll_dte: int = 5) -> Policy:
    if name == "flat":
        return FlatPolicy()
    if name == "always_on":
        return AlwaysOnPolicy()
    if name == "random":
        return RandomPolicy()
    return DeltaSellerPolicy(roll_dte=roll_dte)
