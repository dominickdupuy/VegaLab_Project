"""Two plug-in trading agents for the Synthetic alpha benchmark: a rule and a trivial learner.

    export OPTSPREAD_ALPHA_PLUGINS="examples/alpha_plugin.py:iv_timer,examples/alpha_plugin.py:iv_picker"
    python -m optspread.server --runs runs            # the page's arm picker lists both
    python -m scripts.alpha run --root runs --arms iv_timer iv_picker delta_seller

Not part of the study. A plug-in is an ``AlphaPlugin`` named by an import path; this file
shows both kinds:

* ``iv_timer``, a rule (``policy=``): sell a short strangle, rolled five days before
  expiry, while at-the-money implied vol is above its own trailing average by a margin, and
  stay flat otherwise. It is scored as it stands, like the scripted arms. It times a
  premium that comes and goes (``mr_vrp``); where implied vol never moves (``const_vrp``)
  it never sells.
* ``iv_picker``, a learner (``trainer=``): it "trains" by playing each of three candidate
  rules (flat, always sell the rolled strangle, ``iv_timer``) on a slice of the train paths
  the training seed picks, and keeps the one that earned the most there. It is trained
  like PPO: per training seed, on the signal world's train paths and on the twin's (the
  noise control), so a choice that only paid in sample shows as overfitting.

Both see only what every agent sees: the observation vector (``optspread.env.OBS_NAMES``)
and the step's ``info`` (``position``, ``day``, ...).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from optspread.actions import FLAT, SHORT_STRANGLE
from optspread.alpha.plugins import AlphaPlugin
from optspread.config import RunConfig
from optspread.costs import make_cost_model
from optspread.env import OBS_NAMES, SpreadEnv
from optspread.harness import run_episode
from optspread.policies import FlatPolicy, Info, Observation, Policy
from optspread.scripted import make_seller
from optspread.types import ChainProvider, Position

IV = OBS_NAMES.index("atm_iv")
ROLL_DTE = 5


class IvTimer:
    """Short strangle while ATM implied vol exceeds its trailing ``window``-day average by
    ``margin`` (decimal vol); flat otherwise, and flat for the first ``window`` days."""

    def __init__(self, window: int = 10, margin: float = 0.0025) -> None:
        self.window, self.margin = window, margin
        self._seen: deque[float] = deque(maxlen=window)

    def reset(self, seed: int) -> None:
        self._seen.clear()

    def act(self, obs: Observation, info: Info) -> int:
        iv = float(obs[IV])
        full = len(self._seen) == self.window
        rich = full and iv > float(np.mean(self._seen)) + self.margin
        self._seen.append(iv)
        if not rich:
            return FLAT
        pos: Position | None = info["position"]
        if pos is not None and pos.first_expiry - int(info["day"]) <= ROLL_DTE:
            return FLAT  # roll: close now, reopen tomorrow if still rich
        return SHORT_STRANGLE


def candidates() -> dict[str, Policy]:
    return {
        "flat": FlatPolicy(),
        "always_sell": make_seller(SHORT_STRANGLE, ROLL_DTE),
        "iv_timer": IvTimer(),
    }


def pick(config: RunConfig, provider: ChainProvider) -> Policy:
    """Keep the candidate that earned the most on this seed's slice of the train paths."""
    assert config.train is not None
    env = SpreadEnv(provider, make_cost_model(config.cost), config.env)
    n_paths = int(getattr(provider, "n_paths", 16))
    # Half the train paths (at least one, at most the budget in episodes), starting where
    # the training seed says: seeds see different slices, so they can choose differently.
    n = max(1, min((n_paths + 1) // 2, config.train.total_timesteps // config.env.episode_days))
    seeds = [(config.train.seed * n + i) % n_paths for i in range(n)]
    scores = {
        name: float(np.mean([run_episode(env, pol, s).pnl for s in seeds]))
        for name, pol in candidates().items()
    }
    return candidates()[max(scores, key=lambda k: scores[k])]


iv_timer = AlphaPlugin(
    name="iv_timer",
    label="IV timer",
    policy=IvTimer,
    description="Sells a rolled short strangle while ATM implied vol is above its 10-day "
    "average by a quarter vol point; flat otherwise.",
)

iv_picker = AlphaPlugin(
    name="iv_picker",
    label="IV picker",
    trainer=pick,
    description="Picks flat, always-sell or the IV timer by its P&L on the training paths.",
)
