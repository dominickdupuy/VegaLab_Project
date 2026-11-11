# Bringing your own data, agent, planted effect and cost model

Owner: Hemdutt Rao (market layer and data adapter).

Each layer of the simulator sits behind an interface, so you can swap in your own
piece and keep the rest: the leak suite, the paired grid, the paired test and replay.

| Layer | Interface | Default |
|---|---|---|
| Data | `optspread.data.ChainSource` (makes `ChainProvider`s) | `SyntheticSource` |
| Agent | `optspread.policies.Policy`, or `optspread.train.Trainer` | the scripted arms, `train_ppo` |
| Planted effect | `optspread.effects.PlantedEffect` | `ConstantPremium` |
| Cost model | `optspread.costs.CostModel` | `QuotedSpreadCost` |

`optspread.extend.Components` connects them. It is a runner (config and policy in,
episodes out) that builds each cell's market and costs from your factories, and every
orchestrator accepts one: `experiment.sweep.pilot_matrix`, `experiment.sweep.run_sweep`,
`experiment.schedule.run_grid` and `replay.replay_run`.

**What this does not change.** The confirmatory sweep uses the defaults, and nothing
here adds a config field. As a result the config hash does not record your
components: a run made with them has the same run id as the default world at the same
settings. Put such runs under a root of their own, and replay them through the same
components. Everything below is for exploratory research and is not part of the
pre-registered design.

Every code block on this page runs in `tests/experiment/test_extending.py`, top to bottom in one
namespace, so the page cannot drift from the code. The blocks use small settings to
keep the test fast.

## Setup

A small world and one helper used throughout: the paired test (the detection event of
#26). It compares an arm at a planted premium against the same arm at zero, on the
same paths, with S seed blocks of N episodes.

```python
from pathlib import Path

from optspread.analysis import decide_premium
from optspread.config import CostConfig, EnvConfig, MarketConfig, RunConfig
from optspread.experiment.sweep import pilot_matrix

SMALL = RunConfig(
    seed=500,
    n_episodes=4,
    market=MarketConfig(n_days=21, tenor_days=(10, 21)),
    env=EnvConfig(episode_days=21),
    cost=CostConfig(enabled=True),
)


def at_premium(config, vol_pts):
    return config.model_copy(
        update={"market": config.market.model_copy(update={"iv_premium_vol_pts": vol_pts})}
    )


def paired_test(runner, make_policy, vol_pts, n_seeds=3, n_episodes=4):
    """The arm at ``vol_pts`` against itself at zero: one-sided t on the seed means."""
    treated = pilot_matrix(at_premium(SMALL, vol_pts), n_seeds, n_episodes, make_policy, runner)
    control = pilot_matrix(at_premium(SMALL, 0.0), n_seeds, n_episodes, make_policy, runner)
    return decide_premium(treated, control)
```

## A planted effect

An effect receives the fair chain (every contract priced at the path's vol) and a size
in vol points, and returns the planted IV surface. The size comes from
`MarketConfig.iv_premium_vol_pts`, so the sweep's premium axis drives any effect, and
zero premium means the null world under every effect. Two worked examples ship with
the package: `SkewPremium`, which is larger on out-of-the-money puts, and
`TermPremium`, which is larger at short tenors. Here is a third, written from scratch:

```python
import numpy as np

from optspread.data import SyntheticSource, conformance_report
from optspread.effects import SkewPremium
from optspread.extend import Components, planted
from optspread.leaks import detect_leak, perturbation_check
from optspread.market import SyntheticChainProvider
from optspread.policies import DeltaSellerPolicy
from optspread.rng import SeededStepRNG


class WingPremium:
    """The premium only on strikes more than 5% from spot; the body is priced fair."""

    def apply(self, fair, vol_pts):
        far = np.abs(np.log(fair.strikes / fair.spot)) > 0.05
        return fair.ivs + vol_pts / 100.0 * far[None, :]


market = at_premium(SMALL, 2.0).market
chain = SyntheticChainProvider(market, effect=WingPremium()).reset(SeededStepRNG(0))
print("planted vol points by strike:", np.round((chain.ivs[0] - market.sigma) * 100, 2))

# Conformance: the provider contract, structural invariants, and seed reproducibility.
assert conformance_report(SyntheticSource(market, effect=WingPremium())).passed
# Leaks: the past is unchanged when the future is resampled, and no observation predicts
# the next return (the full per-field suite is leaks.field_coverage).
past_same, _ = perturbation_check(lambda: SyntheticChainProvider(market, WingPremium()), 1, 2, 5)
assert past_same
world = Components(source=planted(WingPremium()))
assert not detect_leak(lambda: world.env(at_premium(SMALL, 2.0)), seeds=range(10)).flagged

# The paired test: the fixed-delta seller at 2 vol points of wing premium against zero,
# then the same for the packaged skew premium. Every block gets the same scripted seller.
def seller(block):
    return DeltaSellerPolicy()


print(paired_test(world, seller, 2.0).render())
print(paired_test(Components(source=planted(SkewPremium())), seller, 2.0).render())
```

`SyntheticChainProvider(market, effect=...)` and `SyntheticSource(market, effect=...)`
take an effect directly. To plant an effect on chains from somewhere else, such as real
data, wrap the provider with
`effects.PlantedChainProvider(provider, effect, market.iv_premium_vol_pts)`. The
synthetic provider already plants its own effect, so wrapping it plants twice unless its
own premium is zero.

## A cost model

A cost model returns the dollar cost of executing one leg at one chain. The factory
receives each cell's `CostConfig`, so the costs-off switch still works:

```python
import math
from dataclasses import dataclass

from optspread.costs import QuotedSpreadCost, ZeroCost
from optspread.scripted import cost_drag_check


@dataclass(frozen=True)
class WingWideningCost:
    """The quoted cost, scaled up in proportion to the strike's distance from spot."""

    quoted: QuotedSpreadCost
    widening: float = 2.0

    def leg_cost(self, chain, leg):
        distance = abs(math.log(leg.strike / chain.spot))
        return self.quoted.leg_cost(chain, leg) * (1.0 + self.widening * distance)


def wing_widening(config):
    return WingWideningCost(QuotedSpreadCost(config)) if config.enabled else ZeroCost()


# The cost-drag check (make drag) on your model: costs change the accounting and nothing
# else, a cash settlement is free, and no structure leaves the book without paying.
drag = cost_drag_check(n_episodes=3, costs=wing_widening(CostConfig()))
assert drag.passed, drag.render()

costly = Components(costs=wing_widening)
print(paired_test(costly, seller, 2.0).render())
```

## Your own data

### A ChainSource

A source makes one fresh provider per episode. A provider returns the chain at `t = 0`
on `reset(rng)` and the next day's chain on each `step()`. The chain at `t` must depend
only on `rng.stream(s)` for `s <= t`. This example serves chains from a daily series
you already hold: it picks the window from stream 0 and prices a flat smile at each
day's vol.

```python
from optspread.effects import ConstantPremium, PlantedChainProvider
from optspread.rng import PATH_CHANNEL
from optspread.types import Chain

days = 300
history_rng = np.random.default_rng(7)
VOLS = 0.15 + 0.10 * (np.sin(np.arange(days) / 20.0) > 0)  # two vol regimes
SPOTS = 4000.0 * np.exp(np.cumsum(VOLS / np.sqrt(252) * history_rng.standard_normal(days)))


class SeriesProvider:
    def __init__(self, market, spots, vols):
        self.market, self.spots, self.vols = market, spots, vols
        self.start, self.day = 0, 0

    @property
    def n_days(self):
        return self.market.n_days

    @property
    def done(self):
        return self.day >= self.market.n_days

    def reset(self, rng):
        self.start = int(rng.stream(0, PATH_CHANNEL).integers(len(self.spots) - self.n_days))
        self.day = 0
        return self.chain()

    def step(self):
        self.day += 1
        return self.chain()

    def chain(self):
        m, i = self.market, self.start + self.day
        centre = round(self.spots[i] / m.strike_step) * m.strike_step
        strikes = centre + m.strike_step * np.arange(-m.n_strikes_each_side, m.n_strikes_each_side + 1.0)
        tenors = np.asarray(m.tenor_days, dtype=np.int64)
        return Chain(
            t=self.day,
            spot=float(self.spots[i]),
            r=m.r,
            strikes=strikes,
            expiry_days=self.day + tenors,
            ivs=np.full((len(tenors), len(strikes)), self.vols[i]),
        )


@dataclass(frozen=True)
class SeriesSource:
    market: MarketConfig

    @property
    def name(self):
        return "my_series"

    @property
    def available(self):
        return True

    def provider(self):
        # The series plants nothing itself, so the grid's premium axis is planted here.
        inner = SeriesProvider(self.market, SPOTS, VOLS)
        return PlantedChainProvider(inner, ConstantPremium(), self.market.iv_premium_vol_pts)


report = conformance_report(SeriesSource(SMALL.market))
print(report.render())
assert report.passed
# A window source draws only stream 0, so resampling later streams leaves the whole
# window in place: the past must not move, and the future is not expected to.
assert all(perturbation_check(SeriesSource(SMALL.market).provider, 3, 4, t)[0] for t in (0, 5, 20))

print(paired_test(Components(source=SeriesSource), seller, 2.0).render())
```

`leaks.detect_leak` needs the true next return, which only the synthetic provider
exposes. For your own data, the leak checks are conformance and the perturbation check.

### Your own surface through the real-data path

The real SPX source (`OptionMetricsSource`) reads two files from `<root>/panel/`, in the
schema of [DATA.md](DATA.md). If you write your own surface in that schema, it is served
like the SPX one, with the same interpolation and the same strike and tenor grid:

| File | Columns |
|---|---|
| `surface_grid.parquet` | `date`, `tenor_days` (calendar days, each of `realdata.panel.TENOR_GRID`), `k` (ln(K / F), each of `K_GRID`), `iv` |
| `spx_daily.parquet` | `date`, `spot`, `carry30` (rate less dividends against `spot`; `rate30` is the fallback), or `spot_parity` with `carry30_parity` as the SPX panel has them, which the engine then prices on instead |

This needs the `data` extra (pandas, pyarrow). The surface also needs more trading days
than the episode length inside the split its episodes are drawn from. A source serves one
split, `train` or `eval` (default `train`), and the store is cut to its dates before
anything is read. On the team's SPX panel those are `realdata.SPLITS`, the dates
pre-registration amendment A8 fixed, and nothing else: a source over it refuses other
ranges. Your own surface has its own dates, so give it `realdata.SplitRanges` that fit
them, with training wholly before evaluation, and `split="train"` for the source you train
on.

```python
from datetime import date

import pandas as pd

from optspread.data import OptionMetricsSource
from optspread.realdata import SplitRanges
from optspread.realdata.panel import K_GRID, TENOR_GRID

dates = pd.bdate_range("2024-01-02", periods=40).date
panel = Path("my-data/panel")
panel.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [
        {"date": d, "tenor_days": tenor, "k": k, "iv": 0.18 - 0.20 * k}  # a downward skew
        for d in dates
        for tenor in TENOR_GRID
        for k in K_GRID
    ]
).to_parquet(panel / "surface_grid.parquet", index=False)
pd.DataFrame({"date": dates, "spot": SPOTS[: len(dates)], "carry30": 0.03}).to_parquet(
    panel / "spx_daily.parquet", index=False
)

# Evaluate on 2024; anything you train on would come from before it.
ranges = SplitRanges(
    train=(date(2023, 1, 1), date(2023, 12, 31)), eval=(date(2024, 1, 1), date(2024, 12, 31))
)
mine = OptionMetricsSource(SMALL.market, root=Path("my-data"), split="eval", ranges=ranges)
assert mine.available and conformance_report(mine).passed
```

The real-data provider does not apply `iv_premium_vol_pts`. To sweep a planted effect
on it, wrap its provider with `PlantedChainProvider`, as `SeriesSource` does above.

## Your own agent

### A Policy

A policy is reset with the episode seed. Each day it sees the observation vector (see
`env.OBS_NAMES`; a learned agent should read only this) and the env's `info` dict, and
returns an action id from `optspread.actions`.

```python
from optspread.actions import FLAT, IRON_CONDOR
from optspread.env import OBS_NAMES
from optspread.harness import evaluate

ATM_IV, RV = OBS_NAMES.index("atm_iv"), OBS_NAMES.index("realized_vol_20d")


class SellWhenRich:
    """Hold an iron condor while the near ATM vol is above ``ratio`` times realised vol."""

    def __init__(self, ratio=1.0):
        self.ratio = ratio

    def reset(self, seed):
        pass

    def act(self, obs, info):
        rich = obs[RV] > 0 and obs[ATM_IV] > self.ratio * obs[RV]
        return IRON_CONDOR if rich else FLAT

    def save(self, path):  # only a trained policy needs this (next block)
        path.write_text(f"SellWhenRich(ratio={self.ratio})\n")


print(evaluate(SMALL, SellWhenRich()))
print(paired_test(Components(), lambda block: SellWhenRich(), 2.0).render())
```

The `make_policy` argument of `pilot_matrix` receives the block index. For a learner,
block i is the policy trained with seed i.

### A Trainer, through the grid

A trainer takes a config and returns a trained policy that can save itself. In
`experiment.schedule.run_grid` it takes the learned arm's slot. That slot is named `ppo` in the
record, and its `TrainConfig` carries the PPO set: your trainer reads what it needs
(here, the seed) and ignores the rest. Train only on `train.TrainingPaths`, which keeps
every training path off the evaluation paths.

```python
from optspread.config import TrainConfig
from optspread.experiment.grid import GridConfig
from optspread.experiment.schedule import run_grid
from optspread.analysis import decide_contrast
from optspread.replay import replay_run
from optspread.train import TrainingPaths

WORLD = Components(source=planted(SkewPremium()), costs=wing_widening)


def episode_return(env, policy):
    obs, info = env.reset()
    total, done = 0.0, False
    while not done:
        obs, reward, done, _, info = env.step(policy.act(obs, info))
        total += reward
    return total


def pick_ratio(config):
    """'Training': the ratio with the best return on four training paths."""
    def score(ratio):
        env = TrainingPaths(WORLD.env(config), config.train.seed)  # the same paths per ratio
        return sum(episode_return(env, SellWhenRich(ratio)) for _ in range(4))

    return SellWhenRich(max((0.9, 1.0, 1.25), key=score))


grid = GridConfig(
    base=SMALL,
    premiums=(0.0, 2.0),
    arms=("delta_seller", "ppo"),
    n_seeds=2,
    train=TrainConfig(total_timesteps=4 * 21, seed=0),
)
root = Path("runs/extend/skew-wing")  # a root of its own
record = run_grid(grid, root, trainer=pick_ratio, runner=WORLD)
print(record.render())
print(decide_contrast(record.contrast(root, 2.0, "ppo")).render())

# Replay rebuilds a run from its manifest and compares bytes: through the same components.
cell = record.cell(2.0, "ppo", 0)
assert replay_run(root / cell.run_id, trainer=pick_ratio, runner=WORLD).matches
```

With two processes or more (`workers=`), the trainer, the runner and its factories must
pickle. That means module-level classes and functions, or a `functools.partial` of one,
not a lambda or a class defined in a notebook.

## What to run on a new piece

| Piece | Check | Call |
|---|---|---|
| Data | provider contract, invariants, reproducibility | `data.conformance_report(source)` |
| Data | no look-ahead | `leaks.perturbation_check(source.provider, seed, other, t)` for several `t` |
| Planted effect | both of the above, plus the observation leak detector | `leaks.detect_leak(lambda: Components(source=planted(effect)).env(config), seeds)`, `leaks.field_coverage(...)` |
| Cost model | the cost-drag identities | `scripted.cost_drag_check(costs=model)` |
| Any | the paired test | `analysis.decide_premium` on `experiment.sweep.pilot_matrix` at p and at zero, or `GridRecord.contrast` into `analysis.decide_contrast` |
| Any | bit-for-bit replay | `replay.replay_run(run_dir, trainer, runner=components)` |
