# Pre-registration

Owner: Dominick Dupuy. Opened 2026-09-20. Committed and tagged at M2 (2026-10-29).

This file records analysis decisions **before** we have seen any pilot data. A
decision rule chosen after the fact can always be made to produce a result, so
the value of this document is entirely in its timestamp. Every entry names the
alternatives we rejected and why, because a decision recorded without its
alternatives is indistinguishable from a decision never made.

Changes after the M2 tag are amendments: they are appended with a date and a
reason, never edited in place.

---

## D1. The decision rule

**Closed 2026-09-14 (advisor meeting).**

The confirmatory claim is tested once, on the pre-specified primary contrast,
with a one-sided test at alpha = 0.05 and the required sample size fixed in
advance from the variance-components pilot at 80% power. The reported result is
the effect estimate with its interval and the detection floor alongside it.

**A null result is the finding.** If the planted effect is not detected and the
design was powered to detect it, that is what gets written up. Nothing in the
analysis is conditioned on the sign of the outcome.

Rejected alternatives:

| Rejected | Why |
|---|---|
| Report the best of several metrics | The multiplicity is unbounded and undeclared, so the nominal alpha is meaningless. Any one of Sharpe, CVaR, terminal wealth and hit rate can be made to look favourable on the same run. |
| Two-sided at alpha = 0.01 with no power statement | A stricter alpha without a stated floor buys nothing. It lowers the false-positive rate and raises the false-negative rate silently, which is the failure mode this project exists to avoid. |
| Decide the rule after the pilot | This is the specific thing pre-registration is for. The pilot fixes the sample size; it does not get to fix the test. |

## D2. The effect-size axis

**Closed 2026-09-14 (advisor meeting).**

Effect size is expressed as the **planted volatility risk premium in implied
volatility points** on the surface (`MarketConfig.iv_premium_vol_pts`). The axis
of every power curve is vol points. Zero vol points is the null world.

This is the axis the advisor pushed hardest on, and the argument that settled it
is that the quantity has to be something we set rather than something we
measure. The premium is an input to the generator, so the correct answer is known
by construction and the detection floor is a statement about the instrument.

Rejected alternatives:

| Rejected | Why |
|---|---|
| Sharpe ratio of the agent | Not a property of the world. It moves with leverage, position sizing and the cost model, so the same planted premium would read as different effect sizes across configurations. |
| Dollar P&L per episode | Scales with initial cash and contract count, so it is not comparable across configurations and cannot be pooled. |
| Realised minus implied volatility spread | Closer to the economics, but it is an *outcome* of the path rather than a knob. We cannot plant a known value of it, so the correct answer would stop being known by construction. |

## D3. The PPO implementation

**Closed 2026-09-14 (advisor meeting).**

Stable-Baselines3 PPO at **fixed hyperparameters**, chosen once from the library
defaults and frozen at the 2026-11-01 cutoff. The agent is an instrument in this
experiment, not the subject of it.

Rejected alternatives:

| Rejected | Why |
|---|---|
| Write PPO in-repo | Our implementation would need its own correctness gates, and any result would then be partly a claim about our PPO. A well-used library removes that variable at no cost to the argument. |
| Stable-Baselines3 with a hyperparameter sweep | Tuning makes the measured effect a property of our search effort. Two teams with the same data and different compute budgets would report different detection floors, which makes the floor uninterpretable. |
| A stronger algorithm chosen later by result | Selecting the algorithm on the outcome is the decision rule failure of D1 wearing different clothes. |

---

## Fixed before any confirmatory run

These follow from D1 to D3 and are recorded here so they are not renegotiated:

- **Primary contrast:** the scripted fixed-delta seller against the trained
  policy, on the same seeds, at the same planted premium.
- **Seeds:** paired across arms. An arm is never compared against a different
  draw of paths.
- **Sample size:** set by the variance-components pilot at M2, from
  `Var(grand mean) = tau^2 / S + sigma^2 / (S * N)`, at 80% power.
- **Costs:** the cost model is frozen at the 2026-11-01 cutoff. Any change after
  that re-runs the affected cells or goes in the defect appendix.

## Open, and why they are not pre-registered yet

- The variance components themselves. They are *measured* at M2, not decided.
  The pilot's job is to supply `tau^2` and `sigma^2`; nothing about the analysis
  depends on their values.

---

## Amendments

Appended, never edited in place. Each names its date, what it changes and why.

### A1. 2026-10-25: the detection event is each arm against itself at zero premium (#26)

**Changes** the *Primary contrast* above. The scripted fixed-delta seller against the
trained policy at the same premium detects an effect in the null world: at zero
premium the seller pays its cost drag, a learner that learns to stay flat does not, and
the flat learner "beats" the seller with nothing planted (the audit reproduced it at
S = 24, N = 40: t = 2.03 against 1.714). A test that fires when the answer is no is not
a test of the premium.

**The event.** For each arm a (the fixed-delta seller and PPO) and each planted premium
p > 0, the paired contrast is a at p minus a at zero, on the same evaluation paths
(seed block i, path j) and, for PPO, the same training seed (block i trains with seed
`train_seed + i` at every premium). The cost drag and the path variance cancel
together, the paired design of D1 is kept, and each arm has its own power curve: the
oracle-to-PPO gap is the horizontal distance between the two. In code,
`optspread.analysis.decide_premium`, contrasts read by `GridRecord.contrast`.

At p = 0 the contrast is identically zero (one arm against itself on the same paths),
so the sweep's leftmost point is degenerate rather than a draw at alpha. The size of the
test is established instead by the null-world case of the section 8 gate
(`optspread.validation.validate_null_world`): a flat learner, a control paying a cost
drag of five standard errors, nothing planted. The event fires there at about 0.05;
the event it replaces fires at about 0.99 and is kept as the gate's negative control. The M2
analysis item requires both.

**Sidedness.** One-sided, as D1 closed it. Project document section 5.2 says two-sided;
D1 supersedes it and rejected two-sided at 0.01 explicitly. The alternative is
directional by construction: a planted premium is an edge for a seller, and the
question is whether the instrument detects it, not whether it detects a loss.

### A2. 2026-10-25: the test (#7)

A one-sided Student t on the S seed-level means of the paired contrast of A1, df =
S - 1, alpha = 0.05, as implemented by `optspread.analysis.decide`. The standard error
is `se^2 = MS_between / (S N)`, the sample variance of the seed means over S, not the
clamped plug-in `VarianceComponents.var_grand_mean`. The verdict is reported with the
estimate and its one-sided 95% lower bound, which is positive exactly when the test
rejects.

### A3. 2026-10-25: the power calculation (#7, #20)

Power and the detection floor are computed with the exact noncentral t
(`optspread.power.power_t`, `detectable_effect_t`): seed means `N(delta, tau^2 +
sigma^2 / N)`, noncentrality `delta / se`. The planning functions on the normal
approximation (`episodes_for`, `seeds_for`, `frontier`, `scripts/pilot_budget.py`)
overstate power at small S and are for orientation only; no design is committed on them.

### A4. 2026-10-25: the training budget (#4)

`total_timesteps = 1,000,000` per seed, for every PPO cell of the pilot and the sweep.
From the learning-curve check (`docs/LEARNING_CURVE.md`, `.json`): PPO at 4 vol points,
3 training seeds (2000 to 2002, outside the pilot's and the sweep's), evaluated every
250,000 steps to 2,000,000 on the same 200 paths. The rule, stated in
`scripts/learning_curve.py` before the run: a checkpoint has levelled off when every
later gain, averaged over seeds, is at most two standard errors (the larger of the
between-seed and the paired within-seed one); the budget is 1,000,000 if the curve has
levelled by then, else 2,000,000. Result: levelled by 1,000,000. The 1,001,472-step checkpoint (where a 1,000,000
budget stops, whole rollouts) gains at most 418 dollars to any later checkpoint
against standard errors of 260 to 360; the mean over seeds is 5,122 there and 5,438 at
2,000,896. The rule's own first levelled point, 251,904, is an artefact of one
lagging seed widening the between-seed error, and the budget is the same either way.

### A5. 2026-10-25: the pilot design (#5, #6)

S = 8 training seeds (0 to 7) by N = 200 episodes per seed (path seeds 4,000 on), both
arms at zero and at 1 vol point, at the budget of A4 and the costs frozen for Nov 1
(#8, `docs/COSTS.md`). Under A1 the contrast needs both worlds, so 0 and 1 vol point
is the design that measures what is tested, and #5's recommendation stands with a
sharper reason: the pilot's tau^2 and sigma^2 are those of the PPO contrast at 1 vol
point (`matrix.npy`), with the seller's (`matrix_oracle.npy`, tau^2 = 0 by
construction) as the negative control. Whether tau^2 moves with the premium is the M4
homogeneity question (#13), tested on the sweep itself. The pilot's seeds are disjoint
from the sweep's (training 1,000 on, paths 100,000 on), so no pilot number enters the
confirmatory data.

### A6. 2026-10-25: how the pilot sets S and N (#6), fixed before the pilot runs

- **The effect to detect**, delta: the fixed-delta seller's own contrast at 1 vol point,
  the grand mean of `matrix_oracle.npy`. It is what a learner earns by finding the
  oracle's trade at that premium, and it is a property of the scripted arm, so choosing
  it reads nothing of PPO's outcome.
- **The variances:** tau^2 and sigma^2 of the PPO contrast (`matrix.npy`), the
  point estimates of `optspread.power.variance_components`.
- **The design:** of N in {200, 500, 1000} and S >= 5, the (S, N) with the fewest
  core-hours for the whole sweep (6 premiums by 2 arms, at the A4 budget and the rates
  measured on the machine that runs it) whose exact power `power_t(delta, tau^2,
  sigma^2, S, N)` is at least 0.80. If no S up to 100 reaches it, S = 100 and N = 1000,
  and the floor at that design is reported as the finding it is.
- If delta is not positive, the seller has no edge at 1 vol point to find, and the
  design is S = 50, N = 200, with the floor reported.

### A7. 2026-10-26: the confirmatory S and N, by the rule of A6 (#6)

The pilot of A5 ran as designed (grid `acc8d0d33b3ba4ac`: 8 seeds by 200 episodes, both
arms at 0 and 1 vol point, 1,000,000 training steps, costs on). The arm means:

| premium | arm | grand mean | se |
|---|---|---|---|
| 0 | ppo | -132 | 242 |
| 0 | delta_seller | -397 | 182 |
| 1 | ppo | 1,320 | 363 |
| 1 | delta_seller | 428 | 176 |

The contrasts of A1, by `optspread.power.variance_components`:

- **PPO** (`matrix.npy`): grand mean 1,452 (se 270), tau^2 = 51,838 (se 311,564),
  sigma^2 = 105,996,836.
- **The fixed-delta seller** (`matrix_oracle.npy`): grand mean 824.3 (se 25.4),
  tau^2 = 1,026 (se 2,767, consistent with zero), sigma^2 = 828,609.

`python -m scripts.confirmatory_design --timesteps 1000000 --train-rate 804 --eval-rate
1538` (the rates measured on the pilot machine) applies A6 with delta = 824.3. The
cheapest design at 80% power for each N:

| N | S | exact power | sweep core-hours |
|---|---|---|---|
| 200 | 7 | 0.809 | 14.7 |
| 500 | 5 | 0.898 | 10.7 |
| 1000 | 5 | 0.981 | 11.0 |

**The confirmatory design is S = 5, N = 500.** Exact power is 0.898 at delta = 824.3,
the exact-t detection floor is 698.3, and the sweep costs 10.7 core-hours at these rates.
S = 5 is the minimum A6 allows, so the rule is bound there, not by power. The sweep runs
`python -m scripts.sweep --seeds 5 --episodes 500 --timesteps 1000000` (training seeds from
1,000, paths from 100,000), disjoint from the pilot. PPO's pilot contrast is not used
here except through its variances.

### A8. 2026-10-28: the dates a strategy run on real chains may read (#28, #36)

**Changes** nothing in the confirmatory sweep, which stays on the synthetic null world
(D2) and has run (A7). It fixes, before any policy is trained or scored on real SPX
chains, which dates such a run may read.

| Range | Dates | Trading days | Who reads it |
|---|---|---|---|
| train | 2010-01-04 to 2018-12-31 | 2,264 | training a policy on real chains |
| eval | 2019-01-02 to 2022-08-24 | 919 | scoring a policy on real chains |
| holdout | 2022-08-25 on | 756 to 2025-08-29 | no strategy run |

- **Train and eval are disjoint.** A policy is never scored on a day it trained on.
  The seed check (`train.check_disjoint`) covers the synthetic world; the date ranges
  cover real chains, which a seed check cannot see.
- **The holdout is not read by any strategy run.** It can be opened only by a later
  amendment, made before it is read, that names the single analysis it is used for.
- **A window never spans a gap.** An episode starts only where its days are
  consecutive on the trading calendar, so a day dropped as unusable can't be
  stepped over silently.
- **Market measurement is not a strategy run.** The VRP mapping (#15) and the
  dashboard's data views read every date and report the split; they decide nothing
  about a policy.

In code: `optspread.realdata.SPLITS` names train and eval and has no entry for the
holdout. `SurfaceStore.from_frames` cuts the panel to its split before reading it,
`OptionMetricsSource` defaults to `train`, and `HistoricalChainProvider` draws starts
only from whole windows and refuses a surface that has none. The panel is the one in
`docs/DATA.md`, built from the team's OptionMetrics extraction (Hugging Face mirror
`Algogators/options_data`): 3,939 trading days, 2010-01-04 to 2025-08-29, none
missing.

2022-08-24 first appeared as the end of the development set in an older extract
(the vegalab repo's). A panel built from that extract was withdrawn; the boundary
is adopted here on purpose, as #36 recommends. The amendment planned in #37 (the
exact planner, the M4 definitions, the bootstrap rule) follows as A9.
