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
