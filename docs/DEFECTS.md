# Defect appendix

Owner: Dominick Dupuy (the final report, [#19]). Kept current from 2026-10-28.

The project document asks for this appendix twice. Section 17 lists "an appendix of
defects caught by the harness, with fixes", as the strongest evidence that the suite
works. Section 12.1 adds that "any post-cutoff change to a stopped component is either
accompanied by re-running the affected sweep cells or documented in the defect appendix
with the results it touches. It is never patched silently." This file does both. It
also records what the cross-platform checks found, the M4 rules that were written after
the sweep, and the limits we know of and did not fix.

Commits are on the week 8 branch (`issue-38-week8-m4`) unless marked otherwise. "The
audits" are the four read-only audits of this week's branch at 834ffb3 (2026-10-28):
data, machinery, numbers and statistics, plus one on docs and process.

## 1. Changes to frozen modules after the cutoff tag

The tag `cutoff-2026-11-01` points at 34872b2 (2026-10-26 00:54 -0400). Since then,
four of the modules on the simulation path have changed, and two new ones have joined
it. None of these changes moved a synthetic run's numbers.

| Module | Commits (all 2026-10-28) | What changed |
|---|---|---|
| `market.py` (+16) | 7c1c863 ([#32]) | The planted premium now goes through `effects.PlantedEffect`. `ConstantPremium` is the default, and `sigma + p/100` is the same IEEE operation as before |
| `harness.py` (+63) | 4911de8 ([#32]), f9c08be ([#21]), b8febe7, 69fae2a ([#28]) | `run_episodes` and `build_env` take a source, a split, a provider and a cost model of your own. The defaults are the old path |
| `scripted.py` (+42) | 4911de8 ([#32]) | The drag check takes your own runner and cost model. The defaults are the old path |
| `train.py` (+76) | 5154ed8 ([#4]), b8febe7, 69fae2a ([#28]), 241e572 ([#23]) | Training on a real-data split (`source=`); the mypy ignore that holds with and without SB3; `should_stop` for cancelled dashboard jobs. A run that is never stopped takes the old code path |
| `data.py` (+63), `effects.py` (new) | 1390828, fc073df, 55edc96, 69fae2a, 48d0201 ([#28]); 7c1c863 ([#32]) | `make_source` and the split checks; the constant premium as an effect |

These are unchanged since the tag: `config`, `rng`, `env`, `portfolio`, `costs`,
`margin`, `pricing`, `types`, `actions` and `policies`. The simulation path did not
change at all between the tag and 8b47f99, the commit Dominick's sweep ran from.

**The proof that they are output-identical.** It is a freeze probe: the same configs run
under `git archive cutoff-2026-11-01` and under the branch, comparing the sha256 of each
`run.json` text. The configs:

- 48 scripted configs: premiums 0, 0.25, 1, 4, -1 and 13.7 × flat, always-on, random
  and the fixed-delta seller × costs on and off;
- one config with a changed market and env;
- `cost_drag_check`;
- one 4,096-step PPO training followed by its evaluation.

That makes 51 checksums. All 51 are identical, and `config_hash(RunConfig())` is
`17c15aa74ce473a9` under both. The machinery audit ran the probe at 834ffb3. It was run
again at 9ade13a, which includes every later change to `train.py` and `data.py`, and
again all 51 matched. Against Dominick's own runs, at the branch head:

- the sweep's resume plan reads "60 on disk, 0 to run" (`grid-7d9d53d340006f02`, every
  cell written at 8b47f99);
- the pilot's reads "32 on disk, 0 to run" (written at 68fb9ec);
- `docs/SWEEP_EXPECTED.json` rebuilds byte for byte from those cells.

So no sweep cell needs re-running.

Gap: nothing in code enforces the freeze. The gates accept any commit at or after the
tag, and only the tests and this probe guard the numbers. The docstrings of
`config.TrainConfig` and `train.py` still say `total_timesteps` is "not pre-registered".
A4 fixed it at 1,000,000. Those docstrings were left as they are so that the frozen
modules are not touched again for prose alone.

## 2. Cross-platform findings

The sweep ran on Windows (Python 3.12.0, numpy 2.5.3, torch 2.14.0, SB3 2.9.0). The
checks below were run on macOS arm64 (Python 3.12.8, numpy 2.5.4, torch 2.15.0, SB3
2.9.2).

- **Scripted runs agree within 1.7e-11.** All 30 seller cells were regenerated on macOS
  and compared with the Windows cells. Every number is within 1.70e-11 relative (at most
  4.66e-10 dollars), with no trade-count mismatch. The bytes differ in every cell,
  because libm rounds the last bit differently. The partner tolerance is 1e-9, about 60
  times that noise. `optspread/reproduction.py` had said 2.1e-12 and "500 times"; it
  now states what was measured (45258d1, [#10]).
- **Scoring a saved policy agrees within 1.8e-12.** Dominick's `policy.zip`, scored on
  macOS, reproduces his Windows `run.json` to 1.8e-12 relative, with 0 trade-count
  mismatches. Evaluation is cross-platform clean.
- **PPO training is not bit-reproducible across platforms.** The reference cell
  `e608f959c816fc46-s102000` was retrained for 1M steps on macOS. Its run id and config
  hash match, but it is a different policy:
  - mean 4,661.39 against 4,958.07;
  - 471 of 500 episodes trade a different number of times;
  - pinned to Dominick's library versions it gives 4,246.41, so pinning does not fix it.

  The partner PPO cell is therefore checked statistically: its mean must be within 3
  standard errors of ours, and it must trade, refuse no open and keep the attribution
  residual under the M4 bound (45258d1, [#10]). `make m3gate ARGS="--replay N"` does not
  sample a PPO cell trained on another platform, and says so (e720b4b, [#11]).
- **Replay is bit for bit only on the machine that made the run.** Dominick's Windows
  cells replay as `DIFFERS` on macOS, while his own `replay.log` reads IDENTICAL. The
  manifest now records the platform (OS, architecture, Python, numpy). `make replay
  ARGS="--rtol 1e-9"` compares a scripted run within that relative tolerance, with
  every count exact, and reports `IDENTICAL`, `WITHIN rtol ...` or `DIFFERS`. The
  Windows seller cells read `WITHIN` on macOS. No tolerance rescues a retrained PPO run
  (f8c7f2a, [#21]).
- **Attribution agrees, but does not reproduce, off the machine.** Two PPO cells
  attributed on macOS match their runs within 3.5e-10 dollars, yet they failed the
  bit-for-bit check. The M4 residual item now requires `agrees` (P&L and cost within
  1e-9 relative, the same trades and steps) and reports `reproduces` beside it
  (10ac7cc, [#35]).
- **One edge episode differs, and it is a real path split.** Re-simulated on macOS for
  the edge, 2 of the 60 confirmatory cells reproduce bit for bit, 57 agree within 1e-9,
  and 1 differs: PPO at 1 vol point, block 0 (`ea5be29e60901c1a-s100000`), in 1 of its
  500 episodes, seed 100148. That episode's saved P&L is 2,510.69 and its re-simulated
  P&L 4,096.75; its cost is 765.17 against 751.03, with 11 trades and 63 steps in both.
  The other 499 agree within 1e-9. It is not a tolerance edge: along the re-simulated
  path the smallest PPO logit margin is 0.099 (median 4.3), far above float noise, and
  no strike pick is near a tie, so one discrete branch went the other way between the
  platforms (git 8b47f99, torch 2.14.0, SB3 2.9.0, numpy 2.5.3 against torch 2.15.0,
  SB3 2.9.2, numpy 2.5.4). `run.json` keeps per-episode totals only, so the branch
  cannot be named. The cell mean moves from 1,524.02 to 1,527.19 (1,586.06 / 500), and
  the PPO edge at 1 vol point from 1,396 (lower bound 615) to 1,397 (616): detected
  either way, so no verdict changes. The dashboard's Attribution page quotes the
  `run.json` edge, as Sweep and the M4 analysis do, with the re-simulated one beside
  it. To settle it, re-simulate seed 100148 on the sweep machine: 2,510.69 there makes
  it cross-platform, anything else makes the original evaluation nondeterministic.

## 3. The M4 rules were written after the sweep

The sweep ran 2026-10-26, 08:23 to 12:01, and its raw arm means were posted in [#9].
`optspread/m4analysis.py` was first written 2026-10-28 at 09:43. A1 and A2 (the event
and the test) are pre-registered. The rest was not pre-registered:

- the power curve's interpolation and its (0, 0) anchor;
- the median-unbiased noncentrality;
- the floor at the first 0.80 crossing, and the gap;
- Brown-Forsythe on seed means;
- the bootstrap agreement tolerance with its Bonferroni split, and the switch to the
  bootstrap rule.

The module's docstring said the tolerance was "fixed before any sweep was read". It now
says when the rules were written (0ee5428, [#12]), and the report says so at the top.
These rules enter the pre-registration only as A9 ([#37], pending), which must say it
was written after the sweep. Until then, PPO's floor (0.88 vol points), the gap (0.821)
and "every level agrees; the t test stands" are not pre-registered numbers.

## 4. A7's planned power against the measured power

A7 chose S = 5 and N = 500 from the pilot's variances: power 0.898 at delta = 824.3, and
a floor of 698.3 dollars. The sweep's own PPO contrast at 1 vol point has tau^2 = 4.088e5
against the pilot's 5.18e4 (the pilot's se on that estimate is 3.1e5), and sigma^2 =
1.311e8. The same design on those variances has power **0.584** and a floor of
**1,114**. The rule was followed correctly; tau^2 from a small pilot is the noisy
quantity the project document warns about (7.3). Do not quote 0.898 and 698 as what the
sweep achieved. The M4 report prints the plan and the measurement side by side
(0ee5428, [#12]).

## 5. Known limits

| Limit | What it means | Where it is stated |
|---|---|---|
| Databento closes are the first bar only | The 317 closing files were extracted before the closing window existed ([#31]), so a contract with no book update in the closing minute is missing that day. Every Databento day is in the holdout, which no strategy run reads | `docs/DATA.md`, Known limits |
| Real-data standard errors are understated | Episodes are windows drawn from about 850 starts, so most of them share days; the per-episode se treats them as independent | `docs/DATA.md`; README, Running on real data |
| Fixed 50-point strike grid | A real chain is laid on `MarketConfig`'s grid: 4.5% of spot apart in 2010, 1.1% in 2021. So delta-targeted strikes are coarse in `train` and fine in `eval`, and a 5-delta put snaps to the edge in March 2020. `MarketConfig` is frozen | `docs/DATA.md`, Known limits |
| Carry is 0 in attribution | `Book` books no interest on cash, so the carry component is identically 0, and its edge is exactly 0 with se 0 | `optspread/attribution.py`, `optspread/edge.py` |
| Vega is not exercised on the synthetic world | The surface is flat, so vega is 0 on every sweep cell. Breaking vega (sign, ×0.01, ×100) leaves the residual share at 1.3% to 1.9%; on a moving surface the same errors give 37% to 3,880%. The residual item checks delta, gamma and theta only, and says so; vega is tested on `MovingSurface` in `tests/resim/test_attribution.py` (10ac7cc, [#35]) | README, The M4 analysis |
| Brown-Forsythe has low power | At 5 groups of 5 seed means, its size is 0.003 at alpha 0.05, and its power is 0.047, 0.22 and 0.47 at variance ratios 4, 10 and 25. PPO's tau^2 spans a factor of 56 across premiums and is not rejected. So "no" is weak evidence that tau^2 is constant, and the report prints size and power beside the verdict (0ee5428, 10ac7cc) | `docs/M4_ANALYSIS.md` (when written) |
| The floor is biased low | The floor is taken at a noisy curve's first crossing of 0.80. The median estimate is 8% to 11% below the true floor (about 0.1 vol point for PPO's 0.88), with recrossings in 10% to 16% of draws. The band [lo, hi] covers 0.97 to 0.99, so it is conservative | `optspread/m4analysis.py` |
| The seller's floor is below the grid | 0.0594 vol points comes entirely from interpolation between (0, 0) and the first sweep point at 0.25, and it is flagged `below_grid`. The gap inherits it | `optspread/m4analysis.py` |
| The bootstrap switch is blunt at S = 5 | Its switch rate is 0.050 under t(3) and 0.062 under lognormal seed means, against 0.025 under normal ones, so it barely sees non-normality. The rule it switches to has size about 0.10. Nothing diverged on the sweep | `optspread/m4analysis.py`; A9 ([#37]) |
| The learning-curve rule's standard error | With the within-seed se alone, the curve still levels at 1,001,472 steps. With the smaller of the two se's, it levels at 1,251,328, which would have made the budget 2M. The rule was stated before the run and followed | `docs/LEARNING_CURVE.md`, A4 |
| No bid/ask filter on real quotes | Wide 2010 quotes put calendar arbitrage into the grid's wings. The provider removes that arbitrage from what it serves (05dd4ed, [#29]), and the grid keeps the quotes as they were | `docs/DATA.md`, Method |


### An open defect in the frozen margin check

`margin.is_defined_risk` (`optspread/margin.py:63-68`) decides whether a structure has bounded risk by comparing the two far-end payoffs at 10x the top strike against an absolute tolerance of 1e-9 dollars. Those payoffs are millions of dollars, so their rounding exceeds 1e-9 and some defined-risk structures are margined as naked. The dashboard's 2026-10-02 example: an iron condor (6900/7250P, 8150/8400C, spot 7,721.12) shows margin $85,254 where its max loss is $30,363.58.

Measured on the confirmatory sweep's evaluation paths (found by the dashboard audit, 2026-10-28):

- Of 2,835,000 structures any arm could open, the check misclassifies 5,890 (0.6% of condors, also at spot 5,000).
- The fixed-delta seller is unaffected: margin never binds on its cash, and all 30 seller cells regenerate byte for byte with a corrected check.
- PPO observes `margin_frac`, so it is affected. Scored with the sweep's saved policies, 53 of 15,000 episodes hold a reclassified condor (363 steps) and one changes P&L: run 4a48a04bb0099da1-s101500, seed 101623, -18,213.69 against -18,245.81 (12 trades against 8). A 50,000-step training run trains a different policy under each check (mean 2,760 against -1,219).

The module is frozen and the confirmatory sweep ran with this check, so it is **not fixed on this branch**: correcting it would change what the confirmatory PPO cells reproduce. The confirmatory results are as the code ran them; the one affected episode moves its cell's mean by $0.06 (one of 500 episodes, by $32), which no decision rests on. A relative-tolerance fix with its test is prepared for after the study, for Dom to sign off.

## 6. Defects the harness caught before this week

These are the four entries [#19] lists, and the others the gates caught on the way.

| Date | Caught by | Defect | Fix |
|---|---|---|---|
| 2026-09-24 | the cost-drag check | A hold-to-expiry seller ended the episode still holding an unexpired structure, marked at mid with no exit charged, so the drag came out 74 to 128 dollars short of the prediction | 907e9ea: expiry (free, cash-settled) and a live close (crosses the spread) are separate paths in the book |
| 2026-09-27 | writing the M1 report | The leak suite had no stated pass criterion. A flat-book probe tested only the price fields, and controls sized with one absolute eps were invisible in four fields | c9de9fd: `leaks.PASS_CRITERION`, a probe policy that trades, and per-field controls sized in units of each field's sd |
| 2026-10-18 | the train path test | SB3 resets with no seed, which the env read as path 0, so PPO would have trained on one path for its whole budget | d69438c ([#3]): training paths at 2^48 and above, one block per training seed, and evaluation seeds refused there |
| 2026-10-19 | the M2 replay item | A replay could pass having compared nothing | 2eeb776 ([#3]): replay checks every recorded artifact and refuses to replay nothing |
| week 6 | the real-data conformance suite | Pre-OSI vendor symbols (one exchange code per contract) collapsed the early-2010 chain | the normaliser in `realdata/optionmetrics.py` ([#3]), later completed so that every row takes its root from the settlement flag (3de1456, 06f9dfc, [#31]) |
| 2026-10-25 | `validation.validate_null_world` | The pre-registered contrast (PPO against the seller at one premium) fired in the null world, because the seller pays its drag and a flat learner does not | a5a5fd8 ([#26]), A1: each arm against itself at zero premium |
| 2026-10-28 | the section 8 gate's numbers | The planner used the normal approximation, overstating power at small S (its floor is 18% low at S = 5) | d6a2f7c ([#20]): planning on the exact noncentral t |
| 2026-10-28 | review of the invariants | The conformance invariants could not see calendar arbitrage | b8f694e ([#29]): `leaks.calendar_violations` in every conformance report |
| 2026-10-28 | a real-chain P&L test | A held leg's mark jumped when the nearest listed tenor changed (about $2,800 on a condor) | 7966075 ([#27]): `TermStructureChain` interpolates total variance |
| 2026-10-28 | the M2 harness item | A run trained by the tests' stand-in learner counted as PPO | e8321a3 ([#31]): the manifest names its trainer |
| 2026-10-28 | a killed worker | A dead worker hung the grid and the job pool | f3d1490, b80bbb8 ([#30]) |

## 7. Defects found by this week's audits, and their fixes

One line each. All the fixes are dated 2026-10-28.

**Data path**
- `split=None` reached every date, holdout included, and was recorded as `eval`. Now a
  ValueError at every entry point: 48d0201 ([#28]).
- Ranges of your own were checked only against `root`, so a symlink, `DEFAULT_ROOT` or
  the low-level API reached the holdout. The team's panel is now known by its digests,
  and `split` is required: 48d0201 ([#28]).
- Action mix on a config ran on `train`, not `eval`: 48d0201 ([#28]).
- The real provider served calendar arbitrage on 21 train days (180 cells; 23 of 40
  seeds), which made M1 and M2 NOT CLEARED on the branch. It now serves an isotonic
  projection in tenor, with 0 failing days: 05dd4ed ([#29]).
- The window gap check missed a day absent from both tables. Gaps are now found on the
  NYSE calendar: 05dd4ed ([#29]).
- carry30 carried the close-to-snapshot gap (-18% to +13%). The engine now prices on the
  parity spot with `carry30_parity` (-4.1% to +6.0%): 05dd4ed, 06f9dfc ([#31]).
- The 2010 PM weeklies sat under the AM root SPX. The root is now the settlement flag:
  06f9dfc ([#31], panel version 4).
- Expiries on the 2018 and 2025 unscheduled closures kept the vendor's date (ATM IV
  3.1% and 1.0%). They now settle the trading day before: 06f9dfc ([#31]).
- Stale SPXW expiries on 2018-02-05 (208 bp off the forward curve) are now dropped:
  06f9dfc ([#31]).
- `docs/DATA.md` had stale v2 statements (three days without IV30, a cell count that
  left out NaN to value). Brought to v3 and v4: 05dd4ed, and this week's docs pass
  ([#31]).
- Two panels were described as "the" 2026-10-28 rebuild. `docs/DATA.md` now names one
  panel of record: 6c91c8f ([#31]).

**Machinery**
- `mypy --strict` failed with the `train` extra installed, so every partner record would
  have read FAIL. Fixed with `type: ignore[misc, unused-ignore]` and a CI job on
  `.[dev,train]`: 241e572 ([#23]).
- Runs made with your own components looked like default runs and were reused, and a
  real-data cell was reused after a panel rebuild. A run now records its components,
  panel, trainer and platform, and every reuse check compares them: f8c7f2a ([#21]).
- `run_sweep` wrote over another source's run, and readers that named no trainer
  accepted a stand-in's PPO cells: f8c7f2a ([#21]).
- Replay had no tolerance mode. It now has `--rtol`: f8c7f2a ([#21]).
- BLAS thread caps were set after numpy loaded, file I/O used the locale encoding, and
  `.DS_Store` marked every run dirty: f8c7f2a ([#21]).
- A Windows partner with `core.autocrlf` failed the expected-file hash. Both sides now
  hash with LF, and `.gitattributes` keeps committed JSON LF: 45258d1 ([#10]).
- The partner record could pass on a commit before the cutoff tag: 45258d1 ([#10]).
- `make gate`, `drag` and `bench` imported another checkout's package in a shared venv,
  and `make reproduce` could pick a Python older than 3.12: 45258d1 ([#10]).
- A cancelled PPO job kept training, the job pool forked from a threaded server, and a
  job killed part way could be reused without its equity paths: 30b3362 ([#30]).
- The dashboard's default runs root was not the orchestrators': 2d4fb5b ([#30]).
- The panel animation windows were unbounded, the pilot's matrix manifest was served as
  a run, and error texts showed absolute paths: 0570a41 ([#30]).
- The M3 sweep item replayed byte for byte only, so it failed off the sweep machine. It
  now compares within the seller tolerance and samples no PPO cell trained elsewhere:
  e720b4b ([#11]).

**Numbers and statistics**
- A PPO cell does not reproduce off Dominick's machine (section 2). The partner PPO
  check is now statistical: 45258d1 ([#10]).
- The M4 attribution item could pass only on Dominick's machine. It now requires
  `agrees`: 10ac7cc ([#35]).
- `vrp.power_at` refused the real PPO curve, which dips by sampling noise, so the M4 VRP
  item failed. The curve is now read from its knots as the analysis builds it, giving
  PPO 0.525 at +0.64: 8ed3d24 ([#15]), 10ac7cc ([#35]).
- Politis-White used K_N = max(5, log10 n). It now uses max(5, sqrt(log10 n)) as the
  paper does: 8ed3d24 ([#15]).
- The M4 rules' timing, A7's measured power, Brown-Forsythe's power, the floor's limits,
  one lambda_80 solver, per-level bootstrap seeds and the section 8 null world at S = 5:
  0ee5428 ([#12]).
- A3 still calls the planner normal-approximation, while the code is exact since
  d6a2f7c. It is to be corrected in A9 ([#37], open).

**Docs** (this week's docs pass, [#38])
- The README's lane map gave `rng.py` and `types.py` to the wrong lane, missed a dozen
  modules, and listed three actions where there are four.
- The README's runtime row for the pilot said 16 cells and 2.3 core-hours. It is 32
  cells and 4.5 core-hours.
- The README, the `--timesteps` help and `pilot_budget.py` called the training budget a
  placeholder. A4 fixes it at 1,000,000.
- Stale lines: "being written in parallel", "the rebuild under way now", "until the W6-3
  build writes it", "What comes next".
- `docs/COSTS.md` labelled the pre-freeze parameters "current" and said the fit had been
  reverted.
- The README's drag table had no `settles` column.

[#3]: https://github.com/dominickdupuy/VegaLab_Project/issues/3
[#4]: https://github.com/dominickdupuy/VegaLab_Project/issues/4
[#9]: https://github.com/dominickdupuy/VegaLab_Project/issues/9
[#10]: https://github.com/dominickdupuy/VegaLab_Project/issues/10
[#11]: https://github.com/dominickdupuy/VegaLab_Project/issues/11
[#12]: https://github.com/dominickdupuy/VegaLab_Project/issues/12
[#15]: https://github.com/dominickdupuy/VegaLab_Project/issues/15
[#19]: https://github.com/dominickdupuy/VegaLab_Project/issues/19
[#20]: https://github.com/dominickdupuy/VegaLab_Project/issues/20
[#21]: https://github.com/dominickdupuy/VegaLab_Project/issues/21
[#23]: https://github.com/dominickdupuy/VegaLab_Project/issues/23
[#26]: https://github.com/dominickdupuy/VegaLab_Project/issues/26
[#27]: https://github.com/dominickdupuy/VegaLab_Project/issues/27
[#28]: https://github.com/dominickdupuy/VegaLab_Project/issues/28
[#29]: https://github.com/dominickdupuy/VegaLab_Project/issues/29
[#30]: https://github.com/dominickdupuy/VegaLab_Project/issues/30
[#31]: https://github.com/dominickdupuy/VegaLab_Project/issues/31
[#32]: https://github.com/dominickdupuy/VegaLab_Project/issues/32
[#35]: https://github.com/dominickdupuy/VegaLab_Project/issues/35
[#37]: https://github.com/dominickdupuy/VegaLab_Project/issues/37
[#38]: https://github.com/dominickdupuy/VegaLab_Project/issues/38
