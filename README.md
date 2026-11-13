# optspread

A leak-tested SPX options simulator used to measure how small a trading effect
our research setup can actually detect.

Vega Lab, CIS 4914 Senior Project, University of Florida, Fall 2026. Team:
Dominick Dupuy (lead), Hemdutt Rao, Ilan Danial. Advisor: Ashish Aggarwal.

The headline deliverable is not a profitable agent. It is a calibrated
instrument and a stated detection floor. See the project document for the full
argument; this README covers what is in the repository today.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,train]"   # Windows
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,train]"      # macOS / Linux
make check                   # ruff + mypy --strict + pytest with a coverage floor
make gate                    # the Wave 0 economic sanity table
make drag                    # scripted sellers against the predicted cost drag
make leaks                   # leak coverage across every observation field
make bench                   # throughput by workload, measured not assumed
make m1                      # docs/M1_GATE_REPORT.md from live runs (re-dates the committed one)
make m2                      # docs/M2_GATE_REPORT.md: what M2 has, and what failed first
make m3gate ARGS="..."       # docs/M3_GATE_REPORT.md: the sweep, the leak floor, the demo, the partner
make reproduce               # a partner team's one command (see below)
make vrp                     # docs/VRP_MAPPING.md: the real SPX premium in planted vol points
make m4 ARGS="--root ..."   # docs/M4_ANALYSIS.md: power curves, floors, the gap (the sweep)
make m4gate ARGS="..."       # docs/M4_GATE_REPORT.md: the M4 items, from the analysis and the sweep
make actionmix ARGS="..."    # which structures a policy trades, from runs or a grid (see below)
make edge ARGS="--grid ..."  # which greeks and structures carry the detected edge (see below)
make train-smoke             # a two-seed PPO pilot on two workers, then every run replayed
make run ARGS="..."          # one run; --source optionmetrics for the real panel (see below)
make pilot ARGS="..."        # the variance-components pilot (see below)
make sweep ARGS="..."        # the full premium x arm x seed sweep (see below)
make replay RUN=<run dir>    # rerun from the manifest; bit-identical, or ARGS="--rtol 1e-9"
make web && make serve       # the dashboard on http://127.0.0.1:8600 (venv with the dashboard extra, + npm; see below)
```

What we found wrong, how it was fixed and what it touched is in
[`docs/DEFECTS.md`](docs/DEFECTS.md), the defect appendix.

## Reproduce our results (partner teams)

All you need is a clone and Python 3.12 or newer, on macOS, Linux or Windows; no access to
our shared folder.

```bash
git clone https://github.com/dominickdupuy/VegaLab_Project.git
cd VegaLab_Project
python scripts/reproduce.py        # python3 on macOS and Linux; or: make reproduce
```

It creates `.venv`, installs the project into it, and runs in turn: ruff, `mypy --strict`
and the test suite (`make check`), the Wave 0 gate (`make gate`), the cost drag (`make
drag`), the leak suite (`make leaks`), the poisoned-environment demo, and three cells of our
confirmatory sweep regenerated from scratch and compared with `docs/SWEEP_EXPECTED.json`
(the sweep's digests and numbers; no market data). Expect about 10 minutes after the
install (8 on an Apple-silicon laptop), five of them the test suite. `--with-ppo` adds one PPO cell, a million training
steps, about 20 minutes more; `--cells N` regenerates more seller cells (about 40 seconds
each); `--team "Your team"` names you in the record.

A seller cell is compared byte for byte and within a tolerance (`optspread/reproduction.py`
states and justifies both: on another platform the bytes differ in the last digit, so the
tolerance is what passes). **The PPO cell is checked statistically**, because PPO training
is not bit-reproducible across platforms, not even at our exact library versions: retrained
on macOS our reference cell is another policy (mean 4,661 against 4,958 on Windows, most
trade counts different). It passes when its mean P&L is within 3 standard errors of ours
and it trades, refuses no open and keeps the attribution residual small. It is optional;
run, it must pass too. It prints PASS or FAIL per step and writes
`docs/M3_PARTNER_REPRODUCTION.json` and `.md`. **Send both back**, by a pull request adding
them or by email, failures included. `make m3gate` reads the JSON for the M3 gate's partner
item (#10, #11): it passes on a clean checkout at or after the `cutoff-2026-11-01` tag with
every step passed.

## The lane map

Lanes per project document sections 9.2 and 10: Hemdutt Rao the market layer and the
data adapter, Ilan Danial the machinery and the dashboard, Dominick Dupuy the statistical
spine. Each module's docstring opens with the same `Lane:` line; `scripts/` follow the
module they drive.

| Module | Lane | What it is |
|---|---|---|
| `rng.py` | Hemdutt | Per-step RNG stream contract. `stream(t)` depends only on `(seed, channel, t)`. |
| `types.py` | Hemdutt | `OptionLeg`, `Chain`, `Position`, and the `ChainProvider` protocol. |
| `pricing.py` | Hemdutt | Black-Scholes prices and analytic greeks, NaN-free at the limits. |
| `market.py` | Hemdutt | Zero-drift GBM with the chain priced off the path vol, plus a plantable premium. |
| `effects.py` | Hemdutt | The `PlantedEffect` interface: the constant premium (D2), and skew and term examples. |
| `costs.py` | Hemdutt | Quoted half-spread plus commission. |
| `margin.py` | Hemdutt | Max-loss for defined risk; Cboe Rule 10.3(c)(5)(A), the broad-based index rule, for undefined risk. |
| `actions.py` | Hemdutt | The locked action library: FLAT, iron condor, short strangle, put credit spread. |
| `portfolio.py` | Hemdutt | Cash and position book; margin checked before an open; cash settlement at expiry. |
| `policies.py` | Hemdutt | Flat, always-on, random, and the fixed-delta seller (the oracle). |
| `gates.py` | Hemdutt | The Wave 0 economic sanity gate. |
| `scripted.py` | Hemdutt | Scripted sellers and the analytic cost-drag check. |
| `data.py` | Hemdutt | The adapter boundary and its conformance suite; `make_source`; the OptionMetrics stub. |
| `extend.py` | Hemdutt | `Components`: your own source and cost model as the runner every orchestrator takes. |
| `vrp.py` | Hemdutt | The real SPX premium on the sweep's axis: forward realised vol, a block bootstrap, a power curve read at it. |
| `realdata/__init__.py` | Hemdutt | The A8 splits (`SPLITS`), the team's panel known by its digests, the loader behind the adapter. |
| `realdata/optionmetrics.py` | Hemdutt | WRDS `opprcd` and `secprd` normalised: roots by settlement flag, expiries moved to settlement days. |
| `realdata/opra.py` | Hemdutt | Databento OPRA batches: the inventory and the SPX and SPXW closing quotes. |
| `realdata/surface.py` | Hemdutt | Quotes to a surface: parity carry, our IV and greeks. |
| `realdata/panel.py` | Hemdutt | The stitched panel: daily series, surface grid, coverage, the vendor seam, the NYSE calendar. |
| `realdata/provider.py` | Hemdutt | Real chains as episodes: one split, a calendar-arbitrage-free surface, the parity spot. |
| `config.py` | Ilan | Frozen pydantic configs and the content hash every run is keyed by. |
| `manifest.py` | Ilan | Run manifests: git SHA, config hash, library versions, platform, data source, trainer, artifact digests. |
| `env.py` | Ilan | Gymnasium env and the append-only observation schema. |
| `harness.py` | Ilan | Episode runner and evaluator; the one path from a config to numbers. |
| `artifacts.py` | Ilan | The run output schema: versioned `RunRecord`, hashed into the manifest. |
| `experiment/sweep.py` | Ilan | Scripted premium sweep on paired seeds. |
| `runs/rundir.py` | Ilan | Run ids and the resume rule every orchestrator shares; one orchestrator per results root. |
| `train.py` | Ilan | The PPO arm at the frozen D3 hyperparameters; training paths kept off evaluation paths. |
| `experiment/` | Ilan | The pilot and sweep grid: premium x arm x seed block (`grid.py`), parallel and resumable (`schedule.py`), the pilot (`pilot.py`). |
| `replay.py` | Ilan | The reproducibility gate: rerun from the manifest, compare `run.json` bytes (or within `--rtol`). |
| `experiment/cli.py` | Ilan | The `--source` flag every command that launches runs takes, read the same way. |
| `dashboard/` | Ilan | What the dashboard reads: run index, gate board, risk figures, live leak demo; `dashboard_data.py` re-exports it for the M2 gate. |
| `dashboard/panel_views.py` | Ilan | Read-side views of the SPX panel: coverage, smile, surface, term structure, VRP, seam. |
| `dashboard/jobs/` | Ilan | Runs launched from the dashboard: knobs to `RunConfig`, a bounded process pool. |
| `api/` | Ilan | The API's typed payloads, each saying where its numbers came from: the runs, the panel, the study, its checks and analyses, the work a view starts; the answer cache (`cache.py`). |
| `server.py` | Ilan | FastAPI over the `api/` payloads (`api/server.py`), serving the static site in `web/out`. |
| `web/` | Ilan | The dashboard: Next.js static export, ECharts, two themes from the deck. |
| `leaks.py` | Dominick | Perturbation property, structural invariants, poisoned env, leak floor. |
| `gatereport.py` | Dominick | Milestone reports: criterion, evidence, what failed first; a JSON sidecar. |
| `power.py` | Dominick | Variance components, power (planning and exact), sample size, budget. |
| `analysis.py` | Dominick | The confirmatory decision rule: one-sided t on paired seed-mean contrasts. |
| `validation.py` | Dominick | The section 8 gate: the pipeline against a closed-form answer. |
| `m2gate.py` | Dominick | One evidence function per M2 deliverable. |
| `m3gate.py` | Dominick | One evidence function per M3 deliverable: the sweep, the leak floor, the demo, the partner. |
| `reproduction.py` | Dominick | What a partner team reproduces from a clean clone, its tolerances, and the record it sends back. |
| `m4analysis.py` | Dominick | The M4 analysis of the sweep: power curves, floors, the oracle-to-PPO gap, tau^2 homogeneity, bootstrap. |
| `m4gate.py` | Dominick | One evidence function per M4 deliverable, read from the analysis, the mapping and the sweep. |
| `attribution.py` | Dominick | Per-step P&L into delta, gamma, vega, theta, carry, cost and a residual; JSON for the attribution view. |
| `edge.py` | Dominick | The detected edge (A1, A2) split by component and by structure held. |
| `actionmix.py` | Dominick | Which structures a policy trades, day by day and across a grid. |

Documents: `docs/PRE_REGISTRATION.md` records the decisions closed at the
September 14 advisor meeting (D1 to D3) with the alternatives we rejected, and the
amendments appended since (A1 to A8: the detection event, the test, the power
calculation, the training budget, the pilot, the rule that set S and N, the
confirmatory S and N, the real-data dates). `docs/M1_GATE_REPORT.md` is generated by
`make m1` and is what the Presentation 1 slides are built from; `docs/M2_GATE_REPORT.md`
comes from `make m2`. Each has a `.json` sidecar from the same build, which is what the
dashboard's gate board reads. `docs/LEARNING_CURVE.md` is the check that set the
training budget (A4); `docs/SWEEP_EXPECTED.json` holds the confirmatory sweep's digests
and numbers for the partner reproduction; `docs/DATA.md` is the real SPX panel and
`docs/DATA_MANIFEST.json` its file digests; `docs/VRP_MAPPING.md` the real premium in
planted vol points; `docs/COSTS.md` the cost model's fit to real quotes;
`docs/DEFECTS.md` the defect appendix.
`docs/EXTENDING.md` shows how to bring your own data, agent, planted effect and cost
model, and how to put each through the leak suite, the paired grid and the paired
test; every example in it runs in the test suite.

## The null world

The chain is priced at the same volatility that drives the underlying path, with
zero drift. Every contract is then worth its discounted expected payoff, so every
strategy has zero expectancy before costs and negative expectancy after. That is
not an informal expectation. It is `make gate`, and the build fails if it breaks.

Representative table at 100 episodes per arm, mean net P&L per 63-day episode:

```
arm           costs       mean P&L        se  trades
flat          False              0         0     0.0
flat          True               0         0     0.0
always_on     False           -509     1,157     3.0
always_on     True            -786     1,162     3.0
random        False           -333       370    70.7
random        True          -4,583       371    70.7
```

Three orderings are asserted: flat sits at exactly zero either way, always-on
reads zero without costs and turns negative with them, and the churning random
policy bleeds fastest. The numbers move with the seed; the inequalities do not.

## Checking the cost model against arithmetic

`make drag` runs four scripted sellers, a hold-to-expiry and a rolled version of
the iron condor and the put credit spread. They are not strategies. Under fair
implied volatility a seller has no premium to collect, so its expected result is
minus its costs and nothing else, which makes the average loss a number that can
be predicted by hand and compared.

```
seller           structure               free   costed     drag  predicted      gap  settles
hold_condor      IRON_CONDOR             -515     -772      257        257        0      1.0
hold_put_spread  PUT_CREDIT_SPREAD     -2,058   -2,186      128        128        0      1.0
roll_condor      IRON_CONDOR            1,716    1,366      350        350        0      0.0
roll_put_spread  PUT_CREDIT_SPREAD       -817   -1,014      197        197        0      0.0
```

`settles` is the mean number of cash settlements per episode: a hold-to-expiry seller
settles its one structure, a rolled seller closes before expiry and never settles.

Three checks: costs change the accounting and nothing else (`pnl_costed` equals
`pnl_free` minus what was charged, per seed, on shared paths); a cash settlement
charges nothing; and no structure leaves the book without paying. The third one
failed on the first run, 74 to 128 dollars an episode short of the prediction. A
hold-to-expiry seller ended the episode still holding an unexpired structure,
marked at mid, with no exit ever charged, so the reported P&L was one crossing
better than anything the episode could have realised. Expiry and a live close are
now distinct paths in the book: the first is free because SPX settles in cash, the
second crosses the spread.

## Proving the world is clean

`optspread.leaks` is the detector, and `tests/stats/test_leaks.py` is the gate.

**Perturbation property.** Generate a path, resample every RNG stream after step
`t`, and assert the chain at `t` is unchanged. This is a direct machine check
that no future information reaches the present. It is only constructible because
randomness is factored per step: one generator drawn sequentially cannot express
"same past, different future".

**Structural invariants.** Consecutive timestamps, ordered strikes and expiries,
positive vols, no-arbitrage price bounds, put-call parity within tolerance, and no
calendar arbitrage. Bounds and parity price each expiry from its own vols, so they
hold for any positive vol: a 60% 21-day row over a 15% 42-day row passed them,
though its ATM call costs 353 points to the 42-day call's 139 (#29). The calendar
check compares total implied variance, sigma^2 T, from each listed expiry to the
next at the same forward moneyness, and fails when it falls. The tolerance is
rounding (1e-12 of w) plus a bound on the check's own interpolation between the
longer expiry's strikes, which is zero on the flat synthetic surface and about
1e-5 of w on a skewed real one; the #29 surface falls by 0.026. It runs wherever the
other invariants run: `conformance_report` for every `ChainSource`, and `make leaks`,
which also shows it firing on its control.

**Poisoned environment.** `PoisonedChainProvider` injects one-step look-ahead:
the spot shown at `t` carries `eps` times the return into `t+1`. The perturbation
check catches it exactly. The statistical detector catches it too, by testing
whether any observation feature at `t` predicts the path's true next return.

**Every field, not just the price fields.** The rule is written down in
`leaks.PASS_CRITERION`, which is the part that was missing: the criterion lived in
its author's head, so the suite read as a pass every time it ran. Stating it
exposed two real gaps.

The first was the probe. Under a flat book, nine of the fourteen observation
fields sit at a constant zero -- every inventory, greek and margin slot -- so a
flat-only probe silently tested the price fields and reported a pass for the rest.
The detector now probes with a policy that trades, whose actions come from
`POLICY_CHANNEL` and so are independent of the path. That leaves one field
untested, `atm_iv`, and it is reported as untested with the reason: the null
world's surface is flat at a constant vol, so the field cannot carry a signal.

The second was the control. Every field now gets its own positive control -- a leak
planted in that field alone, with the finding required to be attributed to that
field rather than a neighbour. The first version planted one absolute `eps` and
four fields failed quietly: `time_frac`, `has_position`, `action_frac` and
`dte_frac` have natural variance that dwarfs a fixed leak, so the control was
invisible in them. Controls are now sized in units of each field's own standard
deviation, which makes `eps` a signal-to-noise ratio that means the same thing
everywhere. At `eps = 0.25` all fourteen fields fire and are attributed correctly.

```
  [PASS] clean world is quiet
  [PASS] every field tested or explained
  [PASS] every field has a working control
```

**Leak detection floor.** The same calibration logic, applied to the detector
itself. Planting spot leaks of decreasing size and recording the smallest still
caught gives the detector a stated floor. At 10 seeds and 63 days it sits at
`eps = 0.10`: a leak carrying a tenth of tomorrow's return is caught, half that is
not. This costs no training compute, which is why it runs in CI.

Passing this suite shows we catch the modes it is built to catch. It does not
prove the absence of unknown leakage anywhere in the codebase.

## Milestones close on a report

`optspread.gatereport` renders a milestone report from live runs rather than
prose, so it cannot drift from the tests. Each item carries its criterion, its
evidence, and on a failure what failed first. There are three verdicts, not two:
the third is `NO CRITERION`, for an item with evidence but no rule another person
could check, and it does not clear a gate whatever the evidence says.

`make m1` writes `docs/M1_GATE_REPORT.md`. The committed report (2026-09-28) reads
**PASS** on all eight items. It read NOT CLEARED until the PPO arm was measured, which is the
report doing its job: partial work is not called done, and the item that was
outstanding was named rather than folded into a passing summary.

Every report is also written as JSON beside the Markdown (`M1_GATE_REPORT.json`),
with a fixed schema: milestone, report date, verdict, and per item the name, owner,
verdict, criterion, evidence and what failed first. The M1 sidecar was read back
from the committed 2026-09-28 report rather than produced by re-running M1, which
would have re-dated it; `GateReport.from_markdown` accepts a report only if it
re-renders the file byte for byte, and a test holds every committed report and its
sidecar to that.

`make m2` writes `docs/M2_GATE_REPORT.md`. Three of its items (the PPO harness,
replay, the dashboard) read other lanes' code, so `optspread.m2gate` imports it by
name and reads NOT CLEARED with the failed import if it is ever absent. The
pre-registration item looks for the earliest `prereg-*` tag (`prereg-m2` at the M2
meeting) and requires the file at HEAD to begin with the tagged text byte for byte:
amendments are appended, never edited in place. The committed M2 report (PASS, 7
of 7) was made on Dominick's Hugging Face-mirror build of the panel, not the panel
of record (`docs/DATA.md`, The panel of record).

`make m3gate ARGS="--runs-root $VEGALAB_DATA/runs --replay 2"` writes
`docs/M3_GATE_REPORT.md` (#11): the A7 sweep complete and valid under the runs root,
written from a clean tree at or after the cutoff tag; the leak floor measured again
and equal to M1's; the poisoned-environment demo called through the server and failing
on purpose; and the partner's `docs/M3_PARTNER_REPRODUCTION.json` (below). Until the
partner team sends its record, the partner item reads NOT CLEARED.

## Validating the analysis itself

Plotting code always produces a plot, so a power curve is not evidence that the
analysis is right. Project document section 8 makes that a gate:
`optspread.validation` runs the pipeline end to end on synthetic paired arms with
tau^2, sigma^2 and the effect set by hand, where the correct detection probability is
a closed form, and compares.

The rule under test is `optspread.analysis.decide`: pair the arms cell by cell (the
shared path cancels), estimate the components, and run a one-sided t on the S
seed-level means of the contrast with S - 1 degrees of freedom. Under normal seed
means its detection probability is exactly a noncentral t, `power.power_t`. The
criterion is `validation.PASS_CRITERION`: at every effect on the grid the rejection
rate over R replications lies within 4 Monte Carlo standard errors of the closed form,
and the estimated sigma^2 and Var(grand mean) are unbiased to the same standard.

```
case: tau^2 = 4, sigma^2 = 100, shared path variance 400, S = 10 x N = 25, one-sided alpha = 0.05
  delta  analytic  empirical   MC se      z   normal
      0    0.0500     0.0468  0.0022  -1.47   0.0500
    0.5    0.1299     0.1272  0.0034  -0.80   0.1388
      1    0.2709     0.2751  0.0044   0.96   0.2992
      2    0.6624     0.6625  0.0047   0.03   0.7228
      3    0.9255     0.9266  0.0026   0.43   0.9563
PASS (every check within 4 Monte Carlo standard errors)
```

The gate has negative controls, as the leak suite does: three rules an analysis
plausibly gets wrong (normal critical values in place of t, arms treated as unpaired,
episodes treated as independent observations) must each fail it, and in the M2
report they do. The subtlest is the first: normal critical values at ten seeds give
a test of size 0.067 rather than 0.05, and the gate misses its closed form by 14
standard errors at delta = 1. The other two miss by more than 200. The controls also
guard the replication count: below about 2,000 per effect the gate is too blunt to
see the first mistake, and the item then fails on the control instead of passing.

**The null world.** A rule can pass all of that and still be the wrong test. The
first pre-registered contrast, PPO against the fixed-delta seller at one premium,
fires with nothing planted: the seller pays its cost drag, a learner that stays
flat does not, and the flat learner "beats" it (#26). The detection event is
therefore each arm against itself at zero premium, on the same paths and, for PPO,
the same training seed (`analysis.decide_premium`), so the drag and the path
variance cancel together. `validation.validate_null_world` is the check: a flat
learner, a control paying a drag of 5 (4.6 standard errors of the replaced event's
contrast), nothing planted. The
pre-registered event fires at 0.050; the event it replaced fires at 0.995 and is
the gate's negative control. At zero itself the event is identically zero (an arm
against itself on the same paths), so the sweep's leftmost point is degenerate
and the size of the test is this check's to show, not the sweep's.

The last column is the finding. The normal approximation, `power`, reads 0.72 at
ten seeds where the test's real power is 0.66; at five seeds its detection floor is
18% too low (the test has 66% power there, not 80%), at three 38%, and it puts the
S = 5 to S = 50 floor ratio at sqrt(10) = 3.2 where the test's is 3.8. So the
planner does not use it (#20): `episodes_for`, `seeds_for`, `frontier`,
`scripts/pilot_budget.py` and the dashboard's power lab all plan on the exact
noncentral t, through `t_noncentrality`, the `delta / se` at which the test on S
seeds has the target power (3.04 at S = 5 against the normal 2.49). A planned
design has the power the committed test will have, at least two seeds, and agrees
with the pre-registered rule of A6. `power`, `detectable_effect` and
`variance_budget` remain, for orientation only.

## Throughput

Measured, not assumed (project document section 11). `make bench` times four
workloads rather than one, because the first version of this benchmark was
misleading: timing the env with a FLAT action gave 37,000 steps/s, which looked
like the environment was free, but a flat book never prices a leg.

```
flat            39,674/s   empty book; the floor, not a planning number
always_on        1,869/s   one structure held and re-marked: the steady state
random           2,233/s   churns, but FLAT a quarter of the time, so it marks less
ppo              1,009/s   includes the training update
```

The cost is **per-step position marking, not trading**. `random` comes out faster
than `always_on` despite 24 times the trades, because one action in four is FLAT
and those steps mark nothing. Re-pricing greeks for every held leg dominates; the
policy's own decision is nanoseconds.

`ppo` belongs next to `random`, not `always_on`, because an untrained policy
explores close to uniformly, so that is the workload it puts on the environment.
The gap between them is the training update: **55% of PPO wall clock**, the rest
being the environment. Both halves are worth optimising if the sweep ever needs to
be faster.

Plan the sweep off **about 1,800 steps per second per core**, the `always_on`
figure, which is what a trained policy will mostly do, and **about 1,000** once
training is included. Both sit inside the document's unvalidated 500 to 3,000
estimate. Re-measure on any machine that will run the sweep.

The learner is the `train` extra rather than a core dependency, so `make check`
does not pull torch: nothing in `optspread/` imports it at module import (`train.py`
imports it inside the calls that train). When it is installed,
`library_versions` records torch and Stable-Baselines3 in every manifest, which says
only that they were installed. What trained a learned run is recorded beside them: its
manifest's `trainer` names the function (`optspread.train.train_ppo` for the real
learner), and the M2 harness and pilot items count only runs the real learner trained,
so a run made with the tests' stand-in learner, which has the same config and run id,
is never taken for PPO (#31). A manifest written before the field existed counts when
its hashed `policy.zip` is a Stable-Baselines3 archive.

## Reproducibility, enforced

A `RunConfig` plus a seed fully determines a run's numerical output. Every run
directory holds two halves: `manifest.json` for provenance and `run.json` for
results, written in that order so the manifest hashes the bytes on disk.
`RunRecord` carries a schema version, and a reader that meets a version it does
not know refuses rather than misreading fields.

`RunManifest.write` validates before it writes. A manifest missing its run id,
git SHA, config hash, config or library versions is refused, as is one whose
`config_hash` has drifted from its own config, and one whose git SHA is unknown.
A run nobody can reproduce cannot reach the results folder, which is a rule the
tooling enforces rather than one the team remembers.

The manifest also records where the chains came from (`data`): the `optspread.data`
source and, for the real `optionmetrics` source, the sha256 of each panel file it read,
keyed as `docs/DATA_MANIFEST.json` keys them. The source is provenance, not config, so
it is not in the config hash and no run id moved when it was added. A manifest written
before it was recorded has no `data` and still validates; every such run ran on the
generator, which was then the only source the harness could build.

`make replay RUN=<run dir>` is the gate of project document 10.7. It reads the
config from the manifest, retrains if the arm is learned, re-evaluates, and
compares the rendered `run.json` with the file on disk byte for byte, not to a
tolerance. A results file edited after its manifest was written fails before the
replay starts, and a mismatch names the first episode that differs. The replay runs
on the recorded source; a real-data run whose panel has been rebuilt or is missing
fails and says so rather than replaying on other data. A PPO run
replays because torch runs one thread per process on CPU, the learner is seeded
from the config, and evaluation takes the greedy action. Bit-identity is promised
on the machine and library versions that produced the run; the replay reports any
version, commit or platform (OS, architecture, Python, numpy, recorded in the manifest) that
has moved since, because those are the first suspects. A learned run is retrained only by the
trainer its manifest names, and a run of components of one's own only through them.

`ARGS="--rtol 1e-9"` replays a scripted run made on another platform, where libm rounds the
last bit otherwise: every number of `run.json` within the relative tolerance, every count
exact, reported as `IDENTICAL`, `WITHIN rtol 1e-09 (max rel diff ...)` or `DIFFERS`. Our
Windows sweep's seller cells replay `WITHIN` on macOS. No tolerance makes a retrained PPO run
pass on another platform: the training itself diverges into another policy, so a PPO cell
replays only where it was trained (`make m3gate ARGS="--replay N"` samples only seller
cells there and says so).

## The PPO arm

Decision D3 is enforced, not just recorded. `PPOHyperparameters` writes out the
Stable-Baselines3 defaults (21 values across `PPO` and `ActorCriticPolicy`), every
one is passed to the library explicitly so an upgrade cannot move them, and
`TrainConfig` refuses any other set. A PPO run's `RunConfig` carries its
`TrainConfig`, so the hyperparameters, `total_timesteps`, the training seed and the
device are all in its config hash. `total_timesteps` is the one number D3 did not
fix, so it has no default anywhere: every command that trains asks for it.
Pre-registration A4 fixed it at 1,000,000 per seed for the pilot and the sweep, from
the learning-curve check (`docs/LEARNING_CURVE.md`). A
scripted run serialises exactly as before, so adding the learner re-keyed no
existing run.

One bug was caught before it shipped. Stable-Baselines3 seeds the first reset and
then resets with no seed, and the environment reads a missing seed as path 0, so a
plain training run would have replayed one path for its whole budget. Training
episodes now draw from their own seed range at 2^48 and above, a separate block per
training seed, and evaluation seeds are refused there, so no policy is scored on a
path it trained on.

## Handing the pilot and sweep to a bigger machine

Each run is one command, resumable, parallel across processes, and writes to any
root. Point `--root` at the team's shared data folder, or leave it out: with
`VEGALAB_DATA` set, the pilot defaults to `$VEGALAB_DATA/runs/pilot` and the sweep to
`$VEGALAB_DATA/runs` (without it, `runs/pilot` and `runs` under the current
directory), which are the roots `make m2` and `make m4` read by default, and the
one `make serve` reads unless `OPTSPREAD_RUNS` names another.

**Install** (Linux, Python 3.12). Take the CPU build of torch: a 64x64 MLP gains
nothing from a GPU, and the CUDA wheels are several gigabytes.

```bash
git clone https://github.com/dominickdupuy/VegaLab_Project.git && cd VegaLab_Project
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,train]"
export PY=.venv/bin/python VEGALAB_DATA=/path/to/shared/folder
make check PY=$PY            # the test ladder, including the real-PPO tests
make train-smoke PY=$PY      # about 30 s: training, workers and replay on this box
$PY -m scripts.bench --steps 20000 --ppo   # re-measure the rates on this machine
```

**The pilot**, then the budget it sets:

```bash
make pilot PY=$PY ARGS="--root $VEGALAB_DATA/runs/pilot --seeds 8 --episodes 200 \
    --timesteps 1000000 --workers 16"
$PY scripts/pilot_budget.py --delta <effect> --matrix $VEGALAB_DATA/runs/pilot/matrix.npy \
    --train-steps 1000000 --cores 16
```

The pilot runs both arms at zero and at `--premium` (default 1 vol point, #5),
because the detection event is each arm against itself at zero (#26). It writes
`matrix.npy` (S x N, PPO at the premium minus PPO at zero, what the test tests) and
`matrix_oracle.npy` (the fixed-delta seller's same contrast, the negative control
whose seed component is zero by construction), each arm's raw matrices beside them,
and prints both sets of variance components.

**The sweep**, at the S = 5 and N = 500 the pilot set (pre-registration A7):

```bash
nohup make sweep PY=$PY ARGS="--root $VEGALAB_DATA/runs --seeds 5 --episodes 500 \
    --timesteps 1000000 --workers 32" > $VEGALAB_DATA/runs/sweep.log 2>&1 &
make replay PY=$PY RUN=$VEGALAB_DATA/runs ARGS="--max 4"   # spot-check afterwards
```

The default grid is 0, 0.25, 0.5, 1, 2 and 4 vol points (`--premiums`) by the
fixed-delta seller and PPO (`--arms`). Within seed block i every arm at every
premium runs the same evaluation paths, and block i's policy is training seed
`--train-seed + i` at every premium. Blocks are disjoint, which is the nested
design the pilot formula assumes. The pilot and the sweep default to different
path and training seeds, so the pilot's data never enters the confirmatory run.

**Runtime.** Add `--dry-run` to either command to print the plan before
committing a machine: cells on disk, cells to run, core-hours at the M1 rates.
At those rates (about 1,000 steps/s/core with the training update, about 1,800
for evaluation) a PPO cell costs `timesteps / 1000` seconds of training plus
`N x 63 / 1800` of evaluation, about 17 minutes at one million steps, and a
scripted cell only the evaluation:

| Run | Cells | Core-hours | Wall, 16 workers | Wall, 32 workers |
|---|---|---|---|---|
| pilot, S=8, N=200, 1M steps (both arms at 0 and 1 vol point) | 32 | 4.5 | 17 min | 17 min |
| sweep, 6 levels, S=5, N=500, 1M steps (A7, the confirmatory run) | 60 | 8.6 | 34 min | 17 min |
| sweep, 6 levels, S=50, N=200, 1M steps | 600 | 84.6 | 5.3 h | 2.8 h |
| sweep, the same at 2M steps | 600 | 168 | 10.6 h | 5.6 h |

The wall clock is the longest-first schedule, not core-hours over workers: a cell
cannot be split, so the pilot takes one PPO cell's time however many workers it
has, and 300 PPO cells on 32 workers take ten rounds, not 9.4. A worker is one process on one core with torch at one
thread, so `--workers` should be the number of physical cores. Cells run longest
first so the PPO cells are not the tail, and the ETA on each progress line rescales
the plan by the wall clock measured so far. One million steps is A4's budget, and
`pilot_budget.py`'s default.

**Outputs and resuming.** Under `--root`, every cell is an ordinary run directory
`<config hash>-s<first seed>/` holding `manifest.json`, `run.json` and, for PPO,
`policy.zip`, all hashed into the manifest; `grid-<hash>.json` lists the cells of
one grid. A killed run loses only the cells in flight: rerun the same command and
finished cells are read back, not recomputed. A failed cell is reported and the
others keep going; the exit status is 1 until every cell is done. Changing any
flag other than `--root`, `--workers` or `--dry-run` changes the cells' hashes, so
it starts new cells beside the old ones rather than overwriting them. The worker
count never changes a number: two workers write the same bytes as one. Run one
command per root at a time.

A worker killed outright, by the out-of-memory killer say, does not hang the run
(#30). The pool notices the dead process and the cells in flight with it are
reported as lost (a pool's other workers go down with it); every finished cell stays
on disk, and the cells not yet started carry on on a fresh pool. A pool that dies
before finishing any cell is not fed again: the rest are reported as not run, and a
rerun with fewer `--workers` picks them up. There is no per-cell timeout, because a
clock cannot tell a stuck cell from a slow PPO one on a loaded machine.

## Running on real data

The stitched SPX panel (`docs/DATA.md`) is a data source like the generator, and every
command that launches runs takes it: `--source optionmetrics` on the single-run
command, the pilot, the sweep and `scripts/oracle_sweep.py`, and `"source":
"optionmetrics"` in a dashboard job (`POST /api/jobs`, or the Launch page's Data toggle, which
offers only the sources `GET /api/jobs/options` finds present). The
panel is read from `$VEGALAB_DATA/panel` (the shared folder when `VEGALAB_DATA` is
unset) and never written; a missing panel is refused before anything runs.

```bash
make run ARGS="--policy delta_seller --episodes 200 --seed 0 --source optionmetrics"
python -m scripts.run --policy ppo --timesteps 1000000 --episodes 200 --source optionmetrics
python -m scripts.sweep --source optionmetrics --premiums 0 --arms delta_seller ppo \
    --seeds 8 --episodes 200 --timesteps 1000000 --workers 8
make replay RUN=$VEGALAB_DATA/runs/optionmetrics/single/<run id>
```

**Where runs land.** The source is not in the config hash (the freeze: no run id
moved), so a real-data run has the run id of its synthetic twin. Real-data runs
therefore go under their own folder of the runs root (`$VEGALAB_DATA/runs` when
`VEGALAB_DATA` is set, else `runs`), unless `--root` names another:

| command | generator | real panel |
|---|---|---|
| `scripts.run` | `<runs root>/single` | `<runs root>/optionmetrics/single` |
| `scripts.pilot` | `<runs root>/pilot` | `<runs root>/optionmetrics/pilot` |
| `scripts.sweep` | `<runs root>` | `<runs root>/optionmetrics` |
| dashboard job | `<runs root>/launched` | `<runs root>/optionmetrics/launched` |

Each run's manifest records the source, the sha256 of each panel file it read and the
split its episodes came from (`DataRecord`). Every reuse check (`rundir.valid_run`, so the
grid's resume and the single run's, and the job runner's) refuses a run on another source
or split, and a directory that already holds one is never overwritten: the cell fails and
says so. A learned cell is also reused only when its manifest names the trainer asked for,
so a stand-in's cell is never taken for the real learner's (#31; a manifest from before
the trainer was recorded counts as the real learner's when its `policy.zip` is a
Stable-Baselines3 archive).

**The split.** Training on 2010-01-01 to 2018-12-31 (`train`), evaluation on
2019-01-01 to 2022-08-24 (`eval`), and nothing from 2022-08-25 on is read: the dates
pre-registration amendment A8 fixed (`optspread.realdata.SPLITS`, #28). They are not a
flag, and the real source refuses other ranges on the team's panel; ranges of your own
are for a surface of your own (`docs/EXTENDING.md`). The scripted arms are evaluated on
`eval`. A PPO arm trains on windows from `train` (`train_ppo(..., source=...)`) and is
evaluated on windows from `eval`, so a policy is never scored on a day it saw in
training. An episode is a window of `n_days + 1` consecutive trading days whose start
the seed draws; no window crosses a day the panel lacks.

**Replay** reruns a real-data run on the source and split its manifest records,
retraining a PPO run on `train`, after hashing the panel again: a panel rebuilt since
the run (version 4 replaced version 3 on 2026-10-28, so a run on version 3 is one) is
a `DIFFERS` that says so, not a replay on other data. A real-data manifest from before
the split was recorded reads as `eval`.

**Limits.**

- The planted premium is not applied to real chains: the surface carries the premium
  the market priced. Cells that differ only in premium see the same chains, so on the
  real source `--premiums 0` is the level that means anything, and the pilot's
  contrast against zero is zero by construction. The generator's `sigma` and `r` are
  unused too; spot and carry come from the panel.
- Windows overlap. The default 63-day episode has about 850 possible starts in the
  evaluation range, drawn with replacement, so two hundred episodes share most of
  their days and their P&Ls are correlated: the reported standard error assumes
  independent episodes and understates the real one.
- Real-data runs are exploratory. The pre-registered analysis is the synthetic sweep
  (D2); `make m2` and `make m4` read the synthetic roots, and the confirmatory grid
  (`grid-7d9d53d340006f02.json`) is unchanged.

## The M4 analysis

`make m4 ARGS="--root $VEGALAB_DATA/runs"` reads the confirmatory sweep (A7:
`grid-7d9d53d340006f02.json`) and writes `docs/M4_ANALYSIS.md` and its JSON sidecar.
`optspread/m4analysis.py` defines every number and documents the sidecar's schema:

- **Decisions (#12).** Each arm at each premium against itself at zero (A1), by the
  pre-registered t (A2), with the variance components and the plug-in exact power.
- **Power curve (#12).** At each premium, the power of a fresh S x N experiment at the
  noncentrality estimated from the observed t (median-unbiased), with the exact 95%
  band; linear in noncentrality between sweep points, from alpha at zero.
- **Floor and gap (#12).** Where each curve first reaches 0.80, with the band's
  crossings and the sweep points either side; the gap is PPO's floor minus the
  fixed-delta seller's. A floor the grid never reaches is reported as above it.
- **tau^2 across premiums (#13).** Brown-Forsythe on seed means, per arm, beside its
  own simulated size and power at the sweep's groups and S (at 5 groups of 5 its size is
  about 0.003 and its power low, so a non-rejection is weak evidence).
- **Bootstrap (#14).** `validation.bootstrap_detection` at every level, graded against
  what normal seed means give at the same S, beside the bootstrap and t intervals.
- **The design's power, measured.** A7's planned power (0.898 at delta = 824.3) and floor
  (698.3 dollars) came from the pilot's variances; the report recomputes both for the
  same S = 5 x N = 500 design on the sweep's own variances, labelled apart from the plan.

**When the rules were written.** The decisions are A1 and A2's, pre-registered. The rest
(the curve's interpolation and anchor, the floor, the gap, Brown-Forsythe, the bootstrap
tolerance and the switch to the bootstrap rule) was written on 2026-10-28, after the
sweep ran (2026-10-26) and after its raw arm means were posted in issue #9, and before
the analysis code read the sweep's files. It is not pre-registered until A9 (#37)
records it, and A9 says it was written after the sweep. The report says so at the top,
and states its limits at S = 5: the bootstrap check barely sees non-normal seed means
and the rule it switches to has size about 0.10; a floor taken at a noisy curve's first
crossing reads about a tenth low; a floor below the first sweep point (the seller's)
rests on interpolation from the anchor at zero.

The command refuses a grid that is not the confirmatory design or has a missing or
invalid cell. `--exploratory` analyses it anyway, labels every output, and needs an
explicit `--out`, so a report built on a test grid never lands in `docs/`.

`make m4gate ARGS="--runs-root $VEGALAB_DATA/runs"` writes `docs/M4_GATE_REPORT.md` and
its sidecar (#35). `optspread/m4gate.py` holds one item per deliverable of the M4 row:
the power curve, bootstrap and tau^2 items read `docs/M4_ANALYSIS.json` and fail unless it
is the confirmatory analysis of the A7 grid with every cell (tau^2 passes on the test
having been run, whatever it finds); the VRP item checks `docs/VRP_MAPPING.json` against
the panel `docs/DATA_MANIFEST.json` records and reads each arm's curve at the mapped
premium and over its interval through `vrp.power_at_noncentrality`, from the curve's knots
by the analysis's own interpolation, so a curve that dips by sampling noise (PPO's does)
is read as it is; the dashboard item builds the server's
app, matches an attribution route and an M4 or power-curve route by name and looks for
both in the built site; the collaboration report is `docs/COLLABORATION.md`; and the
attribution residual of at least one sweep cell per arm must agree with its run (P&L and
cost within 1e-9 relative, the same trades; bit-for-bit reproduction is reported beside
it, since a cell attributed on another platform agrees without reproducing) and stay
under 5% of gross. On the flat synthetic surface the vega term is zero, so that residual
checks delta, gamma and theta, not vega, and the item says so; vega is tested on a
moving surface in `tests/resim/test_attribution.py`. Until #34 lands the analysis, the report reads NOT CLEARED and says why.

## Attribution, the edge and the action mix

Three read-side tools re-simulate finished runs (the same env, policy and seeds; a PPO
cell with its saved `policy.zip`) and never write into a run:

```bash
python -m scripts.attribution $VEGALAB_DATA/runs/<run id>         # one run's P&L by greek (#16)
make actionmix ARGS="--grid $VEGALAB_DATA/runs --arms delta_seller --workers 8"
make edge ARGS="--grid $VEGALAB_DATA/runs --first 50 --workers 2"
```

- **Attribution** (`optspread.attribution`, project document 10.9) splits each step's
  equity change into delta, gamma, vega, theta, carry, cost and a residual, and says
  whether the re-simulation reproduces the run bit for bit or agrees within 1e-9. It
  writes under `$OPTSPREAD_ATTRIBUTION` (default `attribution/`, gitignored). Carry is
  always 0, because the book never books it.
- **The action mix** (`optspread.actionmix`) is which structures a policy holds and
  opens, day by day: held share, opens per episode, refused opens, holding days and
  exits, per run or across a grid by arm and premium, under `$OPTSPREAD_ACTIONMIX`
  (default `actionmix/`). A config run on real data reads the `eval` split.
- **The edge** (`optspread.edge`) attributes the detection event rather than a run:
  each component, and each structure held, of each arm at premium p against itself at
  0 on the same paths (A1), tested on the S block means (A2), so the components'
  edges sum to the P&L edge. It writes `<attribution root>/edge/edge-<hash>.json`, which
  the dashboard's Attribution page reads. The split is descriptive; only the P&L edge
  is the pre-registered test.

## The dashboard

The dashboard only shows what the harness recorded, so every number traces back to a
run. It is a static site (`web/`, Next.js exported to `web/out`) served by
`optspread.server` beside a JSON API, and every card names its source behind an info
icon: a run key and config hash, a gate report file, a panel file, or the module that
computed it.

```bash
pip install -e ".[dev,dashboard]"
make web                     # npm ci, the API types from the server, build web/out, typecheck
make serve                   # http://127.0.0.1:8600; OPTSPREAD_RUNS (default $VEGALAB_DATA/runs, else runs/) and VEGALAB_DATA pick the roots
make web-dev                 # hot reload on :3000 against a running `make serve`
```

`make web` needs Node and npm and also the project venv with the dashboard extra: after
`npm ci` it writes the server's OpenAPI document (`python -m scripts.write_openapi`, which
imports the app; without FastAPI it stops with `ModuleNotFoundError`) and regenerates the
site's API types from it into `web/lib/schema.d.ts` (`npm run api-types`), then builds and
typechecks, so the build fails when a page no longer fits the API. That file is committed
and the target rewrites it: commit it again whenever it changes. With Node alone, build
from the committed file: `cd web && npm ci && npm run build && npm run typecheck`.

| View | Reads | What it shows |
|---|---|---|
| Home | everything | "What do you want to find out?": one card per trader's question beside the live SPX vol surface (the last 40 gridded days played once, then a date scrubber; the power curve with the real premium where there is no panel or no WebGL) and one sentence on the real premium against PPO's floor; a status strip (data, the real premium, the sweep, runs indexed); below the questions, "Can you trust these numbers?": the checks (every gate item's verdict and evidence), replay, the leak detector and the pre-registration (the study design from `docs/PRE_REGISTRATION.md`), each a line with its mark, the reviewer's workflow and the three-line reproduce command |
| Runs | `runs/**/manifest.json`, `run.json` | filter by config hash, seed, arm, premium, commit, sweep or grid, data source; P&L distribution, equity paths when recorded, Sortino, CVaR 5%; drawdown and return on initial cash and peak margin (a re-simulation of the episodes, a real-data run's on the panel and split it recorded, refused once the panel is rebuilt); attribution per episode and per step with the residual and its share (`optspread.attribution`, cached under `$OPTSPREAD_ATTRIBUTION`, never in a run); the action mix (held share, opens per episode, refused opens, holding days, by day of episode; `make actionmix`, read from `$OPTSPREAD_ACTIONMIX`); Replay (IDENTICAL or DIFFERS, with the reason); digests recomputed on read; a paired compare of two runs |
| Launch | `optspread.config` defaults | every knob, presets (PPO when the learner is installed), the data source (synthetic, or OptionMetrics when the panel is present: evaluated on A8's 2019-01-01 to 2022-08-24, PPO trained to 2018-12-31), the wall-time estimate with training at the A7 rates, live progress; the run lands in `<runs root>/launched/` or `<runs root>/optionmetrics/launched/` through `write_run` |
| Power lab | `optspread.power` | exact t power against the planted premium in vol points (A6: $824.3 per vol point; dollars on a toggle), from the A7 presets; the floor against seed count with S = 5 and S = 50 marked, the frontier and its compute; a 3D surface of exact power over premium x seeds |
| Real data | `$VEGALAB_DATA/panel/`, `docs/VRP_MAPPING.json` | four questions, each a view answered in one sentence computed from the panel: the premium the market pays for volatility (the VRP mapping day by day with its quarter's mean, today's priced premium, sub-periods and years on the sweep's axis); the surface on a day (3D from `surface_grid.parquet`, played through the days, its smile and term structure, its level, skew and term slope against every day, the ATM term structure through time); the simulator against the market (where real and simulated differ most); what the fixed-delta seller opens at that close (an illustration, with its payoff and the loss its wings cap, computed from the legs; on a day the frozen margin check margins the condor as naked, #39, the page says so). Data quality folds away below: coverage and missing days, the vendor seam (each overlap day's 95th-percentile IV gap, averaged over the days, the figure `docs/DATA.md` quotes), FRED, the build's version against `docs/DATA_MANIFEST.json`, the quoted smile against the vendor's IV |
| Leak demo | `optspread.leaks` | the detector live on the clean env and its poisoned twin, its floor (caught or missed at each leak size), and a leak planted in any one field or in every field in turn |
| Sweep | `grid-<hash>.json` (or a `sweep.json`) and their cells; `docs/M4_ANALYSIS.json` | per-arm means by premium (± se of block means), every cell, each arm against itself at zero block by block; power curves with bands, floors, the gap, the real premium read off each curve, the decision per premium, tau^2 homogeneity and the bootstrap check, once `make m4` has written the analysis |
| Attribution | `<attribution root>/edge/edge-<hash>.json` (`make edge`) | per arm and premium, each component's edge (delta, gamma, vega, theta, carry, cost, residual) with its se and one-sided 95% bound, diverging and stacked across premiums with the P&L edge marked; the same split by structure held; residual share per cell; reproduces, agrees or differs per cell, each cell linked to its run's own attribution; the grid's action mix by arm and premium (`actionmix/grid-<hash>.json`) |

Every page keeps its settings in the URL, so any view links as it stands (the link
button copies it). A workflow is a question a trader (or a reviewer of our claims) would
ask, walked through the real pages: real edge or luck, tuning the seller, what eats the
edge, the real market's premium, where the P&L comes from, how big a backtest must be,
whether a backtest could be cheating, whether a learner beats a rule, and verifying our
claims (started from the trust band, not a card). Each step says what to do, rings the control to use (three pulses, no overlay),
keeps Next locked until the action is done, and ends with one sentence on what the
result means, with its numbers. A step knows its action was done from the address bar
and from the server's own answers (`web/lib/activity.ts`), never from a page's state;
steps on the showcase's paired runs (`runs/showcase/catalog.json`) say so when a run is
not under the served root.
They read the showcase's questions from `/api/runs/catalog`, name a pair by its group and
the settings that pick each run (`web/lib/catalog.ts`, never by run key) and open the
paired compare; every question of the showcase is walked. A step rings its control by
the hook the page carries (`guide("<page>-<control>")`, `web/lib/guide.ts`). The strip is
one line under the header (two on a phone).
Every figure exports as CSV, JSON or PNG; the hash, commit and source behind it sit
behind the icon beside the export buttons, and every arm, metric and knob explains
itself on hover, focus or tap (`web/lib/glossary.ts`, which merges each page's own
terms; one tooltip, `InfoTip`, everywhere). A filter or selector with one value is not
drawn.

A run directory that does not read cleanly is listed as INVALID or DIGEST MISMATCH
with the reason, never dropped. The panel is read one year's partition and one date
at a time; where `VEGALAB_DATA` has no panel, the real-data views say which files are
missing. PPO appears in Launch only when `optspread.train` and its learner are
installed.
The 3D views (the surface, the term structure through time, the power surface)
need WebGL and load echarts-gl only when one is drawn; drag rotates, the wheel
zooms, and with the chart focused the arrow keys walk the grid and read each point
out as text.
A chart draws once per real change of what it shows: the site binds ECharts itself,
sets an option only when it differs from the last one drawn, never animates or morphs
between two, and counts its draws in the chart's `data-draws` attribute.

Answers are kept while the files they read stand (`optspread/api/cache.py`): each route's
body is keyed by its query and a stamp (path, modification time, size) of the runs root,
the docs or the panel, kept in memory, and for the panel's routes on disk too (under
`<attribution root>/webcache`, or `--cache` / `OPTSPREAD_CACHE`), since a cloud-synced
panel's first read of a file can take minutes. Every kept answer carries an ETag, so a
browser holding it is told 304; the site's hashed assets are immutable. In the browser a
view revisited shows its last answer at once while it is asked again.

Launched jobs share a small process pool. A worker that dies fails the job it was
running and the pool is rebuilt; jobs queued behind it run on the new one, and
later launches are unaffected (#30). At most `OPTSPREAD_MAX_JOBS` jobs (default 8)
are queued or running at once; past that a launch is refused with a 429 rather than
queued. A PPO job saves the policy it trained beside `run.json`, hashed into the manifest
as `schedule.execute_run` saves it, so its capital, attribution and action mix
re-simulate it. `POST /api/jobs/<id>/cancel` cancels a job: a queued one never runs, and a
running one stops between episodes or, if it finishes anyway, reports no run.

A request's window is bounded: at most 252 gridded frames and 4,200 landscape dates
(422 past either). An error names the root a file is missing from (`<runs root>`,
`<data root>`), never a path on the serving machine, and the pilot's matrix manifest
is never served as a run.

## What comes next

Where the project stands against the project document's schedule (section 12):

- **M1 (Oct 1) and M2 (Oct 29): done.** Both gate reports are committed and read PASS.
  The pre-registration is tagged (`prereg-m2`) and has grown by amendment only (A1 to
  A8). The cutoff is tagged (`cutoff-2026-11-01`): the cost model, margin, the action
  library (FLAT, the iron condor, the short strangle and the put credit spread),
  pricing, the observation schema and the PPO hyperparameters are frozen. Every change
  since to a module on the simulation path is listed in `docs/DEFECTS.md` with the
  proof that no run's numbers moved.
- **The sweep: run.** The A7 grid (`grid-7d9d53d340006f02.json`, 60 cells: six premiums
  x two arms x five seed blocks, N = 500, 1,000,000 training steps) ran on Dominick's
  machine after the cutoff tag (#9). `docs/SWEEP_EXPECTED.json` holds its digests.
- **M3 (Nov 13): the code is done; the evidence waits on others.** `make m3gate`, the
  partner command (`scripts/reproduce.py`) and its tolerances are built, and the leak
  floor and the demo items run. Two things remain: the partner team's
  `docs/M3_PARTNER_REPRODUCTION.json` (#10), and the sweep and pilot copied into the
  shared folder (#34), so the sweep item reads them there.
- **M4 (Nov 19): the code is done; the report waits on Dominick.** The analysis
  (`make m4`), the VRP mapping (committed, on the panel of record), attribution and the
  edge, the action mix, the M4 gate (`make m4gate`) and the dashboard's Sweep and
  Attribution pages are built. Remaining: `docs/M4_ANALYSIS.*` from the sweep in the
  shared folder (#34, #12); pre-registration A9 (#37), which records the M4 rules and
  says they were written after the sweep; the statistical choices of #36; the
  collaboration report (#17).
- **M5 (Nov 24).** The adapter conformance suite and the OptionMetrics source are done
  (real SPX data is in: `docs/DATA.md`, one panel of record). Usability testing with
  people outside the team, and the changes it leads to, are #18.
- **Final (Dec 2).** The report, the consolidated gate reports, the demo and the defect
  appendix (`docs/DEFECTS.md`, kept current from now on) are #19.

A strategy run on real chains reads one fixed date range: training to 2018-12-31,
evaluation 2019-01-02 to 2022-08-24, and nothing after that (pre-registration A8, #28).
