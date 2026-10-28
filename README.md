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
make m1                      # rebuild docs/M1_GATE_REPORT.md from live runs
make m2                      # docs/M2_GATE_REPORT.md: what M2 has, and what failed first
make train-smoke             # a two-seed PPO pilot on two workers, then every run replayed
make pilot ARGS="..."        # the variance-components pilot (see below)
make sweep ARGS="..."        # the full premium x arm x seed sweep (see below)
make replay RUN=<run dir>    # rerun from the manifest; fails unless bit-identical
make web && make serve       # the dashboard on http://127.0.0.1:8600 (dashboard extra + npm)
```

## The lane map

| Module | Lane | What it is |
|---|---|---|
| `rng.py` | Dominick | Per-step RNG stream contract. `stream(t)` depends only on `(seed, channel, t)`. |
| `types.py` | Dominick | `OptionLeg`, `Chain`, `Position`, and the `ChainProvider` protocol. |
| `leaks.py` | Dominick | Perturbation property, structural invariants, poisoned env, leak floor. |
| `pricing.py` | Hemdutt | Black-Scholes prices and analytic greeks, NaN-free at the limits. |
| `market.py` | Hemdutt | Zero-drift GBM with the chain priced off the path vol, plus a plantable premium. |
| `costs.py` | Hemdutt | Quoted half-spread plus commission. |
| `margin.py` | Hemdutt | Max-loss for defined risk; Cboe Rule 10.3(c)(5)(A), the broad-based index rule, for undefined risk. |
| `actions.py` | Hemdutt | The locked action library: FLAT, iron condor, short strangle. |
| `portfolio.py` | Hemdutt | Cash and position book; margin checked before an open; cash settlement at expiry. |
| `policies.py` | Hemdutt | Flat, always-on, random, and the fixed-delta seller (the oracle). |
| `gates.py` | Hemdutt | The Wave 0 economic sanity gate. |
| `config.py` | Ilan | Frozen pydantic configs and the content hash every run is keyed by. |
| `manifest.py` | Ilan | Run manifests: git SHA, config hash, library versions, artifact digests. |
| `env.py` | Ilan | Gymnasium env and the append-only observation schema. |
| `harness.py` | Ilan | Episode runner and evaluator; the one path from a config to numbers. |
| `artifacts.py` | Ilan | The run output schema: versioned `RunRecord`, hashed into the manifest. |
| `sweep.py` | Ilan | Scripted premium sweep on paired seeds; the resume rule every orchestrator shares. |
| `train.py` | Ilan | The PPO arm at the frozen D3 hyperparameters; training paths kept off evaluation paths. |
| `orchestrate.py` | Ilan | The pilot and sweep grid: premium x arm x seed block, parallel and resumable. |
| `replay.py` | Ilan | The reproducibility gate: rerun from the manifest, compare `run.json` bytes. |
| `scripted.py` | Hemdutt | Scripted sellers and the analytic cost-drag check. |
| `data.py` | Hemdutt | The adapter boundary and its conformance suite; the OptionMetrics stub. |
| `gatereport.py` | Dominick | Milestone reports: criterion, evidence, what failed first; a JSON sidecar. |
| `power.py` | Dominick | Variance components, power (planning and exact), sample size, budget. |
| `analysis.py` | Dominick | The confirmatory decision rule: one-sided t on paired seed-mean contrasts. |
| `validation.py` | Dominick | The section 8 gate: the pipeline against a closed-form answer. |
| `m2gate.py` | Dominick | One evidence function per M2 deliverable. |
| `dashboard_data.py` | Ilan | What the dashboard reads: run index, gate board, risk figures, live leak demo. |
| `dashboard_panel.py` | Ilan | Read-side views of the SPX panel: coverage, smile, surface, term structure, VRP, seam. |
| `jobs.py` | Ilan | Runs launched from the dashboard: knobs to `RunConfig`, a bounded process pool. |
| `webapi.py` | Ilan | The API's typed payloads, each saying where its numbers came from. |
| `server.py` | Ilan | FastAPI over `webapi`, serving the static site in `web/out`. |
| `web/` | Ilan | The dashboard: Next.js static export, ECharts, two themes from the deck. |

Documents: `docs/PRE_REGISTRATION.md` records the decisions closed at the
September 14 advisor meeting with the alternatives we rejected.
`docs/M1_GATE_REPORT.md` is generated by `make m1` and is what the Presentation 1
slides are built from; `docs/M2_GATE_REPORT.md` comes from `make m2`. Each has a
`.json` sidecar from the same build, which is what the dashboard's gate board reads.

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
seller           structure               free   costed     drag  predicted      gap
hold_condor      IRON_CONDOR             -515     -772      257        257        0
hold_put_spread  PUT_CREDIT_SPREAD     -2,058   -2,186      128        128        0
roll_condor      IRON_CONDOR            1,716    1,366      350        350        0
roll_put_spread  PUT_CREDIT_SPREAD       -817   -1,014      197        197        0
```

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

`optspread.leaks` is the detector, and `tests/test_leaks.py` is the gate.

**Perturbation property.** Generate a path, resample every RNG stream after step
`t`, and assert the chain at `t` is unchanged. This is a direct machine check
that no future information reaches the present. It is only constructible because
randomness is factored per step: one generator drawn sequentially cannot express
"same past, different future".

**Structural invariants.** Consecutive timestamps, ordered strikes and expiries,
positive vols, no-arbitrage price bounds, put-call parity within tolerance.

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

`make m1` writes `docs/M1_GATE_REPORT.md`. It currently reads **PASS** on all
eight items. It read NOT CLEARED until the PPO arm was measured, which is the
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
replay, the dashboard) are being written in parallel, so `optspread.m2gate` imports
them by name and reads NOT CLEARED with the failed import when they are absent. The
pre-registration item looks for the earliest `prereg-*` tag (`prereg-m2` at the M2
meeting) and requires the file at HEAD to begin with the tagged text byte for byte:
amendments are appended, never edited in place.

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
learner, a control paying a drag of five standard errors, nothing planted. The
pre-registered event fires at 0.050; the event it replaced fires at 0.995 and is
the gate's negative control. At zero itself the event is identically zero (an arm
against itself on the same paths), so the sweep's leftmost point is degenerate
and the size of the test is this check's to show, not the sweep's.

The last column is the finding. The planning functions (`episodes_for`,
`seeds_for`, `frontier`) use the normal approximation, which at ten seeds reads
0.72 where the test's real power is 0.66. A sample size from them is optimistic at
small S; read the exact figure from `power_t` before committing a design.

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
`library_versions` records torch and Stable-Baselines3 in every manifest, so the
presence of those two fields is itself the evidence that a learner was involved.

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

`make replay RUN=<run dir>` is the gate of project document 10.7. It reads the
config from the manifest, retrains if the arm is learned, re-evaluates, and
compares the rendered `run.json` with the file on disk byte for byte, not to a
tolerance. A results file edited after its manifest was written fails before the
replay starts, and a mismatch names the first episode that differs. A PPO run
replays because torch runs one thread per process on CPU, the learner is seeded
from the config, and evaluation takes the greedy action. Bit-identity is promised
on the machine and library versions that produced the run; the replay reports any
version or commit that has moved since, because those are the first suspects.

## The PPO arm

Decision D3 is enforced, not just recorded. `PPOHyperparameters` writes out the
Stable-Baselines3 defaults (21 values across `PPO` and `ActorCriticPolicy`), every
one is passed to the library explicitly so an upgrade cannot move them, and
`TrainConfig` refuses any other set. A PPO run's `RunConfig` carries its
`TrainConfig`, so the hyperparameters, `total_timesteps`, the training seed and the
device are all in its config hash. `total_timesteps` is the one number D3 did not
fix, so it has no default anywhere: every command that trains asks for it. A
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
root. Point `--root` at the team's shared data folder.

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
$PY scripts/bench.py --steps 20000 --ppo   # re-measure the rates on this machine
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
| pilot, S=8, N=200, 1M steps | 16 | 2.3 | 17 min | 17 min |
| sweep, 6 levels, S=50, N=200, 1M steps | 600 | 84.6 | 5.3 h | 2.8 h |
| sweep, the same at 2M steps | 600 | 168 | 10.6 h | 5.6 h |

The wall clock is the longest-first schedule, not core-hours over workers: a cell
cannot be split, so the pilot takes one PPO cell's time however many workers it
has, and 300 PPO cells on 32 workers take ten rounds, not 9.4. A worker is one process on one core with torch at one
thread, so `--workers` should be the number of physical cores. Cells run longest
first so the PPO cells are not the tail, and the ETA on each progress line rescales
the plan by the wall clock measured so far. One million steps is the placeholder
`pilot_budget.py` uses, not a pre-registered number.

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

## The dashboard

The dashboard only shows what the harness recorded, so every number traces back to a
run. It is a static site (`web/`, Next.js exported to `web/out`) served by
`optspread.server` beside a JSON API, and every card names its source: a run key and
config hash, a gate report file, a panel file, or the module that computed it.

```bash
pip install -e ".[dev,dashboard]"
make web                     # npm ci, build web/out, typecheck
make serve                   # http://127.0.0.1:8600; OPTSPREAD_RUNS and VEGALAB_DATA pick the roots
make web-dev                 # hot reload on :3000 against a running `make serve`
```

| View | Reads | What it shows |
|---|---|---|
| Overview | everything | the project summary, gate tiles, data coverage, latest runs |
| Runs | `runs/**/manifest.json`, `run.json` | filter by config hash, seed, arm, premium, commit; P&L distribution, equity paths when recorded, Sortino, CVaR 5%, max drawdown, turnover; digests recomputed on read; a paired compare of two runs |
| Launch | `optspread.config` defaults | every knob, presets, live progress; the run lands in `runs/launched/` through `write_run` |
| Gates | `docs/M*_GATE_REPORT.json`, else the `.md` table | criterion, evidence, verdict, what failed first |
| Power lab | `optspread.power` | power against effect size, the floor against seed count with S = 5 and S = 50 marked, the frontier and its compute |
| Real data | `$VEGALAB_DATA/panel/` | coverage with missing days and the Jul-Aug 2025 overlap, smile and surface for any date, ATM term structure, IV30 against RV21 and VRP30, the seam |
| Leak demo | `optspread.leaks` | the detector run live on the clean env and its poisoned twin |
| Sweep | `sweep.json` and its cells | the oracle curve; power curve and attribution wait for the sweep |

A run directory that does not read cleanly is listed as INVALID or DIGEST MISMATCH
with the reason, never dropped. The panel is read one year's partition and one date
at a time; until the W6-3 build writes it, the real-data views say which files are
missing. PPO appears in Launch only when `optspread.train` and its learner are
installed.

## What comes next

The three decisions that gated the pre-registration closed at the September 14
advisor meeting and are recorded in `docs/PRE_REGISTRATION.md` with their rejected
alternatives: the decision rule, the effect-size axis (planted volatility points,
not Sharpe), and Stable-Baselines3 PPO at fixed hyperparameters.

Next: review the gate report with the advisor on September 28 and present M1; then
the variance components pilot, which sets the sweep budget for M2 from
`Var(grand mean) = tau^2/S + sigma^2/(S*N)` at 80% power, now costed against a
measured 1,000 steps/s/core with training included. Undefined-risk margin
hardening, sweep orchestration and the dashboard follow the M1 to M5 schedule. The
action library closes at the November 1 cutoff with whatever is verified; it holds
FLAT, the iron condor, the short strangle and the put credit spread today.

Real SPX data is in: the OptionMetrics panel (3,939 trading days, 2010-01-04 to
2025-08-29) clears the adapter's conformance suite, and `docs/DATA.md` says how to
build it. A strategy run on real chains reads one fixed date range: training to
2018-12-31, evaluation 2019-01-01 to 2022-08-24, and nothing after that
(pre-registration A8, #28).
