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
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
make check                   # ruff + mypy --strict + pytest with a coverage floor
make gate                    # the Wave 0 economic sanity table
make bench                   # environment throughput, measured not assumed
```

## What week one built

| Module | Lane | What it is |
|---|---|---|
| `rng.py` | Dominick | Per-step RNG stream contract. `stream(t)` depends only on `(seed, channel, t)`. |
| `types.py` | Dominick | `OptionLeg`, `Chain`, `Position`, and the `ChainProvider` protocol. |
| `leaks.py` | Dominick | Perturbation property, structural invariants, poisoned env, leak floor. |
| `pricing.py` | Hemdutt | Black-Scholes prices and analytic greeks, NaN-free at the limits. |
| `market.py` | Hemdutt | Zero-drift GBM with the chain priced off the path vol, plus a plantable premium. |
| `costs.py` | Hemdutt | Quoted half-spread plus commission. |
| `margin.py` | Hemdutt | Max-loss for defined risk, Reg-T style for undefined risk. |
| `actions.py` | Hemdutt | The locked action library: FLAT, iron condor, short strangle. |
| `portfolio.py` | Hemdutt | Cash and position book; cash settlement at expiry. |
| `policies.py` | Hemdutt | Flat, always-on, random, and the fixed-delta seller (the oracle). |
| `gates.py` | Hemdutt | The Wave 0 economic sanity gate. |
| `config.py` | Ilan | Frozen pydantic configs and the content hash every run is keyed by. |
| `manifest.py` | Ilan | Run manifests: git SHA, config hash, library versions, artifact digests. |
| `env.py` | Ilan | Gymnasium env and the append-only observation schema. |
| `harness.py` | Ilan | Episode runner and evaluator; the one path from a config to numbers. |

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
always_on     True          -1,049     1,170     3.0
random        False           -333       370    70.7
random        True          -9,843       375    70.7
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
hold_condor      IRON_CONDOR             -515     -999      484        484        0
hold_put_spread  PUT_CREDIT_SPREAD     -2,058   -2,300      242        242        0
roll_condor      IRON_CONDOR            1,716    1,113      603        603        0
roll_put_spread  PUT_CREDIT_SPREAD       -817   -1,187      370        370        0
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

**Leak detection floor.** The same calibration logic, applied to the detector
itself. Planting leaks of decreasing size and recording the smallest still caught
gives the detector a stated floor. At 20 seeds and 63 days it currently sits at
`eps = 0.10`: a leak carrying a tenth of tomorrow's return is caught, half that
is not. This costs no training compute, which is why it runs in CI.

Passing this suite shows we catch the modes it is built to catch. It does not
prove the absence of unknown leakage anywhere in the codebase.

## Throughput

Measured, not assumed (project document section 11). `make bench` times four
workloads rather than one, because the first version of this benchmark was
misleading: timing the env with a FLAT action gave 37,000 steps/s, which looked
like the environment was free, but a flat book never prices a leg.

```
flat            37,071/s   empty book; the floor, not a planning number
always_on        1,772/s   one structure held and re-marked: the steady state
random           2,175/s   churns, but FLAT a quarter of the time, so it marks less
ppo          not measured  stable-baselines3 not installed
```

The cost is **per-step position marking, not trading**. `random` comes out faster
than `always_on` despite 24 times the trades, because one action in four is FLAT
and those steps mark nothing. Re-pricing greeks for every held leg dominates; the
policy's own decision is nanoseconds.

Plan the sweep off **about 1,800 steps per second per core**, the `always_on`
figure, which is what a trained policy will mostly do. That sits inside the
document's 500 to 3,000 estimate. Re-measure on any machine that will run the
sweep. The `ppo` arm is wired at the fixed hyperparameters of decision D3 and
reports a figure once Stable-Baselines3 is installed.

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

## What comes next

Open decisions 7.1 (decision rule), 7.2 (effect-size axis) and 7.4 (PPO
implementation) are due at the advisor meeting before M2 and gate the
pre-registration. Undefined-risk margin hardening, the variance components pilot,
sweep orchestration and the dashboard follow the M1 to M5 schedule in the project
document. The action library closes at the November 1 cutoff with whatever is
verified.
