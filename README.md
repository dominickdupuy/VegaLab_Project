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
always_on     False           -509     1,157     2.0
always_on     True            -846     1,158     2.0
random        False            144       267    54.9
random        True          -8,452       301    54.9
```

Three orderings are asserted: flat sits at exactly zero either way, always-on
reads zero without costs and turns negative with them, and the churning random
policy bleeds fastest. The numbers move with the seed; the inequalities do not.

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

## Reproducibility

A `RunConfig` plus a seed fully determines a run's numerical output. `config_hash`
is the content address; `RunManifest` records the git SHA, the config, library
versions, and a SHA-256 for every artifact. Every figure in the final report will
carry the hash that produced it.

## Throughput

Measured, not assumed (project document section 11): about **1,800 environment
steps per second per core** under a random policy, inside the 500 to 3,000
estimate. Re-measure with `make bench` on any machine that will run the sweep.
The PPO-inclusive figure follows once decision 7.4 is settled.

## What comes next

Open decisions 7.1 (decision rule), 7.2 (effect-size axis) and 7.4 (PPO
implementation) are due at the advisor meeting before M2 and gate the
pre-registration. Undefined-risk margin hardening, the variance components pilot,
sweep orchestration and the dashboard follow the M1 to M5 schedule in the project
document. The action library closes at the November 1 cutoff with whatever is
verified.
