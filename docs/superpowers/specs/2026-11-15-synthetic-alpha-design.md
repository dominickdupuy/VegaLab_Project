# Synthetic alpha benchmark: design

Date: 2026-11-15. Status: approved in conversation; branch `synthetic-alpha` off `main`.

**As built (port onto `main` after GH-17/18/46, branch `synthetic-alpha-port`).** The design
below is unchanged; the names moved to sit beside what `main` gained meanwhile:

- package `optspread/alpha` (not `optspread/synth`: `main` has a Synth data page, a market
  visualiser, at `/synth`); routes `/api/alpha/*`; page `/alpha/`; data under
  `<runs root>/alpha/`; a dataset's manifest is `dataset.json`, never `manifest.json`,
  which marks a run directory for the run browser.
- every learned arm, not only PPO: `train(config, provider=...)` trains PPO, A2C or DQN on
  a `DatasetChainProvider`; `train_ppo`, `train_a2c` and `train_dqn` gained the same
  keyword and nothing else.
- `GET /api/alpha/options` replaces `/generators` (it also carries the dataset knobs and
  the arms); `GET /api/alpha/datasets/{hash}/benchmarks` returns saved results, so the
  page reloads to them from `?dataset=<hash>`.
- the oracle qualifies only at z > 2 against flat on the fresh sample (see Benchmark).

## Purpose

Generate synthetic SPX worlds with a **known, planted source of edge**, freeze them as a
dataset, train the learned arm (PPO) on exactly that dataset, and compare what it earned
against (a) what was attainable from the planted signal and (b) what it should have
earned from pure noise. The comparison is the point: it separates edge captured from
the signal, edge lost, and P&L that only exists in-sample (overfitting).

The user walks through it on one dashboard page: **Generate** the data and see the
planted signal, **Train** an arm on it, **Compare** the scoreboard.

Out of scope for v1: A2C/DQN (they live on the unmerged `worktree-research-features`
branch), new observation fields (`OBS_NAMES` is append-only and needs two-member review),
stochastic-vol and regime-switching generators.

## Generators

Each generator is a pure function of `(params, MarketConfig, StepRNG)` returning a
`SynthPath`: `spot[0..n_days]` and `atm_iv[0..n_days]` (day 0 is the open of the
episode; `spot[0] = spot0`). Every value at day `t` is computed from
`rng.stream(s, PATH_CHANNEL)` for `s <= t` only, so the no-look-ahead contract holds
and is tested with `SplicedStepRNG`.

Per day `t >= 1` the stream yields, in order, `z` (the spot shock) and, where the
generator needs it, `u` (an auxiliary shock). Drawing `z` first keeps the spot shock
identical to `SyntheticChainProvider.log_return`, so at zero strength a generator
reproduces the existing null world bit for bit. `dt = 1 / trading_days_per_year`.

| kind | params (defaults) | spot | atm_iv | the edge, and who can take it |
|---|---|---|---|---|
| `noise` | none | GBM: `log r = -sigma^2 dt/2 + sigma sqrt(dt) z` | `sigma` | none; the control |
| `const_vrp` | `premium_pts` (3.0) | GBM | `sigma + premium_pts/100` | sell always (unconditional) |
| `mr_vrp` | `mean_pts` (0.0), `vol_pts` (4.0), `half_life_days` (10) | GBM | `max(sigma + p_t/100, 0.01)`, `p_t` an OU in vol points | sell when IV is rich, flat when cheap (timing; visible through `atm_iv`) |
| `spot_mr` | `half_life_days` (10) | `log S_t = log S_0 + x_t`, `x_t = phi x_{t-1} + sigma sqrt(dt) z` | `sigma` | multi-day realised vol below implied: hold short vol (unconditional) |
| `drift` | `bps_per_day` (5.0) | GBM plus `bps_per_day/1e4` per day | `sigma` | put credit spread (unconditional) |

`phi = exp(-ln 2 / half_life_days)`. The `mr_vrp` OU: `p_0 = mean_pts + vol_pts * u_0`
(stationary start, `u_0` from stream 0 after its `z`), then
`p_t = mean_pts + phi (p_{t-1} - mean_pts) + vol_pts sqrt(1 - phi^2) u_t`.

The **signal series** shown on the page is the planted quantity per day: the premium in
vol points (`const_vrp`, `mr_vrp`), the OU deviation `x_t` (`spot_mr`), the cumulative
drift (`drift`), zero (`noise`).

**The null twin** of any dataset is the `noise` generator on the same seeds: same spot
shocks, nothing planted.

## Dataset

`SynthSpec`: generator params (pydantic discriminated union on `kind`), the market
(`sigma`, `n_days`; rest of `MarketConfig` at defaults), `n_train` (64), `n_heldout`
(256), `seed` (0). Path seeds: train `seed .. seed+n_train-1`, held-out the next
`n_heldout`; all below `2**40`, far below the training-path range of `optspread.train`.

`build_dataset(spec, root) -> SynthDataset` writes `root/synth/datasets/<hash>/`:

- `paths.npz`: `{signal,twin}_{train,heldout}_{spot,iv,sig}`, each `(n_paths, n_days+1)` float64.
- `dataset.json`: spec, hash, path seeds, sha256 of `paths.npz`, git SHA, created time. Not
  `manifest.json`, which marks a run directory for the run browser.

The hash is `config_hash`-style over the canonical spec, so the same spec is the same
directory and is not rebuilt. `load_dataset(root, hash)` verifies the npz digest against
the manifest and refuses a mismatch.

`DatasetChainProvider(dataset, split, market)` implements `ChainProvider`. `reset(rng)`
selects path `rng.seed mod n_paths` (`rng` must be a `SeededStepRNG`; anything else is
refused), and builds each chain exactly as `SyntheticChainProvider._chain` does but with
the stored spot and a flat surface at the stored `atm_iv`. Evaluation walks seeds
`0..n-1`, each path once. PPO training uses `TrainingPaths`, whose consecutive seeds cycle
the `n_train` frozen paths in order: the agent sees only those paths, repeatedly.

## Benchmark

`run_benchmark(dataset_hash, arm, total_timesteps, root, trainer)`:

1. **Learned arm** (`ppo`, or a scripted arm for a quick run): trained on `signal/train`
   (unchanged D3 hyperparameters; `train_ppo` gains an optional `env` argument and
   nothing else). Evaluated on `signal/train`, `signal/heldout`, `twin/heldout`.
2. **Noise control**: the same arm trained on `twin/train`, evaluated on `twin/train`
   (in sample) and `twin/heldout` (out of sample). Skipped for scripted arms (identical
   to the twin evaluation of the arm itself).
3. **Scripted menu**: flat; hold and roll of iron condor, short strangle and put credit
   spread; for `mr_vrp` also a timing seller (short strangle, rolled, held only while
   `atm_iv - sigma > k` vol points, `k` in {0, 1, 2}). Each is scored on a **fresh
   sample** of 200 paths from the same generator (seeds `seed + 2**30 ..`), never the
   frozen data; the best one is the **oracle**. The oracle is then evaluated on
   `signal/heldout` and `twin/heldout`, paired with the learned arm. A menu policy qualifies as
   oracle only if it beats flat by 2 se on the fresh sample; otherwise flat is the oracle. A menu policy qualifies as
   oracle only if it beats flat by 2 se on the fresh sample; otherwise flat is the oracle
   (short premium is negatively skewed, so small samples flatter sellers).

Derived numbers (every one with a standard error):

- **Signal edge, attainable** = oracle(signal/heldout) - oracle(twin/heldout), paired by path.
- **Signal edge, captured** = arm(signal/heldout) - arm(twin/heldout), paired by path.
- **Capture ratio** = captured / attainable; se by 2,000-sample paired bootstrap over held-out paths, fixed seed.
- **Overfit gap** = arm(signal/train) - arm(signal/heldout); unpaired se.
- **Noise fit** = noise-control(twin/train) - noise-control(twin/heldout); any positive
  in-sample result here is pure memorisation, the cleanest overfitting reading.

Each comparison carries a verdict string for the page, by fixed rules (|z| > 2):
captured / not distinguishable from zero; overfit / no detectable overfit; noise
memorised / noise not memorised.

Output: `root/synth/bench/<dataset_hash>-<arm>-<timesteps>/benchmark.json` (scoreboard
rows `{policy, split, mean, se, n}` plus the derived block). Reused if present.

## Jobs, API

Building a dataset is synchronous (milliseconds to a second). A benchmark is a job on
the server's existing process pool; `SynthJobManager` shares the executor and the
`_Progress` file mechanism of `optspread.jobs`.

| route | |
|---|---|
| `GET /api/synth/options` | kinds, params with defaults, bounds and one-line descriptions; dataset knobs; arms |
| `GET /api/synth/datasets` | manifests of the datasets on disk, newest first |
| `POST /api/synth/datasets` | `SynthSpec` -> manifest (builds or reuses) |
| `GET /api/synth/datasets/{hash}` | manifest plus preview: first 20 paths of spot, iv, signal, and their twins |
| `POST /api/synth/benchmarks` | `{dataset, arm, total_timesteps}` -> job id |
| `GET /api/synth/benchmarks/{job_id}` | status, log tail, and the benchmark when done |

## Page: `web/app/synth/`, tab "Synthetic alpha"

1. **Generate**: generator select, its knobs, sizes, seed; Build. Charts: spot fan
   (signal vs twin), ATM IV, signal series. Text: what was planted and which policy
   should be able to take it.
2. **Train**: arm, timesteps, launch; polled progress and log tail.
3. **Compare**: bar chart of mean P&L per episode with 2 se bars, grouped by split, for
   arm, noise control, oracle, flat. Tiles: capture ratio, overfit gap, noise fit, each
   with its verdict.

## Testing

- `noise` reproduces `SyntheticChainProvider` spot path exactly; every generator at zero
  strength equals `noise` on the same seed; twin shares `z` with signal.
- Planted effect is present with the right sign: `const_vrp` iv offset exact; `drift`
  mean log return within MC error; `spot_mr` terminal variance below GBM's; `mr_vrp`
  premium has the stated stationary mean and sd within MC error.
- No look-ahead for every generator under `SplicedStepRNG`.
- Dataset: hash stable, same spec reused, digest mismatch refused, provider replays the
  stored bytes and passes `leaks.sequence_violations`.
- Benchmark with a fake trainer (no torch): scoreboard shape, derived numbers on a
  constructed case, reuse on second call.
- Server routes via the existing test client fixtures. `npm run build` for the page.
- `make check` stays green (ruff, mypy --strict, coverage floor 85).
