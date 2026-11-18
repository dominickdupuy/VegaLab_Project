# Demo branch

This branch is `main` plus seeded data, so the dashboard shows every view without running
anything first. It is for demonstrations; do not merge it into `main`.

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,train,dashboard]"   # Windows
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,train,dashboard]"      # macOS / Linux
make demo        # http://127.0.0.1:8600, the committed site (web/out) on the seeded data
```

## What is seeded

| Folder | What | Read by |
| --- | --- | --- |
| `runs/sweep/` | The confirmatory sweep of 2026-10-26: 60 runs (premium x arm x seed), its `grid-*.json`, the 32-run variance-components pilot (`pilot/`) and the logs | Sweep, Power lab (measured floors), Runs |
| `runs/showcase/` | `make showcase`: every question the Runs page and the workflows ask (strikes, wings, tenor, roll, costs, account, regime, arms, sample size, PPO training length), with `catalog.json` | Runs, Compare, Launch |
| `runs/launched/`, `runs/synth/`, `runs/alpha/` | Runs launched from the dashboard, synthetic-world runs, the Synthetic alpha datasets, benchmarks and sweeps | Runs, Synthetic alpha |
| `runs/catalog.json` | The showcase catalog, keyed from the runs root, so the Runs page groups runs by question | Runs, workflows |
| `attribution/` | P&L attribution per run (delta, gamma, vega, theta, carry, cost, residual), capital and per-run action mix, for all 57 showcase and launched runs (the sweep's 92 runs are covered by the grid-level edge and mix below, and compute on demand) | A run's Attribution, Capital and Mix tabs |
| `attribution/edge/` | Which greeks and structures carry the detected edge across the sweep grid (`make edge`, first 100 episodes per cell) | Attribution / edge |
| `actionmix/` | Which structures each arm trades across the sweep grid (`make actionmix --lean`) | Sweep, Attribution |
| `web/out/` | The built site, so no Node install is needed | `make demo` |

The Power lab also reads `docs/M4_ANALYSIS.json` (power curves and floors), which `main`
already carries.

Everything here is simulated on the synthetic world. The OptionMetrics panel is licensed
and is **not** committed: the Data pages and the real-SPX runs need it under
`$VEGALAB_DATA/panel`, as on `main`.

## Rebuilding the seed

```bash
make showcase ARGS="--root runs/showcase"
make edge ARGS="--grid runs/sweep --first 100 --workers 3 --out attribution/edge"
make actionmix ARGS="--grid runs/sweep --workers 3 --lean"
# runs/catalog.json: runs/showcase/catalog.json with every key prefixed "showcase/"
# per-run attribution, capital and mix: with `make demo` running, POST
#   /api/runs/attribution, /api/runs/capital and /api/runs/mixview ?key=<run key> per run
```

Two of the sweep's 60 cells (PPO, `4a48a04bb0099da1-s101500` and `ea5be29e60901c1a-s100000`)
do not reproduce their recorded P&L over all 500 episodes when the action mix re-simulates
them (largest difference $32 and $1,586); their first 100 episodes, which the edge
attribution uses, reproduce exactly. The action-mix files record this per cell.

This branch also carries one code fix, in `optspread/attribution.py`: the sweep's PPO
policies store their checkpoint uncompressed inside `policy.zip`, which torch 2.14 cannot
stream ("failed reading file .data/serialization_id"), so the policy is re-packed in
memory before loading. The file on disk and its manifest hash are untouched.
