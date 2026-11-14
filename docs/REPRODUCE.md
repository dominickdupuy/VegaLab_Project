# Reproduce our gates from a clean checkout

For the partner team (issue #10, the COIL exchange). You need a clone and Python 3.12 or
newer, on macOS, Linux or Windows. You do not need our shared folder or any market data.

## Install and run

```bash
git clone https://github.com/dominickdupuy/VegaLab_Project.git
cd VegaLab_Project
python scripts/reproduce.py        # python3 on macOS and Linux; or: make reproduce
```

The one command creates `.venv`, installs the project into it (`pip install -e
".[dev,train]"`), and runs eight steps in order. Expect about 20 minutes after the
install on a laptop; the test suite is most of it (13 minutes on the Windows machine
below, 5 on an Apple-silicon laptop).

| Step | The same as | What it checks |
|---|---|---|
| ruff | `make lint` | style and formatting |
| mypy --strict | `make typecheck` | types |
| pytest | `make test` | the test suite, coverage at least 85% |
| wave0 gate | `make gate` | the sanity table below |
| cost drag | `make drag` | costs against their prediction |
| leak suite | `make leaks` | no look-ahead, and the detector's controls |
| poisoned demo | (in-process) | a planted leak is flagged, a clean world is not |
| sweep cells | (in-process) | three cells of our confirmatory sweep, regenerated |

The first three together are `make check`. Options: `--team "Your team"` names you in
the record; `--cells N` regenerates more seller cells (about 40 seconds each);
`--with-ppo` adds one PPO cell (a million training steps, about 20 minutes more).

To run a step by hand after the install, use the `make` target, or on a machine without
`make` the command the record lists for it, with `.venv/bin/python` (`.venv\Scripts\python`
on Windows), for example `.venv/bin/python -m scripts.wave0_gate`.

## What you should see

Every step prints `PASS`, and the run ends with `M3 partner reproduction: PASS`. The
tables below are from commit 489defa on Windows 11 (Python 3.12.8, numpy 2.5.4, scipy
1.18.1). The dollar figures are rounded to the dollar, so yours should match as printed;
across platforms the underlying numbers differ by at most 1.7e-11 relative.

### `make gate`

```
arm           costs       mean P&L        se  trades
flat          False              0         0     0.0
flat          True               0         0     0.0
always_on     False           -509     1,157     3.0
always_on     True            -786     1,162     3.0
random        False           -333       370    70.7
random        True          -4,583       371    70.7
  [PASS] flat exactly zero
  [PASS] no-cost always-on ~ 0 (|t| <= 3)
  [PASS] with-cost always-on < 0 and < no-cost
  [PASS] random bleeds faster
Wave 0 expectation HOLDS.
```

### `make drag`

```
seller           structure               free   costed     drag  predicted      gap  settles
hold_condor      IRON_CONDOR             -515     -772      257        257        0      1.0
hold_put_spread  PUT_CREDIT_SPREAD     -2,058   -2,186      128        128        0      1.0
roll_condor      IRON_CONDOR            1,716    1,366      350        350        0      0.0
roll_put_spread  PUT_CREDIT_SPREAD       -817   -1,014      197        197        0      0.0
  [PASS] cost identity: costed == free - charged
  [PASS] cash settlement charges nothing
  [PASS] no unpaid exit at the horizon
Cost drag check HOLDS.
```

### `make leaks`

The clean world, one row per observation field (13 of 14 tested, `atm_iv` constant under
the probe), ending:

```
630 pairs, 13 of 14 fields tested, threshold 7.69e-04
worst: unrealized_frac p=1.11e-01
no leak detected
```

Then the controls, where every one of the 14 fields reads `True  True` (a planted leak
is caught, and the right field is named), ending:

```
  [PASS] clean world is quiet
  [PASS] every field tested or explained
  [PASS] every field has a working control
Leak suite PASSES its stated criterion.

spot-leak detection floor: eps = 0.1 on grid (1.0, 0.5, 0.25, 0.1, 0.05)
caught: (True, True, True, True, False)
```

### The poisoned demo and the sweep cells

```
poisoned demo: eps = 0.5 over 10 seeds: poisoned flagged True, its past moved True;
               clean flagged False, its past moved False
sweep cells:   3 cell(s) regenerated: 3 within tolerance
```

Each regenerated fixed-delta seller cell is compared with `docs/SWEEP_EXPECTED.json`
within 1e-9 relative (every count exactly). On Windows the cells are also byte-identical
to ours; on macOS and Linux the bytes differ in the last digit, and the tolerance is what
passes. The three default cells and their mean P&L per episode come from that file:

| Cell | Run | Mean P&L | se |
|---|---|---|---|
| delta_seller at 0 vol points, block 0 | `65046cadd80098cf-s100000` | -391.86 | 404.17 |
| delta_seller at 4 vol points, block 0 | `7ed5d3842e4d92d8-s100000` | 2,401.63 | 325.86 |
| delta_seller at 1 vol point, block 3 | `7198f9de012d36ed-s101500` | 179.30 | 431.31 |

With `--with-ppo`, the PPO cell is checked statistically, not by bytes: PPO training is
not bit-reproducible across platforms, even at the same library versions. It passes when
its mean P&L is within 3 standard errors of ours, it trades, no open is refused and the
attribution residual stays small. `optspread/reproduction.py` states each tolerance and
why.

## What to send back

The run writes `docs/M3_PARTNER_REPRODUCTION.json` and `docs/M3_PARTNER_REPRODUCTION.md`.
Send us both, by a pull request that adds them or by email, failures included: a failure
is what we most need to see. Our M3 gate (`make m3gate`) reads the JSON; its partner item
passes on a clean checkout at or after the `cutoff-2026-11-01` tag with every step passed.

## If something fails

- **A step prints `FAIL`.** The record keeps the last 60 lines of its output; send it as
  it is.
- **Windows, `WinError 1314` in a test.** That was a test making a symbolic link without
  the privilege to; it is fixed as of the commit that adds this file. Pull and rerun.
- **`python` is older than 3.12.** Start the script with a newer one, for example
  `python3.12 scripts/reproduce.py`.
- **The install fails behind a proxy or offline.** The install needs PyPI (and, on
  Linux, download.pytorch.org for the CPU build of torch).
