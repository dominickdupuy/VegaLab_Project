# Real SPX data

Owner: Hemdutt Rao (data adapter). Extraction: Dominick Dupuy.

The simulator's confirmatory sweep stays on the synthetic null world: the planted
premium has to be known by construction (D2). Real data enters in three places,
none of which changes a run's numbers: the `optionmetrics` source behind the
adapter boundary (`optspread.data`), the variance risk premium series that anchors
the detection floor in market units (project document 4.5), and the dashboard's
data views.

Market data is not in this repository. The code that builds it is, together with
this schema; the files live in the team's shared folder, named by `VEGALAB_DATA`
(default: the OneDrive shortcut `Dupuy, Dominick's files - wrds_optionmetrics`).

## Sources

| Source | What | Dates | Where |
|---|---|---|---|
| WRDS OptionMetrics `opprcd` | every SPX and SPXW contract, daily closing best bid/offer, volume, open interest, vendor IV and greeks | 2010-01-04 to 2025-08-29 (the vendor ceiling at extraction) | `opprcd/year=YYYY/SPX.parquet` |
| WRDS OptionMetrics `secprd` | SPX daily close | same | `secprd/year=YYYY/SPX.parquet` |
| Databento OPRA.PILLAR `cbbo-1m` | consolidated best bid/offer at the close: each contract's first bar from the last minute of regular trading (15:59 New York; 12:59 on an early close) through the minute after it | 2025-07-01 on | `databento/spx_close/year=YYYY/<date>.parquet` (SPX and SPXW rows only) |
| FRED `DGS3MO`, `SOFR`, `SP500` | public reference series, used only as a sanity check | as published | `reference/fred/*.csv` |

The two vendors overlap for July and August 2025. Over the overlap both are
priced and compared (the seam, below), and OptionMetrics alone enters the panel,
so no date has two answers. From 2025-09-02 the panel is Databento's.

## The panel of record

**The team's panel is the one in the shared folder**, `$VEGALAB_DATA/panel`, built by
`scripts/build_spx_panel.py` from the WRDS extraction and the Databento closes. It is
version 4, built 2026-10-28: 4,213 trading days, 2010-01-04 to 2026-10-02, none
missing (3,939 from OptionMetrics, 274 from Databento). `docs/DATA_MANIFEST.json`
records it (`panel_version` 4, 341 files, 2.54 GB), and every real-data result in this
repository reads it: the VRP mapping (`docs/VRP_MAPPING.*`), the adapter conformance
item of the gate reports, and every real-data run.

**Another build exists, and it is not this one.** On 2026-10-28 Dominick built a panel
from the Hugging Face mirror (below) with main's code of that day, where `PANEL_VERSION`
was 2: 3,939 trading days, 2010-01-04 to 2025-08-29, OptionMetrics only (the mirror has
no Databento side), `vrp30` median +1.74 vol points, positive on 67% of days. The
committed `docs/M2_GATE_REPORT.md` (its adapter item, PASS 7 of 7, 13352c6) was made on
that build. Its files are not the ones `docs/DATA_MANIFEST.json` records, and it lacks
the version 3 and 4 normaliser changes (Build, below). Pre-registration A8's "3,939
trading days" are the OptionMetrics days, which both builds have, so A8's train, eval
and holdout dates are the same on either; the values on those dates are not.

**Replay refuses a real-data run made on another panel.** A run on the `optionmetrics`
source records the sha256 of each panel file it read (`DataRecord.panel`). Replay hashes
the panel it finds again and, when the digests differ, reports `DIFFERS` with both panel
ids rather than replaying on other data; the sweep's resume (`sweep.valid_run`) treats
such a cell as one to recompute. A run made on the Hugging Face build, or on version 3,
therefore replays only against that build.

The Databento batches in the shared OneDrive mix datasets, schemas and symbol
groups (and include futures for other projects), so every job is read for its own
query first: `databento/inventory.csv` lists each tar with its dataset, schema,
symbols and dates. Jobs above 15 MB (definitions and statistics for whole symbol
groups) are listed by size and not opened; the panel needs only the closing
snapshots.

## Extraction

The OptionMetrics files are pulled from WRDS by `scripts/wrds_extract.py`, which
needs a WRDS account with OptionMetrics access and the `wrds` extra:

```bash
pip install -e ".[dev,data,wrds]"
export WRDS_USERNAME=<your WRDS user>    # required, or pass --username
python -m scripts.wrds_extract "$VEGALAB_DATA" 2010 2025 scripts/wrds_secids.json
```

The account is required: `--username`, else `WRDS_USERNAME`; with neither the pull
stops before connecting, and there is no default account. The arguments are the
output folder, the first and last year (inclusive; one `opprcd<year>` and
`secprd<year>` table each), and the ticker list. The list,
`scripts/wrds_secids.json`, maps each ticker to its OptionMetrics `secid`s (SPX is
108105; a ticker with two secids, such as XSP, is pulled under both) and is the
default when the last argument is left out. The `wrds` client asks for the password
or reads it from `~/.pgpass`.

Every query is capped at `date <= 2025-08-29`, the vendor ceiling at extraction
(`END_DATE` in the script), so a later pull does not silently extend the
OptionMetrics side past the seam. Each (table, ticker, year) is written to a `.part`
file and renamed when complete, a year with no rows leaves an `.empty` marker, and a
rerun skips both, so a dropped connection resumes where it stopped. Progress and
retries go to `download.log` in the output folder.

**The Hugging Face mirror.** The same extraction is mirrored, private and gated, as
`Algogators/options_data` (32 underlyings; access needs your own WRDS OptionMetrics
entitlement, per its README). The SPX files are all the panel needs:

```bash
hf download Algogators/options_data --repo-type=dataset --local-dir "$VEGALAB_DATA"     --include "opprcd/*/SPX.parquet" --include "secprd/*/SPX.parquet"
python scripts/build_spx_panel.py panel --years 2010 2025 --workers 1
```

The mirror's `strike_price` is already in index points (its `fix_strikes.py`
divided out the vendor's x 1000). `optionmetrics.read_opprcd` reads the unit from
the data (`strike_divisor`: the median strike against the year's median close)
rather than assuming it, so either copy builds the same panel. The mirror has no
Databento side, so a panel built from it ends at 2025-08-29, and it is a panel of
your own, not the team's (The panel of record, above).

## Build

```bash
pip install -e ".[dev,data]"
python scripts/build_spx_panel.py inventory      # every Databento job's query
python scripts/build_spx_panel.py extract-opra   # SPX/SPXW closing quotes, one file per day
python scripts/build_spx_panel.py panel          # the stitched panel and its summaries
python scripts/build_spx_panel.py fred           # the FRED sanity check
python scripts/build_spx_panel.py manifest       # docs/DATA_MANIFEST.json
```

The manifest records the sha256 and size of every file the build wrote (`files`)
and the size of every raw `opprcd`/`secprd` input it read (`inputs`), so a panel
that differs can be traced to the code or to the inputs. Hashing the inputs reads
several gigabytes through the sync client, so it is opt-in: `manifest --hash-inputs`
adds their sha256, and without it each input's `sha256` is `null` and
`inputs_hashed` is `false`.

Every stage resumes: a rerun skips what is already written. The tars are read
where they sit and decoded in memory, one at a time.

**The closing window (#31).** `cbbo-1m` writes a bar only for a contract whose book
changed in that minute, so an illiquid contract can be missing from the closing bar.
`extract-opra` keeps each contract's earliest bar from the close (the file's first
bar, 15:59 or 12:59) through one minute after it (`opra.CLOSE_WINDOW`), rather than
the first bar alone. Closing files written before this change hold only the first
bar, and `extract-opra` skips a day already written, so picking it up means moving
the old files aside and extracting again before the panel stage:

```bash
mv "$VEGALAB_DATA/databento/spx_close" "$VEGALAB_DATA/databento/spx_close.first-bar"
python scripts/build_spx_panel.py extract-opra
```

**The version 3 rebuild (#31).** Two normaliser changes alter what 2010 to 2014 build
to: the April 2014 SPX monthly now settles on Thursday 2014-04-17, not Good Friday, and
the SPXPM, SPXQ and 2010 root codes are now SPX or SPXW. IV30 also takes the SPXW row
where both roots list an expiry. `PANEL_VERSION` is 3, so the panel stage ignores the
version 2 cache and rebuilds every year. To rebuild from the shared folder:

```bash
python scripts/build_spx_panel.py panel --years 2010 2026
python scripts/build_spx_panel.py manifest --repo-copy docs/DATA_MANIFEST.json   # --hash-inputs to hash the raw files too
make vrp ARGS="--daily <a local copy of panel/spx_daily.parquet>"
```

Built 2026-10-28, and replaced the same day by version 4.

**The version 4 rebuild (#31, the data audit of 834ffb3). Built 2026-10-28; this is the
panel of record.** Four changes alter what the panel stage writes, so `PANEL_VERSION` is
4 and the stage rebuilt every year (the per-year cache under `panel/_summaries/v<N>` is
keyed on the version and nothing else; see below):

- the root is the settlement flag on every row, so the 3,990 PM-settled 2010 weekly
  rows the vendor files under the OSI root SPX are SPXW (2010 only);
- an expiry dated on a weekday NYSE closure settles the trading day before (SPXW
  2018-12-05 and 2025-01-09, `panel.UNSCHEDULED_CLOSURES`);
- an expiry whose parity forward is more than 100 bp off the day's forward curve is
  dropped as stale (`panel.stale_expiries`; 2018-02-05 and two long expiries, below);
- the daily carry is against the parity spot, `carry30_parity` in place of `carry30`.

The provider changes below (the arbitrage-free surface, the parity spot, the calendar)
need no rebuild; they apply to whatever panel is read. The rebuild, from a checkout with
`VEGALAB_DATA` naming the shared folder (the same commands rebuild it again):

```bash
python scripts/build_spx_panel.py panel --years 2010 2026     # every year: v4 ignores the v3 cache
python scripts/build_spx_panel.py fred                        # fred_check reads the new daily table
python scripts/build_spx_panel.py manifest --repo-copy docs/DATA_MANIFEST.json   # --hash-inputs to hash the raw files too
cp -R "$VEGALAB_DATA/panel" /tmp/panel_v4                      # a local copy, off the sync client
python scripts/build_spx_panel.py check --panel /tmp/panel_v4  # exit 0: no chain fails on train or eval
make vrp ARGS="--daily /tmp/panel_v4/spx_daily.parquet"        # docs/VRP_MAPPING, from the new daily table
```

The panel stage ran 18:16 to 18:36 and the manifest at 19:43; `docs/DATA_MANIFEST.json`
and the regenerated `docs/VRP_MAPPING.*` are committed from it (f87484e). The check stage
passes. Every real-data run recorded against the v3 panel is refused by replay now that
the panel's digests have changed (`DataRecord.panel`); rerun them. What changed against
v3 is under Results of the build. The coverage counts include the stale quotes in
`n_dropped`.

**The cache is keyed on the version alone.** `build_year` reuses a year's cached
summaries (`term`, `counts`, `spot_parity`, `seam`, `surface`) whenever they exist under
the current `PANEL_VERSION`. A change to the normaliser, the pricer, `finalise`,
`atm_term` or `surface_grid` that is not accompanied by a bump is silently not built. The
daily table, the coverage and the files written from the summaries are recomputed every
run, so a change to `daily_table` or `carry30` alone needs no bump, only the panel stage.

## Dates a strategy run may read (#28)

A real-data provider serves one split, never all dates (pre-registration A8):

| Split | Dates | Trading days |
|---|---|---|
| `train` | 2010-01-04 to 2018-12-31 | 2,264 |
| `eval` | 2019-01-02 to 2022-08-24 | 919 |
| holdout (no split) | 2022-08-25 on | 756 to 2025-08-29, plus every Databento day |

`optspread.realdata.SPLITS` names the two ranges, and there is no split for the
dates after the evaluation range. Every way into a provider checks the split at runtime
(`realdata.check_split`): `OptionMetricsSource`, `make_source` (whatever the source),
`harness.build_env` through it, `SurfaceStore.from_frames`, `provider.load_store` and the
`HistoricalChainProvider` constructor all raise a ValueError for anything that is not
`train` or `eval`, `None` included. `split` is required on `from_frames` and `load_store`;
there is no "every date" store. So a manifest's `DataRecord` can only record a real split
(the source cannot be built with another). The store is cut to its split before anything
is read from it.

A window starts only where its `n_days + 1` days are consecutive on the trading
calendar, so an episode never runs across a day dropped as unusable, and a surface with
no whole window is refused. The calendar is the NYSE's (`panel.nyse_trading_days`, the
holiday list 2010 to 2027, checked against all 4,213 panel days) together with the daily
table's own dates, so a day absent from both tables is a gap too. A policy trained on
`train` and scored on `eval` never sees the same day twice; the seed check in
`train.check_disjoint` covers the synthetic world.

Market measurement is not a strategy run: the VRP series and the dashboard's data
views read every date and report the split. Their read of the surface is
`provider.market_surface(grid, daily)`, a `MarketSurface` over every date with its holes
kept, a different type from `SurfaceStore`; the provider refuses it (TypeError).

## Method

Both vendors go through the same code (`optspread.realdata.surface`).

- **Carry from the quotes.** Per (date, root, expiry), call minus put against
  strike over the strikes within 5% of the parity-ATM strike (15% if fewer than
  three pairs) is a line `C - P = D (F - K)`: the discount factor is minus its
  slope, the forward its intercept over `D`. Below 60 days `D` is barely
  identified, so short expiries take the day's median rate from the 60-day to
  1.5-year expiries and refit only the forward. Neither vendor's carry is borrowed
  for the other.
- **IV and greeks with our pricer.** `optspread.pricing.bs_price` on the prepaid
  forward `S = F D` with `r = -ln(D) / T` (Black-76 in the engine's terms),
  inverted by bisection; greeks from `bs_greeks` the same way. The vendor IV is
  kept as `iv_vendor` beside ours.
- **Time** is calendar ACT/365 from the 16:00 close; AM-settled contracts (the SPX
  root) stop 6.5 hours earlier, at the open of their expiry day.
- **Two roots, meaning settlement.** `SPX` is the AM-settled monthly, `SPXW` the
  PM-settled family (weeklies, quarterlies, end-of-month). Every row takes its root
  from the vendor's settlement flag, AM to SPX and PM to SPXW, whatever its symbol
  says: before the OSI migration (2010-02-12) the vendor symbol is the old exchange
  code, to 2014 the vendor carries other roots (`SPXPM`, the PM third-Friday
  monthlies; `SPXQ`, the quarterlies; in 2010, older codes), and in 2010 the PM Friday
  weeklies are OSI symbols on the SPX root (3,990 rows, 2010-05-21 to 2010-12-02). A
  row whose symbol already names the right root is listed; a remapped row that repeats
  a listed contract is dropped (#31, panel version 4).
- **AM expiries on a Saturday.** Until February 2015 the vendor dates an AM monthly
  on the Saturday after its last trading day; it moves to that trading day, the
  Friday, or the Thursday when the Friday is a closure (April 2014: Good Friday,
  2014-04-18). `panel.SATURDAY_ERA_HOLIDAYS` lists the NYSE closures 2010 to 2017
  (#31).
- **Expiries on a closure.** An expiry the vendor dates on a weekday the exchange was
  closed settles at the close of the trading day before: SPXW 2018-12-05 (George H. W.
  Bush) and SPXW 2025-01-09 (Jimmy Carter), the only two in range. Left on the vendor's
  date, each had a day on the clock at its last quote and an ATM IV of 3.1% and 1.0%.
  `panel.UNSCHEDULED_CLOSURES` lists them with Sandy (2012-10-29 and 30); `NYSE_HOLIDAYS`
  now runs complete from 2010 to 2027 (panel version 4).
- **Stale expiries.** Per day, each expiry's parity forward is compared with the line
  through its six nearest expiries (median pairwise slope, median level). One more than
  100 bp off it is stale and every quote on it is dropped, counted with the band in
  `coverage.parquet`. On the v3 term structure that is six expiries on three days: the
  four SPXW expiries of 2018-02-05 about 208 bp below the rest (quotes left from earlier
  in a day the index fell 4%, their ATM IVs 5 to 7 vol points rich, which bumped the
  45-day grid), and the longest SPX expiry on 2010-02-10 and 2012-05-08 (past the grid's
  last tenor). At 75 bp it would take 37, nearly all the longest 2010 expiry, where a
  local line misses the curve (panel version 4).
- **Discount factors above 1.** The parity fit keeps `D` in (0.5, 1.05], so a negative
  implied rate is allowed. It occurs: 3.7% of expiries of 60 days or more, 13 to 17% of
  them in 2010 to 2012 (SPXW 2020-09-30 on 2020-03-17: `D` = 1.0103, `r` = -1.9%). It is
  the quotes' box rate, noise and all, not an error; its effect on IV is 0.1 to 0.7 vol
  points.
- **No bid/ask filter.** A mid comes from any two-sided quote, however wide (in 2010,
  10.3% of out-of-the-money IV rows have a spread above 100% of mid: the 2010-06-18 750
  put on 2010-05-06 is 0.05 / 4.00, an IV of 62% where the vendor gives none). These
  rows are what put calendar arbitrage into the grid's wings; rather than pick a spread
  threshold that changes IV30 and the VRP series, the provider removes the arbitrage
  from what it serves (below), and the grid keeps the market as quoted.
- **Kept:** `|ln(K / F)| <= 0.7` and at most 800 days to expiry.

## Real chains in an episode

`optspread.realdata.provider.HistoricalChainProvider` serves the panel as episodes.
Two rules keep a real episode honest; neither touches the synthetic world.

- **A held leg is marked between tenors (#27).** A leg's expiry stays fixed while
  the listed tenors move with the day, so it soon sits between two of them.
  `Chain.iv_at` takes the nearest one, which on a real term structure switches rows
  at the midpoint and books a one-day P&L step that is only the lookup (about
  $2,800 on a condor in the test's steep curve). Real chains are a
  `TermStructureChain`, whose `iv_at` interpolates total variance (`sigma^2 * T`)
  linearly between the bracketing listed expiries, flat in vol outside them, with
  the nearest listed strike as before. On a listed expiry it returns the listed vol
  exactly. Prices, greeks, costs and margin all read `iv_at`. Synthetic chains stay
  plain `Chain`.
- **No calendar arbitrage (#29).** Total variance at a fixed log-moneyness must not
  fall with tenor. The grid as quoted breaks that on 251 of 4,213 days (mostly deep
  wings, worst between 45 and 60 days), and the provider used to serve 180 strike cells
  on 21 training days with it, so a policy could buy a calendar spread for less than
  nothing. Each day's grid is now made free of it before it is served
  (`provider.calendar_free`): per log-moneyness column, total variance over the quoted
  tenors is replaced by its least-squares projection onto the non-decreasing sequences
  (isotonic regression, pool-adjacent-violators, equal weights), which moves 904 of 1.15M
  quoted cells (0.08%) on those 251 days, by at most 5.3 vol points. The holes are then
  filled along tenor first, total variance linear in time between quoted tenors and flat
  vol outside them, and only then along log-moneyness. The old fill ran along
  log-moneyness first, holding each tenor flat past its own last quoted strike, which
  carried a short tenor's rich wing out under a longer tenor's lower one. Both steps keep
  the order, and so do the provider's interpolations, so every chain clears
  `leaks.calendar_violations`. On the v3 panel: every day of `train` (2,264) and `eval`
  (919), 0 failing (was 21 days and 180 cells on `train`); 0 of seeds 0 to 39 with a
  violating window (was 23 of 40); `conformance_report` passes on both splits.
  `scripts/build_spx_panel.py check` runs the same check on a built panel; on version 4
  it passes with `train` 2,264 days and 2,201 window starts, `eval` 919 days and 856
  starts, 0 failing on either.
- **The spot is the option market's.** The engine prices on the parity spot
  (`spot_parity`, the prepaid forward of the shortest expiry) with the carry against it
  (`carry30_parity`) as `r`, so its forward is the market's at 30 days and, up to the
  curvature of the carry, at every other tenor. The carry against the index close also
  carried the gap between the close and the option snapshot divided by 30 days: +13.3%
  on 2011-11-01 (a 74 bp gap), -18.3% on 2020-03-13 (142 bp), and on those days the
  42-day smile was read at the wrong moneyness and the mark error reversed the next
  day. Against the parity spot it runs from about -4% to +6%. A daily table built before
  version 4 has neither column and keeps its own consistent pair, the close and
  `carry30`.
- **Training and evaluation use different dates (#28).** Seeds cannot keep them
  apart on real data, because any seed maps to any window start. The store is cut
  to its split (above), and whoever builds the source picks one:
  `OptionMetricsSource(split="eval")` or `make_source("optionmetrics",
  split="eval")` (the source's default is `train`; the harness evaluates on `eval`
  and `optspread.train` trains on `train`). The split is not part of any hashed
  config, and the confirmatory sweep does not read real data.

The dates are A8's and are not a parameter on this panel: there are no date flags, and
`SplitRanges` of your own are refused over the team's panel, known by path (no `root`,
`data_root()`, `DEFAULT_ROOT`) or by content: either panel file, symlinks followed, has
a sha256 that `docs/DATA_MANIFEST.json` or the data root's `MANIFEST.json` records
(`realdata.is_team_panel`), so a copy or a symlink of it is refused as well, wherever it
sits. `OptionMetricsSource` refuses when it is built, `load_store` before it reads a
byte, and the `HistoricalChainProvider` constructor refuses a store whose recorded panel
digests are the team's. What is left is reading the parquet files by hand and calling
`SurfaceStore.from_frames` with ranges, which nothing in the package does. Ranges of
your own are for a surface of your own under its own `root`
([EXTENDING.md](EXTENDING.md), "Your own surface"). Every run on the panel records the
split its episodes came from in its manifest (`DataRecord.split`, `eval`; absent for the
generator, and a real-data manifest from before it was recorded reads as `eval`), and
`DataRecord.ranges` when a source cut its splits on ranges of its own; such a record is
never reused or replayed on this panel. A learned arm trains on `train`
(`train_ppo(..., source=...)`) and is evaluated on `eval`, and replay reruns both.
The README's "Running on real data" has the commands and where the runs land.

## Schema

`panel/spx/year=YYYY/part.parquet`, one row per contract per day:

| Column | Meaning |
|---|---|
| date, root, expiry, dte, tau | trading date, root (`SPX` AM-settled or `SPXW` PM-settled: the vendor's settlement flag, whatever the symbol's root), expiry (an AM Saturday listing, or any expiry dated on a closure, moved to the trading day before it), calendar days, years (AM-adjusted) |
| cp, strike | right, strike in index points |
| bid, ask, mid | closing quote; mid only for a two-sided market |
| volume, open_interest | OptionMetrics only (NaN from Databento) |
| forward, discount, carry_fit | parity carry and how it was fitted (`near`, `wide`, `+r_ref`) |
| iv, delta, gamma, vega, theta | ours (vega per unit vol, theta per year) |
| iv_vendor | OptionMetrics' IV, a check, never an input |
| source | `optionmetrics` or `databento` |

Summaries in `panel/`:

| File | Contents |
|---|---|
| `spx_daily.parquet` | date, spot (vendor close, else parity), spot_parity, iv30_atm, rv21, rv21_fwd, vrp30, carry30_parity (`ln(F / spot_parity) / T` at the expiry nearest 30 days; `carry30`, against the close, before version 4), rate30, source, n_quotes |
| `surface_grid.parquet` | IV per (date, tenor in 7..365 calendar days, log-moneyness -0.5..0.3 by 0.025); NaN where nothing was quoted; as quoted, calendar arbitrage included (the provider removes it) |
| `atm_term.parquet` | ATM IV, forward and discount per (date, root, expiry) |
| `coverage.parquet` | every expected trading day, its source, contracts, and whether it is missing |
| `seam_overlap.parquet` | per overlap day: matched contracts and the vendor-to-vendor differences |
| `fred_check.parquet` | per year and source: median rate gap to bills and SOFR, spot gap to the S&P 500 close |

`vrp30` is the 30-day ATM implied volatility less the realised volatility over the
following 21 trading days, in volatility units: the market's premium, measured.

## Known limits

- **Databento closes are the first bar only.** The closing files in the shared folder
  were extracted before the closing window (#31), so a contract with no book update in
  the closing minute is missing that day. Re-extracting all 317 days (2.9 GB of
  `cbbo-1m` tars) would recover them; it was judged not worth it, since it touches only
  the thinnest contracts. The command is under Build.
- **The strike grid is fixed in index points.** A real chain is laid on the
  `MarketConfig` grid: 50 points apart, 16 strikes a side, centred on the spot. That is
  4.5% of spot apart at 1,100 (2010) and 1.1% at 4,700 (2021), covering about ±73% and
  ±17%, so delta-targeted strikes are coarse in most of `train` and fine in `eval`, a
  shift in the action set between the splits, and in March 2020 a 5-delta 42-day put
  lies past the bottom strike and snaps to the edge. Scaling the grid to spot would
  change `MarketConfig`, which is part of the hashed run config and frozen for the
  study; read real-data runs with this in mind.
- **Real-data standard errors are too small.** Episodes are windows drawn from about
  850 possible starts in the evaluation range, so most of them share days, and the
  per-episode standard error a real-data run reports treats them as independent. Read
  real-data runs as illustrations of a strategy on history, not as a powered test; the
  confirmatory test is on the synthetic world, where paths are independent.

## Results of the build

Rebuilt 2026-10-28 at panel version 4 from the shared folder, the panel of record;
`docs/DATA_MANIFEST.json` records the sha256 and size of every file the build wrote
(341 files, 2.54 GB) and the size of every raw input (998 files, 7.60 GB;
`--hash-inputs` was not run, so their sha256 is `null`). Against version 3: spot and
`rv21` are unchanged on every day; `iv30_atm` and `vrp30` moved on 4 days (2 in 2018,
2 in 2024), by at most 0.27 vol points; `carry30_parity` replaces `carry30` and runs
from -4.12% to +5.96% (`carry30` ran from -18.3% to +13.3%). The VRP mapping is
unchanged at +0.64 vol points, 95% [-0.34, +1.47], median +1.71.

Version 3, built earlier the same day (42 minutes on two workers), against the
2026-10-19 version 2 build: spot and realised vol are unchanged on every day; the surface
grid changed only in 2010, 2013 and 2014, in 18,037 of 1.53M cells (15,244 in 2010, 356
in 2013, 2,437 in 2014): 4,045 changed value, by at most 1.1 vol points, and 13,991 went
from no quote to a value;
IV30 moved on 538 days by at most 1.3 vol points (mostly third Fridays from 2017, now
SPXW); carry30 moved on 75 days, all in 2010 to 2014. The Databento closes are still the
first bar only (see Build).

- **Coverage.** 4,213 trading days, 2010-01-04 to 2026-10-02, none missing: 3,939
  from OptionMetrics (to its 2025-08-29 ceiling) and 274 from Databento after it.
  The Databento inventory found 317 daily closing snapshots (2025-07-01 to
  2026-10-02). The band (`|ln(K/F)| <= 0.7`, 800 days) dropped 5.8M quotes, counted
  per day in `coverage.parquet`.
- **Our IV against the vendor's.** On OptionMetrics' own quotes, out of the money
  and beyond 7 days, ours sits within a few hundredths of a vol point of the
  vendor's at the median (daily medians between -0.16 and +0.15 vol points over the
  overlap).
- **The seam.** Over the 43 overlap days (July and August 2025), about 7,250
  contracts a day are quoted by both vendors. Priced the same way, their IVs differ
  by 0.003 vol points at the 95th percentile on the average day (0.013 on the
  worst): each day's 95th percentile of the absolute IV difference over the
  contracts both quote, averaged over the 43 days (the median day's is 0.001). The
  Real data page's seam figure is this one. Their forwards differ by at most 0.1
  basis points: both carry the OPRA
  consolidated closing quote, so the cut-over at 2025-08-29 introduces no step.
- **Against FRED.** After the cut-over the spot is the parity prepaid forward; its
  median gap to the S&P 500 close is 0.1 to 0.6 basis points a year. The
  parity-implied 30-day rate sits above the 3-month bill and SOFR in most years,
  the known premium of SPX box rates over Treasuries: by 25 to 70 basis points
  against bills from 2010 to 2022 (116 against SOFR in 2022, while rates rose
  fastest), narrowing to between -2 and +8 in 2024. `panel/fred_check.parquet`
  has every year.
- **The variance risk premium.** Over the 4,192 days with both (the last 21 have no
  realised vol to follow), 30-day ATM implied volatility averages 15.49% against 14.70%
  realised over the following 21 trading days. The premium (`vrp30`, in volatility
  units) has a median of +1.75 vol points and a mean of +0.79, the mean pulled below the
  median by the days realised vol outran implied (2020 most of all: without it the mean
  is +0.99); it is positive on 67.8% of those days. Every day has a 30-day IV: the three
  days without one in version 2 (2010-02-17, 2010-04-12, 2010-04-13) have one since the
  2010 root codes were mapped to SPX and SPXW. This is the series the VRP mapping (#15)
  starts from.
- **Costs.** `docs/COSTS.md` fits the cost model's two parameters to these quotes.

The Hugging Face mirror build of 2026-10-28 (3,939 days, OptionMetrics only, version 2
code) is not the panel of record; see The panel of record, above.
