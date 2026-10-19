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
| Databento OPRA.PILLAR `cbbo-1m` | consolidated best bid/offer, the bar for the last minute of regular trading (15:59 New York; 12:59 on an early close) | 2025-07-01 on | `databento/spx_close/year=YYYY/<date>.parquet` (SPX and SPXW rows only) |
| FRED `DGS3MO`, `SOFR`, `SP500` | public reference series, used only as a sanity check | as published | `reference/fred/*.csv` |

The two vendors overlap for July and August 2025. Over the overlap both are
priced and compared (the seam, below), and OptionMetrics alone enters the panel,
so no date has two answers. From 2025-09-02 the panel is Databento's.

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
export WRDS_USERNAME=<your WRDS user>    # default: dominickdupuy
python -m scripts.wrds_extract "$VEGALAB_DATA" 2010 2025 scripts/wrds_secids.json
```

The arguments are the output folder, the first and last year (inclusive; one
`opprcd<year>` and `secprd<year>` table each), and the ticker list. The list,
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

## Build

```bash
pip install -e ".[dev,data]"
python scripts/build_spx_panel.py inventory      # every Databento job's query
python scripts/build_spx_panel.py extract-opra   # SPX/SPXW closing quotes, one file per day
python scripts/build_spx_panel.py panel          # the stitched panel and its summaries
python scripts/build_spx_panel.py fred           # the FRED sanity check
```

Every stage resumes: a rerun skips what is already written. The tars are read
where they sit and decoded in memory, one at a time.

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
- **Before the OSI migration** (2010-02-12) the vendor symbol is the old exchange
  code, so those rows take their root from settlement: AM is SPX, PM is SPXW.
- **Kept:** `|ln(K / F)| <= 0.7` and at most 800 days to expiry.

## Schema

`panel/spx/year=YYYY/part.parquet`, one row per contract per day:

| Column | Meaning |
|---|---|
| date, root, expiry, dte, tau | trading date, the vendor's root (SPX and SPXW from the OSI migration on; before it, and for the old SPXPM and SPXQ series to 2014, the vendor's other roots), expiry (an AM Saturday listing moved to its Friday), calendar days, years (AM-adjusted) |
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
| `spx_daily.parquet` | date, spot (vendor close, else parity), spot_parity, iv30_atm, rv21, rv21_fwd, vrp30, carry30, rate30, source, n_quotes |
| `surface_grid.parquet` | IV per (date, tenor in 7..365 calendar days, log-moneyness -0.5..0.3 by 0.025); NaN where nothing was quoted |
| `atm_term.parquet` | ATM IV, forward and discount per (date, root, expiry) |
| `coverage.parquet` | every expected trading day, its source, contracts, and whether it is missing |
| `seam_overlap.parquet` | per overlap day: matched contracts and the vendor-to-vendor differences |
| `fred_check.parquet` | per year and source: median rate gap to bills and SOFR, spot gap to the S&P 500 close |

`vrp30` is the 30-day ATM implied volatility less the realised volatility over the
following 21 trading days, in volatility units: the market's premium, measured.

## Results of the build

Built 2026-10-19 from the shared folder; `docs/DATA_MANIFEST.json` records the
sha256 and size of every file the build wrote.

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
  worst), and their forwards by at most 0.1 basis points: both carry the OPRA
  consolidated closing quote, so the cut-over at 2025-08-29 introduces no step.
- **Against FRED.** After the cut-over the spot is the parity prepaid forward; its
  median gap to the S&P 500 close is 0.1 to 0.6 basis points a year. The
  parity-implied 30-day rate sits above the 3-month bill and SOFR in most years,
  the known premium of SPX box rates over Treasuries: by 25 to 70 basis points
  against bills from 2010 to 2022 (116 against SOFR in 2022, while rates rose
  fastest), narrowing to between -2 and +8 in 2024. `panel/fred_check.parquet`
  has every year.
- **The variance risk premium.** 30-day ATM implied volatility averages 15.5%
  against 14.7% realised over the following 21 days; the premium (`vrp30`) has a
  median of +1.75 vol points (mean +0.78, pulled down by 2020) and is positive on
  67% of days. Three days have no 30-day IV (2010-02-17, 2010-04-12, 2010-04-13):
  no expiry was listed on one side of 30 days, and nothing is extrapolated. This is the series the VRP mapping (#15) starts from.
- **Costs.** `docs/COSTS.md` fits the cost model's two parameters to these quotes.
