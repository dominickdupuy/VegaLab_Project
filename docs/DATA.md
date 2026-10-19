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
| date, root, expiry, dte, tau | trading date, SPX or SPXW, expiry, calendar days, years (AM-adjusted) |
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
