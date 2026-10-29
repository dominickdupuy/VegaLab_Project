/**
 * What every metric and control on the Real data page means, one line each. Merged into
 * the site glossary under "data." (lib/glossary.ts): the page asks for `data.skew`.
 * Keep each entry plain text, one sentence or two.
 */
export const GLOSSARY_DATA: Record<string, string> = {
  // the questions
  q_premium: "How much more the option market charges for volatility than the volatility that then happens, on average and day by day.",
  q_surface: "The real SPX implied-volatility surface on one trading day: its level, its skew across strikes and its slope across tenors.",
  q_sim: "The synthetic world the experiment runs on, set beside the real market on the same day: where its single implied vol misses the market's surface.",
  q_seller: "The structure the fixed-delta seller would open at one real close: its strikes, the credit, the worst case and the margin it posts.",

  // the premium question
  premium: "The volatility risk premium: 30-day at-the-money implied vol less the realised vol of the next 21 trading days, in vol points. Positive means sellers of options were paid for bearing volatility risk.",
  premium_mean: "The mean daily premium over every day with a complete forward window (2010 to a month before the panel ends): the VRP mapping's headline, the number the sweep's premium axis is read against.",
  premium_interval: "95% interval of the mean from a stationary block bootstrap (10,000 draws, Politis-White block of about 58 days): consecutive days share 20 of their 21 forward returns, so days are not independent.",
  premium_median: "The median day's premium. It sits above the mean because the premium is left-skewed: a few volatility spikes (2020 most of all) pull the mean down.",
  days_positive: "The share of days on which implied vol came out above the realised vol that followed.",
  sweep_axis: "The planted premiums the confirmatory sweep tests, in vol points (0, 0.25, 0.5, 1, 2, 4). Where the real premium falls says which sweep cells look like the market.",
  premium_priced: "Today's IV30 less the realised vol of the 21 trading days before it: what the market charges now over what has just happened. Known on the day; the measured premium needs the next 21 days.",
  premium_recent: "The mean daily premium over the latest quarter (63 trading days) whose forward windows are complete, and how it ranks among every such quarter since 2010.",
  premium_smooth: "The trailing mean of the daily premium over the last 63 trading days: one quarter. The daily series is noisy because 21 days of realised vol are.",
  premium_daily: "Each day's premium: IV30 that day less the realised vol over the 21 trading days that followed, divided by c21 = 0.988 to remove the small-sample bias of a 21-day vol.",
  iv30: "30-day at-the-money implied volatility: what the option market charges for the next 30 calendar days, in vol points.",
  rv_fwd: "Realised volatility over the 21 trading days after the day: zero-mean, annualised on 252 days, divided by c21 = 0.988.",
  rv_trailing: "Realised volatility over the 21 trading days before the day, on the same definition: what had just happened.",
  subperiods: "The premium's mean and 95% interval in parts of the sample. A split keeps only days whose whole forward window falls on its side, so the parts share no return.",
  sensitivity: "The headline under other definitions of realised vol: with or without the bias correction, the sample standard deviation, 30 calendar days, variance units, trailing vol.",
  chart_view: "Premium: the daily premium and its smoothed line. Implied and realised: IV30 against the realised vol that followed and the one before.",
  by: "Periods: before and after 2020, and the development extract against its holdout. Years: each calendar year of quote dates.",

  // the surface question
  date: "The trading day. A weekend or holiday picks the trading day before it; the arrows step one trading day.",
  surface: "Implied vol across strike (ln(K/F)) and tenor on the build's fixed grid: 7 to 365 calendar days, ln(K/F) from -0.5 to +0.3. Holes are where nothing was quoted; nothing is filled across a hole.",
  playback: "Plays the surface forward one trading day per frame from the chosen date, the colour scale held still so the motion reads. Pausing keeps the frame's date.",
  x_axis: "ln(K/F), log of strike over the forward (0 at the money, -0.1 about 10% below), or K/F itself.",
  atm30: "30-day at-the-money implied vol on the grid (ln(K/F) = 0), in vol points, and how it ranks against every gridded day.",
  skew: "The 30-day implied vol of a put about 10% below the forward (ln(K/F) = -0.1) less the 30-day at-the-money vol: how much more downside protection costs, in vol points.",
  call_wing: "The 30-day implied vol of a call about 5% above the forward less the at-the-money vol: usually negative, calls are cheaper than the money.",
  term_slope: "At-the-money implied vol a year out less the 30-day one. Positive: upward sloping, the calm shape. Negative: inverted, near-term fear above the long run.",
  smile: "Implied vol across strikes at one tenor of the day's grid: the skew is how far the left side rises above the middle.",
  smile_tenor: "The grid tenor the smile is drawn at, in calendar days.",
  term: "At-the-money implied vol across the grid's tenors on the day, with the put 10% below beside it.",
  landscape: "At-the-money implied vol by date and tenor: the term structure through time. Long windows average consecutive days into at most a few hundred rows.",
  landscape_window: "The stretch of history the landscape draws.",

  // the simulator question: against the market
  sim_level: "The simulator prices every listed contract at one implied vol: its path vol sigma plus the planted premium.",
  planted_premium: "The premium the simulator plants: every contract's IV above the path's own vol, in vol points. The sweep tests 0 to 4.",
  path_vol: "Where the simulator's sigma comes from. Config: MarketConfig's 20%. RV21: the day's trailing 21-day realised vol. Matched to IV30: sigma set so sigma plus the premium equals the day's IV30.",
  cmp_view: "Side by side: the real surface beside the simulator's contracts. Overlay: the simulator's level as a plane over the real surface. Difference: real less simulated over the real grid, the simulator's contracts outlined.",
  diff_mean: "Real less simulated implied vol, averaged over the simulator's contracts the real surface quotes, in vol points.",
  diff_abs: "The mean size of the gap, either direction, in vol points.",
  diff_max: "The largest gap at any one of the simulator's contracts, in vol points.",
  regions: "The simulator's contracts split by moneyness: the put wing below ln(K/F) = -0.03, near the money within 0.03, the call wing above.",

  // the seller question
  iron_condor: "Short a put and a call near the 16-delta strikes, long a put and a call further out (5 delta) as wings: the seller collects a credit and its loss is capped by the wings.",
  credit: "Dollars received at the open for one condor (100 multiplier): the most it can make.",
  max_loss: "The worst case at expiry, computed from the four legs: the wider wing less the credit, in dollars. The wings cap it even on a day the margin check misreads the condor as naked (#39).",
  margin: "Dollars of buying power the structure ties up at the open, by optspread.margin: its max loss when the check reads the risk as defined, the naked-short rule when it does not (on some high-spot days it misreads a condor, #39).",
  breakevens: "The SPX levels at expiry where the condor neither makes nor loses: the short strikes less and plus the credit per share.",
  legs: "Each leg on the real chain the provider builds for the day: strike, strike over spot, delta, implied vol and mid price per share.",

  // data quality
  coverage: "Contracts quoted per trading day by vendor. The panel takes each day from one vendor; a missing day is listed, never interpolated.",
  missing_days: "Trading days with no quote from the vendor that should cover them. A day the other vendor quotes is not missing from the panel.",
  seam: "OptionMetrics and Databento priced the same way on the days both quote (July and August 2025). The figure: on each shared day, the 95th percentile of the IV gap over the contracts both quote, averaged over the days (the worst day beside it), as docs/DATA.md states it.",
  fred: "FRED's public series as a sanity check: the parity-implied 30-day rate against 3-month bills and SOFR (SPX box rates sit above Treasuries), and the spot against the S&P 500 close, per year, in basis points.",
  panel_version: "The build's PANEL_VERSION as docs/DATA_MANIFEST.json records it, and whether the panel files this server reads have the sha256 the manifest records.",
  quoted_smile: "One expiry's out-of-the-money quotes as the vendor sent them: our IV (optspread.pricing) against OptionMetrics' own IV, a check never an input.",
  roots: "SPX settles on the morning's open (AM), SPXW on the close (PM). Both list most expiries.",
  vendor: "The data vendor for the day: OptionMetrics to 2025-08-29, Databento after.",
  expiry: "The listed expiry the quoted smile is drawn for, with its days to expiry and usable quotes.",
};

export type DataTerm = keyof typeof GLOSSARY_DATA;
