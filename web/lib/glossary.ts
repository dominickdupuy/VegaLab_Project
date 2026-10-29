/**
 * The glossary: one plain line for every arm, metric and knob the site shows, read by
 * every tooltip (components/InfoTip.tsx). A term is looked up by its key or by any of
 * its aliases, case and punctuation ignored, so a tile labelled "CVaR 5%" finds
 * `cvar` without the page naming it.
 *
 * The pages' own terms live beside them as plain `{ term: text }` objects
 * (lib/glossary_runs.ts, glossary_pages.ts, glossary_data.ts) and are merged into this
 * one by the `extend(...)` call at the bottom: the Runs pages' terms as they are, the
 * others under their page's name (`<InfoTip term="power.surface" />`,
 * `<Tile info="data.skew" />`). A term two pages share is defined here once.
 */

import { GLOSSARY_DATA } from "./glossary_data";
import { ATTRIBUTION_TIPS, LEAK_TIPS, POWER_TIPS, SWEEP_TIPS } from "./glossary_pages";
import { RUNS_GLOSSARY } from "./glossary_runs";

export type Glossary = Record<string, string>;

interface Entry {
  text: string;
  aliases?: string[];
}

const BASE: Record<string, Entry> = {
  // ---------------------------------------------------------------- arms
  flat: { text: "Flat: never trades. The zero line every other arm is measured against.", aliases: ["Flat"] },
  always_on: {
    text: "Always-on condor: holds one iron condor to expiry and opens the next the day after it settles. Sells premium blindly, whatever the market.",
    aliases: ["Always-on condor", "always on"],
  },
  random: { text: "Random: picks a structure, or flat, uniformly at random each day. It churns, so it shows what costs alone do.", aliases: ["Random"] },
  delta_seller: {
    text: "The fixed-delta seller: sells an iron condor at fixed deltas (short 0.16, wings 0.05 by default), closes it 5 days before expiry and reopens the next day. A simple rule that knows the premium is there: the oracle.",
    aliases: ["Fixed-delta seller", "seller", "fixed delta seller"],
  },
  ppo: {
    text: "A reinforcement-learning agent (PPO) that learns when to open which structure from the same observations. The learner under test.",
    aliases: ["PPO"],
  },
  oracle: { text: "The arm that is built to exploit the planted premium (the fixed-delta seller): the best case a learner is compared with." },

  // ---------------------------------------------------------------- worlds
  synthetic: { text: "A simulated SPX market where we choose the answer: implied vol is set above realised vol by exactly the planted premium.", aliases: ["Synthetic"] },
  optionmetrics: { text: "Real end-of-day SPX option chains from OptionMetrics (WRDS), 2010 onward.", aliases: ["OptionMetrics"] },
  databento: { text: "Real SPX option quotes from Databento OPRA, used to extend and cross-check OptionMetrics.", aliases: ["Databento OPRA", "Databento"] },
  source: {
    text: "Data source: synthetic (a simulated market where we plant the premium) or real SPX history (OptionMetrics), where nothing is planted.",
    aliases: ["Data source", "Data", "World source"],
  },
  premium: {
    text: "The planted premium: how many vol points implied vol sits above the vol the market actually realises. 0 is a fair-priced world; 1 means options are 1 vol point too dear.",
    aliases: ["Planted premium", "premium (vol pts)", "iv premium vol pts", "Planted premium (vol pts)", "Premiums (vol pts)"],
  },
  vrp: {
    text: "The variance risk premium in the real market: 30-day implied vol minus the vol realised over the next 21 trading days. Positive means options were dear.",
    aliases: ["VRP30", "Real SPX premium", "Mean premium", "real premium", "Mean VRP30"],
  },
  vol_points: { text: "Vol points: percentage points of annualised volatility. 20% to 21% implied vol is 1 vol point.", aliases: ["vol pts", "vol pt", "vol points"] },
  null_world: { text: "A world with nothing planted (premium 0): any profit there is luck, and a seller just pays its costs.", aliases: ["Null world", "fair world", "fair-priced world"] },

  // ---------------------------------------------------------------- design
  seeds: {
    text: "S, seeds: independent repetitions of the whole experiment (a new market path set, and for PPO a new training run). The between-seed spread is what most limits detection.",
    aliases: ["S", "Seeds S", "S, seeds", "Distinct seeds", "Seed blocks", "blocks", "Block"],
  },
  episodes: { text: "N, episodes: 63-trading-day trading windows simulated per seed. More episodes average away day-to-day noise, but not seed-to-seed noise.", aliases: ["N", "Episodes", "N, episodes per seed"] },
  tau2: { text: "τ², between-seed variance: how much the average result moves from one seed to the next. It does not shrink with more episodes.", aliases: ["τ²", "tau²", "τ², between-seed variance"] },
  sigma2: { text: "σ², within-seed variance: how much single episodes scatter around their seed's average. It shrinks as N grows.", aliases: ["σ²", "σ², within-seed variance"] },
  alpha: { text: "α: the false-alarm rate allowed, one-sided. At 0.05 a world with no premium is called 'detected' 1 time in 20.", aliases: ["α", "α, one-sided"] },
  config_hash: { text: "A fingerprint of every knob a run used. Two runs with the same hash and seed must give the same numbers.", aliases: ["Config hash", "Distinct configs"] },
  commit: { text: "The git commit of the code that produced the run.", aliases: ["Commit", "Git SHA"] },
  sweep: { text: "The pre-registered grid: every arm at every planted premium, repeated over S seed blocks.", aliases: ["Sweep", "grid", "Cells"] },

  // ---------------------------------------------------------------- metrics
  mean_pnl: { text: "Mean P&L: the average profit per 63-day episode, in dollars, after costs.", aliases: ["Mean P&L", "Mean P&L / episode", "mean P&L per episode", "P&L"] },
  se: { text: "se, standard error: how far the mean could be off from the true value by chance alone. Roughly, a mean within 2 se of zero is not distinguishable from zero.", aliases: ["SE", "standard error", "SE of the grand mean"] },
  t: { text: "t: the estimate divided by its standard error. Above about 2 (one-sided, many episodes) the result is unlikely to be luck.", aliases: ["t", "t stat", "Paired t", "t-stat"] },
  cvar: { text: "CVaR 5%: the average of the worst 5% of episodes. What a bad quarter costs, not just how often one happens.", aliases: ["CVaR 5%", "CVaR"] },
  sortino: { text: "Sortino ratio: mean P&L divided by the spread of losing episodes only. Higher is better; it ignores upside volatility.", aliases: ["Sortino"] },
  drawdown: { text: "Max drawdown: the largest fall from a running peak of cumulative P&L to a later trough.", aliases: ["Drawdown", "Max drawdown", "Drawdown, seed order", "Worst drawdown"] },
  hit_rate: { text: "Hit rate: the share of episodes that made money.", aliases: ["Hit rate"] },
  turnover: { text: "Turnover: trades per episode. Every trade pays the spread and commission.", aliases: ["Turnover"] },
  cost: { text: "Cost: half the bid-ask spread plus commission, paid on every contract traded.", aliases: ["Cost / episode", "Cost", "Cost/ep"] },
  return_on_margin: {
    text: "Return on posted margin: total P&L divided by the peak margin the broker would have held. What the strategy earns on the capital it actually ties up.",
    aliases: ["Return on peak margin", "return on posted margin", "Return on margin"],
  },
  return_on_cash: { text: "Return on initial cash: P&L per episode as a share of the account's starting cash.", aliases: ["Return on initial cash"] },
  power: {
    text: "Power: the chance the test calls an effect 'detected' when it is really there. 80% means a real edge of that size is missed 1 time in 5.",
    aliases: ["Power", "Power at the effect", "Detection probability", "Target power"],
  },
  floor: {
    text: "Detection floor: the smallest planted premium the design detects with 80% power. Anything smaller usually looks like noise.",
    aliases: ["Detection floor", "Floor", "minimum detectable effect", "Detector floor"],
  },
  edge: {
    text: "Edge: an arm's mean P&L at a planted premium minus the same arm's at 0, on the same market paths. The part of the profit the premium pays for.",
    aliases: ["Edge", "P&L edge", "Mean of B − A"],
  },
  contrast: { text: "Contrast: an arm at premium p against itself at 0, seed block by seed block. The pre-registered test is a one-sided t on these.", aliases: ["Contrasts", "Contrast"] },
  gap: { text: "Gap: how much more premium the learner (PPO) needs than the oracle seller before it is detected.", aliases: ["Gap, PPO − oracle", "Gap"] },
  residual: {
    text: "Residual: the part of P&L the greeks do not explain (big moves, discrete re-hedging). A small residual means the attribution is trustworthy.",
    aliases: ["Residual", "residual share"],
  },
  delta_pnl: { text: "Delta P&L: profit from the underlying's move, through the book's delta.", aliases: ["Delta"] },
  gamma_pnl: { text: "Gamma P&L: the convexity term; a short-option book loses on big moves.", aliases: ["Gamma"] },
  vega_pnl: { text: "Vega P&L: profit from implied vol changing, through the book's vega.", aliases: ["Vega"] },
  theta_pnl: { text: "Theta P&L: time decay; a short-option book collects it day by day.", aliases: ["Theta"] },
  carry: { text: "Carry: interest on cash. Always 0 here, because the book never books it.", aliases: ["Carry"] },
  held_share: { text: "Held: the share of trading days the arm had a structure open.", aliases: ["Held", "held share"] },
  opens: { text: "Opens per episode: how often the arm opened a new structure; refused opens were blocked by margin or rules.", aliases: ["Opens per episode", "refused opens"] },
  replay: {
    text: "Replay: rerun a run from its manifest alone and compare the new result with the saved one. IDENTICAL means byte for byte.",
    aliases: ["Replay", "IDENTICAL", "AGREES", "DIFFERS"],
  },
  digest: { text: "Digest: a sha256 of each saved file, recomputed now. A mismatch means the file changed after it was written.", aliases: ["Artifact digests", "DIGEST MISMATCH"] },
  leak: {
    text: "A leak: information from the future getting into what the agent sees. It makes any backtest look profitable. The detector plants one on purpose and must catch it.",
    aliases: ["leak size eps", "eps", "Leak size eps"],
  },
  bonferroni: { text: "Bonferroni threshold: the p-value line, divided by the number of fields tested, that a field must cross to be flagged.", aliases: ["Bonferroni threshold"] },

  // ---------------------------------------------------------------- knobs
  sigma: { text: "Market vol σ: the volatility the simulated SPX actually realises (0.2 = 20%).", aliases: ["sigma", "market vol"] },
  r: { text: "Rate r: the risk-free interest rate used to price the options.", aliases: ["rate"] },
  n_days: { text: "Days per path: trading days simulated in each market path (63 is one quarter).", aliases: ["days per path"] },
  tenor_days: { text: "Tenors: the expiries listed, in trading days.", aliases: ["tenors"] },
  short_delta: { text: "Short delta: the delta of the options sold (0.16 is about one standard deviation out). Smaller is further out: less credit, rarer losses.", aliases: ["short delta"] },
  wing_delta: { text: "Wing delta: the delta of the protective options bought further out; must be smaller than the short delta.", aliases: ["wing delta"] },
  roll_dte: { text: "Roll at DTE: the seller closes its structure this many days before expiry and reopens the next day.", aliases: ["roll dte", "roll at"] },
  expiry_idx: { text: "Opening tenor: which listed expiry new structures open at (0 is the nearest).", aliases: ["expiry idx", "opening tenor"] },
  costs_enabled: { text: "Costs on: charge the spread and commission on every trade. Off shows the world's pure economics.", aliases: ["costs enabled", "costs"] },
  min_half_spread: { text: "Minimum half-spread: the least paid per contract to cross the market, in dollars of option price.", aliases: ["min half spread"] },
  half_spread_frac: { text: "Half-spread fraction: the spread paid as a share of the option's price, when above the minimum.", aliases: ["half spread frac"] },
  commission_per_contract: { text: "Commission: dollars per contract traded.", aliases: ["commission per contract", "commission"] },
  initial_cash: { text: "Initial cash: the account's starting balance.", aliases: ["initial cash"] },
  seed: { text: "Seed: the first market path's random seed; the same seed gives the same paths.", aliases: ["Seed", "Episode seed"] },
  total_timesteps: { text: "Training budget: how many market steps PPO learns from before it is evaluated. The sweep trains 1,000,000.", aliases: ["total timesteps", "PPO steps per seed", "training budget", "training steps"] },
  n_strikes_each_side: { text: "Strikes each side: how many strikes are listed above and below spot.", aliases: ["strikes each side"] },
  strike_step: { text: "Strike step: the distance between listed strikes, in index points.", aliases: ["strike step"] },
};

const index = new Map<string, string>();
const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/[_\-·,/()%:]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

function put(key: string, text: string, aliases: string[] = []): void {
  index.set(norm(key), text);
  for (const a of aliases) if (!index.has(norm(a))) index.set(norm(a), text);
}

for (const [k, e] of Object.entries(BASE)) put(k, e.text, e.aliases);

/** A page's terms under its own name: {seeds: …} becomes {"power.seeds": …}. */
export function prefixed(page: string, dict: Readonly<Record<string, string>>): Glossary {
  return Object.fromEntries(Object.entries(dict).map(([k, v]) => [`${page}.${k}`, v]));
}

/** Merge page glossaries site-wide; a page's entry replaces one of the same key here. */
export function extend(...dicts: Glossary[]): void {
  for (const d of dicts) for (const [k, v] of Object.entries(d)) index.set(norm(k), v);
}

/**
 * The plain line for `term` (a key such as "cvar" or a label such as "CVaR 5%"), from
 * `dict` first when given, else the glossary; undefined when nothing explains it.
 */
export function define(term: string | undefined | null, dict?: Glossary): string | undefined {
  if (!term) return undefined;
  if (dict) {
    const hit = dict[term] ?? Object.entries(dict).find(([k]) => norm(k) === norm(term))?.[1];
    if (hit) return hit;
  }
  return index.get(norm(term));
}

/** Every arm's line, in the order the site lists them. */
export const ARM_ORDER = ["flat", "always_on", "random", "delta_seller", "ppo"] as const;

// ---------------------------------------------------------------- the pages' own terms
// Each page's terms live beside it and merge here once, so every tooltip on the site reads
// one glossary. A term only one page uses is namespaced by that page ("power.surface",
// "data.skew"), so the same word can mean the page's thing there and the general thing
// everywhere else; the Runs pages' terms are site-wide (they replace only aliases here).
extend(RUNS_GLOSSARY, prefixed("power", POWER_TIPS), prefixed("sweep", SWEEP_TIPS), prefixed("attribution", ATTRIBUTION_TIPS), prefixed("leaks", LEAK_TIPS), prefixed("data", GLOSSARY_DATA));
