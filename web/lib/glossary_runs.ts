/**
 * The Runs, run detail, Compare and Launch pages' own terms, {term: text}, merged into
 * the site glossary by `extend(...)` in lib/glossary.ts. A term the site glossary already
 * explains (an arm, a metric, a launch knob) lives there only, so it reads the same on
 * every page; what is here is what only these pages show.
 */
export const RUNS_GLOSSARY: Record<string, string> = {
  // the world
  real: "Real SPX history: episodes start on real closes and trade the day's real surface; nothing is planted.",
  a8_dates: "A8 dates: real data is split in time. PPO trains on the earlier dates and every arm is evaluated only on the later ones, so no run is scored on data it learned from.",
  world: "World: everything about a run except its arm, seeds, episode count and premium. Runs compare fairly only within one world.",
  paired_seeds: "Seeds: an episode's seed fixes its market path. Two runs on the same seeds saw the same markets, so they pair.",
  seed_first: "First seed: episodes use seed, seed+1, ... Runs with the same first seed share paths and pair. PPO also trains on this seed, so it must be below 32,768.",
  // the numbers
  interval: "95% interval: the range the true mean sits in, 19 times out of 20, given this sample.",
  box: "Box: the middle half of episodes (25th to 75th percentile) with the median as a line; whiskers reach the 5th and 95th percentiles; the dot and bar are the mean and its 95% interval.",
  paired_contrast: "Paired contrast: the arm at this premium less the same arm at zero premium, episode by episode on the same paths. The path's luck cancels and only the premium's effect is left.",
  paired_difference: "Paired difference, B − A: run B's P&L less run A's (the baseline) on each seed both ran, averaged. Each pair saw the same market path, so the path's luck cancels and only what differs between the runs is left.",
  contrast_runs: "Over runs: each run pair is one block; the interval uses the spread of the block means (the pre-registered rule).",
  contrast_episodes: "One run pair: the interval uses the episodes only, so it leaves out the run-to-run spread (for PPO, the training seed's).",
  drawdown_seed: "Drawdown in seed order: the worst fall of cumulative P&L when the episodes are strung together by seed.",
  drawdown_episode: "Drawdown in an episode: the worst peak-to-trough fall within one episode.",
  // capital
  equity_path: "Equity path: cumulative P&L day by day within an episode. The band is the 10th to 90th percentile over every episode.",
  margin: "Posted margin: what the book must set aside for what it holds: the max loss for defined risk, the Cboe rule otherwise.",
  peak_margin: "Peak margin: the most margin the book held at any point in the episode.",
  resimulated: "Re-simulated: the episodes are run again from the manifest (a PPO run with its saved policy, not retrained) and checked against run.json.",
  policy_heatmap: "Where it acted: every step of the run's recorded episodes, binned by how far through the episode it fell and the at-the-money implied vol the policy saw then; each cell is the share of its steps that chose the action picked. Recorded by runs launched from the dashboard.",
  // attribution
  attribution: "Attribution: each day's P&L split by source: the greeks (how the options' price moves with the index, with the size of its move, with implied vol and with time), carry, costs and a residual.",
  greeks: "The greeks: how an option's price moves with one thing at a time: delta with the index's move, gamma with the size of that move, vega with implied vol, theta with time passing. Attribution splits P&L along them.",
  "attr.delta": "Delta: P&L from the index moving up or down, at the book's delta (how much its value moves per point of the index).",
  "attr.gamma": "Gamma: P&L from how big the index's move was, whatever its direction. A book that sold options loses it on big days: the price of selling.",
  "attr.vega": "Vega: P&L from implied vol changing.",
  "attr.theta": "Theta: P&L from time passing alone. Options lose value each day towards expiry, so a book that sold them collects it day by day.",
  "attr.carry": "Carry: interest the book credits on cash. Always 0 here, because the book books none.",
  "attr.cost": "Cost: half-spreads and commissions paid.",
  "attr.residual": "Residual: what the greeks do not explain. Small is good.",
  // the action mix
  action_mix: "Action mix: what the policy actually traded: which structure, at what delta and tenor, how long it held and how it left.",
  "structure.FLAT": "Flat: no position.",
  "structure.IRON_CONDOR": "Iron condor: short a put and a call at the short delta, long a further put and call (the wings). Defined risk.",
  "structure.SHORT_STRANGLE": "Short strangle: short a put and a call, no wings. Undefined risk.",
  "structure.PUT_CREDIT_SPREAD": "Put credit spread: short a put, long a further put. Defined risk, bullish.",
  refused: "Refused: opens the book could not margin.",
  holding_days: "Holding days: days from an open until the book was flat again.",
  "exit.policy": "Closed by the policy (a roll or a switch).",
  "exit.expiry": "Settled at expiry.",
  "exit.margin": "Liquidated: equity fell below the margin it needed.",
  "exit.horizon": "Still open on the episode's last day; closed there.",
  // reproduction (these replace the replay line on the verdicts themselves)
  IDENTICAL: "Identical: replayed from its manifest, run.json came back bit for bit.",
  AGREES: "Agrees: every number within a billionth relative: the last bits another platform rounds its own way.",
  "NOT COMPARABLE": "Not comparable: a PPO run retrained on another platform than its own; training amplifies last-bit differences, so it is checked statistically instead.",
  // launch
  n_episodes: "Episodes: more episodes shrink the interval (by the square root).",
};
