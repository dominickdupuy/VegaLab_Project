/**
 * One-line explanations for the Power lab, Sweep, Attribution and Leak detector pages:
 * what a control or chart is and how to read it. Merged into the site glossary under
 * each page's name (lib/glossary.ts), so a page asks for `power.surface` or
 * `sweep.view`; a term these pages share with others (seeds, the floor, the greeks)
 * is defined there once.
 */

export const POWER_TIPS = {
  preset: "Start from a measured design: the PPO contrast or the fixed-delta seller's, both with the variance the pilot measured.",
  unit: "Show effects in vol points of planted premium, or in dollars of mean P&L per episode (1 vol pt = $824.3, the seller's contrast at one vol point).",
  reset: "Back to the confirmatory design: the PPO contrast, 5 seeds of 500 episodes, alpha 0.05, 80% power.",
  effect: "The premium you want to be able to see: the size of the edge, in vol points of planted premium (or dollars per episode).",
  target: "How often the design must catch the effect when it is there. 80% is the convention.",
  steps: "PPO training steps per seed. Zero for a scripted arm, which needs no training.",
  trainRate: "Training throughput measured on the pilot machine, steps per second per core.",
  evalRate: "Evaluation throughput measured on the pilot machine, steps per second per core.",
  cores: "Cores working in parallel; divides the wall-clock time, not the work.",
  surface: "The chance of detecting each premium (left axis) with each number of seeds (right axis), at your N. The orange line is where power reaches the target; the dot is your design.",
  premiumAxis: "How far the premium axis of the surface reaches, in vol points.",
  seedAxis: "How many seeds the surface's seed axis reaches.",
  powerCurve: "Detection probability against the size of the effect, for your S and for S = 5 and S = 50. Where a curve crosses the target line is that design's floor.",
  floorCurve: "The smallest effect detected at the target power, against the number of seeds, at your N. Lower is better.",
  frontier: "For each seed count, the fewest episodes per seed that reach the target power at your effect, and what that costs in compute.",
  episodesNeeded: "Episodes per seed needed to reach the target power at your effect with this many seeds. Infeasible when seed-to-seed noise alone is too large.",
  compute: "Core-hours to train and evaluate every seed of this design at the measured rates.",
} as const;

export const SWEEP_TIPS = {
  view: "Means: each arm's P&L at each premium. Contrasts: each arm against itself at zero premium, seed block by block. Power: how often each arm's edge is detected, from the analysis.",
  arm: "Show one arm or all of them.",
  sweep: "Which grid to read. The confirmatory grid is the pre-registered one; the pilot measured the variance it was sized on.",
  sweepPower: "The power analysis reads the confirmatory grid only.",
  means: "Mean P&L per episode at each planted premium, the grand mean over seed blocks, with ± one standard error.",
  contrasts: "Each dot is one seed block's P&L at premium p minus the same block at premium 0; the line joins their means. Spread between dots is the between-seed noise.",
  event: "The pre-registered test: one-sided t on the block contrasts. Detected when the lower bound is above zero.",
  cells: "One row per run of the grid: arm, premium and seed block, with its mean P&L, cost and trades.",
  powerCurve: "Detection probability against planted premium, measured from the grid; the band is its exact interval. The shaded region is where the real SPX premium lies.",
  realPremium: "Each arm's power read off its curve at the real SPX premium and at both ends of its interval.",
  decisions: "The test at each premium, per arm: estimate, standard error, t, the lower bound and the verdict.",
  homogeneity: "Whether between-seed noise is the same at every premium (Brown-Forsythe). The power curve assumes it is.",
  bootstrap: "A resampling check of the t test: does it reject as often as theory says?",
} as const;

export const ATTRIBUTION_TIPS = {
  edge: "The edge is an arm's P&L at premium p minus the same arm at premium 0, seed block by seed block, split into the greeks that earned it.",
  arm: "Which arm the details below are about.",
  grid: "Which grid's attribution file to read.",
  whisker: "Whiskers: the 95% interval of each component across the seed blocks. A component whose whisker clears zero carries the edge.",
  pnlEdge: "The P&L edge itself, re-simulated; the components sum to it.",
  acrossPremiums: "The same split at every premium, stacked by sign, with the P&L edge as a diamond. Click a premium to read it above.",
  structures: "The edge charged to the structure held over each step: which trades earned it.",
  mix: "Which structures each arm holds, as a share of the steps, at each premium.",
  table: "Every arm and premium: the P&L edge from run.json and the re-simulated components.",
  cells: "One row per run: its mean P&L, the residual share and whether the re-simulation reproduces run.json.",
  reproduction: "The attribution re-simulates every run; bit for bit, within 1e-9, or with episodes beyond it.",
} as const;

export const LEAK_TIPS = {
  eps: "How much of tomorrow's return is planted into today's observation. 0.5 means half of it: a large, obvious leak.",
  seeds: "How many 63-day paths the detector reads. More paths make a small leak easier to see.",
  run: "Runs the detector on a clean and a poisoned simulator, and plants the leak in every field in turn (about ten seconds).",
  caught: "The poisoned simulator must be flagged: some field correlates with tomorrow's return beyond the threshold.",
  everyField: "A leak planted into each observation field alone must be caught, and on that field. A detector that never fired on a field cannot vouch for it.",
  quiet: "The clean simulator must not be flagged: no field may correlate with tomorrow's return beyond the threshold.",
  strength: "How strongly each observation field predicts tomorrow's return, as |t|. Past the threshold line: flagged. Bonferroni across the fields tested, alpha 0.01.",
  threshold: "The |t| a field must exceed to be flagged: alpha 0.01 split across the fields tested.",
  floor: "The smallest planted leak the detector still catches, found by planting smaller and smaller ones.",
  field: "Plant the leak into one field only, sized in that field's own standard deviation, and see whether the detector names it.",
} as const;

