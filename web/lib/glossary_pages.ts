/**
 * One-line explanations for the Power lab, Sweep, Attribution, Leak detector and Synthetic alpha pages:
 * what a control or chart is and how to read it. Merged into the site glossary under
 * each page's name (lib/glossary.ts), so a page asks for `power.surface` or
 * `sweep.view`; a term these pages share with others (seeds, the floor, the greeks)
 * is defined there once.
 */

export const POWER_TIPS = {
  preset: "Start from a measured design: the PPO learner's contrast or the fixed-delta seller's, with the noise (τ² and σ²) the pilot (8 seeds of 200 episodes) measured for that strategy, as amendment A7 of the pre-registration records it. It stays chosen while the noise is that strategy's, whatever the effect and seeds. Choosing one changes only the noise (and the training steps: the seller trains none); the effect, seeds, episodes, false-alarm rate and target power stay as you set them.",
  noise: "The two kinds of noise every number here comes from: seed-to-seed noise (τ²), how much a seed's average moves from one seed to the next, and episode noise (σ²), how much single episodes scatter. More episodes shrink only σ²'s part; only more seeds shrink τ²'s.",
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
  seedsNeeded: "Seeds needed to reach the target power at your effect with this many episodes per seed, when no number of episodes is enough at your seeds; with what that design costs in compute.",
  compute: "Core-hours to train and evaluate every seed of this design at the measured rates.",
} as const;

export const SWEEP_TIPS = {
  view: "Profit by premium: each arm's P&L at each premium. Edge over zero premium: each arm against itself at zero premium, seed block by block. How often it's caught: how often each arm's edge is detected, measured from the sweep (the Power lab predicts it for a design you set).",
  arm: "Show one arm or all of them.",
  sweep: "Which grid to read. The confirmatory grid is the pre-registered one; the pilot measured the variance it was sized on.",
  sweepPower: "The power analysis reads the confirmatory grid only.",
  rate: "The interest rate the grid's options were priced at. The confirmatory study ran at 4% on a market path with no drift, which puts a small loss into some structures by construction (about $406 a put credit spread, $95 an iron condor); each arm against itself at zero premium mostly cancels it, and re-scored in the fair market (rate 0) no verdict changes (amendment A11). Runs launched now default to rate 0, so their numbers differ slightly from the study's.",
  means: "Mean P&L per episode at each planted premium, the grand mean over seed blocks, with ± one standard error.",
  contrasts: "Each dot is one seed block's P&L at premium p minus the same block at premium 0; the line joins their means. Spread between dots is the between-seed noise.",
  event: "The pre-registered test: one-sided t on the block contrasts. Detected when the lower bound is above zero.",
  cells: "One row per run of the grid: arm, premium and seed block, with its mean P&L, cost and trades.",
  powerCurve: "Detection probability against planted premium, measured from the grid; the band is its exact interval. The shaded region is where the real SPX premium lies.",
  medianUnbiased: "Median-unbiased: the line is as likely to sit above the true power as below it. From 5 seeds it can still be far off either way; the band shows how far.",
  realPremium: "Each arm's power read off its curve at the real SPX premium and at both ends of its interval.",
  decisions: "The test at each premium, per arm: estimate, standard error, t, the lower bound and the verdict.",
  homogeneity: "Whether between-seed noise is the same at every premium (Brown-Forsythe). The power curve assumes it is.",
  bootstrap: "A resampling check of the t test: does it reject as often as theory says?",
  estimate: "Estimate: the profit the premium adds, the mean over seed blocks of the arm at premium p minus the same arm at 0, in dollars per episode.",
  lowerBound: "Lower bound: the one-sided 95% lower bound on the estimate, estimate − critical t × se. The edge is detected when it is above zero.",
  critical: "Critical t: the score the test must pass, the one-sided 95% point of a t distribution with S − 1 degrees of freedom.",
  pluginPower: "Plug-in power: the chance of detecting this premium, computed by plugging the estimate and its noise into the power formula; with few seeds it is rough.",
  brownForsythe: "Brown-Forsythe W: tests whether the seed-to-seed spread is the same at every premium; a small p rejects that.",
  bootRate: "How often the bootstrap rejects at this premium, over the resamples.",
  analytic: "How often the t test should reject at this premium, by theory.",
  lift: "Lift: this arm's mean P&L minus its own at the first premium of the sweep.",
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
  eps: "eps: the share of tomorrow's move planted into what the trader sees today. 0.5 is half of it, a large, obvious leak; 0.1 is a tenth.",
  seeds: "How many 63-day paths the detector reads. More paths make a small leak easier to see.",
  run: "Runs the detector on a clean and a poisoned copy of the market, and plants the leak in every field in turn (about ten seconds).",
  caught: "The poisoned copy must be flagged: some field predicts tomorrow's return beyond the alarm line (Bonferroni, alpha 0.01).",
  everyField: "A leak planted into each observation field alone must be caught, and on that field. A detector that never fired on a field cannot vouch for it.",
  quiet: "The clean copy must not be flagged: no field may predict tomorrow's return beyond the alarm line, and resampling the future must leave the past unchanged. A field that never moves in the clean market has nothing to test there.",
  strength: "Signal strength: how well each field predicts tomorrow's return, as |t|. Past the alarm line: flagged. The line is Bonferroni across the fields tested, alpha 0.01, so the chance a clean field crosses it by luck is under 1%.",
  threshold: "The alarm line: the |t| a field must exceed to be flagged, alpha 0.01 split across the fields tested (Bonferroni).",
  floor: "The smallest planted leak the detector still catches, found by planting smaller and smaller ones (eps 1 down to 0.025).",
  field: "Plant the leak into one field only, sized in that field's own standard deviation, and see whether the detector names it.",
} as const;


export const ALPHA_TIPS = {
  arm: "Arm: the strategy being tested. A learner (PPO, A2C, DQN) trains on the train paths first; a scripted rule needs no training and is scored on the same paths.",
  oracle: "Oracle: the best simple rule for this kind of world, picked from a fixed menu of scripted strategies on 200 fresh paths the agent never sees. What it earns is the edge there was to take; if no rule beats flat by 2 se, flat is the oracle and there is no edge.",
  twin: "Noise twin: a copy of every path with the same random shocks and the same costs, but nothing planted. Signal minus twin is what the planted edge paid; anything earned on the twin alone is luck.",
  n_train: "Train paths: the frozen paths a learner trains on, over and over. Its score on them is compared with its score on the held-out paths.",
  n_heldout: "Held-out paths: frozen paths the agent never trains on. Every comparison is scored on them unless a row says train.",
  capture: "Capture ratio: the agent's edge from the signal divided by the attainable edge, so the share of the planted edge it took. 100% matches the best simple rule; 0% took none of it.",
  overfit: "Overfit gap: the agent's profit on its own train paths minus its profit on the held-out paths. Well above zero means it learned those paths, not the edge.",
  noiseFit: "Noise fit: a copy of the learner trained on the noise twin, its profit on its train paths minus on held-out ones. There is nothing to learn on the twin, so anything above zero is memorised noise.",
  attainable: "Attainable edge: the oracle's profit on the signal world minus on its noise twin, per episode. The edge a simple rule can take, and what the capture ratio divides by.",
  captured: "Captured edge (the agent's edge from the signal): the agent's profit on the signal world minus on its noise twin, path by path. Costs and path luck cancel, so what is left came from the planted signal.",
  budget: "Training budget: how many market steps each learner trains for, per seed, before it is scored. The copy trained on the noise twin gets the same budget.",
  seeds: "Training seeds: how many times the learner is trained from scratch on the same frozen paths. One training run can be lucky or unlucky, so with one seed the se counts path noise only.",
  unconditional: "Unconditional: the right trade is the same every day, so a strategy can take the edge without reading anything it observes. The agent only has to find the trade, not when to make it.",
  z: "z: the value divided by its se. Beyond 2 either way the page calls it real; inside that, it could be chance.",
} as const;
