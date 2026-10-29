/**
 * The one typed client for optspread.server.
 *
 * Every shape here mirrors a pydantic model in optspread/webapi.py field for field;
 * the server's own tests pin those models, so a drift shows up as a type error here
 * or a failing test there. Undefined figures arrive as null, never NaN.
 */

import { recordApi } from "./activity";

// ---------- runs ----------

export interface RunRow {
  key: string;
  run_id: string;
  config_hash: string;
  seed: number | null;
  policy: string;
  premium: number | null;
  source: string;
  source_recorded: boolean;
  panel: string;
  git_sha: string;
  commit: string;
  git_dirty: boolean | null;
  created_at: string;
  n_episodes: number | null;
  mean_pnl: number | null;
  se_pnl: number | null;
  t_stat: number | null;
  mean_cost: number | null;
  mean_trades: number | null;
  sweep: string;
  status: "OK" | "UNVERIFIED" | "INVALID" | "DIGEST MISMATCH" | string;
  digests_ok: boolean | null;
  problems: string[];
}

export interface RunList {
  root: string;
  rows: RunRow[];
  facets: {
    config_hashes: string[];
    seeds: number[];
    policies: string[];
    premiums: number[];
    commits: string[];
    sweeps: string[];
    sources: string[];
  };
  trials: { n_runs: number; n_ok: number; n_configs: number; n_seeds: number };
  n_shown: number;
}

export interface Risk {
  n: number;
  mean: number | null;
  se: number | null;
  sortino: number | null;
  cvar5: number | null;
  max_drawdown: number | null;
  hit_rate: number | null;
  turnover: number | null;
}

export interface Histogram {
  edges: number[];
  counts: number[];
}

export interface RunDetail {
  row: RunRow;
  manifest: Record<string, unknown>;
  episodes: { seed: number; pnl: number; cost: number; n_trades: number; steps: number }[];
  risk: Risk;
  histogram: Histogram;
  equity: { file: string; seeds: number[]; paths: number[][]; n_recorded: number } | null;
  artifacts: { path: string; recorded: string; actual: string; ok: boolean }[];
  provenance: { run_dir: string; manifest: string; results: string; replicas: number; sweep_cells: number };
}

export interface Compare {
  a: RunDetail;
  b: RunDetail;
  config_diff: { key: string; a: unknown; b: unknown }[];
  shared_seeds: number;
  diff_mean: number | null;
  diff_se: number | null;
  diff_t: number | null;
  diff_histogram: Histogram;
}

// ---------- jobs ----------

export interface Knob {
  name: string;
  kind: "number" | "integer" | "boolean" | "choice" | "tenors";
  default: unknown;
  minimum: number | null;
  maximum: number | null;
  exclusive_minimum: number | null;
  exclusive_maximum: number | null;
  choices: string[];
  group: string;
  help: string;
  /** What the form calls it. */
  label: string;
  /** Read by the generator only: a real chain brings its own surface, path and carry. */
  synthetic_only: boolean;
}

export type Spec = Record<string, unknown>;

export interface Preset {
  name: string;
  label: string;
  description: string;
  spec: Spec;
}

export interface JobOptions {
  arms: string[];
  ppo_available: boolean;
  /** The data a job can run on here: synthetic always, optionmetrics when the panel is present. */
  sources: string[];
  /** Where a job on each source lands, relative to the runs root. */
  landing: Record<string, string>;
  /** A8: the real data's train (PPO) and eval dates, ISO [first, last]. */
  splits: Record<string, [string, string]>;
  knobs: Knob[];
  presets: Preset[];
  defaults: Spec;
  ppo_preset: Preset | null;
}

export interface Estimate {
  eval_steps: number;
  train_steps: number;
  eval_hours: number;
  train_hours: number;
  total_hours: number;
  train_rate: number;
  eval_rate: number;
  source: string;
}

export interface Job {
  job_id: string;
  state: "queued" | "running" | "done" | "failed";
  phase: string;
  done: number;
  total: number;
  fraction: number;
  run_id: string;
  run_key: string;
  created_at: string;
  policy: string;
  error: string;
  log_tail: string[];
}

// ---------- power lab ----------

export interface PowerParams {
  tau2: number;
  sigma2: number;
  n_seeds: number;
  n_episodes: number;
  delta: number;
  alpha: number;
  target_power: number;
  train_steps: number;
  episode_days: number;
  train_rate: number;
  eval_rate: number;
  cores: number;
}

export interface Design {
  n_seeds: number;
  n_episodes: number | null;
  power: number | null;
  train_hours: number | null;
  eval_hours: number | null;
  total_hours: number | null;
}

export interface Power {
  params: PowerParams;
  se: number;
  var_grand_mean: number;
  variance_budget: number;
  power: number;
  floor: number;
  episodes_needed: number | null;
  seeds_needed: number;
  curve: { delta: number[]; power: number[]; power_s5: number[]; power_s50: number[] };
  floor_vs_seeds: { seeds: number[]; floor: number[]; floor_s5: number; floor_s50: number; ratio: number };
  frontier: Design[];
  design: Design;
  source: string;
}

// ---------- leak demo ----------

export interface LeakSide {
  flagged: boolean;
  worst_feature: string;
  worst_t: number | null;
  p_value: number | null;
  threshold: number;
  n_pairs: number;
  n_tested: number;
  past_unchanged: boolean;
  features: { name: string; tested: boolean; t_stat: number | null; p_value: number | null; reason: string }[];
  /** The |t| a field must exceed to be flagged. */
  t_critical?: number | null;
}

export interface LeakDemo {
  eps: number;
  seeds: number[];
  clean: LeakSide;
  poisoned: LeakSide;
  caught: boolean;
  seconds: number;
  criterion: string;
  source: string;
}

// ---------- sweeps ----------

export interface SweepSummary {
  key: string;
  sweep_hash: string;
  premiums: number[];
  arms: string[];
  n_cells: number;
  seed: number;
  n_episodes: number;
  kind: "sweep" | "grid";
  n_seeds: number;
  n_present: number;
}

export interface GridCell {
  premium: number;
  arm: string;
  block: number;
  run_id: string;
  key: string;
  train_seed: number | null;
  mean_pnl: number | null;
  se_pnl: number | null;
  mean_cost: number | null;
  mean_trades: number | null;
  present: boolean;
}

export interface ArmMean {
  premium: number;
  mean: number | null;
  se: number | null;
  n_blocks: number;
  block_means: (number | null)[];
}

export interface Contrast {
  premium: number;
  estimate: number | null;
  se: number | null;
  t: number | null;
  df: number;
  critical: number | null;
  lower_bound: number | null;
  detected: boolean;
  block_means: (number | null)[];
  reason: string;
}

export interface GridDetail {
  grid_hash: string;
  n_seeds: number;
  n_episodes: number;
  premiums: number[];
  arms: string[];
  total_timesteps: number | null;
  first_path_seed: number;
  costs: boolean;
  cells: GridCell[];
  means: Record<string, ArmMean[]>;
  contrasts: Record<string, Contrast[]>;
  missing: string[];
  git_shas: string[];
}

export interface SweepList {
  root: string;
  sweeps: SweepSummary[];
}

export interface Sweep {
  summary: SweepSummary;
  oracle: Record<string, { premium: number; mean_pnl: number; se_pnl: number; lift: number; run_id: string }[]>;
  source: string;
  kind: "sweep" | "grid";
  grid: GridDetail | null;
}

// ---------- the SPX panel ----------

export interface DataStatus {
  root: string;
  built: boolean;
  missing: string[];
  years: number[];
  files: { path: string; present: boolean; modified: string | null }[];
}

export interface Span {
  source: string;
  first: string;
  last: string;
  n_days: number;
  n_missing: number;
  median_contracts: number;
}

export interface Coverage {
  spans: Span[];
  missing: [string, string][];
  overlap: [string, string, number] | null;
  strips: { source: string; date: string[]; n_contracts: number[]; missing: boolean[] }[];
  source: string;
}

export interface ChainIndex {
  date: string;
  sources: string[];
  roots: string[];
  expiries: { expiry: string; dte: number; n_quotes: number; forward: number | null; carry_fit: string }[];
  default_expiry: string | null;
  source: string;
}

export interface Smile {
  date: string;
  expiry: string;
  dte: number;
  data_source: string;
  moneyness: (number | null)[];
  strike: (number | null)[];
  iv: (number | null)[];
  iv_vendor: (number | null)[];
  cp: string[];
  has_vendor: boolean;
  forward: number | null;
  source: string;
}

export interface Surface {
  date: string;
  data_source: string;
  moneyness: number[];
  dte: number[];
  iv: (number | null)[][];
  source: string;
}

export interface Term {
  date: string;
  data_source: string;
  expiry: string[];
  dte: number[];
  atm_iv: (number | null)[];
  source: string;
}

export interface Series {
  data_source: string;
  date: string[];
  spot: (number | null)[];
  spot_parity: (number | null)[];
  iv30: (number | null)[];
  rv21: (number | null)[];
  rv21_fwd: (number | null)[];
  vrp30: (number | null)[];
}

export interface VrpStats {
  n_days: number;
  mean_vol_pts: number | null;
  median_vol_pts: number | null;
  share_positive: number | null;
  first: string | null;
  last: string | null;
}

export interface Daily {
  series: Series[];
  stats: Record<string, VrpStats>;
  source: string;
}

export interface Seam {
  available: boolean;
  date: string[];
  stats: Record<string, (number | null)[]>;
  medians: Record<string, number | null>;
  /** The headline: each overlap day's 95th-percentile |IV gap|, averaged over the days (docs/DATA.md's figure). */
  p95_mean: number | null;
  /** The worst day's. */
  p95_worst: number | null;
  iv30_sources: string[];
  iv30_date: string[];
  iv30_diff_vol_pts: (number | null)[];
  source: string;
}

// ---------- the 3D views ----------

export interface Grid {
  date: string;
  k: number[];
  tenor_days: number[];
  iv: (number | null)[][];
  source: string;
}

export interface GridFrames {
  k: number[];
  tenor_days: number[];
  frames: { date: string; iv: (number | null)[][] }[];
  more: boolean;
  source: string;
}

export type SigmaFrom = "config" | "rv21" | "atm";

export interface Simulated {
  date: string;
  spot: number;
  spot_from: string;
  sigma: number;
  sigma_from: SigmaFrom;
  premium: number;
  level: number;
  r: number;
  strike: number[];
  tenor_trading_days: number[];
  tenor_days: number[];
  k: number[][];
  iv: number[][];
  real: (number | null)[][];
  mean_diff: number | null;
  mean_abs_diff: number | null;
  max_abs_diff: number | null;
  source: string;
}

export interface Landscape {
  date: string[];
  tenor_days: number[];
  iv: (number | null)[][];
  stride: number;
  n_days: number;
  first: string | null;
  last: string | null;
  source: string;
}

export interface PowerSurfaceParams {
  tau2: number;
  sigma2: number;
  n_episodes: number;
  delta: number;
  alpha: number;
  target_power: number;
  max_premium: number;
  max_seeds: number;
}

export interface PowerSurface {
  params: PowerSurfaceParams;
  premium: number[];
  seeds: number[];
  power: (number | null)[][];
  contour_premium: (number | null)[];
  source: string;
}

// ---------- the study and its analyses ----------

export interface Study {
  test: string;
  alpha: number;
  event: string;
  n_seeds: number;
  n_episodes: number;
  train_steps: number;
  episode_days: number;
  premiums: number[];
  arms: string[];
  grid_hash: string;
  delta_per_vol_pt: number;
  tau2: number;
  sigma2: number;
  seller_tau2: number;
  seller_sigma2: number;
  target_power: number;
  power: number;
  floor: number;
  floor_vol_pts: number;
  train_rate: number;
  eval_rate: number;
  core_hours: number;
  doc: string;
  source: string;
  /** The same design at the sweep's own variances (M4_ANALYSIS.json design_check); null until it is written. */
  measured: MeasuredDesign | null;
}

export interface MeasuredDesign {
  arm: string;
  premium: number;
  tau2: number;
  sigma2: number;
  power: number;
  floor: number;
  floor_vol_pts: number;
  file: string;
}

export interface Check {
  name: string;
  /** What is checked, in plain words: the name the page shows. */
  label: string;
  verdict: string;
  failed_first: string;
  criterion: string;
  evidence: string;
  owner: string;
  /** The readable report: its markdown when there is one. */
  report: string;
  sidecar: string;
  report_date: string;
}

export interface VrpEstimate {
  label: string;
  n_days: number;
  first: string;
  last: string;
  mean: number | null;
  mean_low: number | null;
  mean_high: number | null;
  median: number | null;
  median_low: number | null;
  median_high: number | null;
  share_positive: number | null;
  block: number | null;
}

export interface VrpMapping {
  generated: string;
  source: string;
  source_sha256: string;
  definition: Record<string, unknown>;
  bootstrap: Record<string, unknown>;
  headline: VrpEstimate;
  power_inputs: { estimate: number; interval: [number, number] };
  sweep_axis: number[];
  bracket: [number | null, number | null];
  sensitivity: VrpEstimate[];
  blocks: VrpEstimate[];
  periods: VrpEstimate[];
  years: VrpEstimate[];
}

export interface Vrp {
  available: boolean;
  file: string;
  mapping: VrpMapping | null;
  error: string;
}

export interface CurvePoint {
  premium: number;
  measured: boolean;
  noncentrality: number | null;
  noncentrality_lo: number | null;
  noncentrality_hi: number | null;
  power: number | null;
  power_lo: number | null;
  power_hi: number | null;
  power_plugin: number | null;
}

export interface M4Decision {
  premium: number;
  estimate: number | null;
  se: number | null;
  t: number | null;
  df: number;
  critical: number | null;
  lower_bound: number | null;
  detected: boolean;
  t_interval: [number | null, number | null];
  power_plugin: number | null;
  seed_means: (number | null)[];
  components: Record<string, number | boolean | null>;
}

export interface Homogeneity {
  test: string;
  premiums: number[];
  statistic: number | null;
  df: [number, number];
  p_value: number | null;
  rejected: boolean;
  levels: { premium: number; tau2: number | null; sigma2_over_n: number | null; seed_mean_variance: number | null }[];
}

export interface BootLevel {
  premium: number;
  rate: number | null;
  analytic: number | null;
  gap: number | null;
  reference: [number | null, number | null];
  p_value: number | null;
  threshold: number | null;
  agrees: boolean;
  t_interval: [number | null, number | null];
  boot_interval: [number | null, number | null];
  width_ratio: number | null;
  reference_width_ratio: number | null;
  t_lower: number | null;
  boot_lower: number | null;
  boot_detected: boolean;
}

export interface M4Arm {
  role: "oracle" | "learned" | "other";
  decisions: M4Decision[];
  curve: { points: CurvePoint[]; interpolated: { premium: number; power: number | null; power_lo: number | null; power_hi: number | null }[] };
  floor: {
    premium: number | null;
    lo: number | null;
    hi: number | null;
    status: "reached" | "below_grid" | "not_reached";
    bracket: [number | null, number | null];
    recrossings: number[];
    max_power: number | null;
  };
  homogeneity: { contrast?: Homogeneity; arm?: Homogeneity };
  bootstrap: { rule_in_force: "t" | "bootstrap"; diverged_at: number[]; levels: BootLevel[] };
}

export interface M4Analysis {
  schema_version: number;
  report_date: string;
  confirmatory: boolean;
  deviations: string[];
  provenance: {
    root: string;
    grid_file: string;
    grid_hash: string;
    confirmatory_grid_hash: string;
    analysis_git_sha: string;
    analysis_git_dirty: boolean;
    sweep_git_shas: string[];
    sweep_git_dirty: boolean;
    cells_total: number;
    cells_used: number;
  };
  design: { n_seeds: number; n_episodes: number; blocks: number[]; premiums: number[]; arms: string[]; total_timesteps: number | null; costs: boolean };
  settings: { alpha: number; df: number; critical: number; target_power: number; curve_level: number; n_boot: number; reference_replications: number };
  arm_order: string[];
  arms: Record<string, M4Arm>;
  gap: null | { oracle: string; learned: string; value: number | null; kind: "point" | "lower_bound" | "undefined"; band_range: [number | null, number | null]; bracket: [number | null, number | null] };
  bootstrap_rule_size: { n_seeds: number; replications: number; size: number | null; mc_se: number | null };
}

export interface MappedPower {
  arm: string;
  premium: number;
  low: number;
  high: number;
  power: number | null;
  power_low: number | null;
  power_high: number | null;
  clipped: boolean;
  reason: string;
}

export interface M4 {
  available: boolean;
  file: string;
  analysis: M4Analysis | null;
  vrp: { estimate: number; interval: [number, number]; label: string; file: string } | null;
  mapped: MappedPower[];
  waiting_for: string;
  error: string;
}

// ---------- one run: attribution, capital, replay ----------

export const COMPONENTS = ["delta", "gamma", "vega", "theta", "carry", "cost", "residual"] as const;
export type Component = (typeof COMPONENTS)[number];
export type Terms = Record<Component | "pnl" | "market", number | null>;

export interface AttributionResult {
  run_id: string;
  config_hash: string;
  policy: string;
  premium_vol_pts: number;
  source: string;
  run_dir: string | null;
  reproduces: boolean | null;
  /** Every episode within PNL_RTOL of run.json, same trades and steps; null with no run, or written before. */
  agrees?: boolean | null;
  totals: Terms;
  gross: number | null;
  abs_residual: number | null;
  residual_share: number | null;
  max_abs_step_residual: number | null;
  episodes: { seed: number; n_steps: number; n_trades: number; reproduces: boolean | null; agrees?: boolean | null; totals: Terms; gross: number | null; abs_residual: number | null; residual_share: number | null }[];
}

export interface Attribution {
  state: "cached" | "computing" | "absent" | "failed";
  can_compute: boolean;
  reason: string;
  file: string;
  result: AttributionResult | null;
  seed: number | null;
  steps: { day: number; spot: number; spot_next: number; action_id: number; n_legs: number; terms: Terms }[];
  seconds: number | null;
}

export interface Capital {
  state: "done" | "computing" | "absent" | "failed";
  can_compute: boolean;
  reason: string;
  initial_cash: number | null;
  n: number;
  reproduces: boolean | null;
  agrees: boolean | null;
  rtol: number | null;
  n_differ: number;
  max_abs_diff: number | null;
  mean_return_on_cash: number | null;
  mean_return_on_margin: number | null;
  mean_peak_margin: number | null;
  max_peak_margin: number | null;
  median_drawdown: number | null;
  worst_drawdown: number | null;
  worst_drawdown_on_cash: number | null;
  method: string;
  episodes: { seed: number; pnl: number; max_drawdown: number; peak_margin: number; return_on_cash: number; return_on_margin: number | null }[];
  seconds: number | null;
}

export interface Replay {
  state: "idle" | "running" | "done" | "failed";
  can_replay: boolean;
  reason: string;
  verdict: "" | "IDENTICAL" | "AGREES" | "DIFFERS";
  /** The tolerance asked for; null for bytes only (a PPO retrain). */
  rtol: number | null;
  max_rel_diff: number | null;
  /** False: a retrained PPO run on another platform than the one it was trained on. */
  comparable: boolean;
  note: string;
  retrains: boolean;
  train_steps: number | null;
  estimate_hours: number | null;
  seconds: number | null;
}

// ---------- the edge of a grid: components and structures (optspread.edge) ----------

export const STRUCTURES = ["FLAT", "IRON_CONDOR", "SHORT_STRANGLE", "PUT_CREDIT_SPREAD"] as const;
export type Reproduction = "reproduces" | "agrees" | "differs";

/** One contrast's A2 test: the estimate, its se and the one-sided 1 - alpha interval. */
export interface Edge {
  estimate: number | null;
  se: number | null;
  t_stat: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
  rejected: boolean;
  block_means: (number | null)[];
}

export interface EdgeCell {
  premium: number;
  arm: string;
  block: number;
  run_id: string;
  seed: number;
  train_seed: number | null;
  n_episodes: number;
  reproduces: boolean;
  agrees: boolean;
  reproduction: Reproduction;
  max_abs_pnl_diff: number | null;
  mean: Terms;
  by_structure: Record<string, Terms>;
  held_share: Record<string, number | null>;
  gross: number | null;
  abs_residual: number | null;
  residual_share: number | null;
  max_abs_step_residual: number | null;
}

export interface ArmEdge {
  arm: string;
  premium: number;
  blocks: { block: number; run_id: string; null_run_id: string; seed: number; train_seed: number | null }[];
  pnl: Edge;
  pnl_saved: Edge;
  components: Record<string, Edge>;
  by_structure: Record<string, Record<string, Edge>>;
  identity_gap: number | null;
  structure_gap: number | null;
  residual_edge_share: number | null;
  residual_share: number | null;
}

export interface GridEdge {
  schema_version: number;
  grid_hash: string;
  grid_file: string;
  source: string;
  arms: string[];
  premiums: number[];
  n_seeds: number;
  n_episodes: number;
  first: number | null;
  alpha: number;
  critical: number;
  components: string[];
  structures: string[];
  reproduction: Reproduction;
  reproduces: boolean;
  agrees: boolean;
  max_abs_pnl_diff: number | null;
  cells: EdgeCell[];
  edges: ArmEdge[];
}

export interface EdgeFile {
  name: string;
  grid_hash: string;
  first: number | null;
  source: string;
  arms: string[];
  premiums: number[];
  n_seeds: number;
  n_episodes: number;
  reproduction: string;
  error: string;
}

export interface EdgeList {
  files: EdgeFile[];
  grids: { grid_hash: string; key: string; has_edge: boolean; edge_command: string; actionmix_command: string }[];
  folder: string;
  command: string;
}

export interface EdgeDetail {
  name: string;
  folder: string;
  edge: GridEdge;
  run_keys: Record<string, string | null>;
  grid_key: string | null;
  actionmix_command: string;
  /** For a cell that differs, how many of its episodes do, from its run's cached attribution. */
  episode_counts: Record<string, { n_episodes: number; n_differ: number }>;
}

// ---------- the action mix (optspread.actionmix) ----------

export interface StructureMix {
  n_opens: number;
  open_share: number | null;
  mean_tenor_days: number | null;
  mean_short_abs_delta: number | null;
  mean_long_abs_delta: number | null;
  mean_credit: number | null;
  mean_margin: number | null;
  mean_holding_days: number | null;
  exits: Record<string, number>;
}

export interface MixSummary {
  n_episodes: number;
  n_steps: number;
  actions: Record<string, number>;
  action_share: Record<string, number | null>;
  held: Record<string, number>;
  held_share: Record<string, number | null>;
  events: Record<string, number>;
  n_opens: number;
  n_refused: number;
  opens_per_episode: number | null;
  refused_per_episode: number | null;
  refused_share: number | null;
  structures: Record<string, StructureMix>;
  holding_days: { mean: number | null; p10: number | null; p50: number | null; p90: number | null; max: number | null } | null;
  exits: Record<string, number>;
  by_day: { day: number; actions: number[]; held: number[]; opens: number; refused: number }[];
}

export interface RunActionMix {
  schema_version: number;
  run_id: string;
  config_hash: string;
  policy: string;
  premium_vol_pts: number;
  source: string;
  run_dir: string | null;
  reproduces: boolean | null;
  agrees: boolean | null;
  max_abs_pnl_diff: number | null;
  action_names: string[];
  summary: MixSummary;
}

export interface GridMixRow {
  premium: number;
  arm: string;
  reproduces: boolean | null;
  agrees: boolean | null;
  summary: MixSummary;
}

export interface GridActionMix {
  schema_version: number;
  grid_hash: string;
  grid_file: string;
  source: string;
  arms: string[];
  premiums: number[];
  n_seeds: number;
  reproduces: boolean;
  agrees: boolean;
  cells: (GridMixRow & { block: number; run_id: string })[];
  pooled: (GridMixRow & { run_ids: string[] })[];
}

export interface ActionMix {
  kind: "run" | "grid";
  path: string;
  run: RunActionMix | null;
  grid: GridActionMix | null;
}

// ---------- leaks: the floor and the fields ----------

export interface LeakFloor {
  eps: number[];
  flagged: boolean[];
  floor: number | null;
  seeds: number[];
  seconds: number;
  source: string;
}

export interface LeakField {
  field: string;
  eps: number;
  scale: number;
  seeds: number[];
  flagged: boolean;
  named_correctly: boolean;
  worst_feature: string;
  p_value: number | null;
  threshold: number;
  features: LeakSide["features"];
  seconds: number;
  source: string;
  t_critical?: number | null;
}

export interface LeakCoverage {
  eps: number;
  seeds: number[];
  clean_flagged: boolean;
  passed: boolean;
  uncovered: string[];
  misattributed: string[];
  controls: { field: string; flagged: boolean; named_correctly: boolean; p_value: number | null }[];
  seconds: number;
  source: string;
}

// ---------- the seller on a real close ----------

export interface Seller {
  date: string;
  structure: string;
  spot: number;
  r: number;
  tenor_trading_days: number;
  tenor_calendar_days: number;
  short_delta: number;
  wing_delta: number;
  legs: { right: string; strike: number; qty: number; price: number; iv: number; delta: number; moneyness: number }[];
  credit: number;
  /** $, defined risk only, as the frozen margin check decides it. */
  max_loss: number | null;
  /** $, the worst expiry loss the legs cap whenever each short has a wing beyond it. */
  capped_loss: number | null;
  /** The legs cap the loss, but the frozen margin check margins them as naked (#39). */
  margined_as_naked: boolean;
  margin: number;
  breakevens: number[];
  note: string;
  source: string;
}

// ---------- overview ----------

export interface Overview {
  data: DataStatus;
  coverage: Span[];
  overlap: [string, string, number] | null;
  n_runs: number;
  n_ok: number;
  latest: RunRow[];
  sweeps: number;
  ppo_available: boolean;
}

export interface Health {
  ok: boolean;
  version: string;
  git_sha: string;
}

// ---------- the client ----------

export class ApiError extends Error {
  constructor(
    public status: number, // 0 = the server could not be reached
    message: string,
    /** What a 404 from the panel names as not built yet. */
    public missing: string[] = [],
  ) {
    super(message);
    this.name = "ApiError";
  }
  /** The panel has not been built: a state to show, not a failure. */
  get notBuilt(): boolean {
    return this.status === 404 && this.message.startsWith("panel not built yet");
  }
}

export type Params = Record<string, string | number | boolean | undefined | null | (string | number)[]>;

function withQuery(path: string, params?: Params): string {
  if (!params) return path;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) for (const item of v) q.append(k, String(item));
    else q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `${path}?${s}` : path;
}

function detailText(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => (typeof d === "string" ? d : JSON.stringify(d))).join("; ");
  return JSON.stringify(detail);
}

export async function request<T>(method: "GET" | "POST", path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "Can't reach the optspread server. Start it with `make serve`.");
  }
  if (!res.ok) {
    let message = res.statusText || `HTTP ${res.status}`;
    let missing: string[] = [];
    try {
      const data: unknown = await res.json();
      if (data && typeof data === "object") {
        const d = data as { detail?: unknown; missing?: unknown };
        if (d.detail !== undefined) message = detailText(d.detail);
        if (Array.isArray(d.missing)) missing = d.missing.map(String);
      }
    } catch {
      /* not JSON: keep the status text */
    }
    throw new ApiError(res.status, message, missing);
  }
  const text = await res.text();
  const data = sameAnswer<T>(method, path, text);
  recordApi(method, path, body, data); // what the workflows check a step's action against
  return data;
}

// The last body of each GET, by URL: an answer the server repeats word for word comes back
// as the very object it was the first time, so a page asked again draws nothing new.
const lastGets = new Map<string, { text: string; data: unknown }>();
const MAX_GETS = 200;

function sameAnswer<T>(method: string, path: string, text: string): T {
  if (method !== "GET") return JSON.parse(text) as T;
  const hit = lastGets.get(path);
  if (hit && hit.text === text) return hit.data as T;
  const data = JSON.parse(text) as unknown;
  lastGets.delete(path);
  lastGets.set(path, { text, data });
  while (lastGets.size > MAX_GETS) {
    const oldest = lastGets.keys().next().value;
    if (oldest === undefined) break;
    lastGets.delete(oldest);
  }
  return data as T;
}

export const get = <T>(path: string, params?: Params, signal?: AbortSignal) => request<T>("GET", withQuery(path, params), undefined, signal);

export interface RunFilters {
  config_hash?: string[];
  seed?: number[];
  policy?: string[];
  premium?: number[];
  commit?: string[];
  sweep?: string[];
  source?: string[];
  q?: string;
}

export interface ChainQuery {
  date: string;
  source?: string;
  roots?: string[];
}

export const api = {
  health: (s?: AbortSignal) => get<Health>("/api/health", undefined, s),
  overview: (s?: AbortSignal) => get<Overview>("/api/overview", undefined, s),
  runs: (f: RunFilters, s?: AbortSignal) => get<RunList>("/api/runs", f as Params, s),
  run: (key: string, s?: AbortSignal) => get<RunDetail>("/api/runs/detail", { key }, s),
  compare: (a: string, b: string, s?: AbortSignal) => get<Compare>("/api/runs/compare", { a, b }, s),
  jobOptions: (s?: AbortSignal) => get<JobOptions>("/api/jobs/options", undefined, s),
  jobs: (s?: AbortSignal) => get<Job[]>("/api/jobs", undefined, s),
  job: (id: string, s?: AbortSignal) => get<Job>(`/api/jobs/${encodeURIComponent(id)}`, undefined, s),
  launch: (spec: Spec) => request<Job>("POST", "/api/jobs", spec),
  power: (p: Partial<PowerParams>, s?: AbortSignal) => get<Power>("/api/power", p as Params, s),
  leakDemo: (eps: number, n_seeds: number) => request<LeakDemo>("POST", "/api/leaks/demo", { eps, n_seeds }),
  sweeps: (s?: AbortSignal) => get<SweepList>("/api/sweeps", undefined, s),
  sweep: (key: string, s?: AbortSignal) => get<Sweep>("/api/sweeps/detail", { key }, s),
  dataStatus: (s?: AbortSignal) => get<DataStatus>("/api/data/status", undefined, s),
  coverage: (s?: AbortSignal) => get<Coverage>("/api/data/coverage", undefined, s),
  dates: (source?: string, s?: AbortSignal) => get<string[]>("/api/data/dates", { source }, s),
  chain: (q: ChainQuery, s?: AbortSignal) => get<ChainIndex>("/api/data/chain", q as unknown as Params, s),
  smile: (q: ChainQuery & { expiry?: string }, s?: AbortSignal) => get<Smile>("/api/data/smile", q as unknown as Params, s),
  surface: (q: ChainQuery & { min_dte?: number; max_dte?: number }, s?: AbortSignal) =>
    get<Surface>("/api/data/surface", q as unknown as Params, s),
  term: (q: ChainQuery, s?: AbortSignal) => get<Term>("/api/data/term", q as unknown as Params, s),
  daily: (start?: string, end?: string, s?: AbortSignal) => get<Daily>("/api/data/daily", { start, end }, s),
  seam: (s?: AbortSignal) => get<Seam>("/api/data/seam", undefined, s),
  gridDates: (s?: AbortSignal) => get<string[]>("/api/data/grid/dates", undefined, s),
  grid: (date: string, s?: AbortSignal) => get<Grid>("/api/data/grid", { date }, s),
  gridFrames: (start: string, max_frames: number, s?: AbortSignal) => get<GridFrames>("/api/data/grid/frames", { start, max_frames }, s),
  simulated: (q: { date: string; premium: number; sigma_from: SigmaFrom }, s?: AbortSignal) => get<Simulated>("/api/data/simulated", q, s),
  landscape: (q: { start?: string; end?: string; max_dates?: number }, s?: AbortSignal) => get<Landscape>("/api/data/landscape", q, s),
  powerSurface: (p: Partial<PowerSurfaceParams>, s?: AbortSignal) => get<PowerSurface>("/api/power/surface", p as Params, s),
  study: (s?: AbortSignal) => get<Study>("/api/study", undefined, s),
  checks: (s?: AbortSignal) => get<Check[]>("/api/checks", undefined, s),
  vrp: (s?: AbortSignal) => get<Vrp>("/api/vrp", undefined, s),
  m4: (s?: AbortSignal) => get<M4>("/api/m4", undefined, s),
  attribution: (key: string, seed?: number, s?: AbortSignal) => get<Attribution>("/api/runs/attribution", { key, seed }, s),
  attribute: (key: string) => request<Attribution>("POST", withQuery("/api/runs/attribution", { key })),
  capital: (key: string, s?: AbortSignal) => get<Capital>("/api/runs/capital", { key }, s),
  computeCapital: (key: string) => request<Capital>("POST", withQuery("/api/runs/capital", { key })),
  replay: (key: string, s?: AbortSignal) => get<Replay>("/api/runs/replay", { key }, s),
  startReplay: (key: string) => request<Replay>("POST", withQuery("/api/runs/replay", { key })),
  leakFields: (s?: AbortSignal) => get<string[]>("/api/leaks/fields", undefined, s),
  leakFloor: (n_seeds: number, eps_grid?: number[]) => request<LeakFloor>("POST", "/api/leaks/floor", { n_seeds, eps_grid }),
  leakField: (field: string, eps: number, n_seeds: number) => request<LeakField>("POST", "/api/leaks/field", { field, eps, n_seeds }),
  leakCoverage: (eps: number, n_seeds: number) => request<LeakCoverage>("POST", "/api/leaks/coverage", { eps, n_seeds }),
  seller: (date: string, s?: AbortSignal) => get<Seller>("/api/data/seller", { date }, s),
  edges: (s?: AbortSignal) => get<EdgeList>("/api/edge", undefined, s),
  edge: (grid: string, first?: number | null, s?: AbortSignal) => get<EdgeDetail>("/api/edge/detail", { grid, first }, s),
  runMix: (run: string, source?: string, s?: AbortSignal) => get<ActionMix>("/api/runs/actionmix", { run, source }, s),
  gridMix: (grid: string, source?: string, s?: AbortSignal) => get<ActionMix>("/api/runs/actionmix", { grid, source }, s),
  launchOptions: (s?: AbortSignal) => get<JobOptions>("/api/launch/options", undefined, s),
  estimate: (q: { n_episodes: number; n_days: number; total_timesteps?: number | null }, s?: AbortSignal) => get<Estimate>("/api/launch/estimate", q, s),
};

/** A report or record in docs/, as the server serves it. */
export const docUrl = (name: string) => `/api/docs/${encodeURIComponent(name)}`;
