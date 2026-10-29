/**
 * The Runs, Compare and Launch payloads added on top of `api.ts`: each mirrors a pydantic
 * model in optspread/webapi.py or webviews.py field for field.
 */
import { type Capital, type Compare, type Estimate, get, type MixSummary, type RunActionMix, type RunFilters, request, type Spec } from "./api";

type Params = Parameters<typeof get>[1];

// ---------- P&L by premium per arm, and each arm against itself at zero ----------

export interface World {
  id: string;
  label: string;
  source: string;
  n_runs: number;
  arms: string[];
}

export interface PremiumGroup {
  arm: string;
  /** A learned arm's training budget; "" for a scripted arm. */
  variant: string;
  world: string;
  source: string;
  premium: number;
  n_runs: number;
  n_episodes: number;
  mean: number | null;
  se: number | null;
  ci_low: number | null;
  ci_high: number | null;
  interval: "runs" | "episodes";
  p5: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p95: number | null;
  run_keys: string[];
}

export interface PremiumContrast {
  arm: string;
  variant: string;
  world: string;
  source: string;
  premium: number;
  n_blocks: number;
  n_pairs: number;
  estimate: number | null;
  se: number | null;
  df: number | null;
  ci_low: number | null;
  ci_high: number | null;
  lower_bound: number | null;
  detected: boolean;
  interval: "runs" | "episodes" | "";
  reason: string;
}

export interface Premiums {
  n_runs: number;
  n_skipped: number;
  worlds: World[];
  groups: PremiumGroup[];
  contrasts: PremiumContrast[];
}

// ---------- the catalog: a designed matrix of runs ----------

export interface CatalogRun {
  key: string;
  label: string;
  settings: Record<string, unknown>;
  present: boolean;
  status: string;
  policy: string;
  premium: number | null;
  source: string;
  n_episodes: number | null;
  mean_pnl: number | null;
  se_pnl: number | null;
}

export interface CatalogPair {
  a: string;
  b: string;
  title: string;
  varies: string[];
  premium: number | null;
  /** b − a on the seeds both ran, with its two-sided 95% interval. */
  n_shared: number;
  diff_mean: number | null;
  diff_ci_low: number | null;
  diff_ci_high: number | null;
  clear: boolean;
}

export interface CatalogGroup {
  group: string;
  question: string;
  runs: CatalogRun[];
  pairs: CatalogPair[];
}

export interface Catalog {
  available: boolean;
  file: string;
  seed: number | null;
  groups: CatalogGroup[];
  labels: Record<string, string>;
  questions: Record<string, string>;
  groups_of: Record<string, string>;
  problem: string;
}

// ---------- capital, with the re-simulated equity and margin ----------

export interface Band {
  mean: (number | null)[];
  p10: (number | null)[];
  p50: (number | null)[];
  p90: (number | null)[];
}

export type CapitalPaths = Capital & {
  path_seeds: number[];
  paths: number[][];
  equity_band: Band | null;
  margin_band: Band | null;
  file: string;
};

// ---------- the action mix of one run, with its holdings ----------

export interface Holding {
  seed: number;
  structure: string;
  open_day: number;
  exit_day: number;
  days: number;
  exit: "policy" | "expiry" | "margin" | "horizon" | string;
  tenor_days: number;
  short_delta: number | null;
  long_delta: number | null;
  credit: number;
  margin: number;
  strikes: number[];
}

export interface MixDetail {
  n_holdings: number;
  days_held: Record<string, number[]>;
  holdings: Holding[];
  example: { seed: number; pnl: number; held: string[]; events: string[]; holdings: Holding[] } | null;
}

export interface MixView {
  state: "cached" | "lean" | "computing" | "absent" | "failed";
  can_compute: boolean;
  reason: string;
  file: string;
  mix: RunActionMix | null;
  detail: MixDetail | null;
  seconds: number | null;
}

export type { MixSummary };

// ---------- compare, paired ----------

export type PairedCompare = Compare & {
  diff_df: number | null;
  diff_ci_low: number | null;
  diff_ci_high: number | null;
  b_better_share: number | null;
  diffs: { seed: number; a: number; b: number; diff: number }[];
  verdict: string;
  a_label: string;
  b_label: string;
  group: string;
  question: string;
};

// ---------- the launch preview ----------

export interface PreviewLeg {
  right: string;
  strike: number;
  qty: number;
  price: number;
  iv: number;
  delta: number;
  moneyness: number;
}

export interface DayZero {
  policy: string;
  reference: boolean;
  structure: string;
  plain: string;
  spot: number;
  tenor_days: number | null;
  legs: PreviewLeg[];
  credit: number | null;
  margin: number | null;
  max_loss: number | null;
  breakevens: number[];
}

export interface Peer {
  key: string;
  policy: string;
  premium: number | null;
  source: string;
  n_episodes: number | null;
  mean_pnl: number | null;
  relation: "same run" | "control" | "same arm" | "other arm";
  paired: boolean;
  note: string;
}

export interface Preview {
  ok: boolean;
  errors: string[];
  run_id: string;
  config_hash: string;
  exists: boolean;
  run_key: string;
  world: string;
  implied_vol: number | null;
  day0: DayZero | null;
  day0_note: string;
  estimate: Estimate | null;
  peers: Peer[];
}

export const runsApi = {
  premiums: (f: RunFilters, s?: AbortSignal) => get<Premiums>("/api/runs/premiums", f as Params, s),
  catalog: (s?: AbortSignal) => get<Catalog>("/api/runs/catalog", undefined, s),
  capital: (key: string, s?: AbortSignal) => get<CapitalPaths>("/api/runs/capital", { key }, s),
  mix: (key: string, s?: AbortSignal) => get<MixView>("/api/runs/mixview", { key }, s),
  computeMix: (key: string) => request<MixView>("POST", `/api/runs/mixview?key=${encodeURIComponent(key)}`),
  compare: (a: string, b: string, group?: string, s?: AbortSignal) => get<PairedCompare>("/api/runs/compare", { a, b, group }, s),
  preview: (spec: Spec, s?: AbortSignal) => request<Preview>("POST", "/api/launch/preview", spec, s),
};
