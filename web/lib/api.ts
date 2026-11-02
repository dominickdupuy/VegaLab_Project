/**
 * The one typed client for optspread.server.
 *
 * Every shape here mirrors a pydantic model in optspread/webapi.py field for field;
 * the server's own tests pin those models, so a drift shows up as a type error here
 * or a failing test there. Undefined figures arrive as null, never NaN.
 */

// ---------- runs ----------

export interface RunRow {
  key: string;
  run_id: string;
  config_hash: string;
  seed: number | null;
  policy: string;
  premium: number | null;
  source: string;
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

export interface PolicyHeatmap {
  atm_iv_edges: number[];
  time_frac_edges: number[];
  action_names: string[];
  counts: number[][][];  // [time_bin][iv_bin][action_id]
  n_steps: number;
  source: string;
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
}

export type Spec = Record<string, unknown>;

export interface JobOptions {
  arms: string[];
  ppo_available: boolean;
  knobs: Knob[];
  presets: { name: string; label: string; description: string; spec: Spec }[];
  defaults: Spec;
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

export interface Pending {
  status: string;
  waiting_for: string;
}

export interface SweepSummary {
  key: string;
  sweep_hash: string;
  premiums: number[];
  arms: string[];
  n_cells: number;
  seed: number;
  n_episodes: number;
}

export interface SweepList {
  root: string;
  sweeps: SweepSummary[];
  power_curve: Pending;
  attribution: Pending;
}

export interface Sweep {
  summary: SweepSummary;
  oracle: Record<string, { premium: number; mean_pnl: number; se_pnl: number; lift: number; run_id: string }[]>;
  power_curve: Pending;
  attribution: Pending;
  source: string;
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
  iv30_sources: string[];
  iv30_date: string[];
  iv30_diff_vol_pts: (number | null)[];
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

type Params = Record<string, string | number | boolean | undefined | null | (string | number)[]>;

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

async function request<T>(method: "GET" | "POST", path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
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
  return (await res.json()) as T;
}

const get = <T>(path: string, params?: Params, signal?: AbortSignal) => request<T>("GET", withQuery(path, params), undefined, signal);

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
  policyHeatmap: (key: string, s?: AbortSignal) => get<PolicyHeatmap>("/api/runs/policy", { key }, s),
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
};
