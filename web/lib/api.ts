/**
 * The one typed client for optspread.server.
 *
 * Every payload type is an alias onto ./schema.d.ts, which openapi-typescript generates
 * from the server's OpenAPI document (`npm run api-types`; `make web` writes the document
 * and regenerates it before the build), so a page that no longer fits the API fails the
 * type check. The names are the ones the pages import. Kept by hand: the helpers, and the
 * payloads Python sends as untyped dicts (Attribution, EdgeDetail, M4, Vrp and their
 * parts), whose schema says only "object". Undefined figures arrive as null, never NaN.
 */

import { recordApi } from "./activity";
import type { components } from "./schema";

type Schemas = components["schemas"];

// ---------- runs ----------

export type RunRow = Schemas["RunRowOut"];

export type RunList = Schemas["RunListOut"];

export type Risk = Schemas["RiskOut"];

// Narrowed by hand: the schema says a histogram edge, an equity point, a seed's difference
// and a scripted sweep's cell figures can be null (a value that is not finite is sent as
// null), but the pages that draw them do not handle null yet (app/runs/page.tsx,
// app/sweep/page.tsx, components/CompareRuns.tsx, components/RunExtras.tsx). These keep
// the number they had until those pages do; every other field is generated.
export type Histogram = Omit<Schemas["HistogramOut"], "edges"> & { edges: number[] };

export type RunDetail = Omit<Schemas["RunDetailOut"], "histogram" | "equity"> & {
  histogram: Histogram;
  equity: (Omit<Schemas["EquityOut"], "paths"> & { paths: number[][] }) | null;
};

export type Compare = Omit<Schemas["CompareOut"], "a" | "b" | "diff_histogram" | "diffs"> & {
  a: RunDetail;
  b: RunDetail;
  diff_histogram: Histogram;
  diffs: (Omit<Schemas["SeedDiffOut"], "diff"> & { diff: number })[];
};

// ---------- jobs ----------

export type Knob = Schemas["KnobOut"];

export type Spec = Record<string, unknown>;

export type Preset = Schemas["PresetOut"];

export type JobOptions = Schemas["JobOptionsOut"];

export type Estimate = Schemas["EstimateOut"];

export type Job = Schemas["JobOut"];

// ---------- power lab ----------

export type PowerParams = Schemas["PowerParams"];

export type Design = Schemas["DesignOut"];

export type Power = Schemas["PowerOut"];

// ---------- leak demo ----------

export type LeakSide = Schemas["LeakSideOut"];

export type LeakDemo = Schemas["LeakDemoOut"];

// ---------- sweeps ----------

export type SweepSummary = Schemas["SweepSummaryOut"];

export type GridCell = Schemas["GridCellOut"];

export type ArmMean = Schemas["ArmMeanOut"];

export type Contrast = Schemas["ContrastOut"];

export type GridDetail = Schemas["GridDetailOut"];

export type SweepList = Schemas["SweepListOut"];

// Narrowed by hand, as Histogram above: a scripted sweep's oracle point keeps numbers.
type OraclePoint = Omit<Schemas["OraclePointOut"], "mean_pnl" | "se_pnl" | "lift"> & { mean_pnl: number; se_pnl: number; lift: number };

export type Sweep = Omit<Schemas["SweepOut"], "oracle"> & { oracle: Record<string, OraclePoint[]> };

// ---------- the SPX panel ----------

export type DataStatus = Schemas["DataStatusOut"];

export type Span = Schemas["SpanOut"];

export type Coverage = Schemas["CoverageOut"];

export type ChainIndex = Schemas["ChainIndexOut"];

export type Smile = Schemas["SmileOut"];

export type Surface = Schemas["SurfaceOut"];

export type Term = Schemas["TermOut"];

export type Series = Schemas["SeriesOut"];

export type VrpStats = Schemas["VrpStatsOut"];

export type Daily = Schemas["DailyOut"];

export type Seam = Schemas["SeamOut"];

// ---------- the 3D views ----------

export type Grid = Schemas["GridOut"];

export type GridFrames = Schemas["GridFramesOut"];

export type SigmaFrom = "config" | "rv21" | "atm";

export type Simulated = Schemas["SimulatedOut"];

export type Landscape = Schemas["LandscapeOut"];

export type PowerSurfaceParams = Schemas["PowerSurfaceParams"];

export type PowerSurface = Schemas["PowerSurfaceOut"];

// ---------- the study and its analyses ----------

export type Study = Schemas["StudyOut"];

export type MeasuredDesign = Schemas["MeasuredDesignOut"];

export type Check = Schemas["CheckOut"];

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

export type MappedPower = Schemas["MappedPowerOut"];

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

export type Capital = Schemas["CapitalOut"];

export type Replay = Schemas["ReplayOut"];

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

export type EdgeFile = Schemas["EdgeFileOut"];

export type EdgeList = Schemas["EdgeListOut"];

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

export type StructureMix = Schemas["StructureMix"];

export type MixSummary = Schemas["MixSummary"];

export type RunActionMix = Schemas["RunActionMix"];

export interface GridMixRow {
  premium: number;
  arm: string;
  reproduces: boolean | null;
  agrees: boolean | null;
  summary: MixSummary;
}

export type GridActionMix = Schemas["GridActionMix"];

export type ActionMix = Schemas["ActionMixOut"];

// ---------- leaks: the floor and the fields ----------

export type LeakFloor = Schemas["LeakFloorOut"];

export type LeakField = Schemas["LeakFieldOut"];

export type LeakCoverage = Schemas["LeakCoverageOut"];

// ---------- the seller on a real close ----------

export type Seller = Schemas["SellerOut"];

// ---------- overview ----------

export type Overview = Schemas["OverviewOut"];

export type Health = Schemas["HealthOut"];

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
