/**
 * The showcase's questions (runs/showcase/catalog.json, served as /api/runs/catalog), as
 * the workflows use them: one question, two runs that differ in one setting, same seeds.
 *
 * A pair is named by its catalog group and by the settings that pick each run, never by
 * run keys, so a re-made showcase (new hashes, a new seed) still answers every step. The
 * pairs are the catalog's own (a run against its group's baseline at the same premium),
 * so a step can point at the Compare button the Runs page draws for it; the strip opens
 * the paired compare (/api/runs/compare). A pair the served root's catalog does not have
 * blocks its step with the reason, rather than breaking.
 *
 * Every group of the showcase is covered: strikes, wings, tenor and roll (Tune the
 * seller); costs, account and regime (What eats the edge); real (A premium to sell
 * today); sample (How big must my backtest be); arms and training (Does a learner beat a
 * rule).
 */
import type { Catalog, CatalogGroup, CatalogRun } from "./api_runs";

export interface PairRun {
  key: string;
  label: string;
  present: boolean;
}

export interface Pair {
  id: string;
  group: string;
  question: string;
  /** The group's baseline: the run the catalog compares the other against. */
  a: PairRun;
  b: PairRun;
}

type Pick = (r: CatalogRun) => boolean;

export interface PairSpec {
  id: string;
  group: string;
  a: Pick;
  b: Pick;
}

const num = (r: CatalogRun, k: string) => (r.settings[k] === undefined ? undefined : Number(r.settings[k]));
const has = (k: string, v: number | boolean | string) => (r: CatalogRun) => r.settings[k] === v || (typeof v === "number" && num(r, k) === v);
const prem = (p: number) => (r: CatalogRun) => (r.premium ?? num(r, "iv_premium_vol_pts") ?? 0) === p;
const all = (...ps: Pick[]): Pick => (r) => ps.every((p) => p(r));
/** The run with none of the knobs the group varies: its default. */
const plain = (...keys: string[]): Pick => (r) => keys.every((k) => r.settings[k] === undefined);
const arm = (a: string) => (r: CatalogRun) => (r.policy || String(r.settings.policy ?? "")) === a;

const COST_KEYS = ["costs_enabled", "half_spread_frac", "min_half_spread", "commission_per_contract"];

export const PAIRS = {
  strikes: { id: "strikes", group: "strikes", a: all(has("short_delta", 0.16), prem(1)), b: all(has("short_delta", 0.35), prem(1)) },
  wings: { id: "wings", group: "wings", a: has("wing_delta", 0.05), b: has("wing_delta", 0.1) },
  tenor: { id: "tenor", group: "tenor", a: all(has("expiry_idx", 1), prem(1)), b: all(has("expiry_idx", 0), prem(1)) },
  roll: { id: "roll", group: "roll", a: has("roll_dte", 5), b: has("roll_dte", 0) },
  costs: { id: "costs", group: "costs", a: all(plain(...COST_KEYS), prem(1)), b: all(has("costs_enabled", false), prem(1)) },
  spreads: { id: "spreads", group: "costs", a: all(plain(...COST_KEYS), prem(1)), b: all((r) => num(r, "half_spread_frac") !== undefined, prem(1)) },
  commission: { id: "commission", group: "costs", a: all(plain(...COST_KEYS), prem(1)), b: all((r) => num(r, "commission_per_contract") !== undefined, prem(1)) },
  account: { id: "account", group: "account", a: has("initial_cash", 1_000_000), b: has("initial_cash", 50_000) },
  regime: { id: "regime", group: "regime", a: has("sigma", 0.2), b: has("sigma", 0.3) },
  real: { id: "real", group: "real", a: all(arm("delta_seller"), has("short_delta", 0.16)), b: arm("flat") },
  arms: { id: "arms", group: "arms", a: all(arm("delta_seller"), prem(1)), b: all(arm("ppo"), prem(1)) },
  sample: { id: "sample", group: "sample", a: has("n_episodes", 200), b: has("n_episodes", 50) },
  training: { id: "training", group: "training", a: has("total_timesteps", 200_000), b: has("total_timesteps", 500_000) },
} satisfies Record<string, PairSpec>;

const run = (r: CatalogRun): PairRun => ({ key: r.key, label: r.label, present: r.present });

/** The pair in the served catalog, or why there is none. */
export function resolvePair(cat: Catalog | undefined, spec: PairSpec): Pair | string | undefined {
  if (!cat) return undefined; // still loading
  if (!cat.available) return "this runs root has no catalog.json (the showcase's questions)";
  const g: CatalogGroup | undefined = cat.groups.find((x) => x.group === spec.group);
  if (!g) return `the catalog has no “${spec.group}” question`;
  const a = g.runs.find(spec.a);
  const b = g.runs.find(spec.b);
  if (!a || !b) return `the catalog's “${spec.group}” question lacks one of the two runs this step compares`;
  return { id: spec.id, group: g.group, question: g.question, a: run(a), b: run(b) };
}

/** The run directory's last part, which the runs table names its rows by. */
export const runIdOf = (key: string) => key.split("/").pop() ?? key;

/** The guide name of a pair's Compare button on the Runs page (components/RunCatalog.tsx). */
export const pairGuide = (a: string, b: string) => `runs-pair-${a}|${b}`;

/** The plain name of a catalog group, as the Runs page's question tabs show it. */
export const GROUP_LABEL: Record<string, string> = {
  strikes: "Strikes",
  wings: "Wings",
  tenor: "Tenor",
  roll: "Roll",
  costs: "Costs",
  account: "Account size",
  regime: "Vol regime",
  real: "Real SPX",
  arms: "Arms",
  sample: "Sample size",
  training: "Training",
};

export const groupName = (g: string) => GROUP_LABEL[g] ?? g.charAt(0).toUpperCase() + g.slice(1);
