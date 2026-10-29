/**
 * Guided workflows, written for a trader (and one for a reviewer of our claims).
 *
 * A step opens a real page, says exactly what to do there, rings the control to use,
 * and keeps Next locked until the action is done. It then says in one plain sentence
 * what the result means, with the numbers the reader just produced. The strip under
 * the header (components/Workflow.tsx) runs them; nothing on a page depends on being
 * in a workflow, and no step draws over the page.
 *
 * How a step knows its action was done. `done(ctx)` reads only:
 *   - the address bar (every page keeps its settings in the query string), and
 *   - the activity log (lib/activity.ts): every answer the server gave this visit, with
 *     its query and body, and the exports. So "a run launched and finished" is a
 *     POST /api/jobs since the step began whose job the server later reports done;
 *     "the detector ran at eps <= 0.1" is a POST /api/leaks/demo with that body.
 * Nothing is read from a page's own state, so a page can be laid out anew and its
 * steps still work, on the showcase root or any other, as long as it uses lib/api.ts
 * and keeps its query parameters.
 *
 * The contract the pages keep (rename one here if a page renames it):
 *   query   /launch/ preset · /power/ preset, n_seeds, n_episodes, delta · /leaks/ view,
 *           eps · /sweep/ view, arm · /attribution/ arm, p · /runs/ key, a, b, question ·
 *           /data/ q (premium|surface|sim|seller), date, sigma · / checks, study (the
 *           homepage's panels)
 *   targets a control is found by data-guide="<name>" first (lib/guide.ts), else by
 *           its visible text or label (see Target). Names are <page>-<control>:
 *           launch-run, launch-preset-<name> · runs-question-<group>, runs-pair-<a>|<b> (a
 *           catalog pair's Compare), runs-pick-<run key>, runs-compare, runs-open,
 *           runs-replay, runs-by-day, runs-episode · power-effect, power-seeds,
 *           power-episodes, power-preset · leaks-run, leaks-eps, leaks-view, leaks-floor,
 *           leaks-field, leaks-plant · sweep-arm, sweep-view · attribution-arm,
 *           attribution-premium · data-date, data-next-day, data-sigma-from ·
 *           home-checks, home-study · export (every figure's export bar).
 */

import type { Activity, Call } from "./activity";
import type { Attribution, Check, EdgeDetail, Grid, Job, LeakDemo, LeakField, LeakFloor, M4, Power, Replay, RunDetail, Seller, Simulated, Study, Sweep, Term, Vrp } from "./api";
import { api } from "./api";
import { type Catalog, type PairedCompare, runsApi } from "./api_runs";
import { groupName, type Pair, PAIRS, type PairSpec, pairGuide, resolvePair, runIdOf } from "./catalog";
import { MINUS, num, pct, prem, short, usd } from "./format";

/** $ of mean P&L per episode for one vol point of premium (pre-registration A6). */
export const PER_VOL = 824.3;
/** The PPO contrast's variance components (A7), for a comparison the strip draws itself. */
const A7 = { tau2: 51_838, sigma2: 105_996_836, n_seeds: 5, n_episodes: 500, delta: PER_VOL, alpha: 0.05, target_power: 0.8, train_steps: 1_000_000, episode_days: 63, train_rate: 804, eval_rate: 1538, cores: 1 };

/** Where a step's control is: data-guide first, then its text, its label or a selector. */
export interface Target {
  /** The data-guide name (lib/guide.ts). */
  guide: string;
  /** A button, radio or link whose visible text matches. */
  text?: RegExp;
  /** A <label> (with its input) whose text matches. */
  label?: RegExp;
  /** An element whose aria-label matches. */
  aria?: RegExp;
  css?: string;
  /** Ring every match, not only the first. */
  all?: boolean;
}

export interface Ctx {
  path: string;
  q: URLSearchParams;
  /** When this step began, ms. */
  since: number;
  /** The newest answer to `call` since the step began. */
  call(c: Call): Activity | undefined;
  /** Every answer to `call` since the step began, oldest first. */
  calls(c: Call): Activity[];
  /** The newest answer to `call` before the step began. */
  before(c: Call): Activity | undefined;
  /** UI actions since the step began ("export"). */
  did(kind: string): Activity | undefined;
  /** What an earlier step kept when it was done. */
  kept(stepId: string): Record<string, unknown> | undefined;
  /** A read the strip makes itself, cached by key: undefined until it has answered. */
  get<T>(key: string, load: () => Promise<T>): T | undefined;
}

export interface Step {
  id: string;
  href: string;
  /** What to do, in the imperative: "Press Launch run". May read ctx for numbers. */
  task: string | ((ctx: Ctx) => string);
  /** The control to ring; several when the action needs more than one. */
  target?: Target | Target[] | ((ctx: Ctx) => Target | Target[] | undefined);
  done: (ctx: Ctx) => boolean;
  /** The one-sentence reading of what the reader just produced; null while it loads. */
  read?: (ctx: Ctx) => string | null;
  /** What later steps may need from this one, saved when it is done. */
  keep?: (ctx: Ctx) => Record<string, unknown> | undefined;
  /**
   * Open the step's own query on top of the reader's current one, even when the page is
   * the same (a question-led page switching view, keeping the date chosen). Without it,
   * a step on the same page as the last one leaves the page exactly as the reader left it.
   */
  merge?: boolean;
  /** Always open the step's own href, even from the same page (the runs table after a comparison). */
  open?: boolean;
  /** Why the action cannot be done on this server (Next unlocks, saying so), or null. */
  blocked?: (ctx: Ctx) => string | null;
}

export interface Workflow {
  id: string;
  /** The card's title. */
  title: string;
  /** The question it answers, one line. */
  question: string;
  minutes: number;
  /** What you will do, three short lines. */
  youWill: string[];
  icon: "edge" | "market" | "pnl" | "size" | "leak" | "learner" | "review" | "tune" | "costs";
  /** Shown first and largest on the homepage. */
  featured?: boolean;
  reviewer?: boolean;
  steps: Step[];
}

// ---------------------------------------------------------------- helpers

const vp = (x: number | null | undefined, d = 2) => (x === null || x === undefined || !Number.isFinite(x) ? "–" : `${x < 0 ? MINUS : ""}${Math.abs(x).toFixed(d)}`);
const at = (a: Activity | undefined) => a?.data as never;
const body = (a: Activity) => (a.body ?? {}) as Record<string, unknown>;
const premOf = (b: Record<string, unknown>) => Number(b.iv_premium_vol_pts ?? 0);
const policyOf = (b: Record<string, unknown>) => String(b.policy ?? "delta_seller");
const near = (a: number, b: number, tol = 1e-6) => Math.abs(a - b) <= tol;
const ARM: Record<string, string> = { delta_seller: "the fixed-delta seller", ppo: "PPO", always_on: "the always-on condor", random: "the random arm", flat: "flat" };

/** The newest job launched in this step that matches and that the server reports done. */
function finished(ctx: Ctx, match: (b: Record<string, unknown>) => boolean): (Job & { spec: Record<string, unknown> }) | undefined {
  const posts = ctx.calls("POST /api/jobs").filter((a) => match(body(a)));
  const states = [...ctx.calls("GET /api/jobs"), ...ctx.calls("GET /api/jobs/*")];
  for (const p of [...posts].reverse()) {
    const id = (p.data as Job | undefined)?.job_id;
    if (!id) continue;
    let job = p.data as Job;
    for (const s of states) {
      if (s.at < p.at) continue;
      const list = Array.isArray(s.data) ? (s.data as Job[]) : [s.data as Job];
      const hit = list.find((j) => j?.job_id === id);
      if (hit) job = hit;
    }
    if (job.state === "done" && job.run_key) return { ...job, spec: body(p) };
  }
  return undefined;
}

/** A job of this step that failed, so the strip can say why rather than wait forever. */
function failed(ctx: Ctx): string | null {
  const states = [...ctx.calls("GET /api/jobs"), ...ctx.calls("GET /api/jobs/*")];
  const ids = new Set(ctx.calls("POST /api/jobs").map((p) => (p.data as Job | undefined)?.job_id));
  for (const s of [...states].reverse()) {
    const list = Array.isArray(s.data) ? (s.data as Job[]) : [s.data as Job];
    const bad = list.find((j) => j && ids.has(j.job_id) && j.state === "failed");
    if (bad) return `the run failed: ${bad.error || bad.phase}`;
  }
  return null;
}

function runOf(ctx: Ctx, key: unknown): RunDetail | undefined {
  return typeof key === "string" && key ? ctx.get(`run|${key}`, () => api.run(key)) : undefined;
}

/** The power answer of this step that satisfies `ok`, newest first. */
function powerWhere(ctx: Ctx, ok: (p: Power) => boolean): Power | undefined {
  return ctx
    .calls("GET /api/power")
    .map((a) => a.data as Power)
    .reverse()
    .find((p) => p && ok(p));
}

const latestPower = (ctx: Ctx) => ctx.call("GET /api/power")?.data as Power | undefined;

/** Dates the data views were asked for in this step, and the one shown before it began. */
function dates(ctx: Ctx): { now: string | undefined; changed: boolean } {
  const routes: Call[] = ["GET /api/data/term", "GET /api/data/seller", "GET /api/data/grid", "GET /api/data/chain", "GET /api/data/surface"];
  const seen = routes.flatMap((r) => ctx.calls(r)).sort((a, b) => a.at - b.at).map((a) => a.query?.date).filter((d): d is string => Boolean(d));
  const prior = routes.map((r) => ctx.before(r)).filter((a): a is Activity => Boolean(a)).sort((a, b) => a.at - b.at).pop()?.query?.date;
  const now = seen[seen.length - 1];
  return { now, changed: prior ? seen.some((d) => d !== prior) : new Set(seen).size > 1 };
}

/** At-the-money vol per tenor from the gridded surface: the column nearest ln(K/F) = 0. */
function atmOfGrid(g: Grid): { date: string; pts: { d: number; v: number | null }[] } {
  let j = 0;
  g.k.forEach((k, i) => {
    if (Math.abs(k) < Math.abs(g.k[j] ?? Infinity)) j = i;
  });
  return { date: g.date, pts: g.tenor_days.map((d, i) => ({ d, v: g.iv[i]?.[j] ?? null })) };
}

function leakRead(d: LeakDemo): string {
  if (d.caught && !d.clean.flagged)
    return `Caught: with tomorrow's return planted at eps ${d.eps}, the poisoned simulator was flagged on ${d.poisoned.worst_feature} (|t| ${num(Math.abs(d.poisoned.worst_t ?? 0), 1)}) while the clean one stayed quiet, so a backtest that peeks ahead cannot hide here.`;
  if (d.caught) return `The poisoned simulator was caught on ${d.poisoned.worst_feature}, but the clean one was flagged too: a false alarm, which the pass criterion counts as a failure.`;
  return `Missed: at eps ${d.eps} the planted leak stayed under the line with ${d.seeds.length} paths, an honest miss; smaller leaks need more data to see.`;
}


/** Whether a run is under this server's runs root: false on a 404, true otherwise. */
function present(ctx: Ctx, key: string): boolean | undefined {
  return ctx.get(`exists|${key}`, () => api.run(key).then(
    () => true,
    (e: unknown) => !(e && typeof e === "object" && (e as { status?: number }).status === 404),
  ));
}

/** The served root's catalog (/api/runs/catalog), read once by the strip. */
const catalogOf = (ctx: Ctx) => ctx.get<Catalog>("catalog", () => runsApi.catalog());

/** The pair a step compares, from the served catalog: undefined while it loads, a reason when it is missing. */
function pairOf(ctx: Ctx, spec: PairSpec): Pair | string | undefined {
  return resolvePair(catalogOf(ctx), spec);
}

const isPair = (p: Pair | string | undefined): p is Pair => typeof p === "object";

/** The paired comparison of the pair (B against the baseline A), whichever order the reader opened them in. */
function pairCompare(ctx: Ctx, p: Pair): { a: RunDetail; b: RunDetail; d: Paired } | undefined {
  const hit = ctx
    .calls("GET /api/runs/compare")
    .map((c) => c.data as PairedCompare)
    .filter((x) => x && new Set([x.a.row.key, x.b.row.key]).size === 2 && [x.a.row.key, x.b.row.key].every((k) => k === p.a.key || k === p.b.key))
    .pop();
  if (!hit) return undefined;
  const flip = hit.a.row.key !== p.a.key;
  const sign = flip ? -1 : 1;
  const lo = hit.diff_ci_low === null || hit.diff_ci_low === undefined ? null : sign * hit.diff_ci_low;
  const hi = hit.diff_ci_high === null || hit.diff_ci_high === undefined ? null : sign * hit.diff_ci_high;
  const [l, h] = lo !== null && hi !== null && lo > hi ? [hi, lo] : [lo, hi];
  return {
    a: flip ? hit.b : hit.a,
    b: flip ? hit.a : hit.b,
    d: { diff: sign * (hit.diff_mean ?? 0), lo: l, hi: h, clear: l !== null && h !== null && (l > 0 || h < 0), n: hit.shared_seeds },
  };
}

/** B less A per episode on the shared seeds, with its two-sided 95% interval. */
interface Paired {
  diff: number;
  lo: number | null;
  hi: number | null;
  /** The interval excludes zero. */
  clear: boolean;
  n: number;
}

const signedUsd = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${x >= 0 ? "+" : MINUS}${usd(Math.abs(x))}`);
/** "95% −$120 to +$480" */
const ci = (d: Paired) => (d.lo === null || d.hi === null ? "no interval" : `95% ${signedUsd(d.lo)} to ${signedUsd(d.hi)}`);

/**
 * A step on one of the catalog's questions: open it on the Runs page and press Compare
 * on the pair (or tick both runs in the table), then one sentence on the paired result.
 */
function compareStep(spec: PairSpec, say: (a: RunDetail, b: RunDetail, d: Paired, p: Pair) => string, task?: (p: Pair) => string): Step {
  return {
    id: spec.id,
    href: `/runs/?question=${spec.group}`,
    open: true,
    task: (c) => {
      const p = pairOf(c, spec);
      if (!isPair(p)) return `Open the ${groupName(spec.group)} question on the Runs page and compare its two runs.`;
      return task ? task(p) : `${p.question} Under ${groupName(p.group)}, press Compare on “${p.b.label}” (against “${p.a.label}”).`;
    },
    target: (c) => {
      const p = pairOf(c, spec);
      if (!isPair(p) || (c.q.get("a") && c.q.get("b"))) return undefined;
      return [{ guide: pairGuide(p.a.key, p.b.key) }, { guide: `runs-pick-${p.a.key}`, aria: new RegExp(`^Compare ${runIdOf(p.a.key)}$`) }, { guide: `runs-pick-${p.b.key}`, aria: new RegExp(`^Compare ${runIdOf(p.b.key)}$`) }];
    },
    done: (c) => {
      const p = pairOf(c, spec);
      return isPair(p) && Boolean(pairCompare(c, p));
    },
    blocked: (c) => {
      const p = pairOf(c, spec);
      if (typeof p === "string") return p;
      const miss = isPair(p) ? [p.a, p.b].find((r) => !r.present) : undefined;
      return miss ? `the run “${miss.label}” (${miss.key}) is not under this runs root` : null;
    },
    read: (c) => {
      const p = pairOf(c, spec);
      const r = isPair(p) ? pairCompare(c, p) : undefined;
      return isPair(p) && r ? say(r.a, r.b, r.d, p) : null;
    },
  };
}

const noise = (d: Paired) => (d.clear ? "" : ", though the paired interval still includes zero at this sample size");

/** "$447 more per episode" or "$447 less". */
const diffText = (x: number) => `${usd(Math.abs(x))} ${x >= 0 ? "more" : "less"}`;

/** The plain verdict on a change: did it earn more, and did its tail get worse? */
function tradeOff(d: Paired, tailB: number | null, tailA: number | null, change: string): string {
  const worse = (tailB ?? 0) < (tailA ?? 0);
  if (!d.clear) return `${change} made no difference to the mean you could tell from luck at this sample size${worse ? ", but it fattened the tail" : ""}`;
  if (d.diff > 0) return worse ? `${change} earned more, at the price of a fatter tail` : `${change} earned more without a worse tail`;
  return worse ? `${change} earned less and fattened the tail` : `${change} earned less, with a thinner tail`;
}

// ---------------------------------------------------------------- the workflows

export const WORKFLOWS: Workflow[] = [
  {
    id: "edge",
    featured: true,
    title: "Real edge, or luck?",
    question: "Would a backtest tell one vol point of premium from noise?",
    minutes: 5,
    icon: "edge",
    youWill: ["Launch the seller in a fair world and with 1 vol pt planted", "Pair the two runs on the same market paths", "Find how many seeds a convincing test needs"],
    steps: [
      {
        id: "fair",
        href: "/launch/?preset=null_world",
        task: "Launch the fixed-delta seller in a fair-priced world: press Launch and wait for it to finish.",
        target: { guide: "launch-run", text: /^\s*launch( run)?\s*$/i },
        done: (c) => Boolean(finished(c, (b) => policyOf(b) === "delta_seller" && near(premOf(b), 0))),
        blocked: failed,
        keep: (c) => ({ key: finished(c, (b) => policyOf(b) === "delta_seller" && near(premOf(b), 0))?.run_key }),
        read: (c) => {
          const r = runOf(c, finished(c, (b) => policyOf(b) === "delta_seller" && near(premOf(b), 0))?.run_key);
          if (!r) return null;
          const m = r.risk.mean ?? 0;
          return m < 0
            ? `With nothing planted the seller averaged ${usd(m)} per episode (t = ${num(r.row.t_stat, 2)}) after ${usd(r.row.mean_cost)} of costs: a fair-priced world pays a premium seller nothing to cover them.`
            : `With nothing planted the seller made ${usd(m)} per episode, but t = ${num(r.row.t_stat, 2)}: in a fair-priced world any profit is luck.`;
        },
      },
      {
        id: "planted",
        href: "/launch/?preset=null_world",
        task: "Now plant the premium: pick the “Planted 1 vol pt” quick start, then press Launch.",
        target: (c) => (c.q.get("preset") === "planted_1" || c.q.get("iv_premium_vol_pts") === "1" ? { guide: "launch-run", text: /^\s*launch( run)?\s*$/i } : { guide: "launch-preset-planted_1", text: /Planted 1 vol pt/ }),
        done: (c) => Boolean(finished(c, (b) => policyOf(b) === "delta_seller" && near(premOf(b), 1))),
        blocked: failed,
        keep: (c) => ({ key: finished(c, (b) => policyOf(b) === "delta_seller" && near(premOf(b), 1))?.run_key }),
        read: (c) => {
          const r = runOf(c, finished(c, (b) => policyOf(b) === "delta_seller" && near(premOf(b), 1))?.run_key);
          const f = runOf(c, c.kept("fair")?.key);
          if (!r) return null;
          if (!f) return `With 1 vol pt planted the seller made ${usd(r.risk.mean)} per episode (t = ${num(r.row.t_stat, 2)}).`;
          const d = (r.risk.mean ?? 0) - (f.risk.mean ?? 0);
          return `One vol point of premium moved the seller from ${usd(f.risk.mean)} to ${usd(r.risk.mean)} per episode, ${d >= 0 ? "+" : MINUS}${usd(Math.abs(d))}; next, check that this is more than luck.`;
        },
      },
      {
        id: "pair",
        href: "/runs/?policy=delta_seller",
        task: (c) => {
          const a = runOf(c, c.kept("fair")?.key);
          const b = runOf(c, c.kept("planted")?.key);
          const ids = a && b ? ` (${short(a.row.config_hash, 10)}-s${a.row.seed} and ${short(b.row.config_hash, 10)}-s${b.row.seed})` : "";
          return `Tick your two runs${ids} in the table and press Compare selected.`;
        },
        target: (c) => {
          const a = runOf(c, c.kept("fair")?.key);
          const b = runOf(c, c.kept("planted")?.key);
          const picked = c.path.startsWith("/runs") && c.q.get("a");
          if (picked) return undefined;
          return a && b
            ? [{ guide: `runs-pick-${a.row.key}`, aria: new RegExp(`^Compare ${a.row.run_id}$`) }, { guide: `runs-pick-${b.row.key}`, aria: new RegExp(`^Compare ${b.row.run_id}$`) }, { guide: "runs-compare", text: /compare selected/i, all: true }]
            : { guide: "runs-compare", text: /compare selected/i, all: true };
        },
        done: (c) => Boolean(c.q.get("a") && c.q.get("b") && c.call("GET /api/runs/compare")),
        read: (c) => {
          const x = at(c.call("GET /api/runs/compare")) as PairedCompare | undefined;
          if (!x) return null;
          const flip = (x.a.row.premium ?? 0) > (x.b.row.premium ?? 0);
          const sg = flip ? -1 : 1;
          const d = sg * (x.diff_mean ?? 0);
          const lo = x.diff_ci_low === null ? null : sg * (flip ? (x.diff_ci_high ?? 0) : x.diff_ci_low);
          const hi = x.diff_ci_high === null ? null : sg * (flip ? (x.diff_ci_low ?? 0) : x.diff_ci_high);
          const clear = lo !== null && hi !== null && (lo > 0 || hi < 0);
          return `Paired on ${num(x.shared_seeds)} shared market paths, the higher-premium run beats the other by ${usd(d)} per episode (${lo === null || hi === null ? "no interval" : `95% ${signedUsd(lo)} to ${signedUsd(hi)}`}): ${clear ? "an interval clear of zero, so the premium shows up at this sample size" : "the interval includes zero, so this sample cannot tell the edge from noise"}.`;
        },
      },
      {
        id: "size",
        href: "/power/?unit=vol",
        task: "Size the test for a learned strategy: set Effect to detect to 0.25 vol pts, then raise Seeds, S until power at the effect reads 80% or more.",
        target: (c) => ((latestPower(c)?.params.delta ?? PER_VOL) > 0.25 * PER_VOL + 0.5 ? { guide: "power-effect", label: /Effect to detect/ } : { guide: "power-seeds", label: /^\s*Seeds, S/ }),
        done: (c) => Boolean(powerWhere(c, (p) => p.params.delta <= 0.25 * PER_VOL + 0.5 && p.power >= 0.8)),
        read: (c) => {
          const p = powerWhere(c, (x) => x.params.delta <= 0.25 * PER_VOL + 0.5 && x.power >= 0.8);
          if (!p) return null;
          return `To catch a ${vp(p.params.delta / PER_VOL)}-vol-pt edge 4 times in 5 takes S = ${p.params.n_seeds} seeds of ${num(p.params.n_episodes)} episodes (${pct(p.power, 0)} power, about ${p.design.total_hours === null ? "–" : `${num(p.design.total_hours, 0)} core-hours`}), far beyond the 5 seeds a typical published backtest reports.`;
        },
      },
    ],
  },
  {
    id: "tune",
    title: "Tune the seller",
    question: "Which strikes, wings, expiry and roll earn the most for the risk taken?",
    minutes: 5,
    icon: "tune",
    featured: true,
    youWill: ["Compare short deltas on the same market paths", "Trade wing width against the worst case", "Pick an expiry and a roll"],
    steps: [
      compareStep(PAIRS.strikes, (a, b, d) => `At 1 vol pt planted, selling at 0.35 delta averaged ${usd(b.risk.mean)} per episode against ${usd(a.risk.mean)} at the default 0.16 (${signedUsd(d.diff)} on the same paths, ${ci(d)}), and its worst 5% of episodes ${usd(b.risk.cvar5)} against ${usd(a.risk.cvar5)}: ${tradeOff(d, b.risk.cvar5, a.risk.cvar5, "selling closer to the money")}.`),
      compareStep(PAIRS.wings, (a, b, d) => `Narrow wings (0.10 delta) averaged ${usd(b.risk.mean)} per episode against ${usd(a.risk.mean)} with the default 0.05 (${signedUsd(d.diff)}, ${ci(d)}), with the worst 5% at ${usd(b.risk.cvar5)} against ${usd(a.risk.cvar5)}: ${tradeOff(d, b.risk.cvar5, a.risk.cvar5, "narrower wings")}.`),
      compareStep(PAIRS.tenor, (a, b, d) => `The 21-day condor averaged ${usd(b.risk.mean)} per episode against ${usd(a.risk.mean)} for the default 42-day one (${signedUsd(d.diff)}, ${ci(d)}), trading ${num(b.risk.turnover, 1)} times an episode against ${num(a.risk.turnover, 1)}: ${!d.clear ? "no difference in the mean you could tell from luck at this sample size" : d.diff > 0 ? "the shorter expiry earned more, even though it rolls more often and pays more in costs" : "the longer expiry earned more, partly by rolling less and paying less in costs"}.`),
      compareStep(PAIRS.roll, (a, b, d) => `Holding to expiry averaged ${usd(b.risk.mean)} per episode against ${usd(a.risk.mean)} rolling 5 days early, the default (${signedUsd(d.diff)}, ${ci(d)}), with the worst 5% at ${usd(b.risk.cvar5)} against ${usd(a.risk.cvar5)}: ${tradeOff(d, b.risk.cvar5, a.risk.cvar5, "holding to expiry")}.`),
    ],
  },
  {
    id: "costs",
    title: "What eats the edge?",
    question: "How much of a 1-vol-point edge survives costs, a small account and a wilder market?",
    minutes: 5,
    icon: "costs",
    youWill: ["Turn costs off and see what they take", "Double the spreads, then the commission", "Shrink the account and change the vol regime"],
    steps: [
      compareStep(PAIRS.costs, (a, b, d) => `Costs take ${usd(Math.abs(d.diff))} per episode off the seller (${usd(b.risk.mean)} without them, ${usd(a.risk.mean)} with; ${ci(d)}): that is what a planted premium has to beat before it shows as profit.`),
      compareStep(PAIRS.spreads, (a, b, d) => `Doubling the spreads moved the seller from ${usd(a.risk.mean)} to ${usd(b.risk.mean)} per episode (${signedUsd(d.diff)}, ${ci(d)}): at ${num(a.risk.turnover, 1)} trades an episode, execution is a first-order part of the edge.`),
      compareStep(PAIRS.commission, (a, b, d) => `Doubling the commission moved the seller from ${usd(a.risk.mean)} to ${usd(b.risk.mean)} per episode (${signedUsd(d.diff)}): ${Math.abs(d.diff) < 0.1 * Math.abs(a.row.mean_cost ?? 1) ? "beside the bid-ask spread, commission is small change for a condor seller" : "at this turnover commission is a real part of the cost"}.`),
      compareStep(PAIRS.account, (a, b, d) => `With $50,000 the seller traded ${num(b.risk.turnover, 1)} times an episode against ${num(a.risk.turnover, 1)} with $1,000,000, averaging ${usd(b.risk.mean)} against ${usd(a.risk.mean)} (${ci(d)}): ${(b.risk.turnover ?? 0) < (a.risk.turnover ?? 0) ? "below a certain size, margin refuses some of the trades the strategy needs" : "at this size margin did not bind"}.`),
      compareStep(PAIRS.regime, (a, b, d) => `With the same 1 vol pt planted, the seller averaged ${usd(b.risk.mean)} per episode in a 30% vol world against ${usd(a.risk.mean)} in the default 20% one, with the worst 5% at ${usd(b.risk.cvar5)} against ${usd(a.risk.cvar5)}: ${(b.risk.cvar5 ?? 0) < (a.risk.cvar5 ?? 0) ? "the same premium comes with far bigger swings when vol is high" : "the regime changed little here"}${noise(d)}.`),
    ],
  },
  {
    id: "market",
    featured: true,
    title: "A premium to sell today?",
    question: "Are SPX options dear, and what would a seller collect on a given day?",
    minutes: 4,
    icon: "market",
    youWill: ["Pick a trading day and read its vol curve", "See what the seller would have opened", "Check the market's premium against what a backtest can see"],
    steps: [
      {
        id: "day",
        href: "/data/?q=surface",
        task: "Pick a day to look at: set Trading day to 2020-03-16 (the Covid crash), or any trading day you like.",
        target: { guide: "data-date", label: /^\s*Trading day/ },
        done: (c) => dates(c).changed,
        blocked: (c) => {
          const s = at(c.call("GET /api/data/status") ?? c.before("GET /api/data/status")) as { built?: boolean } | undefined;
          return s && s.built === false ? "the SPX panel is not built on this server" : null;
        },
        read: (c) => {
          const now = dates(c).now;
          const grid = c.calls("GET /api/data/grid").map((a) => a.data as Grid).filter((g) => g && (!now || g.date === now)).pop();
          const term = c.calls("GET /api/data/term").map((a) => a.data as Term).filter((x) => x && (!now || x.date === now)).pop();
          const atm = grid ? atmOfGrid(grid) : term ? { date: term.date, pts: term.dte.map((d, i) => ({ d, v: term.atm_iv[i] })) } : null;
          if (!atm) return null;
          const pts = atm.pts.filter((p): p is { d: number; v: number } => typeof p.v === "number");
          if (pts.length < 2) return `On ${atm.date} the chain has too few expiries to read a term structure.`;
          const one = pts.reduce((b, p) => (Math.abs(p.d - 30) < Math.abs(b.d - 30) ? p : b));
          const far = pts[pts.length - 1] as { d: number; v: number };
          return `On ${atm.date} one-month at-the-money vol was ${vp(one.v * 100, 1)}% against ${vp(far.v * 100, 1)}% at ${num(far.d)} days: ${one.v > far.v ? "inverted, the market paying up for near-term protection, the stress a premium seller is paid to carry" : "upward sloping, a calm market"}.`;
        },
      },
      {
        id: "seller",
        href: "/data/?q=seller",
        merge: true,
        task: "See what the fixed-delta seller would open at a close: step to another trading day (→, or the date box) and read the condor it opens.",
        target: [{ guide: "data-next-day", aria: /Next trading day/ }],
        done: (c) => Boolean(c.call("GET /api/data/seller")) && dates(c).changed,
        read: (c) => {
          const s = at(c.call("GET /api/data/seller")) as Seller | undefined;
          if (!s) return null;
          const risk = s.capped_loss ?? s.max_loss ?? s.margin;
          const name = s.structure.toLowerCase().replaceAll("_", " ");
          return `On ${s.date} the seller would open ${/^[aeiou]/.test(name) ? "an" : "a"} ${name} for a ${usd(s.credit)} credit with ${usd(risk)} at risk: ${pct(risk ? s.credit / risk : null, 1)} of the worst case paid up front, kept if SPX stays between ${s.breakevens.map((b) => num(b)).join(" and ")}.`;
        },
      },
      {
        id: "sim",
        href: "/data/?q=sim",
        merge: true,
        task: "Set the simulator to that day's vol: pick “matched to IV30” under Path vol.",
        target: { guide: "data-sigma-from", text: /matched to IV30|^\s*ATM\s*$/i },
        done: (c) => c.q.get("sigma") === "atm" || c.calls("GET /api/data/simulated").some((a) => a.query?.sigma_from === "atm"),
        read: (c) => {
          const s = c.calls("GET /api/data/simulated").filter((a) => a.query?.sigma_from === "atm").map((a) => a.data as Simulated).pop();
          if (!s) return null;
          return `Matched to the day's level (${vp(s.level * 100, 1)}%), the simulator still misses the real chain by ${vp((s.mean_abs_diff ?? 0) * 100)} vol pts on average and ${vp((s.max_abs_diff ?? 0) * 100)} at worst: the skew is the part of the market the simulator does not plant.`;
        },
      },
      {
        id: "detect",
        href: "/power/?unit=vol",
        task: (c) => {
          const v = (c.get("vrp", () => api.vrp()) as Vrp | undefined)?.mapping?.headline.mean;
          return `Is the market's premium big enough to see? Set Effect to detect to the real average premium${v === undefined || v === null ? "" : `, ${vp(v)} vol pts`}.`;
        },
        target: { guide: "power-effect", label: /Effect to detect/ },
        done: (c) => {
          const v = (c.get("vrp", () => api.vrp()) as Vrp | undefined)?.mapping?.headline.mean;
          return v !== undefined && v !== null && Boolean(powerWhere(c, (p) => Math.abs(p.params.delta / PER_VOL - v) <= 0.051));
        },
        blocked: (c) => {
          const vrp = c.get("vrp", () => api.vrp()) as Vrp | undefined;
          return vrp && !vrp.available ? "the real premium (docs/VRP_MAPPING.json) is not on this server" : null;
        },
        read: (c) => {
          const v = (c.get("vrp", () => api.vrp()) as Vrp | undefined)?.mapping?.headline;
          const p = v ? powerWhere(c, (x) => Math.abs(x.params.delta / PER_VOL - (v.mean ?? 0)) <= 0.051) : undefined;
          if (!v || !p) return null;
          return `At the market's average premium of ${vp(v.mean)} vol pts (95% ${vp(v.mean_low)} to ${vp(v.mean_high)}), an S = ${p.params.n_seeds} backtest catches it ${pct(p.power, 0)} of the time${p.power < 0.5 ? ": most backtests of that size would call a real premium noise" : p.power < 0.8 ? `, short of the 80% a design should have: ${Math.round((1 - p.power) * 10)} in 10 backtests of that size would call a real premium noise` : ": large enough to see at this size"}.`;
        },
      },
      compareStep(PAIRS.real, (a, b, d) => {
        const edge = -d.diff; // the seller less flat
        const clear = d.clear && edge > 0;
        return `On real SPX closes from 2019 to 2022-08 the seller made ${usd(a.risk.mean)} per episode against flat's ${usd(b.risk.mean)} (${signedUsd(edge)} on the same days, ${d.lo === null || d.hi === null ? "no interval" : `95% ${signedUsd(-d.hi)} to ${signedUsd(-d.lo)}`}), with its worst 5% of episodes at ${usd(a.risk.cvar5)}: ${clear ? "the real premium paid, with real tail risk" : "on this history the premium did not clearly pay for the tail"}.`;
      }),
    ],
  },
  {
    id: "pnl",
    title: "Where does the P&L come from?",
    question: "Which greeks pay the seller, and does the learner earn it the same way?",
    minutes: 4,
    icon: "pnl",
    youWill: ["Split the seller's edge into delta, gamma, vega, theta and cost", "Compare PPO's split at the same premium", "Open one episode day by day"],
    steps: [
      {
        id: "seller",
        href: "/attribution/",
        task: (c) => {
          const g = (at(c.call("GET /api/edge/detail") ?? c.before("GET /api/edge/detail")) as EdgeDetail | undefined)?.edge;
          const low = g?.premiums.filter((p) => p > 0)[0];
          return `Pick ${low === undefined ? "the smallest premium" : `${prem(low)} vol pts`} in Planted premium, then the fixed-delta seller in Details for.`;
        },
        target: (c) => (c.q.get("p") ? { guide: "attribution-arm", label: /^\s*Details for/ } : { guide: "attribution-premium", label: /^\s*Planted premium/ }),
        done: (c) => {
          const g = (at(c.call("GET /api/edge/detail") ?? c.before("GET /api/edge/detail")) as EdgeDetail | undefined)?.edge;
          const low = g?.premiums.filter((p) => p > 0)[0];
          const arm = c.q.get("arm") || g?.arms[0];
          return Boolean(g && low !== undefined && arm === "delta_seller" && near(Number(c.q.get("p")), low));
        },
        blocked: (c) => {
          const l = at(c.call("GET /api/edge") ?? c.before("GET /api/edge")) as { files?: unknown[] } | undefined;
          return l && !(l.files ?? []).length ? "no edge file under this server's attribution root (make edge)" : null;
        },
        keep: (c) => ({ p: Number(c.q.get("p")) }),
        read: (c) => edgeRead(c, "delta_seller", Number(c.q.get("p"))),
      },
      {
        id: "ppo",
        href: "/attribution/",
        task: "Now pick PPO in Details for, keeping the premium.",
        target: { guide: "attribution-arm", label: /^\s*Details for/ },
        done: (c) => c.q.get("arm") === "ppo",
        blocked: (c) => {
          const g = (at(c.call("GET /api/edge/detail") ?? c.before("GET /api/edge/detail")) as EdgeDetail | undefined)?.edge;
          return g && !g.arms.includes("ppo") ? "this grid has no PPO arm" : null;
        },
        read: (c) => {
          const g = (at(c.call("GET /api/edge/detail") ?? c.before("GET /api/edge/detail")) as EdgeDetail | undefined)?.edge;
          const p = Number(c.q.get("p") || c.kept("seller")?.p);
          const ppo = g?.edges.find((e) => e.arm === "ppo" && near(e.premium, p));
          const sel = g?.edges.find((e) => e.arm === "delta_seller" && near(e.premium, p));
          if (!ppo || !sel) return null;
          const a = ppo.pnl.estimate ?? 0;
          const b = sel.pnl.estimate ?? 0;
          return `At ${prem(p)} vol pts PPO's edge is ${usd(a)} per episode against the seller's ${usd(b)}: ${a < b ? `the learner keeps ${b > 0 ? pct(a / b, 0) : "less"} of what the rule earns${top(ppo) ? `, mostly through ${top(ppo)}` : ""}` : "the learner earns at least as much as the rule"}.`;
        },
      },
      {
        id: "episode",
        href: "/runs/?key=latest&policy=delta_seller",
        task: "Open one episode of a seller run: under Where the P&L comes from, open By episode and by day and pick an episode.",
        target: [
          { guide: "runs-by-day", text: /by episode and by day/i },
          { guide: "runs-episode", label: /^\s*episode/i },
        ],
        done: (c) => c.calls("GET /api/runs/attribution").some((a) => a.query?.seed !== undefined && (a.data as Attribution | undefined)?.result),
        read: (c) => {
          const a = c.calls("GET /api/runs/attribution").filter((x) => x.query?.seed !== undefined).map((x) => x.data as Attribution).pop();
          const e = a?.result?.episodes.find((x) => x.seed === a.seed);
          if (!a || !e) return null;
          const t = e.totals;
          return `In episode ${a.seed} the seller made ${usd(t.pnl)}: theta ${usd(t.theta)}, gamma ${usd(t.gamma)}, vega ${usd(t.vega)}, delta ${usd(t.delta)}, costs ${usd(t.cost)}, and the residual is ${pct(e.residual_share, 1)} of gross, so the greeks account for the P&L.`;
        },
      },
    ],
  },
  {
    id: "size",
    title: "How big must my backtest be?",
    question: "How many seeds and episodes before a result means something?",
    minutes: 3,
    icon: "size",
    youWill: ["Add seeds and watch the detection floor fall", "Cut episodes and see what changes", "Compare a learner's noise with a rule's, then two real runs of different size"],
    steps: [
      {
        id: "seeds",
        href: "/power/?unit=vol",
        task: "Drag Seeds, S from 5 up to 20 or more.",
        target: { guide: "power-seeds", label: /^\s*Seeds, S/ },
        done: (c) => Boolean(powerWhere(c, (p) => p.params.n_seeds >= 20)),
        keep: (c) => ({ floor: latestPower(c)?.floor, S: latestPower(c)?.params.n_seeds }),
        read: (c) => {
          const p = powerWhere(c, (x) => x.params.n_seeds >= 20);
          if (!p) return null;
          return `At S = ${p.params.n_seeds} the smallest edge you can reliably see drops from ${vp(p.floor_vs_seeds.floor_s5 / PER_VOL)} to ${vp(p.floor / PER_VOL)} vol pts, ${num(p.floor_vs_seeds.floor_s5 / p.floor, 1)} times finer than a 5-seed backtest.`;
        },
      },
      {
        id: "episodes",
        href: "/power/?unit=vol",
        task: "Now cut Episodes per seed, N to 100 or fewer.",
        target: { guide: "power-episodes", label: /Episodes per seed, N/ },
        done: (c) => Boolean(powerWhere(c, (p) => p.params.n_episodes <= 100)),
        read: (c) => {
          const p = powerWhere(c, (x) => x.params.n_episodes <= 100);
          const before = Number(c.kept("seeds")?.floor);
          if (!p) return null;
          const within = p.params.sigma2 / (p.params.n_seeds * p.params.n_episodes);
          const share = within / (within + p.params.tau2 / p.params.n_seeds);
          return `${Number.isFinite(before) ? `The floor moved from ${vp(before / PER_VOL)} to ${vp(p.floor / PER_VOL)} vol pts: ` : ""}at N = ${num(p.params.n_episodes)} episode noise is ${pct(share, 0)} of the variance, so ${share > 0.5 ? "more episodes would still help" : "more seeds, not more episodes, is what helps"}.`;
        },
      },
      {
        id: "rule",
        href: "/power/?unit=vol",
        task: "Switch Start from to “Seller contrast (A7)”: the same test for a simple rule.",
        target: { guide: "power-preset", text: /Seller contrast/ },
        done: (c) => Boolean(powerWhere(c, (p) => near(p.params.tau2, 1026, 0.5))),
        read: (c) => {
          const s = powerWhere(c, (p) => near(p.params.tau2, 1026, 0.5));
          const l = c.get("power|a7", () => api.power({ ...A7, n_seeds: s?.params.n_seeds ?? 5, n_episodes: s?.params.n_episodes ?? 500 }));
          if (!s || !l) return null;
          return `At S = ${s.params.n_seeds} × N = ${num(s.params.n_episodes)} the rule's floor is ${vp(s.floor / PER_VOL)} vol pts against PPO's ${vp(l.floor / PER_VOL)}: a learner is ${num(l.floor / s.floor, 0)} times noisier, so it needs far more seeds for the same claim.`;
        },
      },
      compareStep(
        PAIRS.sample,
        (a, b, d) => {
          const half = (r: RunDetail) => (r.risk.se === null ? null : 1.96 * r.risk.se);
          const zero = (r: RunDetail) => (r.risk.mean ?? 0) - (half(r) ?? 0) <= 0;
          return `The same seller at 0.5 vol pts made ${usd(b.risk.mean)} ± ${usd(half(b))} per episode over ${num(b.risk.n)} episodes and ${usd(a.risk.mean)} ± ${usd(half(a))} over ${num(a.risk.n)}: four times the episodes halves the interval, ${zero(a) && zero(b) ? "and neither run alone tells a half-point edge from zero" : zero(b) ? "and only the larger run tells the edge from zero" : "and both clear zero"}; on the ${num(d.n)} paths they share the two ${Math.abs(d.diff) < 0.5 ? "agree exactly" : `agree to ${usd(Math.abs(d.diff))}`}, because more episodes add paths, they do not change the old ones.`;
        },
        (p) => `Now see it in real runs: under ${groupName(p.group)}, press Compare on “${p.b.label}” (against “${p.a.label}”).`,
      ),
    ],
  },
  {
    id: "leaks",
    title: "Could my backtest be cheating?",
    question: "Would a look-ahead bug in the data be caught before it fakes a profit?",
    minutes: 3,
    icon: "leak",
    youWill: ["Plant tomorrow's return in a world and run the detector", "Shrink the leak until it slips through", "Find the detector's floor"],
    steps: [
      {
        id: "demo",
        href: "/leaks/",
        task: "Press Run the detector: it tests a clean simulator and a twin we poisoned with tomorrow's return.",
        target: { guide: "leaks-run", text: /run the detector/i },
        done: (c) => Boolean(c.call("POST /api/leaks/demo")),
        read: (c) => {
          const d = at(c.call("POST /api/leaks/demo")) as LeakDemo | undefined;
          return d ? leakRead(d) : null;
        },
      },
      {
        id: "small",
        href: "/leaks/",
        task: "Shrink the leak: drag Leak size, eps down to 0.10 or less and press Run again.",
        target: (c) => (c.call("POST /api/leaks/demo") || Number(c.q.get("eps") ?? 0.5) > 0.1 + 1e-9 ? { guide: "leaks-eps", label: /Leak size/ } : { guide: "leaks-run", text: /run (the detector|again)/i }),
        done: (c) => c.calls("POST /api/leaks/demo").some((a) => Number(body(a).eps) <= 0.1 + 1e-9),
        read: (c) => {
          const d = c.calls("POST /api/leaks/demo").filter((a) => Number(body(a).eps) <= 0.1 + 1e-9).map((a) => a.data as LeakDemo).pop();
          return d ? leakRead(d) : null;
        },
      },
      {
        id: "floor",
        href: "/leaks/",
        task: "Find the floor: switch the view to Smallest leak caught and press Find it.",
        target: (c) => (c.q.get("view") === "floor" ? { guide: "leaks-floor", text: /^\s*(find it|run again)/i } : { guide: "leaks-view", text: /smallest leak caught/i }),
        done: (c) => Boolean(c.call("POST /api/leaks/floor")),
        read: (c) => {
          const f = at(c.call("POST /api/leaks/floor")) as LeakFloor | undefined;
          if (!f) return null;
          return f.floor === null
            ? `With ${f.seeds.length} paths the detector caught none of the planted leaks: more paths are needed before it can vouch for a backtest.`
            : `With ${f.seeds.length} paths the detector catches ${f.flagged.filter(Boolean).length} of ${f.eps.length} leak sizes, down to eps ${f.floor} of a day's return; anything smaller can slip through, so a clean verdict means "no leak above ${f.floor}".`;
        },
      },
      {
        id: "field",
        href: "/leaks/",
        task: "Plant it in one field only: switch the view to Every field, pick atm_iv in One field and press Plant in atm_iv.",
        target: (c) => (c.q.get("view") !== "field" ? { guide: "leaks-view", text: /every field/i } : c.q.get("field") !== "atm_iv" ? { guide: "leaks-field", label: /^\s*One field/ } : { guide: "leaks-plant", text: /^\s*plant in/i }),
        done: (c) => Boolean(c.call("POST /api/leaks/field")),
        read: (c) => {
          const f = at(c.call("POST /api/leaks/field")) as LeakField | undefined;
          if (!f) return null;
          return f.flagged
            ? `Planted only in ${f.field}, the leak was caught${f.named_correctly ? " and pinned on the right field" : `, though blamed on ${f.worst_feature}`} (p ${(f.p_value ?? 0) < 1e-6 ? "< 0.000001" : `= ${num(f.p_value ?? 0, 6)}`}): the detector says where to look, not just that something is wrong.`
            : `Planted only in ${f.field} at ${f.eps} sd, the leak was missed: a one-field leak this small needs more paths.`;
        },
      },
    ],
  },
  {
    id: "learner",
    featured: true,
    title: "Does a learner beat a rule?",
    question: "Does PPO find the planted premium as reliably as the simple seller?",
    minutes: 5,
    icon: "learner",
    youWill: ["Read PPO's P&L across planted premiums and test each against zero", "Compare the two detection floors", "Pair PPO with the seller, and with more training"],
    steps: [
      {
        id: "means",
        href: "/sweep/",
        task: "Pick PPO in Arm.",
        target: { guide: "sweep-arm", label: /^\s*Arm/ },
        done: (c) => c.q.get("arm") === "ppo",
        blocked: (c) => {
          const s = at(c.call("GET /api/sweeps") ?? c.before("GET /api/sweeps")) as { sweeps?: { arms: string[] }[] } | undefined;
          return s && !(s.sweeps ?? []).some((x) => x.arms.includes("ppo")) ? "no sweep with a PPO arm under this runs root" : null;
        },
        read: (c) => {
          const s = at(c.call("GET /api/sweeps/detail") ?? c.before("GET /api/sweeps/detail")) as Sweep | undefined;
          const ppo = [...(s?.oracle.ppo ?? [])].sort((a, b) => a.premium - b.premium);
          const sel = [...(s?.oracle.delta_seller ?? [])].sort((a, b) => a.premium - b.premium);
          const lo = ppo[0];
          const hi = ppo[ppo.length - 1];
          if (!lo || !hi) return null;
          const s0 = sel[0];
          const s1 = sel[sel.length - 1];
          return `PPO's mean P&L goes from ${usd(lo.mean_pnl)} at ${prem(lo.premium)} to ${usd(hi.mean_pnl)} at ${prem(hi.premium)} vol pts${s0 && s1 ? `, against the seller's ${usd(s0.mean_pnl)} to ${usd(s1.mean_pnl)}` : ""}: it does respond to the premium; the question is whether reliably.`;
        },
      },
      {
        id: "contrasts",
        href: "/sweep/",
        task: "Switch the view to Contrasts: each premium tested against the same arm at 0.",
        target: { guide: "sweep-view", text: /^\s*contrasts\s*$/i },
        done: (c) => c.q.get("view") === "contrasts",
        read: (c) => {
          const g = (at(c.call("GET /api/sweeps/detail") ?? c.before("GET /api/sweeps/detail")) as Sweep | undefined)?.grid;
          const arm = c.q.get("arm") || "ppo";
          const cs = (g?.contrasts[arm] ?? []).filter((x) => x.estimate !== null);
          if (!cs.length) return null;
          const yes = cs.filter((x) => x.detected).map((x) => prem(x.premium));
          const no = cs.filter((x) => !x.detected).map((x) => prem(x.premium));
          return `Against itself at 0, ${ARM[arm] ?? arm}'s edge is detected at ${yes.length ? `${yes.join(", ")} vol pts` : "no premium"}${no.length ? ` and not at ${no.join(", ")}` : ""}, on ${g?.n_seeds} seed blocks with a one-sided t.`;
        },
      },
      {
        id: "floors",
        href: "/sweep/",
        task: "Switch the view to Power: the chance of detection at each premium, for both arms.",
        target: { guide: "sweep-view", text: /^\s*power\s*$/i },
        done: (c) => c.q.get("view") === "power",
        blocked: (c) => {
          const m = at(c.call("GET /api/m4") ?? c.before("GET /api/m4")) as M4 | undefined;
          return m && !m.available ? "the M4 analysis has not been written on this server" : null;
        },
        read: (c) => {
          const m = at(c.call("GET /api/m4") ?? c.before("GET /api/m4")) as M4 | undefined;
          const a = m?.analysis;
          if (!a) return null;
          const f = (arm: string) => a.arms[arm]?.floor;
          const fs = f("delta_seller");
          const fp = f("ppo");
          const show = (x: typeof fs) => (!x ? "–" : x.status === "not_reached" ? `above ${vp(x.bracket[0])}` : vp(x.premium));
          return `At ${pct(a.settings.target_power)} power the seller is detected from ${show(fs)} vol pts and PPO from ${show(fp)}${a.gap?.value !== null && a.gap?.value !== undefined ? `: the learner needs ${a.gap.kind === "lower_bound" ? "at least " : ""}${vp(a.gap.value)} vol pts more premium before a ${a.design.n_seeds}-seed test sees it` : ""}.`;
        },
      },
      compareStep(PAIRS.arms, (a, b, d) => `At 1 vol pt planted, on the same ${num(d.n)} paths, PPO made ${usd(b.risk.mean)} per episode against the seller's ${usd(a.risk.mean)} (${signedUsd(d.diff)}, ${ci(d)}): ${d.clear ? (d.diff > 0 ? "on this pair the learner out-earned the rule" : "on this pair the rule beat the learner") : "on one run pair the two cannot be told apart, which is why the sweep runs many seeds"}.`),
      compareStep(
        PAIRS.training,
        (a, b, d) => `Trained 500,000 steps, PPO made ${usd(b.risk.mean)} per episode at 2 vol pts against ${usd(a.risk.mean)} after 200,000 (${signedUsd(d.diff)}, ${ci(d)}): ${!d.clear ? "more training did not clearly help on this pair; one training seed's luck is as large as the effect" : d.diff > 0 ? "more training paid" : "more training did worse here: one training seed is a noisy guide"}.`,
        (p) => `Does more training help? Under ${groupName(p.group)}, press Compare on “${p.b.label}” (against “${p.a.label}”).`,
      ),
    ],
  },
  {
    id: "reviewer",
    title: "Verify our claims",
    question: "Is every number backed by evidence, reproducible and free of look-ahead?",
    minutes: 6,
    icon: "review",
    reviewer: true,
    youWill: ["Read each check's verdict and the rule fixed in advance", "Replay a run and compare it byte for byte", "Run the leak detector and export the headline"],
    steps: [
      {
        id: "checks",
        href: "/",
        task: "Open the checks: press Checks in “Can you trust these numbers?”, under the questions.",
        target: { guide: "home-checks", text: /^\s*checks/i },
        done: (c) => c.path === "/" && c.q.get("checks") === "1",
        read: (c) => {
          const rows = at(c.call("GET /api/checks") ?? c.before("GET /api/checks")) as Check[] | undefined;
          if (!rows) return null;
          const open = rows.filter((r) => r.verdict !== "PASS");
          return `${rows.length - open.length} of ${rows.length} checks pass${open[0] ? `; the first open one is “${open[0].label}”${open[0].failed_first ? ` (${open[0].failed_first})` : ""}` : ""}. Each verdict is derived from its report's own criterion and evidence, not typed in.`;
        },
      },
      {
        id: "study",
        href: "/",
        task: "Open Pre-registration in “Can you trust these numbers?”: the rule fixed before any data.",
        target: { guide: "home-study", text: /pre-registration/i },
        done: (c) => c.path === "/" && c.q.get("study") === "1",
        read: (c) => {
          const s = at(c.call("GET /api/study") ?? c.before("GET /api/study")) as Study | undefined;
          if (!s) return null;
          return `Fixed in advance: a ${s.test}, one-sided α = ${s.alpha}, on S = ${s.n_seeds} seed blocks × N = ${num(s.n_episodes)} episodes at premiums ${s.premiums.join(", ")} vol pts; the sweep is judged against this, not against what looked good afterwards.`;
        },
      },
      {
        id: "run",
        href: "/runs/?policy=delta_seller",
        task: "Every run ever made is listed here, failures included (shown: the seller's). Open one: click a run id in the table.",
        target: { guide: "runs-open", css: 'table a[href*="key="]' },
        done: (c) => c.path.startsWith("/runs") && Boolean(c.q.get("key")) && Boolean(c.call("GET /api/runs/detail")),
        read: (c) => {
          const d = at(c.call("GET /api/runs/detail")) as RunDetail | undefined;
          if (!d) return null;
          const ok = d.artifacts.filter((a) => a.ok).length;
          return `${short(d.row.config_hash, 10)}-s${d.row.seed}: made at git ${short(d.row.git_sha, 7)}${d.row.git_dirty ? " (dirty tree)" : ""} from config ${short(d.row.config_hash, 10)}; ${ok} of ${d.artifacts.length} file digests recomputed just now match the record${ok === d.artifacts.length ? ", so nothing was edited after the run" : ""}.`;
        },
      },
      {
        id: "replay",
        href: "/runs/",
        task: "Reproduce it: press Replay. It reruns the run from its manifest alone (a PPO run retrains first and asks before).",
        target: { guide: "runs-replay", text: /^\s*(replay|start the replay|replay now)/i },
        done: (c) => [...c.calls("GET /api/runs/replay"), ...c.calls("POST /api/runs/replay")].some((a) => (a.data as Replay | undefined)?.state === "done" && a.at >= (c.calls("POST /api/runs/replay")[0]?.at ?? Infinity)),
        blocked: (c) => {
          const r = at(c.call("GET /api/runs/replay") ?? c.before("GET /api/runs/replay")) as Replay | undefined;
          return r && !r.can_replay && r.state !== "done" ? `this run cannot be replayed here: ${r.reason}` : r?.state === "failed" ? `the replay failed: ${r.note || r.reason}` : null;
        },
        read: (c) => {
          const r = [...c.calls("GET /api/runs/replay"), ...c.calls("POST /api/runs/replay")].map((a) => a.data as Replay).filter((x) => x?.state === "done").pop();
          if (!r) return null;
          return r.verdict === "IDENTICAL"
            ? `IDENTICAL: rerun from its manifest, the run came back byte for byte the same; python scripts/reproduce.py does this for every sweep cell from a clean clone.`
            : r.verdict === "AGREES"
              ? `AGREES: rerun from its manifest, every number matched within ${r.rtol} relative (largest ${num(r.max_rel_diff ?? 0, 12)}), as a retrained or cross-platform run should.`
              : `DIFFERS: the rerun did not match (${r.note || "see the replay card"}), which is a finding to report.`;
        },
      },
      {
        id: "leak",
        href: "/leaks/",
        task: "Check it could not see the future: press Run the detector.",
        target: { guide: "leaks-run", text: /run the detector/i },
        done: (c) => Boolean(c.call("POST /api/leaks/demo")),
        read: (c) => {
          const d = at(c.call("POST /api/leaks/demo")) as LeakDemo | undefined;
          return d ? leakRead(d) : null;
        },
      },
      {
        id: "export",
        href: "/sweep/?view=power",
        task: "Take the headline with you: export a figure or table under it (CSV, JSON or PNG).",
        target: { guide: "export", css: '[aria-label^="Export"]' },
        done: (c) => Boolean(c.did("export")),
        read: (c) => {
          const e = c.did("export");
          const m = at(c.call("GET /api/m4") ?? c.before("GET /api/m4")) as M4 | undefined;
          if (!e) return null;
          const g = m?.analysis?.provenance;
          return `Exported “${String(e.name)}”${g ? ` from grid ${g.grid_hash} at git ${short(g.analysis_git_sha, 7)}` : ""}; the source icon beside each figure names the runs it was computed from, so any number can be traced back.`;
        },
      },
    ],
  },
];

function top(e: { components: Record<string, { estimate: number | null }> }): string {
  const [c] = Object.entries(e.components)
    .filter(([k]) => k !== "residual")
    .sort((a, b) => Math.abs(b[1].estimate ?? 0) - Math.abs(a[1].estimate ?? 0));
  return c ? c[0] : "";
}

function edgeRead(c: Ctx, arm: string, p: number): string | null {
  const g = (at(c.call("GET /api/edge/detail") ?? c.before("GET /api/edge/detail")) as EdgeDetail | undefined)?.edge;
  const e = g?.edges.find((x) => x.arm === arm && near(x.premium, p));
  if (!e) return null;
  const parts = Object.entries(e.components)
    .filter(([k, v]) => k !== "residual" && v.estimate !== null)
    .sort((a, b) => Math.abs(b[1].estimate ?? 0) - Math.abs(a[1].estimate ?? 0))
    .slice(0, 2)
    .map(([k, v]) => `${k} ${(v.estimate ?? 0) >= 0 ? "+" : MINUS}${usd(Math.abs(v.estimate ?? 0))}`);
  return `At ${prem(p)} vol pts the seller's edge over itself at 0 is ${usd(e.pnl.estimate)} per episode (one-sided 95% ≥ ${usd(e.pnl.lower_bound)}), led by ${parts.join(" and ")}: the planted premium is paid through the options' greeks, not luck.`;
}

export function workflow(id: string | null | undefined): Workflow | undefined {
  return WORKFLOWS.find((w) => w.id === id);
}

/** A step's URL with the workflow in it. */
export function stepHref(w: Workflow, i: number): string {
  const step = w.steps[i];
  if (!step) return "/";
  const [path, hash] = step.href.split("#");
  const sep = (path ?? "").includes("?") ? "&" : "?";
  return `${path}${sep}wf=${w.id}&step=${i + 1}${hash ? `#${hash}` : ""}`;
}

/** The page a step lives on: its href's path. */
export function stepPath(s: Step): string {
  return (s.href.split(/[?#]/)[0] ?? "/") || "/";
}
