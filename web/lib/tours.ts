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
 *           /data/ q (premium|surface|sim|seller), date, sigma · /workbench/ checks, study (the
 *           the workbench's panels) · /alpha/ dataset, gen and the knobs by name (premium_pts,
 *           n_days, n_train, n_heldout, ...), arm, timesteps, seeds
 *   targets a control is found by data-guide="<name>" first (lib/guide.ts), else by
 *           its visible text or label (see Target). Names are <page>-<control>:
 *           launch-run, launch-preset-<name> · runs-question-<group>, runs-pair-<a>|<b> (a
 *           catalog pair's Compare), runs-pick-<run key>, runs-compare, runs-open,
 *           runs-replay, runs-by-day, runs-episode · power-effect, power-seeds,
 *           power-episodes, power-preset · leaks-run, leaks-eps, leaks-view, leaks-floor,
 *           leaks-field, leaks-plant · sweep-arm, sweep-view · attribution-arm,
 *           attribution-premium · data-date, data-next-day, data-sigma-from ·
 *           home-checks, home-study · alpha-generator, alpha-generate,
 *           alpha-dataset-<hash> (an earlier dataset's row), alpha-arm, alpha-steps,
 *           alpha-seeds, alpha-train, alpha-bench-<arm>-<steps>-s<seeds> (a result's row,
 *           data-selected when it is the one shown), alpha-floor-arm-<arm>, alpha-floor-run,
 *           alpha-real-run · export (every figure's export bar).
 */

import type { Activity, Call } from "./activity";
import type { AlphaDataset, AlphaJob, AlphaOptions, Attribution, Benchmark, BenchStat, Check, EdgeDetail, Grid, Job, LeakCoverage, LeakDemo, LeakField, LeakFloor, M4, Power, Replay, RunDetail, Seller, Simulated, Study, Sweep, Term, Vrp } from "./api";
import { api } from "./api";
import { type Catalog, type PairedCompare, runsApi } from "./api_runs";
import { groupName, notBuilt, type Pair, PAIRS, type PairSpec, pairGuide, resolvePair, runIdOf } from "./catalog";
import { MINUS, num, pct, prem, short, usd } from "./format";
import { fieldWord, luck, shareOf, shareWord } from "./leakwords";
import type { components } from "./schema";

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
  /** Ring this one only while no other target of the step is on the page. */
  fallback?: boolean;
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
  /** Why the action cannot be done on this server (Next unlocks, saying so), or null: a
   * lower-case fragment the strip puts after "Not on this server:", or a full sentence it shows as is. */
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
  icon: "edge" | "market" | "pnl" | "size" | "leak" | "learner" | "review" | "tune" | "costs" | "planted";
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

/**
 * The reader picked a day in this step. The data page keeps a date in its address only
 * when it is not the latest, so one change counts even when the answer for the day it
 * opened on was cut short (a quick change aborts it, and the log then has one date only).
 */
const pickedDay = (c: Ctx) => dates(c).changed || Boolean(c.q.get("date") && dates(c).now === c.q.get("date"));

/** At-the-money vol per tenor from the gridded surface: the column nearest ln(K/F) = 0. */
function atmOfGrid(g: Grid): { date: string; pts: { d: number; v: number | null }[] } {
  let j = 0;
  g.k.forEach((k, i) => {
    if (Math.abs(k) < Math.abs(g.k[j] ?? Infinity)) j = i;
  });
  return { date: g.date, pts: g.tenor_days.map((d, i) => ({ d, v: g.iv[i]?.[j] ?? null })) };
}

/**
 * The detector's run in a sentence, in the page's own verdict: caught on the poisoned copy,
 * and (once the page has planted it in each field alone at the same size) in every field.
 */
function leakRead(d: LeakDemo, cov?: LeakCoverage): string {
  const caught = cov ? cov.controls.filter((x) => x.flagged && x.named_correctly).length : null;
  if (d.caught && !d.clean.flagged) {
    const head = `with ${shareOf(d.eps)} of tomorrow's move leaking into today, the poisoned copy set the alarm off on ${fieldWord(d.poisoned.worst_feature)} (signal ${num(Math.abs(d.poisoned.worst_t ?? 0), 1)}) while the clean one stayed quiet`;
    if (cov && !cov.passed)
      return `Partly caught, so the test reads FAIL: ${head}, but planted in each field alone it was caught in only ${caught} of ${cov.controls.length}; a leak this small slips past in some fields with ${d.seeds.length} paths.`;
    return `Caught${cov ? ` in all ${cov.controls.length} fields too` : ""}: ${head}, so a backtest that peeks ahead cannot hide here.`;
  }
  if (d.caught) return `The poisoned copy set the alarm off on ${fieldWord(d.poisoned.worst_feature)}, but the clean one did too: a false alarm, which counts as a failure.`;
  return `Missed: with ${shareOf(d.eps)} of tomorrow leaking and ${d.seeds.length} paths, the alarm stayed quiet, an honest miss; smaller leaks need more data to see.`;
}

/** The page's every-field check at the same size as this demo, when it has answered. */
const coverageAt = (c: Ctx, eps: number) =>
  c
    .calls("POST /api/leaks/coverage")
    .filter((a) => near(Number(body(a).eps), eps, 1e-9))
    .map((a) => a.data as LeakCoverage)
    .pop();


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

/** A comparison is open on the Runs page, and it is not this pair. */
function otherPair(c: Ctx, p: Pair): boolean {
  const a = c.q.get("a");
  const b = c.q.get("b");
  if (!c.path.startsWith("/runs") || !a || !b) return false;
  const want = new Set([p.a.key, p.b.key]);
  return !(want.has(a) && want.has(b) && a !== b);
}

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
      const ask = task ? task(p) : `${p.question} Under ${groupName(p.group)}, press Compare on “${p.b.label}” (against “${p.a.label}”).`;
      return otherPair(c, p) ? `This is another comparison: go back to the questions, then ${ask.charAt(0).toLowerCase()}${ask.slice(1)}` : ask;
    },
    target: (c) => {
      const p = pairOf(c, spec);
      if (!isPair(p)) return undefined;
      // another pair open: the way back to the questions
      if (otherPair(c, p)) return { guide: "runs-back", text: /back to the questions|all runs/i };
      if (c.q.get("a") && c.q.get("b")) return undefined;
      // the question's own Compare button; the table's ticks only while the questions are not drawn
      return [
        { guide: pairGuide(p.a.key, p.b.key) },
        { guide: `runs-pick-${p.a.key}`, aria: new RegExp(`^Compare ${runIdOf(p.a.key)}$`), fallback: true },
        { guide: `runs-pick-${p.b.key}`, aria: new RegExp(`^Compare ${runIdOf(p.b.key)}$`), fallback: true },
      ];
    },
    done: (c) => {
      const p = pairOf(c, spec);
      return isPair(p) && Boolean(pairCompare(c, p));
    },
    blocked: (c) => {
      const p = pairOf(c, spec);
      if (typeof p === "string") return p;
      const miss = isPair(p) ? [p.a, p.b].find((r) => !r.present) : undefined;
      return miss ? notBuilt(catalogOf(c), `The example run “${miss.label}” (${miss.key}) isn't`) : null;
    },
    read: (c) => {
      const p = pairOf(c, spec);
      const r = isPair(p) ? pairCompare(c, p) : undefined;
      return isPair(p) && r ? say(r.a, r.b, r.d, p) : null;
    },
  };
}

/** "$447 more per episode" or "$447 less". */
const diffText = (x: number) => `${usd(Math.abs(x))} ${x >= 0 ? "more" : "less"}`;

/**
 * How the worst 5% of episodes moved, B against A: "fatter" or "thinner" by `by` (a share
 * of A's), or null when it moved less than a tenth, too little to call.
 */
function tailMove(tailB: number | null | undefined, tailA: number | null | undefined): { way: "fatter" | "thinner"; by: number } | null {
  if (tailB === null || tailB === undefined || tailA === null || tailA === undefined || tailA === 0) return null;
  const by = Math.abs(tailB - tailA) / Math.abs(tailA);
  if (by < 0.1) return null;
  return { way: tailB < tailA ? "fatter" : "thinner", by };
}

/** "its tail came out 62% fatter", "… about half as fat", or that it barely moved. */
const tailWords = (m: { way: "fatter" | "thinner"; by: number } | null) =>
  !m ? "the tail barely moved" : `its tail came out ${m.way === "thinner" && m.by >= 0.4 && m.by <= 0.6 ? "materially thinner, about half as fat" : `${pct(m.by, 0)} ${m.way}`}`;

/** The plain verdict on a change: did it earn more, and did its tail move (by a tenth or more)? */
function tradeOff(d: Paired, tailB: number | null, tailA: number | null, change: string): string {
  const m = tailMove(tailB, tailA);
  const tail = tailWords(m);
  if (!d.clear) return `${change} made no difference to the mean you could tell from luck at this sample size, ${m ? "but" : "and"} ${tail}`;
  return `${change} earned ${d.diff > 0 ? "more" : "less"}, and ${tail}`;
}


// ---------------------------------------------------------------- synthetic alpha

type AlphaReal = components["schemas"]["AlphaRealOut"];
type AlphaRealJob = components["schemas"]["AlphaRealJobOut"];
type RealTest = components["schemas"]["RealTest"];

/**
 * The small world the Synthetic alpha workflow opens the page on, every knob in its query:
 * 4 vol pts of premium on 20-day paths, 16 to train on and 64 held out; PPO at 4,096 steps on
 * 2 seeds. Small enough that each step finishes in seconds to a minute.
 */
const ALPHA_START = "/alpha/?gen=const_vrp&premium_pts=4&n_days=20&n_train=16&n_heldout=64&arm=ppo&timesteps=4096&seeds=2";
const ALPHA_ARM: Record<string, string> = { ...ARM, a2c: "A2C", dqn: "DQN" };
const genOf = (d: AlphaDataset | undefined) => (d?.spec as { generator?: { kind?: string } } | undefined)?.generator?.kind;
const zOf = (s: BenchStat | null | undefined) => (s && s.value !== null && s.value !== undefined && s.se ? s.value / s.se : 0);
const pmUsd = (s: BenchStat) => (s.value === null || s.value === undefined ? "none" : `${usd(s.value)} ± ${usd(2 * (s.se ?? 0))}`);
const g3 = (x: number) => (Math.abs(x) >= 10 ? num(x, 1) : String(Number(x.toPrecision(3))));
/** "1 vol point", "2 vol points", "0.7 vol points". */
const sized = (x: number, unit: string, show = g3) => `${show(x)} ${x === 1 ? unit.replace(/points$/, "point") : unit}`;

/** The newest job of this step's `post` that matches and that the server reports done, with the body it was sent. */
function alphaJob(ctx: Ctx, post: Call, poll: Call, match: (b: Record<string, unknown>) => boolean): { job: AlphaJob; body: Record<string, unknown> } | undefined {
  const posts = ctx.calls(post).filter((a) => match(body(a)));
  const polls = ctx.calls(poll);
  for (const p of [...posts].reverse()) {
    const first = p.data as AlphaJob | undefined;
    if (!first?.job_id) continue;
    let job = first;
    for (const s of polls) if (s.at >= p.at && (s.data as AlphaJob | undefined)?.job_id === first.job_id) job = s.data as AlphaJob;
    if (job.state === "done") return { job, body: body(p) };
  }
  return undefined;
}

/** A job of this step that failed or was cancelled, so the strip says why rather than wait. */
function alphaFailed(ctx: Ctx, post: Call, poll: Call): string | null {
  const ids = new Set(ctx.calls(post).map((p) => (p.data as AlphaJob | undefined)?.job_id));
  const bad = ctx
    .calls(poll)
    .map((a) => a.data as AlphaJob | undefined)
    .filter((j) => j && ids.has(j.job_id) && (j.state === "failed" || j.state === "cancelled"))
    .pop();
  if (!bad) return null;
  return bad.state === "cancelled" ? `the job was ${bad.error || "cancelled"}` : `the job failed: ${bad.error || bad.phase}`;
}

/** The dataset the first step made, and whether the page shows another one now. */
function alphaDs(c: Ctx): { hash: string; other: boolean } | undefined {
  const hash = c.kept("dataset")?.hash;
  if (typeof hash !== "string" || !hash) return undefined;
  const open = c.q.get("dataset");
  return { hash, other: Boolean(c.path.startsWith("/alpha") && open && open !== hash) };
}

/** "This is another dataset: open 1a2b3c4d again from Earlier datasets, then …", or the task as it is. */
function onAlphaDs(c: Ctx, task: string): string {
  const d = alphaDs(c);
  return d?.other ? `This is another dataset: open ${short(d.hash)} again from Earlier datasets, then ${task.charAt(0).toLowerCase()}${task.slice(1)}` : task;
}

/** No dataset open (a reload of a bare /alpha/): the way to one, while a step's own control is not drawn. */
const noDataset = (c: Ctx): Target[] => (c.q.get("dataset") ? [] : [{ guide: "alpha-generate", text: /^\s*generate and freeze\s*$/i, fallback: true }]);

/** Another dataset open: its way back, the workflow's row under Earlier datasets. */
function backToAlphaDs(c: Ctx): Target | undefined {
  const d = alphaDs(c);
  return d?.other ? { guide: `alpha-dataset-${d.hash}`, css: `tr[data-guide="alpha-dataset-${d.hash}"]` } : undefined;
}

const alphaOptions = (c: Ctx) => at(c.call("GET /api/alpha/options") ?? c.before("GET /api/alpha/options")) as AlphaOptions | undefined;

/** Capture ratio, overfit gap and noise fit of a benchmark, in one plain sentence. */
function alphaBenchRead(b: Benchmark): string {
  const arm = ALPHA_ARM[b.arm] ?? b.arm;
  const seeds = b.n_seeds ?? 1;
  const r = b.capture_ratio;
  const att = b.attainable.value ?? 0;
  const take =
    r.value === null || r.value === undefined
      ? `the oracle found no measurable edge to take, so there is no capture ratio`
      : zOf(b.captured) > 2
        ? `${arm} took ${pct(r.value)} ± ${pct(2 * (r.se ?? 0))} of the ${usd(att)} per episode the oracle earned (capture ratio)`
        : zOf(b.captured) < -2
          ? `${arm} did worse where the edge was planted than on its noise twin, so it took none of the oracle's ${usd(att)} per episode (capture ratio ${pct(r.value)} ± ${pct(2 * (r.se ?? 0))})`
          : `${arm}'s capture ratio, ${pct(r.value)} ± ${pct(2 * (r.se ?? 0))} of the oracle's ${usd(att)} per episode, cannot be told from zero`;
  const over = zOf(b.overfit_gap) > 2 ? `it did ${usd(b.overfit_gap.value)} per episode better on its own train paths than on new ones (overfit gap), so it learned those paths` : `its overfit gap, ${pmUsd(b.overfit_gap)}, shows no clear overfitting`;
  const n = b.noise_fit;
  const noise = !n ? "" : zOf(n) > 2 ? `, and a copy trained on pure noise “earned” ${usd(n.value)} per episode in sample only (noise fit): it memorises noise` : `, and the copy trained on pure noise shows no memorised noise (noise fit ${pmUsd(n)})`;
  return `On ${seeds} seed${seeds === 1 ? "" : "s"} of ${num(b.total_timesteps ?? 0)} steps, ${take}; ${over}${noise}.`;
}

/** The seller's detection floor from an edge sweep, in one plain sentence that says how rough it is. */
function alphaFloorRead(sw: NonNullable<AlphaJob["sweep"]>): string | null {
  const f = sw.floor;
  if (!f) return null;
  const unit = sw.edge_unit ?? "units";
  const who = ALPHA_ARM[sw.spec.arm] ?? sw.spec.arm;
  const on = `${f.n} ${f.unit ?? "figures"}`;
  const sizes = sw.spec.values.map(Number).filter((v) => v > 0);
  const list = `${sizes.map((v) => g3(v)).join(", ")} ${unit}`;
  const g2 = (x: number) => String(Number(x.toPrecision(2)));
  const Who = who.charAt(0).toUpperCase() + who.slice(1);
  const head =
    f.status === "untestable"
      ? `There is no floor for ${who}: each size needs at least two seeds or path blocks to test`
      : f.status === "not_reached"
        ? `${Who} is not found reliably at any planted size swept (${list}): even at the largest its power is ${pct(f.max_power ?? 0)}, under 80%, so its floor is above ${sized(f.bracket?.[0] ?? Math.max(...sizes), unit)}`
        : f.status === "below_grid"
          ? `${Who} is found reliably at every planted size swept (${list}), so its floor is below the smallest, ${sized(f.bracket?.[1] ?? Math.min(...sizes), unit)}: about ${sized(f.edge ?? 0, unit, g2)} on the line drawn to zero`
          : `${Who} is found 80% of the time from about ${sized(f.edge ?? 0, unit, g2)} of planted edge, between the swept ${g3(f.bracket?.[0] ?? 0)} and ${sized(f.bracket?.[1] ?? 0, unit)}`;
  const alarm = f.null_detected ? `, though it also “found” an edge at 0, where none is planted, so the curve is suspect` : "";
  return `${head}${alarm}; each size is tested on only ${on}, so read the floor as a bracket between swept sizes, not a precise number.`;
}

/** Whether a real-data test is of the result the train step made (any, before it has kept one). */
function ofKept(c: Ctx, r: RealTest): boolean {
  const d = alphaDs(c);
  const key = c.kept("train")?.key;
  return (!d || r.dataset === d.hash) && (typeof key !== "string" || key === `${r.arm}-${r.total_timesteps ?? 0}-s${r.n_seeds}`);
}

/**
 * The real-data test of this step: from a job it finished, or a result already on disk
 * for that benchmark, which the page may have read before the step began (it reads the
 * selected result's test as soon as it shows it).
 */
function alphaRealOf(c: Ctx): RealTest | undefined {
  const job = alphaJob(c, "POST /api/alpha/real", "GET /api/alpha/real/jobs/*", () => true)?.job as AlphaRealJob | undefined;
  if (job?.result && ofKept(c, job.result)) return job.result;
  const reads = [c.before("GET /api/alpha/real"), ...c.calls("GET /api/alpha/real")];
  return reads
    .map((a) => (a?.data as AlphaReal | undefined)?.result ?? undefined)
    .filter((r): r is RealTest => Boolean(r) && ofKept(c, r as RealTest))
    .pop();
}

function alphaRealRead(r: RealTest): string {
  const arm = r.rows.find((x) => x.policy === "arm");
  if (!arm) return "";
  const who = ALPHA_ARM[r.arm] ?? r.arm;
  const z = zOf(r.edge);
  const vs = z > 2 ? "beat flat by more than 2 se there" : z < -2 ? "lost against flat by more than 2 se there" : "cannot be told from flat there";
  const redo = r.reproduced === false ? " (its rebuilt policies did not reproduce the benchmark, so these are policies retrained here)" : "";
  return `On ${num(r.n_windows)} overlapping real SPX windows of ${r.n_days} days (${r.first_date} to ${r.last_date}), ${who}${redo} made ${usd(arm.real_mean)} ± ${usd(2 * arm.real_se)} a window against ${usd(arm.synthetic_mean)} ± ${usd(2 * arm.synthetic_se)} on held-out synthetic paths, and ${vs}; real chains carry the market's own premium, not the planted one, so this is exploratory.`;
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
        task: "Size the test for a learned strategy: set Effect to detect to 0.25 vol pts, then raise Seeds (S) until power at the effect reads 80% or more.",
        target: (c) => ((latestPower(c)?.params.delta ?? PER_VOL) > 0.25 * PER_VOL + 0.5 ? { guide: "power-effect", label: /Effect to detect/ } : { guide: "power-seeds", label: /^\s*Seeds \(S\)/ }),
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
      compareStep(PAIRS.regime, (a, b, d) => {
        const m = tailMove(b.risk.cvar5, a.risk.cvar5);
        const mean = d.clear ? (d.diff < 0 ? "the wilder market cost the seller money" : "the wilder market paid the seller more") : `the wilder market moved the mean by ${signedUsd(d.diff)}, but the paired interval still includes zero at this sample size`;
        return `With the same 1 vol pt planted, the seller averaged ${usd(b.risk.mean)} per episode in a 30% vol world against ${usd(a.risk.mean)} in the default 20% one (${signedUsd(d.diff)} on the same paths, ${ci(d)}), with the worst 5% at ${usd(b.risk.cvar5)} against ${usd(a.risk.cvar5)}: ${mean}, and ${tailWords(m)}.`;
      }),
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
        done: pickedDay,
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
          const hit = v ? powerWhere(c, (x) => Math.abs(x.params.delta / PER_VOL - (v.mean ?? 0)) <= 0.051) : undefined;
          if (!v || !hit || v.mean === null || v.mean === undefined) return null;
          // the slider moves in 0.05s: the power at the premium itself, with the reader's design and noise
          const exact = { ...hit.params, delta: Math.round(v.mean * PER_VOL * 100) / 100 };
          const p = c.get(`power|vrp|${JSON.stringify(exact)}`, () => api.power(exact)) ?? hit;
          const whose = hit.params.tau2 === A7.tau2 && hit.params.sigma2 === A7.sigma2 ? "with PPO's noise" : near(hit.params.tau2, 1026, 0.5) ? "with the fixed-delta seller's noise" : "with the noise you set";
          const m4 = c.get("m4", () => api.m4()) as M4 | undefined;
          const study = m4?.mapped?.length ? `; the study's sweep measured ${m4.mapped.map((r) => `${pct(r.power, 0)} for ${ARM[r.arm] ?? r.arm}`).join(" and ")} at this premium` : "";
          return `At the market's average premium of ${vp(v.mean)} vol pts (95% ${vp(v.mean_low)} to ${vp(v.mean_high)}), a backtest of S = ${p.params.n_seeds} seeds × ${num(p.params.n_episodes)} episodes ${whose} catches it ${pct(p.power, 0)} of the time${p.power < 0.5 ? ", so most backtests of that size would call a real premium noise" : p.power < 0.8 ? `, short of the 80% a design should have` : ", large enough to see at this size"}${study}.`;
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
        task: "Pick the smallest premium above zero in Planted premium, with the fixed-delta seller in Details for (the default).",
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
          const via = greeks(ppo);
          const how = via ? `; it earns ${via}` : "";
          if (!ppo.pnl.rejected)
            return `At ${prem(p)} vol pts PPO's estimated edge is ${usd(a)} per episode against the seller's ${usd(b)}, but it is not detected (one-sided 95% ≥ ${signedUsd(ppo.pnl.lower_bound)}): the sweep cannot yet tell it from zero, let alone from the rule${how}.`;
          return `At ${prem(p)} vol pts PPO's edge is ${usd(a)} per episode against the seller's ${usd(b)}, detected (one-sided 95% ≥ ${signedUsd(ppo.pnl.lower_bound)}): ${a < b ? `the learner keeps ${b > 0 ? pct(a / b, 0) : "less"} of what the rule earns` : "the learner earns more than the rule here"}${how}.`;
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
        task: "Drag Seeds (S) from 5 up to 20 or more.",
        target: { guide: "power-seeds", label: /^\s*Seeds \(S\)/ },
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
        task: "Now cut Episodes per seed (N) to 100 or fewer.",
        target: { guide: "power-episodes", label: /Episodes per seed \(N\)/ },
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
        task: "Switch Start from to “Seller contrast”: the same test for a simple rule.",
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
          const ha = half(a);
          const hb = half(b);
          const times = (b.risk.n ?? 0) > 0 ? (a.risk.n ?? 0) / (b.risk.n ?? 1) : null;
          const narrower = ha && hb ? hb / ha : null;
          return `The same seller at 0.5 vol pts made ${usd(b.risk.mean)} ± ${usd(hb)} per episode over ${num(b.risk.n)} episodes and ${usd(a.risk.mean)} ± ${usd(ha)} over ${num(a.risk.n)}: ${times === null || narrower === null ? "more episodes narrow the interval" : `${num(times, 0)} times the episodes makes the interval ${num(narrower, 1)} times narrower (about √${num(times, 0)})`}, ${zero(a) && zero(b) ? "and neither run alone tells a half-point edge from zero" : zero(b) ? "and only the larger run tells the edge from zero" : "and both clear zero"}; on the ${num(d.n)} paths they share the two ${Math.abs(d.diff) < 0.5 ? "agree exactly" : `agree to ${usd(Math.abs(d.diff))}`}, because more episodes add paths, they do not change the old ones.`;
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
        // the steps run in order: a reader already on a later step's view is told this one comes first
        task: (c) =>
          c.q.get("view") === "floor" || c.q.get("view") === "field"
            ? `Step 1 of 4 first: press Run the detector above; ${c.q.get("view") === "floor" ? "Smallest leak caught is step 3" : "Every field is step 4"}.`
            : "Press Run the detector: it tests a clean copy of the market and a twin we poisoned with tomorrow's move.",
        target: { guide: "leaks-run", text: /run the detector/i },
        done: (c) => Boolean(c.call("POST /api/leaks/demo")),
        read: (c) => {
          const d = at(c.call("POST /api/leaks/demo")) as LeakDemo | undefined;
          return d ? leakRead(d, coverageAt(c, d.eps)) : null;
        },
      },
      {
        id: "small",
        href: "/leaks/",
        task: "Shrink the leak: drag How much of tomorrow leaks down to a tenth (0.1) or less and press Run again.",
        target: (c) => (c.call("POST /api/leaks/demo") || Number(c.q.get("eps") ?? 0.5) > 0.1 + 1e-9 ? { guide: "leaks-eps", label: /How much of tomorrow leaks/ } : { guide: "leaks-run", text: /run (the detector|again)/i }),
        // done once the page has also planted it in each field alone at that size, so the reading says what the page's verdict says
        done: (c) => c.calls("POST /api/leaks/demo").some((a) => Number(body(a).eps) <= 0.1 + 1e-9 && Boolean(coverageAt(c, Number(body(a).eps)))),
        read: (c) => {
          const d = c.calls("POST /api/leaks/demo").filter((a) => Number(body(a).eps) <= 0.1 + 1e-9).map((a) => a.data as LeakDemo).pop();
          return d ? leakRead(d, coverageAt(c, d.eps)) : null;
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
            : `With ${f.seeds.length} paths the alarm catches ${f.flagged.filter(Boolean).length} of ${f.eps.length} leak sizes, down to ${shareOf(f.floor)} of tomorrow's move; anything smaller can slip through, so a quiet alarm means "no leak of ${shareWord(f.floor)} or more".`;
        },
      },
      {
        id: "field",
        href: "/leaks/",
        task: "Plant it in one field only: switch the view to Every field, pick at-the-money implied vol (atm_iv) in One field and press Plant.",
        target: (c) => (c.q.get("view") !== "field" ? { guide: "leaks-view", text: /every field/i } : c.q.get("field") !== "atm_iv" ? { guide: "leaks-field", label: /^\s*One field/ } : { guide: "leaks-plant", text: /^\s*plant in/i }),
        done: (c) => Boolean(c.call("POST /api/leaks/field")),
        read: (c) => {
          const f = at(c.call("POST /api/leaks/field")) as LeakField | undefined;
          if (!f) return null;
          return f.flagged
            ? `Planted only in ${fieldWord(f.field)}, the leak was caught${f.named_correctly ? " and pinned on the right field" : `, though blamed on ${fieldWord(f.worst_feature)}`} (chance this is luck: ${luck(f.p_value)}): the detector says where to look, not just that something is wrong.`
            : `Planted only in ${fieldWord(f.field)} at ${shareWord(f.eps)} of its usual swing, the leak was missed: a one-field leak this small needs more paths.`;
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
        href: "/sweep/?view=cells",
        task: "Open Profit by premium and pick PPO in Arm.",
        // the Arm field is off in the power view the page opens on: ring the view switch first
        target: (c) => (["cells", "contrasts"].includes(c.q.get("view") ?? "") ? { guide: "sweep-arm", label: /^\s*Arm/ } : { guide: "sweep-view", text: /^\s*profit by premium\s*$/i }),
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
        href: "/sweep/?view=cells",
        task: "Switch the view to Edge over zero premium: each premium tested against the same arm at 0.",
        target: { guide: "sweep-view", text: /^\s*edge over zero premium\s*$/i },
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
        href: "/sweep/?view=contrasts",
        task: "Switch the view to How often it's caught: the chance of detection at each premium, for both arms.",
        target: { guide: "sweep-view", text: /^\s*how often it.s caught\s*$/i },
        // the page's default view, so the URL drops it: a bare /sweep/ is the power view
        done: (c) => c.path.startsWith("/sweep") && (c.q.get("view") ?? "power") === "power",
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
          const show = (x: typeof fs) => (!x ? "–" : x.status === "not_reached" ? `above ${vp(x.bracket[0])} vol pts` : x.status === "below_grid" ? `${vp(x.premium)} vol pts (estimated from the sweep, below its smallest premium)` : `${vp(x.premium)} vol pts`);
          return `At ${pct(a.settings.target_power)} power the seller is detected from ${show(fs)} and PPO from ${show(fp)}${a.gap?.value !== null && a.gap?.value !== undefined ? `: the learner needs ${a.gap.kind === "lower_bound" ? "at least " : ""}${vp(a.gap.value)} vol pts more premium before a ${a.design.n_seeds}-seed test sees it` : ""}.`;
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
    id: "alpha",
    featured: true,
    title: "Does an agent find a planted edge?",
    question: "Plant a premium of known size: how much of it does PPO take, and how big must it be to be found?",
    minutes: 4,
    icon: "planted",
    youWill: ["Freeze a small world with 4 vol pts planted, and its noise twin", "Train PPO on it and read what it took, overfit and memorised", "Find the seller's detection floor, then try real SPX days"],
    steps: [
      {
        id: "dataset",
        href: ALPHA_START,
        task: "The form is filled in with a small world: Constant variance premium, 4 vol pts planted, 20-day paths, 16 to train on and 64 held out. Press Generate and freeze.",
        target: { guide: "alpha-generate", text: /^\s*generate and freeze\s*$/i },
        done: (c) => c.calls("POST /api/alpha/datasets").some((a) => genOf(a.data as AlphaDataset) === "const_vrp"),
        keep: (c) => ({
          hash: c
            .calls("POST /api/alpha/datasets")
            .map((a) => a.data as AlphaDataset)
            .filter((d) => genOf(d) === "const_vrp")
            .pop()?.hash,
        }),
        read: (c) => {
          const d = c
            .calls("POST /api/alpha/datasets")
            .map((a) => a.data as AlphaDataset)
            .filter((x) => genOf(x) === "const_vrp")
            .pop();
          if (!d) return null;
          const pts = Number((d.spec as { generator?: { premium_pts?: number } }).generator?.premium_pts);
          const planted = Number.isFinite(pts) ? `implied vol sits ${g3(pts)} vol pts above the vol that drives the path` : `${d.edge.charAt(0).toLowerCase()}${d.edge.slice(1).replace(/\.$/, "")}`;
          return `Dataset ${short(d.hash)} is frozen: ${num(d.n_train)} train and ${num(d.n_heldout)} held-out paths of ${num(Number(d.spec.n_days))} days where ${planted}, beside its noise twin, the same random shocks with nothing planted; an agent only ever trains on the ${num(d.n_train)} train paths.`;
        },
      },
      {
        id: "train",
        href: "/alpha/",
        task: (c) => onAlphaDs(c, "Train PPO on it: Arm PPO, 4,096 timesteps and 2 training seeds are filled in. Press Train and score; it trains four times (each seed on the signal and on the twin), so allow a minute."),
        target: (c) => backToAlphaDs(c) ?? [{ guide: "alpha-train", text: /^\s*train and score\s*$/i }, ...noDataset(c)],
        done: (c) => Boolean(alphaJob(c, "POST /api/alpha/benchmarks", "GET /api/alpha/jobs/*", (b) => b.arm === "ppo" && Number(b.n_seeds) >= 2)?.job.benchmark),
        blocked: (c) => {
          const o = alphaOptions(c);
          if (o && !o.learners_available) return "PPO is not installed on this server (the train extra)";
          return alphaFailed(c, "POST /api/alpha/benchmarks", "GET /api/alpha/jobs/*");
        },
        keep: (c) => {
          const b = alphaJob(c, "POST /api/alpha/benchmarks", "GET /api/alpha/jobs/*", (x) => x.arm === "ppo" && Number(x.n_seeds) >= 2)?.job.benchmark;
          return b ? { key: `${b.arm}-${b.total_timesteps ?? 0}-s${b.n_seeds ?? 1}` } : undefined;
        },
        read: (c) => {
          const b = alphaJob(c, "POST /api/alpha/benchmarks", "GET /api/alpha/jobs/*", (x) => x.arm === "ppo" && Number(x.n_seeds) >= 2)?.job.benchmark;
          return b ? alphaBenchRead(b) : null;
        },
      },
      {
        id: "floor",
        href: "/alpha/",
        task: (c) => onAlphaDs(c, "How big must the edge be? In that card tick only the fixed-delta seller (untick PPO), keep the sizes 0, 1, 2, 4 and press Find each arm's floor. A rule needs no training, so it takes seconds."),
        target: (c) =>
          backToAlphaDs(c) ?? [
            // the checkboxes only while they are wrong: PPO still ticked, the seller not yet
            { guide: "alpha-floor-untick-ppo", css: '[data-guide="alpha-floor-arm-ppo"]:has(input:checked)' },
            { guide: "alpha-floor-tick-seller", css: '[data-guide="alpha-floor-arm-delta_seller"]:has(input:not(:checked))' },
            { guide: "alpha-floor-run", text: /find each arm.s floor/i },
            ...noDataset(c),
          ],
        done: (c) => Boolean(alphaJob(c, "POST /api/alpha/sweeps", "GET /api/alpha/jobs/*", (b) => b.arm === "delta_seller" && b.axis === "edge")?.job.sweep?.floor),
        blocked: (c) => alphaFailed(c, "POST /api/alpha/sweeps", "GET /api/alpha/jobs/*"),
        read: (c) => {
          const sw = alphaJob(c, "POST /api/alpha/sweeps", "GET /api/alpha/jobs/*", (b) => b.arm === "delta_seller" && b.axis === "edge")?.job.sweep;
          return sw ? alphaFloorRead(sw) : null;
        },
      },
      {
        id: "real",
        href: "/alpha/",
        task: (c) => onAlphaDs(c, "Optional: score PPO on real SPX days. With PPO's row picked in Agents on this dataset, press Test on real SPX days in On real SPX days. It retrains each seed, then plays every real window from 2019 to 2022, a minute or two."),
        target: (c) => {
          const back = backToAlphaDs(c);
          if (back) return back;
          const key = c.kept("train")?.key;
          return [
            // PPO's row while another result is the one shown
            ...(typeof key === "string" ? [{ guide: "alpha-pick-ppo", css: `tr[data-guide="alpha-bench-${key}"]:not([data-selected])` }] : []),
            { guide: "alpha-real-run", text: /^\s*test on real spx days\s*$/i },
            ...noDataset(c),
          ];
        },
        done: (c) => Boolean(alphaRealOf(c)),
        blocked: (c) => {
          const r = at(c.call("GET /api/alpha/real") ?? c.before("GET /api/alpha/real")) as AlphaReal | undefined;
          if (r && !r.available) return "there is no real SPX panel on this server, so this optional step is skipped";
          if (r?.refused) return r.refused;
          const bad = c
            .calls("GET /api/alpha/real/jobs/*")
            .map((a) => a.data as AlphaRealJob | undefined)
            .filter((j) => j?.state === "failed")
            .pop();
          return bad ? `the real-data test failed: ${bad.error || bad.phase}` : null;
        },
        read: (c) => {
          const r = alphaRealOf(c);
          return r ? alphaRealRead(r) : null;
        },
      },
    ],
  },
  {
    id: "reviewer",
    title: "Can you trust these numbers?",
    question: "Is every number backed by evidence, reproducible and free of look-ahead?",
    minutes: 6,
    icon: "review",
    reviewer: true,
    youWill: ["Read each check's verdict and the rule fixed in advance", "Replay a run and compare it byte for byte", "Run the leak detector and export the headline"],
    steps: [
      {
        id: "checks",
        href: "/workbench/",
        task: "Open the checks: press Checks in “Can you trust these numbers?”, under the questions.",
        target: { guide: "home-checks", text: /^\s*checks/i },
        done: (c) => c.path.startsWith("/workbench") && c.q.get("checks") === "1",
        read: (c) => {
          const rows = at(c.call("GET /api/checks") ?? c.before("GET /api/checks")) as Check[] | undefined;
          if (!rows) return null;
          const open = rows.filter((r) => r.verdict !== "PASS");
          return `${rows.length - open.length} of ${rows.length} checks pass${open[0] ? `; the first open one is “${open[0].label}”${open[0].failed_first ? ` (${open[0].failed_first})` : ""}` : ""}. Each verdict is derived from its report's own criterion and evidence, not typed in.`;
        },
      },
      {
        id: "study",
        href: "/workbench/",
        task: "Open Pre-registration in “Can you trust these numbers?”: the rule fixed before any data.",
        target: { guide: "home-study", text: /pre-registration/i },
        done: (c) => c.path.startsWith("/workbench") && c.q.get("study") === "1",
        read: (c) => {
          const s = at(c.call("GET /api/study") ?? c.before("GET /api/study")) as Study | undefined;
          if (!s) return null;
          return `Fixed in advance: a ${s.test}, α = ${s.alpha}, on S = ${s.n_seeds} seed blocks × N = ${num(s.n_episodes)} episodes at premiums ${s.premiums.join(", ")} vol pts; the sweep is judged against this, not against what looked good afterwards.`;
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
          return d ? leakRead(d, coverageAt(c, d.eps)) : null;
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

/** The greek an edge is earned through, and the one that gives most of it back: "through theta (+$1,909), giving back gamma (−$1,700)". */
function greeks(e: { components: Record<string, { estimate: number | null }> }): string {
  const xs = Object.entries(e.components)
    .filter(([k, v]) => k !== "residual" && v.estimate !== null)
    .sort((a, b) => Math.abs(b[1].estimate ?? 0) - Math.abs(a[1].estimate ?? 0));
  const gain = xs.find(([, v]) => (v.estimate ?? 0) > 0);
  const loss = xs.find(([, v]) => (v.estimate ?? 0) < 0);
  if (!gain) return "";
  const back = loss && Math.abs(loss[1].estimate ?? 0) > 0.25 * (gain[1].estimate ?? 0) ? `, giving back ${loss[0]} (${signedUsd(loss[1].estimate)})` : "";
  return `through ${gain[0]} (${signedUsd(gain[1].estimate)})${back}`;
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

/** A workflow's first step with `start=1`, the link a workflow card opens; "/workbench/" for an unknown id. */
export function tourHref(id: string): string {
  const w = WORKFLOWS.find((x) => x.id === id);
  return w ? `${stepHref(w, 0)}&start=1` : "/workbench/";
}

/** The page a step lives on: its href's path. */
export function stepPath(s: Step): string {
  return (s.href.split(/[?#]/)[0] ?? "/") || "/";
}
