"use client";

import { useMemo, useState } from "react";
import { Segmented } from "@/components/Bits";
import { InfoTip, Labelled } from "@/components/InfoTip";
import ui from "@/components/ui.module.css";
import type { M4, M4Analysis, Study, Vrp } from "@/lib/api";
import { ARM_LABEL } from "@/lib/chart";
import { MINUS, num, pct, prem, signed, usd } from "@/lib/format";
import { tCritical, tPower } from "@/lib/tpower";
import { Backtests, Fools } from "./Story";
import s from "./hook.module.css";

/**
 * The problem on Home: on the left, why a backtest alone says little (Dom's step 5 and
 * the three ways a backtest fools you); on the right, the question most studies skip and
 * a widget to move. Plant an edge of a size, pick how many seeds the study runs, and see
 * how often the study's own test would catch it, beside two dozen simulated studies drawn
 * from that test's sampling distribution. Nothing here links out: every action is in
 * "Our solution".
 *
 * The power is the study's, from the sweep (docs/M4_ANALYSIS.json, served at /api/m4):
 * each arm's measured curve gives the noncentrality at the sweep's S seeds, linear in
 * premium between its knots (as optspread.vrp.power_at_noncentrality reads it); at fixed
 * N episodes the grand mean's standard error goes as 1/sqrt(S), so at S seeds it is that
 * times sqrt(S / S_sweep), and the power is the exact noncentral t on S - 1 df
 * (lib/tpower.ts). At the real premium and the sweep's S this is the number "What we
 * found" states. Dollars are vol points times the study's $ per vol point (A6, /api/study).
 */

type Arm = "ppo" | "delta_seller";
const ARMS: readonly { value: Arm; label: string }[] = [
  { value: "ppo", label: "PPO (learns)" },
  { value: "delta_seller", label: "Simple rule" },
];
/** The Power lab's seeds axis (optspread/api/power.py FLOOR_SEEDS): dense where power moves fastest. */
const SEED_STOPS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100];
const EDGE_MAX = 2;
const EDGE_STEP = 0.05;
const DOTS = 24;

/** The arm's curve at the sweep's S: premium knots and their noncentrality. */
function knots(a: M4Analysis, arm: Arm): { x: number[]; lam: number[] } | null {
  const pts = (a.arms[arm]?.curve.points ?? []).filter((p) => p.noncentrality !== null && Number.isFinite(p.noncentrality));
  if (pts.length < 2) return null;
  return { x: pts.map((p) => p.premium), lam: pts.map((p) => p.noncentrality as number) };
}

/** np.interp: linear between knots, held at the ends. */
function interp(x: number[], y: number[], at: number): number {
  const n = x.length;
  if (at <= (x[0] ?? 0)) return y[0] ?? 0;
  if (at >= (x[n - 1] ?? 0)) return y[n - 1] ?? 0;
  let i = 1;
  while (i < n - 1 && (x[i] ?? 0) < at) i++;
  const x0 = x[i - 1] ?? 0;
  const x1 = x[i] ?? 0;
  const y0 = y[i - 1] ?? 0;
  const y1 = y[i] ?? 0;
  return y0 + ((y1 - y0) * (at - x0)) / (x1 - x0 || 1);
}

interface Model {
  /** noncentrality at `seeds` seeds */
  lam: number;
  /** the planted edge, $ of mean P&L per episode */
  edge: number;
  /** the standard error of the study's grand mean, $ */
  se: number;
  power: number;
  critical: number;
}

/**
 * The measured curve held level where it dips: the running maximum, over edge size, of the
 * noncentrality read at the sweep's S. The sweep's PPO curve dips near 0.5 vol pts from
 * sampling noise; held level, a bigger edge is never caught less often. Where the curve
 * already rises (the real premium, the seller's whole range) it is the curve itself.
 */
function held(k: { x: number[]; lam: number[] }, premium: number): number {
  let top = interp(k.x, k.lam, premium);
  k.x.forEach((x, i) => {
    if (x <= premium) top = Math.max(top, k.lam[i] ?? top);
  });
  return top;
}

function model(k: { x: number[]; lam: number[] }, premium: number, seeds: number, base: number, perVol: number, alpha: number): Model {
  const scale = Math.sqrt(seeds / base);
  const lam5 = held(k, premium);
  // the standard error the curve implies at this premium; at zero, the first segment's (the noncentrality is linear through the anchor)
  const first = k.x.findIndex((v) => v > 0);
  const se5 = premium > 0 && lam5 > 0 ? (premium * perVol) / lam5 : first > 0 && (k.lam[first] ?? 0) > 0 ? ((k.x[first] ?? 0) * perVol) / (k.lam[first] ?? 1) : perVol;
  const lam = lam5 * scale;
  return { lam, edge: premium * perVol, se: se5 / scale, power: tPower(lam, seeds, alpha), critical: tCritical(seeds - 1, alpha) };
}

/** The standard normal's quantile (Acklam's rational approximation, relative error about 1e-9). */
function normalQuantile(q: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const [a0, a1, a2, a3, a4, a5] = a as [number, number, number, number, number, number];
  const [b0, b1, b2, b3, b4] = b as [number, number, number, number, number];
  const [c0, c1, c2, c3, c4, c5] = c as [number, number, number, number, number, number];
  const [d0, d1, d2, d3] = d as [number, number, number, number];
  const lo = 0.02425;
  if (q < lo) {
    const r = Math.sqrt(-2 * Math.log(q));
    return (((((c0 * r + c1) * r + c2) * r + c3) * r + c4) * r + c5) / ((((d0 * r + d1) * r + d2) * r + d3) * r + 1);
  }
  if (q > 1 - lo) return -normalQuantile(1 - q);
  const u = q - 0.5;
  const r = u * u;
  return ((((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * u) / (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1);
}

interface Study1 {
  /** the study's estimate of the edge, $ per episode */
  est: number;
  caught: boolean;
}

/**
 * Two dozen studies, one at each of the evenly spaced quantiles (i + 0.5) / N of the
 * estimate's sampling distribution: the edge plus normal noise of the grand mean's
 * standard error. The test catches the share p of all studies, so the top round(p N) are
 * filled and the filled share is the headline's. Deterministic: study i is always the
 * same quantile, so moving the edge slides the dots and adding seeds pulls them in.
 */
function draw(m: Model): Study1[] {
  const caught = Math.round(m.power * DOTS);
  return Array.from({ length: DOTS }, (_, i) => ({ est: m.edge + m.se * normalQuantile((i + 0.5) / DOTS), caught: i >= DOTS - caught }));
}

function stops(real: number | null): number[] {
  const out = Array.from({ length: Math.round(EDGE_MAX / EDGE_STEP) + 1 }, (_, i) => Math.round(i * EDGE_STEP * 100) / 100);
  if (real !== null && real > 0 && real < EDGE_MAX && !out.includes(real)) out.push(real);
  return out.sort((a, b) => a - b);
}

const nearest = (xs: number[], v: number) => xs.reduce((best, x, i) => (Math.abs(x - v) < Math.abs((xs[best] ?? 0) - v) ? i : best), 0);

// the strip, in SVG user units: the studies stacked over a line of $ per episode
const W = 560;
const H = 132;
const AXIS = 100;
const R = 5.5;
const PADX = 14;
const ROWS = 5;

/**
 * Where each study sits: across at its estimate, stacked into the first free row of its
 * slot, the nearest slot with room when a slot is full, so a tight cluster (the seller)
 * reads as a block of dots rather than one dot drawn 24 times.
 */
function layout(studies: Study1[], x: (v: number) => number): { cx: number; cy: number }[] {
  const slot = 2 * R + 1.5;
  const used = new Map<number, number>();
  const out: { cx: number; cy: number }[] = new Array(studies.length);
  const order = studies.map((d, i) => ({ i, at: x(d.est) })).sort((a, b) => a.at - b.at);
  for (const { i, at } of order) {
    const home = Math.round(at / slot);
    let k = home;
    for (let step = 0; step < 200; step++) {
      k = home + (step % 2 ? (step + 1) / 2 : -step / 2);
      if ((used.get(k) ?? 0) < ROWS) break;
    }
    const row = used.get(k) ?? 0;
    used.set(k, row + 1);
    out[i] = { cx: Math.max(PADX, Math.min(W - PADX, k * slot)), cy: AXIS - R - 3 - row * (2 * R + 2) };
  }
  return out;
}

function Plot({ arm, studies, m, domain, seeds, premium }: { arm: Arm; studies: Study1[]; m: Model; domain: [number, number]; seeds: number; premium: number }) {
  const [lo, hi] = domain;
  const x = (v: number) => PADX + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (W - 2 * PADX);
  const at = layout(studies, x);
  const caught = studies.filter((d) => d.caught).length;
  const colour = arm === "ppo" ? s.ppo : s.seller;
  const x0 = x(0);
  const xe = x(m.edge);
  return (
    <svg className={s.plot} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${DOTS} simulated ${seeds}-seed studies of ${ARM_LABEL[arm]} with ${prem(premium)} vol pts planted: ${caught} of them caught by the test`}>
      <line x1={PADX} x2={W - PADX} y1={AXIS} y2={AXIS} className={s.zero} />
      <line x1={x0} x2={x0} y1={22} y2={AXIS + 10} className={s.zeroEdge} />
      {/* zero's label reads leftward from its line, the planted edge's rightward from its own, so they never meet */}
      <g className={`${s.move} ${colour}`} style={{ transform: `translateX(${xe}px)` }}>
        <line x1={0} x2={0} y1={22} y2={AXIS + 10} className={s.edge} />
        <text x={5} y={AXIS + 24} className={s.edgeLab}>
          {premium === 0 ? "planted edge = zero" : `planted edge ${usd(m.edge)}`}
        </text>
      </g>
      {premium === 0 ? null : (
        <text x={x0 - 5} y={AXIS + 24} className={s.zeroLab} textAnchor="end">
          zero edge
        </text>
      )}
      <text x={W - PADX} y={12} className={s.legendLab} textAnchor="end">
        ● caught ○ missed
      </text>
      {studies.map((d, i) => (
        <circle key={i} r={R} cx={0} cy={0} className={`${s.dot} ${colour} ${d.caught ? s.caught : s.missed}`} style={{ transform: `translate(${at[i]?.cx ?? 0}px, ${at[i]?.cy ?? 0}px)` }}>
          <title>{`A study measuring ${d.est < 0 ? MINUS : "+"}${usd(Math.abs(d.est))} per episode: ${d.caught ? "caught" : "missed"}`}</title>
        </circle>
      ))}
    </svg>
  );
}

function Stepped({ id, label, term, values, at, onChange, show }: { id: string; label: string; term: string; values: number[]; at: number; onChange: (i: number) => void; show: string }) {
  return (
    <div className={s.slider}>
      <span className={s.sliderLab}>
        <Labelled term={term}>
          <label htmlFor={id}>{label}</label>
        </Labelled>
      </span>
      <input id={id} className={ui.range} type="range" min={0} max={values.length - 1} step={1} value={at} aria-valuetext={show} onChange={(e) => onChange(Number(e.target.value))} />
      <span className={s.val}>{show}</span>
    </div>
  );
}

/** "almost every time (99%)", "61% of the time". */
const often = (p: number) => (p >= 0.95 ? `almost every time (${pct(p, 0)})` : `${pct(p, 0)} of the time`);

export function Hook({ m4, study, vrp }: { m4?: M4; study?: Study; vrp?: Vrp }) {
  const a = m4?.available ? m4.analysis : null;
  const h = vrp?.mapping?.headline;
  const realMean = m4?.vrp?.estimate ?? h?.mean ?? null;
  const realLo = m4?.vrp?.interval[0] ?? h?.mean_low ?? null;
  const realHi = m4?.vrp?.interval[1] ?? h?.mean_high ?? null;
  // the real premium as the site prints it (two decimals), so the default reads as "What we found" does
  const real = realMean === null ? null : Math.round(realMean * 100) / 100;
  const edges = useMemo(() => stops(real), [real]);
  const base = a?.design.n_seeds ?? null;
  const [arm, setArm] = useState<Arm>("ppo");
  const [edgeAt, setEdgeAt] = useState<number | null>(null);
  const [seedAt, setSeedAt] = useState<number | null>(null);
  const ei = edgeAt ?? nearest(edges, real ?? 1);
  const si = seedAt ?? nearest(SEED_STOPS, base ?? 5);
  const premium = edges[ei] ?? 0;
  const seeds = SEED_STOPS[si] ?? 5;
  const perVol = study?.delta_per_vol_pt;
  const alpha = a?.settings.alpha;
  const curves = useMemo(() => (a ? { ppo: knots(a, "ppo"), delta_seller: knots(a, "delta_seller") } : null), [a]);
  const k = curves?.[arm] ?? null;
  const other = curves?.[arm === "ppo" ? "delta_seller" : "ppo"] ?? null;
  const m = a && k && base && perVol && alpha ? model(k, premium, seeds, base, perVol, alpha) : null;
  const mOther = a && other && base && perVol && alpha ? model(other, premium, seeds, base, perVol, alpha) : null;
  const studies = m ? draw(m) : [];
  // one scale for both arms and every setting, so the dots move rather than the axis: the noisier arm at the fewest seeds
  const domain = useMemo<[number, number] | null>(() => {
    if (!curves?.ppo || !base || !perVol || !alpha) return null;
    const ppo = curves.ppo;
    const worst = Math.max(...edges.map((p) => model(ppo, p, SEED_STOPS[0] ?? 2, base, perVol, alpha).se));
    return [-2.4 * worst, EDGE_MAX * perVol + 2.4 * worst];
  }, [curves, base, perVol, alpha, edges]);

  // the left column: step 5 of the basics, the real premium, and the ways a backtest fools you
  const text = (
    <div className={s.text}>
      <h3 className={s.title}>Every answer about which strategy to choose comes from a backtest, and luck scatters it.</h3>
      <p>
        The edge an option seller is chasing is small and noisy.
        {realMean !== null ? (
          <>
            {" "}
            On real SPX data we measure a premium of about {signed(realMean)} vol points <InfoTip term="vrp" />, and its 95% interval runs from {signed(realLo)} to {signed(realHi)}
            {realLo !== null && realLo < 0 ? ": we cannot even be sure it is positive." : "."}
          </>
        ) : null}
      </p>
      <p>So a profitable backtest is weak evidence on its own, whether it comes from a simple rule or a learning agent.</p>
      <Backtests />
      <Fools />
    </div>
  );
  const ask = <p className={s.q}>What most studies never report is how small an edge their setup could reliably detect, and whether a learning agent finds a real edge as well as a simple rule does.</p>;

  if (!a || !m || !domain || !perVol || !base || !alpha) {
    return (
      <div className={`${ui.card} ${s.hook}`}>
        {text}
        <div className={s.right}>
          {ask}
          <div className={s.widget}>
            <p className={s.sub}>{m4?.error || (m4 && !m4.available) ? "The sweep's analysis is not on this server, so the study's noise cannot be shown." : "Loading the study's measured noise…"}</p>
          </div>
        </div>
      </div>
    );
  }

  const caught = studies.filter((d) => d.caught).length;
  const who = arm === "ppo" ? "the learning agent (PPO)" : "the simple rule (the fixed-delta seller)";
  const gap = mOther && premium > 0 ? (arm === "ppo" ? mOther.power - m.power : m.power - mOther.power) : 0;
  const foot =
    !mOther || premium === 0
      ? "With nothing planted, whatever is caught is luck: the test's false-alarm rate."
      : arm === "ppo"
        ? `The simple rule catches the same edge ${often(mOther.power)}${gap > 0.05 ? ": that gap is the cost of learning." : "."}`
        : `The learning agent (PPO) catches the same edge ${gap > 0.05 ? "only " : ""}${pct(mOther.power, 0)} of the time${gap > 0.05 ? ": that gap is the cost of learning." : "."}`;
  return (
    <div className={`${ui.card} ${s.hook}`}>
      {text}
      <div className={s.right}>
        {ask}
        <div className={s.widget}>
        <div className={s.lead}>
        <p className={s.headline} aria-live="polite" data-hook="headline">
          {premium === 0 ? (
            <>
              With no edge, a {seeds}-seed study still “catches” one <em>{pct(m.power, 0)}</em> of the time
            </>
          ) : (
            <>
              A {seeds}-seed study catches this edge <em>{pct(m.power, 0)}</em> of the time
            </>
          )}
        </p>
        <p className={s.sub}>
          Each dot is one simulated backtest of {who} at this edge. Filled dots are “caught”: a one-sided t test at the {pct(alpha, 0)} level says the edge is real.{" "}
          <InfoTip
            label="What caught means"
            title="Caught"
            text={`The study's own test, fixed before the sweep: a one-sided t test on the ${seeds} seed means at the ${pct(alpha, 0)} level. A study catches the edge when its t statistic clears ${num(m.critical, 2)}, the 95th percentile of t on ${seeds - 1} degrees of freedom. The headline is the exact share over all possible studies, from ${ARM_LABEL[arm]}'s measured power curve in the sweep (${m4?.file ?? "M4_ANALYSIS.json"}), scaled from ${base} seeds by √(S/${base}); the ${DOTS} dots sit at evenly spaced quantiles, so ${caught} of them are filled. Dollars use the study's ${usd(perVol)} per vol point. The study's measured curve dips near 0.5 vol pts from sampling noise (see What we found); here it is held level so a bigger edge is never caught less often.`}
          />
        </p>
        </div>
        <div className={s.controls}>
          <Stepped
            id="hook-edge"
            label="Edge size (vol points)"
            term="vol_points"
            values={edges}
            at={ei}
            onChange={setEdgeAt}
            show={premium === real ? `${prem(premium)} · the real premium` : `${prem(premium)} · ${usd(premium * perVol)} per episode`}
          />
          <Stepped id="hook-seeds" label="Seeds" term="seeds" values={SEED_STOPS} at={si} onChange={setSeedAt} show={seeds === base ? `${seeds} · the study's design` : String(seeds)} />
          <div className={s.seg}>
            <Segmented label="Strategy" value={arm} options={ARMS} onChange={setArm} />
          </div>
        </div>
        <Plot arm={arm} studies={studies} m={m} domain={domain} seeds={seeds} premium={premium} />
        <p className={s.foot}>{foot}</p>
        </div>
      </div>
    </div>
  );
}
