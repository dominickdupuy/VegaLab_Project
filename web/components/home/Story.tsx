"use client";

import { type ReactNode, useState } from "react";
import s from "./story.module.css";

/**
 * The basics on Home, told in order: what an option is, why sellers earn a premium, the
 * simple rule that collects it, the agent that tries to beat the rule, and the backtest
 * that judges both, which is where the problem starts. Each has a drawing; they are
 * grouped into three columns, all visible at once.
 * The drawings are schematic, not data: the colours follow the charts (--s1 the seller
 * and implied vol, --s2 the agent and realised vol), and text keeps the text colours.
 */

const W = 360;
const H = 200;

function Fig({ label, children }: { label: string; children: ReactNode }) {
  return (
    <svg className={s.fig} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {children}
    </svg>
  );
}

/** 1. A short put: the premium kept above the strike, a loss that grows below it. */
function OptionFig() {
  const zero = 120;
  const k = 190;
  const be = 30 + (k - 30) * ((190 - zero) / (190 - 80)); // where the payoff crosses zero
  return (
    <Fig label="A put seller's profit at expiry: flat at the premium above the strike, falling below it as the index drops">
      <line x1="30" y1={zero} x2="340" y2={zero} className={s.axis} />
      <line x1="30" y1="20" x2="30" y2="185" className={s.axis} />
      <polygon points={`${be},${zero} ${k},80 340,80 340,${zero}`} className={s.fill1} />
      <polygon points={`30,190 ${be},${zero} 30,${zero}`} className={s.fill2} />
      <polyline points={`30,190 ${k},80 340,80`} className={s.line1} />
      <line x1={k} y1="72" x2={k} y2={zero + 6} className={s.dash} />
      <text x={k} y="64" className={s.lab} textAnchor="middle">
        strike
      </text>
      <text x="268" y="70" className={s.lab} textAnchor="middle">
        premium kept
      </text>
      <text x={be + 14} y="166" className={s.lab}>
        ← loss if the index falls
      </text>
      <text x="340" y={zero + 16} className={s.cap} textAnchor="end">
        S&amp;P 500 at expiry →
      </text>
      <text x="36" y="30" className={s.cap}>
        seller's profit
      </text>
    </Fig>
  );
}

// a fixed, hand-shaped path: realised vol under implied most of the time, one crash above it
const IMPLIED = [78, 76, 79, 77, 75, 78, 80, 77, 76, 79, 78, 76, 74, 77, 79, 78];
const REALISED = [104, 112, 99, 108, 115, 102, 110, 98, 106, 111, 101, 96, 22, 70, 104, 108];

/** 2. Implied vol above realised vol, the gap shaded, and the crash where it flips. */
function PremiumFig() {
  const x = (i: number) => 30 + (i * 310) / (IMPLIED.length - 1);
  const imp = IMPLIED.map((y, i) => `${x(i)},${y}`).join(" ");
  const rea = REALISED.map((y, i) => `${x(i)},${y}`).join(" ");
  const gap = `${imp} ${REALISED.map((y, i) => `${x(i)},${y}`).reverse().join(" ")}`;
  return (
    <Fig label="Over time, the volatility options price in sits above the volatility that happens, except in a crash">
      <line x1="30" y1="175" x2="340" y2="175" className={s.axis} />
      <polygon points={gap} className={s.fillGap} />
      <polyline points={rea} className={s.line2} />
      <polyline points={imp} className={s.line1} />
      <text x="34" y="60" className={s.lab}>
        implied vol: what options charge
      </text>
      <text x="34" y="146" className={s.lab}>
        realised vol: what happened
      </text>
      <text x={x(4)} y="97" className={s.labStrong} textAnchor="middle">
        the premium
      </text>
      <text x={x(12) + 8} y="24" className={s.lab}>
        crash
      </text>
      <text x="340" y="191" className={s.cap} textAnchor="end">
        time →
      </text>
    </Fig>
  );
}

/** 3. Where the seller sells: a strike out in the tail of where the index might end. */
function SellerFig() {
  const mid = 210;
  const sd = 48;
  const base = 170;
  const y = (v: number) => base - 120 * Math.exp(-0.5 * ((v - mid) / sd) ** 2);
  const pts: [number, number][] = [];
  for (let v = 30; v <= 345; v += 5) pts.push([v, y(v)]);
  const k = mid - 1.3 * sd;
  const curve = pts.map(([a, b]) => `${a},${b}`).join(" ");
  const tail = `30,${base} ${pts
    .filter(([a]) => a <= k)
    .map(([a, b]) => `${a},${b}`)
    .join(" ")} ${k},${y(k)} ${k},${base}`;
  const body = `${k},${base} ${k},${y(k)} ${pts
    .filter(([a]) => a > k)
    .map(([a, b]) => `${a},${b}`)
    .join(" ")} 345,${base}`;
  return (
    <Fig label="The distribution of where the index ends; the seller's strike sits in the left tail, so most outcomes keep the premium">
      <polygon points={body} className={s.fill1} />
      <polygon points={tail} className={s.fill2} />
      <polyline points={curve} className={s.line1} />
      <line x1="30" y1={base} x2="345" y2={base} className={s.axis} />
      <line x1={mid} y1="40" x2={mid} y2={base} className={s.dash} />
      <line x1={k} y1="60" x2={k} y2={base} className={s.mark} />
      <text x={mid + 4} y="36" className={s.cap}>
        today
      </text>
      <text x={k} y="52" className={s.labStrong} textAnchor="middle">
        sell here
      </text>
      <text x="218" y="163" className={s.lab}>
        usually: keep it
      </text>
      <text x="34" y="190" className={s.lab}>
        rarely: pay out
      </text>
    </Fig>
  );
}

/** 4. The reinforcement learning loop: the market shows a state, the agent acts, a reward comes back. */
function AgentFig() {
  return (
    <Fig label="The learning loop: the market shows the agent prices and positions, the agent chooses which options to sell, and its profit after costs comes back as the reward">
      <defs>
        <marker id="story-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" className={s.head} />
        </marker>
      </defs>
      <rect x="20" y="70" width="120" height="60" rx="10" className={s.box1} />
      <rect x="220" y="70" width="120" height="60" rx="10" className={s.box2} />
      <text x="80" y="96" className={s.boxT} textAnchor="middle">
        Market
      </text>
      <text x="80" y="114" className={s.cap} textAnchor="middle">
        simulated SPX
      </text>
      <text x="280" y="96" className={s.boxT} textAnchor="middle">
        Agent
      </text>
      <text x="280" y="114" className={s.cap} textAnchor="middle">
        PPO
      </text>
      <path d="M100 66 C 140 22, 220 22, 260 66" className={s.arrow} markerEnd="url(#story-arrow)" />
      <path d="M260 134 C 220 178, 140 178, 100 134" className={s.arrow} markerEnd="url(#story-arrow)" />
      <text x="180" y="24" className={s.lab} textAnchor="middle">
        sees prices, vols, positions
      </text>
      <text x="180" y="176" className={s.lab} textAnchor="middle">
        acts: which options to sell
      </text>
      <text x="180" y="196" className={s.cap} textAnchor="middle">
        reward: profit after costs
      </text>
    </Fig>
  );
}

/** A seeded walk, so the drawing is the same on every load. */
function walk(seed: number, n: number, drift: number): number[] {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [0];
  for (let i = 1; i < n; i++) out.push((out[i - 1] ?? 0) + drift + (rand() - 0.5) * 2.2);
  return out;
}

/** End labels nudged apart so none overlaps another, keeping their order top to bottom. */
function spread<T extends { y: number }>(labels: T[], gap = 15): T[] {
  const out = [...labels].sort((a, b) => a.y - b.y).map((l) => ({ ...l }));
  for (let i = 1; i < out.length; i++) {
    const prev = out[i - 1];
    const cur = out[i];
    if (prev && cur && cur.y - prev.y < gap) cur.y = prev.y + gap;
  }
  return out;
}

/** 5. The same strategy on different random markets: the results fan out around a small true edge. */
function BacktestFig() {
  const h = H;
  const n = 40;
  const runs = [11, 23, 37, 41, 58, 64, 79, 83].map((seed) => walk(seed, n, 0.08));
  const all = runs.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  // the paths stop short of the right edge, so the end labels sit inside the drawing
  const x = (i: number) => 24 + (i * 262) / (n - 1);
  const y = (v: number) => h - 30 - ((v - lo) / (hi - lo || 1)) * (h - 60);
  const ends = runs.map((r) => r[n - 1] ?? 0);
  const best = ends.indexOf(Math.max(...ends));
  const worst = ends.indexOf(Math.min(...ends));
  const path = (r: number[]) => r.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return (
    <Fig label="Eight backtests of the same strategy on different random markets: some end well up, some down, around a small true edge">
      <line x1="24" y1={y(0)} x2={x(n - 1)} y2={y(0)} className={s.axis} />
      {runs.map((r, i) => (i === best || i === worst ? null : <polyline key={i} points={path(r)} className={s.faint} />))}
      <line x1="24" y1={y(0)} x2={x(n - 1)} y2={y(0.08 * (n - 1))} className={s.dash} />
      <polyline points={path(runs[best] ?? [])} className={s.line1} />
      <polyline points={path(runs[worst] ?? [])} className={s.line2} />
      {spread([
        { key: "lucky", y: y(ends[best] ?? 0), cls: s.lab },
        { key: "true edge", y: y(0.08 * (n - 1)), cls: s.cap },
        { key: "unlucky", y: y(ends[worst] ?? 0), cls: s.lab },
      ]).map((l) => (
        <text key={l.key} x={x(n - 1) + 6} y={l.y + 4} className={l.cls}>
          {l.key}
        </text>
      ))}
      <text x="24" y={h - 6} className={s.cap}>
        same strategy, same edge, eight random markets
      </text>
    </Fig>
  );
}

/** Card 3's list: the three ways a backtest turns noise into profit, each with a small icon. */
const FOOLS: { name: string; text: string; icon: ReactNode }[] = [
  {
    name: "A lucky draw",
    text: "One good run looks like skill.",
    icon: (
      <>
        <rect x="3" y="3" width="22" height="22" rx="5" className={s.box1} />
        <circle cx="9.5" cy="9.5" r="2" className={s.pip} />
        <circle cx="14" cy="14" r="2" className={s.pip} />
        <circle cx="18.5" cy="18.5" r="2" className={s.pip} />
      </>
    ),
  },
  {
    name: "Costs left out",
    text: "Profit that never paid for its trades.",
    icon: (
      <>
        <circle cx="14" cy="14" r="11" className={s.box1} />
        <text x="14" y="19" textAnchor="middle" className={s.coin}>
          $
        </text>
      </>
    ),
  },
  {
    name: "Peeking at tomorrow",
    text: "A bug that lets future prices leak in.",
    icon: (
      <>
        <rect x="4" y="6" width="20" height="18" rx="3" className={s.box2} />
        <path d="M4 11h20M10 3.5V8M18 3.5V8" className={s.cal} />
        <path d="M9 18h8m-3-3 3 3-3 3" className={s.cal} />
      </>
    ),
  },
];

/**
 * The basics as three columns, all visible at once and ending level: options and their
 * premium (Dom's steps 1 and 2), the two ways to collect it (3 and 4), and the backtest
 * that judges both (5) and the three ways it fools you, which leads on to the problem. A click, Enter or Space marks a
 * card, one at a time, so a reader can keep their place while talking it through; a link
 * inside a card never marks it.
 */
export function Story() {
  const [on, setOn] = useState<number | null>(null);
  const toggle = (i: number) => setOn((x) => (x === i ? null : i));
  const card = (i: number, body: ReactNode, after?: ReactNode) => (
    <article
      className={`${s.col} ${on === i ? s.colOn : ""}`}
      onClick={(e) => {
        if (!(e.target as HTMLElement).closest("a")) toggle(i);
      }}
    >
      <div
        role="button"
        tabIndex={0}
        aria-pressed={on === i}
        className={s.hit}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle(i);
          }
        }}
      >
        {body}
      </div>
      {after}
    </article>
  );
  return (
    <div className={s.cols}>
      {card(
        0,
        <>
          <h3>Options, and why they&apos;re priced a little high</h3>
          <p>An option is a bet on how far a price moves. A put seller is paid a premium up front, keeps it if the S&amp;P 500 stays above the strike, and pays out if it falls below.</p>
          <div className={s.figWrap}>
            <OptionFig />
          </div>
          <p>Options are usually priced for bigger swings than then happen. That gap, the variance risk premium, is what a seller earns on average, and gives back in a crash.</p>
          <div className={s.figWrap}>
            <PremiumFig />
          </div>
        </>,
      )}
      {card(
        1,
        <>
          <h3>Two ways to collect it: a simple rule, or a learner</h3>
          <p>The fixed-delta seller sells options a set distance from today&apos;s price and rolls them on a schedule. It learns nothing; it is our baseline.</p>
          <div className={s.figWrap}>
            <SellerFig />
          </div>
          <p>A reinforcement learning agent (PPO) sees the market each day, chooses what to sell, and is rewarded with its profit after costs.</p>
          <div className={s.figWrap}>
            <AgentFig />
          </div>
        </>,
      )}
      {card(
        2,
        <>
          <h3>Every answer comes from a backtest, and luck scatters it</h3>
          <p>To judge either one you run it on market data. Run the same strategy on different random markets and the results scatter: with an edge this small, luck can look like skill, and skill like luck.</p>
          <div className={s.figWrap}>
            <BacktestFig />
          </div>
          <div className={s.fools}>
            <p className={s.foolsTitle}>Three ways a backtest fools you</p>
            <ul>
              {FOOLS.map((f) => (
                <li key={f.name}>
                  <svg className={s.icon} viewBox="0 0 28 28" width="28" height="28" aria-hidden="true">
                    {f.icon}
                  </svg>
                  <span>
                    <b>{f.name}</b>
                    {f.text}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </>,
        <p className={s.ask}>
          <b>So how do you tell a real edge from luck?</b> <a href="#problem">See the problem ↓</a>
        </p>,
      )}
    </div>
  );
}
