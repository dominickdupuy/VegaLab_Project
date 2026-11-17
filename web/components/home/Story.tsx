import type { ReactNode } from "react";
import s from "./story.module.css";

/**
 * The basics on Home, told in order: what an option is, why sellers earn a premium, the
 * simple rule that collects it, the agent that tries to beat the rule, and the backtest
 * that judges both, which is where the problem starts. Each chapter has a drawing.
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
  const n = 40;
  const runs = [11, 23, 37, 41, 58, 64, 79, 83].map((seed) => walk(seed, n, 0.08));
  const all = runs.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const x = (i: number) => 30 + (i * 300) / (n - 1);
  const y = (v: number) => 170 - ((v - lo) / (hi - lo || 1)) * 140;
  const ends = runs.map((r) => r[n - 1] ?? 0);
  const best = ends.indexOf(Math.max(...ends));
  const worst = ends.indexOf(Math.min(...ends));
  const path = (r: number[]) => r.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return (
    <Fig label="Eight backtests of the same strategy on different random markets: some end well up, some down, around a small true edge">
      <line x1="30" y1={y(0)} x2="330" y2={y(0)} className={s.axis} />
      {runs.map((r, i) => (i === best || i === worst ? null : <polyline key={i} points={path(r)} className={s.faint} />))}
      <line x1="30" y1={y(0)} x2="330" y2={y(0.08 * (n - 1))} className={s.dash} />
      <polyline points={path(runs[best] ?? [])} className={s.line1} />
      <polyline points={path(runs[worst] ?? [])} className={s.line2} />
      {spread([
        { key: "lucky", y: y(ends[best] ?? 0), cls: s.lab },
        { key: "true edge", y: y(0.08 * (n - 1)), cls: s.cap },
        { key: "unlucky", y: y(ends[worst] ?? 0), cls: s.lab },
      ]).map((l) => (
        <text key={l.key} x="336" y={l.y + 4} className={l.cls}>
          {l.key}
        </text>
      ))}
      <text x="30" y="194" className={s.cap}>
        same strategy, same edge, eight random markets
      </text>
    </Fig>
  );
}

const CHAPTERS: { title: string; text: ReactNode; fig: ReactNode }[] = [
  {
    title: "An option is a bet on how far a price moves",
    text: "An option gives the right, not the obligation, to buy (a call) or sell (a put) at a set price, the strike, until a set date. SPX options are on the S&P 500. The seller of a put is paid a premium up front, keeps it if the index stays above the strike, and pays out if it falls below.",
    fig: <OptionFig />,
  },
  {
    title: "Options are usually priced a little high",
    text: "Every option's price implies a volatility, how much the market expects prices to swing. The swings that then happen are usually smaller. That gap is the variance risk premium, measured in vol points. Sellers earn it on average and give much of it back in a crash.",
    fig: <PremiumFig />,
  },
  {
    title: "A simple rule collects it",
    text: "The fixed-delta seller sells options a set distance from today's price, out where the index rarely ends up, and rolls them on a schedule. It doesn't learn anything. It is our baseline: the edge a simple rule already gets.",
    fig: <SellerFig />,
  },
  {
    title: "Can a learning agent do better?",
    text: "A reinforcement learning agent learns by trial and error. Each day it sees the market, picks which options to sell (or none), and is rewarded with its profit after costs. Over many simulated episodes it leans toward what paid. Ours uses PPO, a standard algorithm.",
    fig: <AgentFig />,
  },
  {
    title: "Every answer comes from a backtest",
    text: "To judge the rule or the agent, you run it on market data and see what it made. Run the same strategy on different random markets, though, and the results scatter. With an edge this small, luck can look like skill, and skill can look like luck.",
    fig: <BacktestFig />,
  },
];

export function Story() {
  return (
    <>
      <ol className={s.story}>
        {CHAPTERS.map((c, i) => (
          <li key={c.title} className={s.chapter}>
            <span className={s.num} aria-hidden="true">
              {i + 1}
            </span>
            <div className={s.copy}>
              <h3>{c.title}</h3>
              <p>{c.text}</p>
            </div>
            <div className={s.figWrap}>{c.fig}</div>
          </li>
        ))}
      </ol>
      <p className={s.next}>
        Which leads to the problem: <a href="#problem">how do you tell a real edge from luck? ↓</a>
      </p>
    </>
  );
}
