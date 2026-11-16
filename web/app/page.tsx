"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { StudyFinding } from "@/components/home/StudyFinding";
import { api, docUrl } from "@/lib/api";
import { num, signed } from "@/lib/format";
import { tourHref } from "@/lib/tours";
import { useFetch } from "@/lib/useFetch";
import s from "./landing.module.css";

/**
 * The landing page, in the shape of a product's: a hero with the way in, what the
 * platform does, how, what it has found, who it is for, questions, and the way in again.
 * The working surface is the Workbench (/workbench/); the controls are taught at /tutorial/.
 */

const ICONS: Record<string, ReactNode> = {
  planted: <path d="M12 21v-9M12 12c0-4 3-7 7-7 0 4-3 7-7 7zM12 15c0-3-2.5-5.5-6-5.5 0 3 2.5 5.5 6 5.5zM7 21h10" />,
  pair: <path d="M4 17l5-6 4 3 7-8M4 20l5-3 4 2 7-5" />,
  power: <path d="M4 20V10M9 20V4M14 20v-7M19 20v-11M3 20h18" />,
  agent: <path d="M12 3a4 4 0 0 1 4 4v1a3 3 0 0 1 2 5 3 3 0 0 1-3 5h-6a3 3 0 0 1-3-5 3 3 0 0 1 2-5V7a4 4 0 0 1 4-4zM12 8v10" />,
  leak: <path d="M12 3c3 4 6 7.5 6 11a6 6 0 0 1-12 0c0-3.5 3-7 6-11zM9.5 14.5h5" />,
  replay: <path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4M12 8v4l3 2" />,
};

const FEATURES: { icon: string; title: string; text: string; href: string; cta: string }[] = [
  { icon: "planted", title: "Markets with a known answer", text: "Simulate SPX with an edge of the size you choose, from zero up, so every test can be scored against the truth before it sees real data.", href: "/synth/", cta: "Synth data" },
  { icon: "pair", title: "Paired comparisons", text: "Two strategies on the same market paths. Luck cancels, so what is left is the difference between them, with its uncertainty.", href: "/runs/", cta: "Runs" },
  { icon: "power", title: "Power before you run", text: "How many seeds and episodes a test needs to catch an edge of a given size, from variance the pilot measured.", href: "/power/", cta: "Power lab" },
  { icon: "agent", title: "Agent benchmark", text: "Plant an edge, train a learning agent on exactly that data and score it on held-out days against a rule-based seller.", href: "/alpha/", cta: "Synthetic alpha" },
  { icon: "leak", title: "Look-ahead detection", text: "Plant a leak in any field of the data and see it caught, so a backtest cannot quietly profit from tomorrow's prices.", href: "/leaks/", cta: "Leak detector" },
  { icon: "replay", title: "Every run reproducible", text: "Each run records its commit, config and seed, and can be replayed from that record and compared byte for byte.", href: "/runs/", cta: "Runs" },
];

const STEPS: { title: string; text: string }[] = [
  { title: "Plant", text: "Pick a market and set the edge: none, one vol point, or anything between." },
  { title: "Run", text: "Launch the delta-hedged seller or a learning agent on it, with real costs." },
  { title: "Verify", text: "Compare on paired seeds, check the power, and run the leak and replay checks." },
];

const AUDIENCE: { title: string; text: string; tour: string; label: string }[] = [
  { title: "Options researchers", text: "Is the variance risk premium on SPX real after costs, and how big must a backtest be to tell?", tour: "edge", label: "Real edge, or luck?" },
  { title: "RL researchers", text: "Does an agent find an edge of known size, or does it fit noise in its training days?", tour: "alpha", label: "Does an agent find a planted edge?" },
  { title: "Reviewers", text: "Is every number backed by evidence, reproducible and free of look-ahead?", tour: "reviewer", label: "Can you trust these numbers?" },
];

const FAQ: { q: string; a: string }[] = [
  { q: "What is the variance risk premium?", a: "The gap between the volatility options price in and the volatility the market then delivers. When options are dear, a seller collects it, at the cost of large losses in a crash." },
  { q: "Why a simulated market?", a: "On real data nobody knows the true edge, so a method that finds one cannot be told from one that fits noise. In the simulator the edge is set by hand, so each method is graded first." },
  { q: "Is this trading advice?", a: "No. It is a research platform for testing whether an edge is real; it places no trades and makes no recommendation." },
  { q: "What do I need to run it?", a: "Python with the project's venv and Node. make web builds the site and make serve starts it on port 8600. The real SPX panel is optional; the simulator needs no data." },
  { q: "Where do I start?", a: "The ten-minute tutorial. It walks through the site and ends with a comparison you ran yourself." },
];

/** An illustration of the Workbench: a power chart and a verdict, drawn, not data. */
function ProductShot() {
  return (
    <div className={s.shot} aria-hidden="true">
      <div className={s.shotBar}>
        <i />
        <i />
        <i />
        <span>Vega Lab · Power lab</span>
      </div>
      <div className={s.shotBody}>
        <div className={s.shotChart}>
          <svg viewBox="0 0 320 180" preserveAspectRatio="none">
            <line x1="20" y1="44" x2="310" y2="44" className={s.shotTarget} />
            <path d="M20 168 C 90 166, 120 120, 160 80 S 250 30, 310 24" className={s.shotA} />
            <path d="M20 168 C 120 167, 170 140, 210 100 S 280 50, 310 44" className={s.shotB} />
            <path d="M20 168 L310 168" className={s.shotAxis} />
          </svg>
          <span className={s.shotLabel}>power vs. edge size · 80% target</span>
        </div>
        <div className={s.shotSide}>
          <span className={s.pill}>PASS</span>
          <b>Paired test</b>
          <span>same paths · 5 seeds</span>
          <span className={s.bar}>
            <span style={{ width: "72%" }} />
          </span>
          <span className={s.bar}>
            <span style={{ width: "48%" }} />
          </span>
        </div>
      </div>
    </div>
  );
}

function Stats() {
  const ov = useFetch((x) => api.overview(x), "overview");
  const vrp = useFetch((x) => api.vrp(x), "vrp");
  const spans = ov.data?.coverage ?? [];
  const first = spans.map((c) => c.first).sort()[0];
  const last = spans.map((c) => c.last).sort().slice(-1)[0];
  const h = vrp.data?.mapping?.headline;
  const items: { value: string; label: string }[] = [
    first && last ? { value: `${first.slice(0, 4)}–${last.slice(0, 4)}`, label: "real SPX option days" } : { value: "SPX", label: "simulated with a known edge" },
    h ? { value: `${signed(h.mean)}`, label: "vol pts: the measured premium" } : { value: "Paired", label: "seeds, so luck cancels" },
    (ov.data?.n_runs ?? 0) >= 20 ? { value: num(ov.data?.n_runs ?? 0), label: "runs, each replayable" } : { value: "Every", label: "run replayable from its record" },
    { value: "Fixed", label: "test written before the sweep" },
  ];
  return (
    <section className={s.stats} aria-label="At a glance">
      {items.map((x) => (
        <div key={x.label} className={s.stat}>
          <b>{x.value}</b>
          <span>{x.label}</span>
        </div>
      ))}
    </section>
  );
}

function Results() {
  const m4 = useFetch((x) => api.m4(x), "m4");
  const vrp = useFetch((x) => api.vrp(x), "vrp");
  const ready = Boolean(m4.data?.available && m4.data.analysis);
  return (
    <section id="results" className={s.section}>
      <div className={s.head}>
        <p className={s.kicker}>Results, not promises</p>
        <h2>What the study has found</h2>
        <p>Read live from this server: the measured power curves, the detection floors and the real premium.</p>
      </div>
      {ready ? (
        <StudyFinding m4={m4.data} vrp={vrp.data} />
      ) : (
        <div className={s.empty}>
          {m4.error
            ? "The server did not answer. Start it with make serve and reload to see the current result."
            : m4.data
              ? "The sweep's analysis is not on this server yet. When it is, the result appears here."
              : "Loading the current result…"}
        </div>
      )}
    </section>
  );
}

function Icon({ name }: { name: string }) {
  return (
    <span className={s.icon} aria-hidden="true">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        {ICONS[name]}
      </svg>
    </span>
  );
}

export default function Landing() {
  return (
    <>
      <section className={s.hero}>
        <div className={s.heroInner}>
          <div className={s.heroCopy}>
            <span className={s.badge}>Research platform for SPX options</span>
            <h1>Know if your options edge is real before you trust it.</h1>
            <p>
              Vega Lab tests the variance risk premium where the answer is known first: plant an edge in a simulated SPX market, run a seller or a learning agent on it, and
              see whether the result survives costs, luck and look-ahead. Then read the same tests on real SPX days.
            </p>
            <div className={s.heroGo}>
              <Link href="/workbench/" className={s.btnMain}>
                Open the app
              </Link>
              <Link href="/tutorial/" className={s.btnGhost}>
                Take the 10-minute tutorial →
              </Link>
            </div>
            <p className={s.heroFine}>Runs locally · no account · every number reproducible</p>
          </div>
          <ProductShot />
        </div>
      </section>

      <Stats />

      <section id="features" className={s.section}>
        <div className={s.head}>
          <p className={s.kicker}>Features</p>
          <h2>Everything a backtest needs to be believed</h2>
          <p>Each tool answers one question a skeptical reader would ask, and each is a page you can open now.</p>
        </div>
        <div className={s.features}>
          {FEATURES.map((f) => (
            <Link key={f.title} href={f.href} className={s.feature}>
              <Icon name={f.icon} />
              <h3>{f.title}</h3>
              <p>{f.text}</p>
              <span className={s.more}>{f.cta} →</span>
            </Link>
          ))}
        </div>
      </section>

      <section id="how" className={`${s.section} ${s.band}`}>
        <div className={s.head}>
          <p className={s.kicker}>How it works</p>
          <h2>Plant, run, verify</h2>
          <p>The same three steps for every question, on the simulator first and on real data after.</p>
        </div>
        <ol className={s.steps}>
          {STEPS.map((x, i) => (
            <li key={x.title} className={s.step}>
              <span className={s.stepNum}>{i + 1}</span>
              <h3>{x.title}</h3>
              <p>{x.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <Results />

      <section className={s.section}>
        <div className={s.head}>
          <p className={s.kicker}>Who it is for</p>
          <h2>Start from your question</h2>
          <p>Each one opens a guided walk-through that runs real experiments with you, one step at a time.</p>
        </div>
        <div className={s.audience}>
          {AUDIENCE.map((x) => (
            <div key={x.title} className={s.who}>
              <h3>{x.title}</h3>
              <p>{x.text}</p>
              <Link href={tourHref(x.tour)} className={s.more}>
                {x.label} →
              </Link>
            </div>
          ))}
        </div>
      </section>

      <section id="faq" className={s.section}>
        <div className={s.head}>
          <p className={s.kicker}>FAQ</p>
          <h2>Questions</h2>
        </div>
        <div className={s.faq}>
          {FAQ.map((x) => (
            <details key={x.q} className={s.qa}>
              <summary>{x.q}</summary>
              <p>{x.a}</p>
            </details>
          ))}
        </div>
      </section>

      <section className={s.final}>
        <h2>See it for yourself in ten minutes.</h2>
        <p>The tutorial ends with a paired comparison you ran on the platform yourself.</p>
        <div className={s.heroGo}>
          <Link href="/tutorial/" className={s.btnMain}>
            Start the tutorial
          </Link>
          <a href={docUrl("REPRODUCE.md")} target="_blank" rel="noreferrer" className={s.btnGhost}>
            Reproduce the study →
          </a>
        </div>
      </section>
    </>
  );
}
