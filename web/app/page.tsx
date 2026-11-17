"use client";

import Link from "next/link";
import { Flow } from "@/components/home/Flow";
import { Hook } from "@/components/home/Hook";
import { Story } from "@/components/home/Story";
import { StudyFinding } from "@/components/home/StudyFinding";
import ui from "@/components/ui.module.css";
import { api, docUrl, type M4, type Vrp } from "@/lib/api";
import { pct } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import s from "./home.module.css";

/**
 * Home: the basics a newcomer asks first, in two cards of words beside drawings; the problem the project takes on,
 * why a backtest alone says little beside a widget to move; our answer to it as a flow of four steps, each with its
 * live result and one action; the result so far; and the way into the app. The guided questions live on the
 * Workbench (/workbench/); the controls are taught at /tutorial/.
 */

const FAQ: { q: string; a: string }[] = [
  { q: "Is this trading advice?", a: "No. It is a research tool. It places no trades and recommends none." },
  { q: "Why simulate instead of using real data?", a: "On real data nobody knows the true edge, so a method that found one cannot be told from a method that fit noise. In the simulator we set the edge, so each method is graded first." },
  { q: "Does it use real market data at all?", a: "Yes. The Real data page reads the SPX options panel when it is built on the server, and the premium it measures is what the study's result is compared against." },
  { q: "What do I need to run it locally?", a: "The project's Python venv and Node. Run make web, then make serve, and open port 8600. The simulator needs no market data." },
];

/** The study's result in one sentence, from the served analysis; nothing until it is there. */
function headline(m4?: M4, vrp?: Vrp): string | null {
  const a = m4?.available ? m4.analysis : undefined;
  if (!m4 || !a) return null;
  const est = m4.vrp?.estimate ?? vrp?.mapping?.headline?.mean;
  const atReal = m4.mapped.find((r) => r.arm === "ppo")?.power;
  if (est === undefined || est === null || atReal === undefined || atReal === null) return null;
  return `Our current result: SPX options have paid about ${est.toFixed(2)} vol points of premium, and at that size a ${a.design.n_seeds}-seed study catches the PPO agent's edge ${pct(atReal, 0)} of the time.`;
}

export default function Home() {
  const m4 = useFetch((x) => api.m4(x), "m4");
  const vrp = useFetch((x) => api.vrp(x), "vrp");
  const study = useFetch((x) => api.study(x), "study");
  const ready = Boolean(m4.data?.available && m4.data.analysis);
  const line = headline(m4.data, vrp.data);

  return (
    <>
      <section className={s.hero}>
        <div className={s.heroCopy}>
          <h1>Test whether an options edge is real before you trade on it.</h1>
          <p>
            Vega Lab is a backtesting lab for selling SPX volatility. You plant an edge of known size in a simulated market, run a strategy on it, and check whether a test of
            that size would have noticed. Then you run the same checks on real SPX data.
          </p>
          {line ? <p className={s.result}>{line}</p> : null}
          <div className={s.heroGo}>
            <Link href="/workbench/" className={s.btnMain}>
              Open the Workbench
            </Link>
            <Link href="/tutorial/" className={s.heroLink}>
              New here? Start with the tutorial →
            </Link>
          </div>
        </div>
        <figure className={s.shot}>
          <img src="/landing/workbench.png" alt="The Workbench: guided questions on the left, the study's power curves and detection floors on the right" width={1280} height={800} />
        </figure>
      </section>

      <section className={s.section} aria-labelledby="basics">
        <h2 id="basics" className={ui.sectionTitle}>
          The basics
        </h2>
        <Story />
      </section>

      <section className={s.section} aria-labelledby="problem">
        <h2 id="problem" className={ui.sectionTitle}>
          The problem
        </h2>
        <Hook m4={m4.data} study={study.data} vrp={vrp.data} />
      </section>

      <section className={s.section} aria-labelledby="solution">
        <h2 id="solution" className={ui.sectionTitle}>
          Our solution
        </h2>
        <p className={s.lede}>
          The deliverable is not a profitable agent. It is a calibrated instrument and a stated detection floor, built in four steps, each a page in the app.
        </p>
        <Flow m4={m4.data} vrp={vrp.data} />
      </section>

      <section className={s.section} aria-labelledby="found">
        <h2 id="found" className={ui.sectionTitle}>
          What we found
        </h2>
        {ready ? (
          <StudyFinding m4={m4.data} vrp={vrp.data} />
        ) : (
          <p className={s.lede}>{m4.error ? "The server did not answer, so the result cannot be shown." : m4.data ? "The sweep's analysis is not on this server yet." : "Loading the result…"}</p>
        )}
      </section>

      <section className={s.section} aria-labelledby="use">
        <h2 id="use" className={ui.sectionTitle}>
          Use the app
        </h2>
        <div className={s.use}>
          <Link href="/tutorial/" className={`${ui.card} ${s.useCard}`}>
            <h3 className={ui.cardTitle}>Tutorial</h3>
            <p>Ten minutes: how the app is laid out, the order a study runs in, and a first comparison you run yourself.</p>
            <span className={s.more}>Take the tutorial →</span>
          </Link>
          <Link href="/workbench/" className={`${ui.card} ${s.useCard}`}>
            <h3 className={ui.cardTitle}>Workbench</h3>
            <p>Pick a question, such as “Real edge, or luck?”, and a guided walk-through runs the experiment with you, step by step.</p>
            <span className={s.more}>Open the Workbench →</span>
          </Link>
          <a href={docUrl("REPRODUCE.md")} target="_blank" rel="noreferrer" className={`${ui.card} ${s.useCard}`}>
            <h3 className={ui.cardTitle}>Reproduce our numbers</h3>
            <p>The commands, the data each step needs, and what to expect, for a partner team rerunning the study.</p>
            <span className={s.more}>REPRODUCE.md →</span>
          </a>
        </div>
      </section>

      <section className={s.section} aria-labelledby="faq">
        <h2 id="faq" className={ui.sectionTitle}>
          FAQ
        </h2>
        <dl className={s.faq}>
          {FAQ.map((x) => (
            <div key={x.q}>
              <dt>{x.q}</dt>
              <dd>{x.a}</dd>
            </div>
          ))}
        </dl>
      </section>
    </>
  );
}
