"use client";

import Link from "next/link";
import { PageHead } from "@/components/Bits";
import { StudyFinding } from "@/components/home/StudyFinding";
import ui from "@/components/ui.module.css";
import { api, docUrl } from "@/lib/api";
import { signed } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import s from "./home.module.css";

/**
 * Home: the basics a newcomer asks first, the problem the project takes on, our answer to
 * it, the result so far, and the way into the app. The guided questions live on the
 * Workbench (/workbench/); the controls are taught at /tutorial/.
 */

const BASICS: { q: string; a: string }[] = [
  {
    q: "What is an option?",
    a: "A contract that gives the right, but not the obligation, to buy (a call) or sell (a put) something at a set price, the strike, until a set date, the expiry. SPX options are on the S&P 500 index. The buyer pays the seller a price up front for that right.",
  },
  {
    q: "What is the variance risk premium?",
    a: "Volatility is how much a price swings. Every option's price implies a volatility; the swings that then happen are the realized volatility. SPX options have tended to price in more than shows up, so option sellers earn the difference on average and give much of it back in crashes. That gap is the premium, measured here in vol points.",
  },
  {
    q: "What is a reinforcement learning agent?",
    a: "A program that learns a strategy by trial and error. Each day it sees the market, picks an action (which options to sell, or none) and is paid a reward, its profit after costs. Over many simulated episodes it leans toward what paid. Ours uses PPO, a standard reinforcement learning algorithm.",
  },
  {
    q: "What is a backtest?",
    a: "Running a strategy on past or simulated data to see what it would have made. It is how a strategy is judged before money is at risk, and it is easy to fool yourself with: too few runs, a lucky stretch, or code that quietly peeks at tomorrow's prices.",
  },
  {
    q: "What is the fixed-delta seller?",
    a: "Our baseline: a fixed rule that sells options a set distance from the current price and rolls them on a schedule. It does not learn. If the agent cannot match it, the agent has not found anything a simple rule would miss.",
  },
];

const SOLUTION: { title: string; text: string; go: { href: string; label: string }[] }[] = [
  {
    title: "Plant an edge of known size",
    text: "A leak-tested simulator of the SPX options market where we set the premium ourselves, from zero up. The right answer is known before any test runs.",
    go: [
      { href: "/synth/", label: "Synth data" },
      { href: "/leaks/", label: "Leak detector" },
    ],
  },
  {
    title: "Run both strategies on the same markets",
    text: "The fixed-delta seller and the PPO agent trade identical market paths, with costs, across many seeds. Pairing them cancels the luck of the draw.",
    go: [
      { href: "/launch/", label: "Launch" },
      { href: "/runs/", label: "Runs" },
    ],
  },
  {
    title: "Find each strategy's detection floor",
    text: "A test fixed before the data, run over a sweep of planted premiums, gives the smallest edge each strategy is caught earning 80% of the time.",
    go: [
      { href: "/sweep/", label: "Sweep" },
      { href: "/power/", label: "Power lab" },
    ],
  },
  {
    title: "Hold the real market up to it",
    text: "We measure the premium real SPX options paid and place it on the same scale, which says whether a study of this size could have seen it.",
    go: [{ href: "/data/?q=premium", label: "Real data" }],
  },
];

const FAQ: { q: string; a: string }[] = [
  { q: "Is this trading advice?", a: "No. It is a research tool. It places no trades and recommends none." },
  { q: "Why simulate instead of using real data?", a: "On real data nobody knows the true edge, so a method that found one cannot be told from a method that fit noise. In the simulator we set the edge, so each method is graded first." },
  { q: "Does it use real market data at all?", a: "Yes. The Real data page reads the SPX options panel when it is built on the server, and the premium it measures is what the study's result is compared against." },
  { q: "What do I need to run it locally?", a: "The project's Python venv and Node. Run make web, then make serve, and open port 8600. The simulator needs no market data." },
];

export default function Home() {
  const m4 = useFetch((x) => api.m4(x), "m4");
  const vrp = useFetch((x) => api.vrp(x), "vrp");
  const h = vrp.data?.mapping?.headline;
  const ready = Boolean(m4.data?.available && m4.data.analysis);

  return (
    <>
      <PageHead
        eyebrow="Vega Lab"
        title="Can a backtest tell a real options edge from luck?"
        lede="We built a simulator of the SPX options market to measure the smallest edge a study like ours can actually detect, and whether a reinforcement learning agent finds it as well as a simple rule. Start with the basics below, or go straight to the app."
        right={
          <>
            <Link href="/tutorial/" className={ui.btn}>
              Take the tutorial
            </Link>
            <Link href="/workbench/" className={`${ui.btn} ${ui.btnPrimary}`}>
              Open the Workbench →
            </Link>
          </>
        }
      />

      <section className={s.section} aria-labelledby="basics">
        <h2 id="basics" className={ui.sectionTitle}>
          The basics
        </h2>
        <dl className={s.basics}>
          {BASICS.map((x) => (
            <div key={x.q} className={`${ui.card} ${s.basic}`}>
              <dt>{x.q}</dt>
              <dd>{x.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className={s.section} aria-labelledby="problem">
        <h2 id="problem" className={ui.sectionTitle}>
          The problem
        </h2>
        <div className={`${ui.card} ${s.problem}`}>
          <p>
            The edge an option seller is chasing is small and noisy.{" "}
            {h ? (
              <>
                On real SPX data we measure a premium of about <b>{signed(h.mean)} vol points</b>, and its 95% interval runs from {signed(h.mean_low)} to {signed(h.mean_high)}: we
                cannot even be sure it is positive.
              </>
            ) : (
              "A seller's profit from one day to the next swings by far more than the premium itself."
            )}
          </p>
          <p>
            Backtests make it worse. A few lucky runs, costs left out, or a bug that lets the strategy see tomorrow's prices can each turn noise into an apparent profit. So
            when a backtest, or a reinforcement learning agent, reports that it made money, that alone says little.
          </p>
          <p className={s.ask}>
            What is missing is a number most studies never state: <b>how small an edge could this setup reliably detect?</b> And does a learning agent find a real edge as
            well as a simple rule does?
          </p>
        </div>
      </section>

      <section className={s.section} aria-labelledby="solution">
        <h2 id="solution" className={ui.sectionTitle}>
          Our solution
        </h2>
        <p className={s.lede}>
          The deliverable is not a profitable agent. It is a calibrated instrument and a stated detection floor, built in four steps, each a page in the app.
        </p>
        <ol className={s.steps}>
          {SOLUTION.map((x, i) => (
            <li key={x.title} className={`${ui.card} ${s.step}`}>
              <span className={s.num} aria-hidden="true">
                {i + 1}
              </span>
              <h3 className={ui.cardTitle}>{x.title}</h3>
              <p>{x.text}</p>
              <span className={s.go}>
                {x.go.map((g) => (
                  <Link key={g.href} href={g.href} className={`${ui.btn} ${ui.btnSmall}`}>
                    {g.label} →
                  </Link>
                ))}
              </span>
            </li>
          ))}
        </ol>
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
