"use client";

import Link from "next/link";
import { StudyFinding } from "@/components/home/StudyFinding";
import { InfoTip } from "@/components/InfoTip";
import ui from "@/components/ui.module.css";
import { api, docUrl } from "@/lib/api";
import { tourHref } from "@/lib/tours";
import { useFetch } from "@/lib/useFetch";
import s from "./landing.module.css";

/**
 * The landing page: what the platform asks, how it answers, what it has found so far and
 * where a researcher starts. The working surface (the guided questions, the live SPX
 * surface, the checks) is the Workbench at /workbench/; the controls are taught at /tutorial/.
 */

type Go = { href: string; label: string };
const STUDIES: { title: string; question: string; how: string; pages: Go[]; tour: string }[] = [
  {
    title: "Is the premium real?",
    question: "Do SPX options price in more volatility than the market delivers, and would a backtest of this size tell that apart from noise?",
    how: "The measured premium on real SPX days, and a sweep over many seeds of a seller run on markets with the premium planted or not.",
    pages: [
      { href: "/data/?q=premium", label: "Real data" },
      { href: "/sweep/", label: "Sweep" },
    ],
    tour: "edge",
  },
  {
    title: "Can an agent capture it?",
    question: "Given an edge of known size, does a learning agent find it, or does it fit noise in its training days?",
    how: "Plant an edge, train an agent on exactly that data, and score it on held-out days against the rule-based seller and the edge's floor.",
    pages: [{ href: "/alpha/", label: "Synthetic alpha" }],
    tour: "alpha",
  },
  {
    title: "What eats the edge?",
    question: "How much of one vol point survives the bid-ask spread, commissions, a small account and a wilder market?",
    how: "Run the same seller with and without costs on the same market paths, so the difference is the cost and nothing else.",
    pages: [
      { href: "/launch/", label: "Launch" },
      { href: "/runs/", label: "Runs" },
    ],
    tour: "costs",
  },
  {
    title: "How big must a test be?",
    question: "How many seeds and episodes before a result of this size means something?",
    how: "The power of the paired test for a given effect, from the variance components the pilot measured.",
    pages: [{ href: "/power/", label: "Power lab" }],
    tour: "size",
  },
];

const TRUST: { title: string; text: string; go: Go }[] = [
  {
    title: "Known answers first",
    text: "Every method is first run on a simulated market where the size of the edge is set by hand, so a test that misses it, or finds one that is not there, shows up before any real data is read.",
    go: { href: "/synth/", label: "Synth data" },
  },
  {
    title: "Paired seeds, a rule fixed in advance",
    text: "Two strategies are compared on the same market paths, so luck cancels. The test, its α and its sample size were written down before the sweep ran.",
    go: { href: "/workbench/?study=1#study", label: "Pre-registration" },
  },
  {
    title: "Checked for leaks, replayable",
    text: "A detector plants look-ahead bugs in every field and shows each is caught. Any run can be replayed from its record and compared byte for byte.",
    go: { href: "/leaks/", label: "Leak detector" },
  },
];

function Finding() {
  const m4 = useFetch((x) => api.m4(x), "m4");
  const vrp = useFetch((x) => api.vrp(x), "vrp");
  const ready = Boolean(m4.data?.available && m4.data.analysis);
  return (
    <section className={s.section} aria-labelledby="finding-title">
      <h2 id="finding-title" className={ui.sectionTitle}>
        What it has found so far
      </h2>
      {ready ? (
        <StudyFinding m4={m4.data} vrp={vrp.data} />
      ) : (
        <p className={s.note}>
          {m4.error
            ? "The server did not answer, so the current result cannot be shown. Start it with make serve and reload."
            : m4.data
              ? "The sweep's analysis is not on this server yet. Once it is, its result appears here and on the Workbench."
              : "Loading the current result…"}
        </p>
      )}
    </section>
  );
}

export default function Landing() {
  return (
    <>
      <section className={s.hero}>
        <p className={ui.eyebrow}>Vega Lab · optspread platform</p>
        <h1 className={s.heroTitle}>Is the variance risk premium a real edge, and can a learning agent find it?</h1>
        <p className={s.heroLede}>
          SPX options tend to price in more volatility than the market then delivers. <InfoTip term="vrp" /> This platform measures how large that premium is, whether a
          seller keeps it after costs, and whether a reinforcement-learning agent can find it. Every method is tested first on a simulated SPX market where the edge's size is
          known, then on real SPX days, against a test fixed in advance.
        </p>
        <div className={s.heroGo}>
          <Link href="/workbench/" className={`${ui.btn} ${ui.btnPrimary}`}>
            Open the workbench →
          </Link>
          <Link href="/tutorial/" className={ui.btn}>
            Take the tutorial
          </Link>
          <span className={s.heroHint}>New here? The tutorial takes ten minutes and ends with a result you produced.</span>
        </div>
      </section>

      <Finding />

      <section className={s.section} aria-labelledby="study-title">
        <h2 id="study-title" className={ui.sectionTitle}>
          What you can study
        </h2>
        <ol className={s.grid}>
          {STUDIES.map((x) => (
            <li key={x.title} className={`${ui.card} ${s.study}`}>
              <h3 className={ui.cardTitle}>{x.title}</h3>
              <p className={s.question}>{x.question}</p>
              <p className={s.how}>{x.how}</p>
              <span className={s.go}>
                <Link href={tourHref(x.tour)} className={`${ui.btn} ${ui.btnSmall} ${ui.btnPrimary}`}>
                  Guided walk-through
                </Link>
                {x.pages.map((g) => (
                  <Link key={g.href} href={g.href} className={`${ui.btn} ${ui.btnSmall}`}>
                    {g.label} →
                  </Link>
                ))}
              </span>
            </li>
          ))}
        </ol>
        <p className={s.note}>
          More questions, the live SPX surface and every check are on the <Link href="/workbench/">Workbench</Link>.
        </p>
      </section>

      <section className={s.section} aria-labelledby="trust-title">
        <h2 id="trust-title" className={ui.sectionTitle}>
          Why the numbers can be trusted
        </h2>
        <ul className={s.trust}>
          {TRUST.map((x) => (
            <li key={x.title} className={s.trustItem}>
              <h3 className={s.trustTitle}>{x.title}</h3>
              <p className={s.how}>{x.text}</p>
              <Link href={x.go.href}>{x.go.label} →</Link>
            </li>
          ))}
        </ul>
        <p className={s.note}>
          Each check's verdict, with its evidence, is in <Link href="/workbench/#trust">Can you trust these numbers?</Link> on the Workbench.
        </p>
      </section>

      <section className={s.section} aria-labelledby="repro-title">
        <h2 id="repro-title" className={ui.sectionTitle}>
          Run it yourself
        </h2>
        <div className={s.repro}>
          <pre className={s.code}>
            <code>{"make web && make serve   # the dashboard on http://127.0.0.1:8600\nmake reproduce            # the study's numbers from scratch"}</code>
          </pre>
          <p className={s.how}>
            The steps, the data each needs and what to expect are in{" "}
            <a href={docUrl("REPRODUCE.md")} target="_blank" rel="noreferrer">
              REPRODUCE.md
            </a>
            ; the test fixed in advance is in{" "}
            <a href={docUrl("PRE_REGISTRATION.md")} target="_blank" rel="noreferrer">
              PRE_REGISTRATION.md
            </a>
            .
          </p>
        </div>
      </section>
    </>
  );
}
