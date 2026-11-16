import type { Metadata } from "next";
import Link from "next/link";
import ui from "@/components/ui.module.css";
import { tourHref } from "@/lib/tours";
import s from "./tutorial.module.css";

export const metadata: Metadata = { title: "Tutorial" };

/**
 * The tutorial: how to find your way around the site, the research loop it runs (what
 * How it works used to hold), and a first session to follow. Reached from the landing
 * page, the VEGA LAB label and the footer; /how-it-works/ sends its readers here.
 */

type Go = { href: string; label: string };

/** The header, left to right: the numbers on the drawing and what each part does. */
const HEADER: { name: string; text: string }[] = [
  { name: "optspread platform", text: "Back to the landing page, from anywhere." },
  { name: "VEGA LAB", text: "Opens this tutorial." },
  { name: "The tabs", text: "One page each. Workbench is the place to start work: the guided questions, the live SPX surface and the checks." },
  { name: "Server status", text: "Green with the commit the server runs, amber while it connects, red if it is down. Hover it for the full commit." },
  { name: "Copy link", text: "Copies the address of exactly what you are looking at. Every page keeps its settings in the address, so the link reopens the same view." },
  { name: "Theme", text: "Light or dark. Charts follow." },
];

const HABITS: { name: string; text: string }[] = [
  {
    name: "The workflow strip",
    text: "A guided walk-through runs in a strip under the header, never over the page. It names the step, rings the control to use with Show me, and keeps Next locked until the step is actually done. Leave with × at any time; your runs stay.",
  },
  { name: "Info tips", text: "An ⓘ next to a term explains it in a sentence: vol points, seeds, power, the floor. Hover or tap it." },
  { name: "Settings live in the address", text: "Change a control and the address changes with it. Reload, go back or share the link and the view is the same." },
  { name: "Runs name their code", text: "Every run records the commit, the config hash and the seed that made it, under Provenance on the Runs page, and can be replayed from that record." },
  { name: "Exports", text: "Each figure has an export bar, so a number in your notes can be traced to the view it came from." },
];

/** The research loop, in the order a study runs it. */
const LOOP: { title: string; text: string; go: Go[]; tour?: string }[] = [
  {
    title: "The market",
    text: "A simulated SPX market where you plant an edge of known size, so the right answer is known before you test.",
    go: [{ href: "/synth/", label: "Synth data" }],
  },
  {
    title: "Run a strategy",
    text: "Launch the fixed-delta seller or a learning agent on it and read what it made per episode, after costs.",
    go: [{ href: "/launch/", label: "Launch" }],
    tour: "tune",
  },
  {
    title: "Does the agent find it?",
    text: "Plant an edge of known size, train an agent on exactly that data, and see how much of the edge it captures and how much it overfits.",
    go: [{ href: "/alpha/", label: "Synthetic alpha" }],
    tour: "alpha",
  },
  {
    title: "Compare runs",
    text: "Put two runs on the same market paths side by side, so luck cancels and only their difference is left.",
    go: [{ href: "/runs/", label: "Runs" }],
    tour: "learner",
  },
  {
    title: "Is the edge real?",
    text: "See how big an edge a study of this size can catch, and what the sweep over many seeds found.",
    go: [
      { href: "/power/", label: "Power lab" },
      { href: "/sweep/", label: "Sweep" },
    ],
    tour: "size",
  },
  {
    title: "Can you trust it?",
    text: "Leak tests, a replay of each run from its record and the checks show whether the numbers hold.",
    go: [
      { href: "/leaks/", label: "Leak detector" },
      { href: "/workbench/#trust", label: "The checks" },
    ],
    tour: "leaks",
  },
];

/** The first session: the "Real edge, or luck?" walk-through, said in advance. */
const FIRST: string[] = [
  "Check the server status dot in the header is green. If it is red, start the server with make serve and reload.",
  "Press Start the first session below. The workflow strip appears under the header and opens the Launch page.",
  "Launch the fixed-delta seller in a fair-priced world: no premium, so it should make nothing on average. Wait for the run to finish; Next unlocks by itself.",
  "Launch it again with the “Planted 1 vol pt” quick start: now the options are dear by one vol point.",
  "On Runs, compare the two on the same market paths. The strip reads the difference out to you, with its uncertainty.",
  "On the Power lab, raise the seeds until a quarter of a vol point is caught 80% of the time. That is the size of test a learned strategy needs.",
  "Press Finish. You have now run the study's core comparison yourself; the Workbench's other questions follow the same pattern.",
];

const SECTIONS: { id: string; label: string }[] = [
  { id: "way", label: "1. Finding your way" },
  { id: "loop", label: "2. The research loop" },
  { id: "first", label: "3. Your first ten minutes" },
];

function HeaderDrawing() {
  // a simplified header, its parts numbered to the list beside it; drawn in HTML so it follows the theme
  return (
    <div className={s.drawing} aria-hidden="true">
      <span className={s.mockBrand}>
        <span className={s.mockName}>
          optspread platform <b className={s.pin}>1</b>
        </span>
        <span className={s.mockLab}>
          VEGA LAB <b className={s.pin}>2</b>
        </span>
      </span>
      <span className={s.mockTabs}>
        <span className={s.mockTabOn}>Workbench</span>
        <span>Synthetic alpha</span>
        <span>Runs</span>
        <span>Launch</span>
        <span>…</span>
        <b className={s.pin}>3</b>
      </span>
      <span className={s.mockRight}>
        <span className={s.mockDot} />
        <b className={s.pin}>4</b>
        <span className={s.mockIcon}>⧉</span>
        <b className={s.pin}>5</b>
        <span className={s.mockIcon}>☾</span>
        <b className={s.pin}>6</b>
      </span>
    </div>
  );
}

export default function Tutorial() {
  return (
    <>
      <section className={s.hero}>
        <div className={s.heroInner}>
          <h1>Learn Vega Lab by running a real test</h1>
          <p>About ten minutes. It covers how the app is laid out and the order a study runs in, then walks you through a first comparison you run yourself.</p>
          <div className={s.heroGo}>
            <Link href={tourHref("edge")} className={s.btnMain}>
              Start the first session →
            </Link>
            <Link href="/workbench/" className={s.btnGhost}>
              Open the app
            </Link>
          </div>
          <ol className={s.progress}>
            {SECTIONS.map((x) => (
              <li key={x.id}>
                <a href={`#${x.id}`}>{x.label}</a>
              </li>
            ))}
          </ol>
        </div>
      </section>
      <div className={`${s.layout} ${s.wrap}`}>
        <nav className={s.toc} aria-label="On this page">
          <p className={s.tocHead}>On this page</p>
          {SECTIONS.map((x) => (
            <a key={x.id} href={`#${x.id}`}>
              {x.label}
            </a>
          ))}
          <Link href="/workbench/" className={s.tocOut}>
            Open the workbench →
          </Link>
        </nav>

        <div className={s.body}>
          <section id="way" className={s.section} aria-labelledby="way-title">
            <h2 id="way-title" className={ui.sectionTitle}>
              1. Finding your way
            </h2>
            <p className={s.intro}>
              Every page of the app shares one header. Open the app in another tab and follow along; the numbers on the drawing match the list.
            </p>
            <div className={`${ui.card} ${s.headerCard}`}>
              <HeaderDrawing />
              <ol className={s.legend}>
                {HEADER.map((x) => (
                  <li key={x.name}>
                    <b>{x.name}</b> {x.text}
                  </li>
                ))}
              </ol>
            </div>
            <dl className={s.habits}>
              {HABITS.map((x) => (
                <div key={x.name} className={s.habit}>
                  <dt>{x.name}</dt>
                  <dd>{x.text}</dd>
                </div>
              ))}
            </dl>
          </section>

          <section id="loop" className={s.section} aria-labelledby="loop-title">
            <h2 id="loop-title" className={ui.sectionTitle}>
              2. The research loop
            </h2>
            <p className={s.intro}>Plant an edge in a market where the answer is known, run a strategy on it, and see whether the result holds up. Each step is a page, and most have a guided walk-through.</p>
            <ol className={s.steps}>
              {LOOP.map((x, i) => (
                <li key={x.title} className={`${ui.card} ${s.step}`}>
                  <span className={s.num} aria-hidden="true">
                    {i + 1}
                  </span>
                  <h3 className={ui.cardTitle}>{x.title}</h3>
                  <p className={s.text}>{x.text}</p>
                  <span className={s.go}>
                    {x.go.map((g) => (
                      <Link key={g.href} href={g.href} className={`${ui.btn} ${ui.btnSmall}`}>
                        {g.label} →
                      </Link>
                    ))}
                    {x.tour ? (
                      <Link href={tourHref(x.tour)} className={`${ui.btn} ${ui.btnSmall} ${ui.btnPrimary}`}>
                        Walk me through it
                      </Link>
                    ) : null}
                  </span>
                </li>
              ))}
            </ol>
          </section>

          <section id="first" className={s.section} aria-labelledby="first-title">
            <h2 id="first-title" className={ui.sectionTitle}>
              3. Your first ten minutes
            </h2>
            <p className={s.intro}>
              The guided question <i>Real edge, or luck?</i>, step by step, so you know what is coming. It launches two real runs, which take a minute or two each.
            </p>
            <ol className={s.first}>
              {FIRST.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ol>
            <p className={s.go}>
              <Link href={tourHref("edge")} className={`${ui.btn} ${ui.btnPrimary}`}>
                Start the first session →
              </Link>
              <Link href={tourHref("reviewer")} className={ui.btn}>
                Or review the evidence first
              </Link>
            </p>
          </section>
        </div>
      </div>
    </>
  );
}
