"use client";

import Link from "next/link";
import { StudyFinding } from "@/components/home/StudyFinding";
import { api, docUrl, type M4, type Vrp } from "@/lib/api";
import { pct } from "@/lib/format";
import { tourHref, WORKFLOWS } from "@/lib/tours";
import { useFetch } from "@/lib/useFetch";
import s from "./landing.module.css";

/**
 * The landing page. Kept plain on purpose: a real screenshot, the study's own numbers,
 * a list of what is in the app and the walk-throughs with their real lengths. The
 * working surface is the Workbench (/workbench/); the controls are taught at /tutorial/.
 */

const TOOLS: { title: string; text: string; href: string }[] = [
  { title: "Synth data", text: "A simulated SPX market. You set the size of the edge, including zero, so a test can be checked against the true answer.", href: "/synth/" },
  { title: "Launch and Runs", text: "Run the delta-hedged seller or a PPO agent, then compare two runs on the same market paths so luck cancels out.", href: "/runs/" },
  { title: "Power lab", text: "How many seeds and episodes you need before an edge of a given size shows up reliably.", href: "/power/" },
  { title: "Synthetic alpha", text: "Plant an edge, train an agent on that data and score it on days it hasn't seen.", href: "/alpha/" },
  { title: "Leak detector", text: "Plants a look-ahead bug in each data field and checks that it gets caught.", href: "/leaks/" },
  { title: "Real data", text: "The premium SPX options actually paid, day by day, and what a seller would have collected.", href: "/data/" },
];

const FAQ: { q: string; a: string }[] = [
  { q: "What is the variance risk premium?", a: "The gap between the volatility SPX options price in and the volatility that then shows up. Sellers collect it most of the time and give a lot of it back in crashes." },
  { q: "Why simulate first?", a: "On real data nobody knows the true edge, so you can't tell a method that found one from a method that fit noise. In the simulator we set it ourselves." },
  { q: "Is this trading advice?", a: "No. It's a research tool. It doesn't place trades or recommend any." },
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

export default function Landing() {
  const m4 = useFetch((x) => api.m4(x), "m4");
  const vrp = useFetch((x) => api.vrp(x), "vrp");
  const line = headline(m4.data, vrp.data);
  const ready = Boolean(m4.data?.available && m4.data.analysis);
  const tours = WORKFLOWS.filter((w) => !w.reviewer);

  return (
    <>
      <section className={s.hero}>
        <div className={s.heroInner}>
          <div className={s.heroCopy}>
            <h1>Test whether an options edge is real before you trade on it.</h1>
            <p>
              Vega Lab is a backtesting lab for selling SPX volatility. You plant an edge of known size in a simulated market, run a strategy on it, and check whether a test
              of that size would have noticed. Then you run the same checks on real SPX data.
            </p>
            {line ? <p className={s.result}>{line}</p> : null}
            <div className={s.heroGo}>
              <Link href="/workbench/" className={s.btnMain}>
                Open the app
              </Link>
              <Link href="/tutorial/" className={s.link}>
                New here? Start with the tutorial →
              </Link>
            </div>
          </div>
          <figure className={s.shot}>
            <img src="/landing/workbench.png" alt="The Workbench: guided questions on the left, the study's power curves and detection floors on the right" width={1280} height={800} />
          </figure>
        </div>
      </section>

      <section id="features" className={s.section}>
        <h2>What's in the app</h2>
        <dl className={s.tools}>
          {TOOLS.map((t) => (
            <div key={t.title} className={s.tool}>
              <dt>
                <Link href={t.href}>{t.title}</Link>
              </dt>
              <dd>{t.text}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section id="results" className={s.section}>
        <h2>Latest result</h2>
        <p className={s.lede}>Read live from this server. It updates when the sweep is re-run.</p>
        {ready ? (
          <StudyFinding m4={m4.data} vrp={vrp.data} />
        ) : (
          <p className={s.lede}>{m4.error ? "The server didn't answer. Start it with make serve and reload." : m4.data ? "The sweep's analysis isn't on this server yet." : "Loading…"}</p>
        )}
      </section>

      <section className={s.section}>
        <h2>Guided walk-throughs</h2>
        <p className={s.lede}>Each one opens the right page and walks you through a real experiment, one step at a time.</p>
        <table className={s.tours}>
          <tbody>
            {tours.map((w) => (
              <tr key={w.id}>
                <td>
                  <Link href={tourHref(w.id)}>{w.title}</Link>
                </td>
                <td className={s.q}>{w.question}</td>
                <td className={s.min}>{w.minutes} min</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section id="faq" className={s.section}>
        <h2>Questions</h2>
        <dl className={s.faq}>
          {FAQ.map((x) => (
            <div key={x.q}>
              <dt>{x.q}</dt>
              <dd>{x.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className={`${s.section} ${s.end}`}>
        <p>
          <Link href="/workbench/" className={s.btnMain}>
            Open the app
          </Link>
          <a href={docUrl("REPRODUCE.md")} target="_blank" rel="noreferrer" className={s.link}>
            How to reproduce our numbers →
          </a>
        </p>
      </section>
    </>
  );
}
