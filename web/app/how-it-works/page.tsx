import type { Metadata } from "next";
import Link from "next/link";
import { PageHead } from "@/components/Bits";
import ui from "@/components/ui.module.css";
import s from "./how.module.css";

export const metadata: Metadata = { title: "How it works" };

/**
 * The platform in five steps, each one sentence and the page that does it. Reached from
 * the footer and the "Vega Lab" label in the header, not the nav: the home page stays
 * the place to start, so the way back to it sits at the top.
 */
const STEPS: { title: string; text: string; go: { href: string; label: string }[] }[] = [
  { title: "The market", text: "A simulated SPX market where you plant an edge of known size, so the right answer is known before you test.", go: [{ href: "/synth/", label: "Synth data" }] },
  { title: "Run a strategy", text: "Launch the fixed-delta seller or a learning agent on it and read what it made per episode, after costs.", go: [{ href: "/launch/", label: "Launch" }] },
  { title: "Compare runs", text: "Put two runs on the same market paths side by side, so luck cancels and only their difference is left.", go: [{ href: "/runs/", label: "Runs" }] },
  {
    title: "Is the edge real?",
    text: "See how big an edge a study of this size can catch, and what the sweep over many seeds found.",
    go: [
      { href: "/power/", label: "Power lab" },
      { href: "/sweep/", label: "Sweep" },
    ],
  },
  {
    title: "Can you trust it?",
    text: "Leak tests, a replay of each run from its record and the checks show whether the numbers hold.",
    go: [
      { href: "/leaks/", label: "Leak detector" },
      { href: "/#trust", label: "The checks" },
    ],
  },
];

function Back() {
  return (
    <Link href="/" className={`${ui.btn} ${ui.btnPrimary}`}>
      ← Back to the dashboard
    </Link>
  );
}

export default function HowItWorks() {
  return (
    <>
      <PageHead eyebrow="Vega Lab" title="How it works" lede="Plant an edge in a market where the answer is known, run a strategy on it, and see whether the result holds up." right={<Back />} />
      <ol className={s.steps}>
        {STEPS.map((x, i) => (
          <li key={x.title} className={`${ui.card} ${s.step}`}>
            <span className={s.num} aria-hidden="true">
              {i + 1}
            </span>
            <h2 className={ui.cardTitle}>{x.title}</h2>
            <p className={s.text}>{x.text}</p>
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
      <p className={s.more}>
        Rather be walked through it? The <Link href="/">guided questions on the home page</Link> take you through each step with real runs.
      </p>
    </>
  );
}
