"use client";

import { type ReactNode, useEffect, useState } from "react";
import { aboutName, InfoTip } from "./InfoTip";
import { Tile } from "./Bits";
import lab from "./lab.module.css";
import ui from "./ui.module.css";

/**
 * The pieces every page shares besides Bits: a fold for details on demand, the plain
 * reading of a view, a row of key numbers, and an arm as it reads in a sentence.
 * Tooltips are InfoTip's (components/InfoTip.tsx), reading the one glossary.
 */

const ARM_PHRASE: Record<string, string> = {
  delta_seller: "the fixed-delta seller",
  ppo: "PPO",
  a2c: "A2C",
  dqn: "DQN",
  always_on: "the always-on condor",
  random: "the random arm",
  flat: "the flat arm",
};

/** An arm as it reads mid-sentence: "the fixed-delta seller", "PPO". */
export function armPhrase(arm: string): string {
  return ARM_PHRASE[arm] ?? arm;
}

/** The same at the start of a sentence. */
export function ArmPhrase(arm: string): string {
  const s = armPhrase(arm);
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Details on demand: closed by default, opened by a click or by a link to its #id.
 * Its content is drawn only once it has been opened, so a closed fold costs no request
 * and no chart. `term` is the glossary key behind the info icon beside the title: the
 * title toggles the fold, the icon only opens its tip (its click never toggles), and the
 * icon's name does not repeat the title, so a search for the title finds the title.
 */
export function Fold({
  id,
  title,
  sub,
  term,
  open: initial = false,
  onToggle,
  children,
}: {
  id?: string;
  title: ReactNode;
  sub?: ReactNode;
  term?: string;
  open?: boolean;
  onToggle?: (open: boolean) => void;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(initial);
  const [seen, setSeen] = useState(initial);
  useEffect(() => {
    if (!id) return;
    // opened by a link to it: on arrival, and on a #id link followed on the page
    const go = () => {
      if (window.location.hash.slice(1) === id) {
        setOpen(true);
        setSeen(true);
      }
    };
    go();
    window.addEventListener("hashchange", go);
    return () => window.removeEventListener("hashchange", go);
  }, [id]);
  return (
    <details
      id={id}
      className={lab.fold}
      open={open}
      onToggle={(e) => {
        const o = (e.currentTarget as HTMLDetailsElement).open;
        setOpen(o);
        if (o) setSeen(true);
        onToggle?.(o);
      }}
    >
      <summary className={lab.foldHead}>
        <span className={lab.foldTitle}>{title}</span>
        {term ? <InfoTip term={term} label={aboutName(title, term)} /> : null}
        {sub ? <span className={lab.foldSub}>{sub}</span> : null}
      </summary>
      {seen ? <div className={lab.foldBody}>{children}</div> : null}
    </details>
  );
}

/** What the view says, in one plain sentence; "Reading the data…" while it loads. */
export function Reading({ children, note, loading }: { children?: ReactNode; note?: ReactNode; loading?: boolean }) {
  if (children === undefined || children === null || children === "") return loading ? <p className={`${ui.reading} ${ui.readingPending}`}>Reading the data…</p> : null;
  return (
    <div className={ui.readingWrap} aria-live="polite">
      <p className={ui.reading}>{children}</p>
      {note ? <p className={ui.readingNote}>{note}</p> : null}
    </div>
  );
}

export interface Fact {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  /** The glossary key behind the label's info icon; the label itself by default. */
  term?: string;
}

/** A row of key numbers: the site's one tile, four to a row. */
export function Facts({ items }: { items: Fact[] }) {
  return (
    <div className={ui.grid4}>
      {items.map((f) => (
        <Tile key={f.label} label={f.label} value={f.value} note={f.note} info={f.term} />
      ))}
    </div>
  );
}
