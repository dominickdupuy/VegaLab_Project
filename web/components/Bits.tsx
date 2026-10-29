import type { ReactNode } from "react";
import { hasChoice } from "@/lib/choices";
import { define, type Glossary } from "@/lib/glossary";
import { InfoTip } from "./InfoTip";
import ui from "./ui.module.css";

export function PageHead({ eyebrow, title, lede, right }: { eyebrow: string; title: string; lede?: ReactNode; right?: ReactNode }) {
  return (
    <header className={ui.pageHead}>
      <div>
        <p className={ui.eyebrow}>{eyebrow}</p>
        <h1 className={ui.title}>{title}</h1>
        {lede ? <p className={ui.lede}>{lede}</p> : null}
      </div>
      {right ? <div className={ui.row}>{right}</div> : null}
    </header>
  );
}

export function Card({
  title,
  sub,
  right,
  source,
  info,
  children,
  className,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  right?: ReactNode;
  /**
   * Where the numbers in this card came from: a file, a run, a module. Shown behind a
   * small icon in the card's head, never as a caption on the page.
   */
  source?: ReactNode;
  /** What the card shows: a glossary term, or the text itself. A string title is looked up by default; false for none. */
  info?: string | ReactNode | false;
  children: ReactNode;
  className?: string;
}) {
  const term = info === undefined ? (typeof title === "string" ? title : undefined) : typeof info === "string" ? info : undefined;
  // a string the glossary does not know is the explanation itself
  const tip = info === false ? null : term && define(term) ? <InfoTip term={term} /> : info ? <InfoTip text={info} label="About this card" /> : null;
  return (
    <section className={`${ui.card} ${className ?? ""}`}>
      {title || right || source ? (
        <div className={ui.cardHead}>
          <div className={ui.cardHeadText}>
            {title ? (
              <h2 className={ui.cardTitle}>
                {title}
                {tip}
              </h2>
            ) : null}
            {sub ? <p className={ui.cardSub}>{sub}</p> : null}
          </div>
          {right || source ? (
            <div className={`${ui.row} ${ui.cardHeadRight}`}>
              {right}
              {source ? <Source>{source}</Source> : null}
            </div>
          ) : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

/** Provenance: an icon whose tooltip names the file, run or module behind the numbers. */
export function Source({ children }: { children: ReactNode }) {
  return <InfoTip icon="source" title="Where this comes from" label="Where this comes from" text={<span className={ui.tipSource}>{children}</span>} />;
}

/**
 * A figure: a label, a value and a short note. Every tile on every page looks the same;
 * a label the glossary knows gets its line behind an info icon.
 */
export function Tile({ label, value, note, info }: { label: string; value: ReactNode; note?: ReactNode; info?: string | false; /** @deprecated every tile looks the same */ accent?: boolean }) {
  const term = info === false ? undefined : (info ?? label);
  return (
    <div className={ui.tile}>
      <span className={ui.tileLabel}>
        {label}
        {term && define(term) ? <InfoTip term={term} label={`About ${label}`} /> : null}
      </span>
      <span className={ui.tileValue}>{value}</span>
      {note ? <span className={ui.tileNote}>{note}</span> : null}
    </div>
  );
}

const VERDICT_CLASS: Record<string, string | undefined> = {
  PASS: ui.badgePass,
  OK: ui.badgePass,
  done: ui.badgePass,
  FAIL: ui.badgeFail,
  INVALID: ui.badgeFail,
  "DIGEST MISMATCH": ui.badgeFail,
  UNVERIFIED: ui.badgeWarn,
  failed: ui.badgeFail,
  "NOT CLEARED": ui.badgeWarn,
  "NO CRITERION": ui.badgeWarn,
  running: ui.badgeWarn,
  queued: ui.badgeNeutral,
  IDENTICAL: ui.badgePass,
  AGREES: ui.badgePass,
  DIFFERS: ui.badgeFail,
  "NOT COMPARABLE": ui.badgeNeutral,
  "EPISODES DIFFER": ui.badgeWarn,
};

const VERDICT_ICON: Record<string, string> = {
  PASS: "✓",
  OK: "✓",
  done: "✓",
  FAIL: "✕",
  INVALID: "✕",
  "DIGEST MISMATCH": "✕",
  UNVERIFIED: "?",
  failed: "✕",
  "NOT CLEARED": "!",
  "NO CRITERION": "?",
  running: "…",
  queued: "…",
  IDENTICAL: "✓",
  AGREES: "≈",
  DIFFERS: "✕",
  "NOT COMPARABLE": "≠",
  "EPISODES DIFFER": "!",
};

/** A verdict or status: always an icon and a word, never colour alone. */
export function Verdict({ value }: { value: string }) {
  return (
    <span className={`${ui.badge} ${VERDICT_CLASS[value] ?? ui.badgeNeutral}`}>
      <span aria-hidden="true">{VERDICT_ICON[value] ?? "•"}</span>
      {value}
    </span>
  );
}

/** A small label: an arm, a source, a status. One the glossary knows explains itself on hover, focus or tap. */
export function Chip({ children, color, term }: { children: ReactNode; color?: string; term?: string | false }) {
  const key = term === false ? undefined : (term ?? (typeof children === "string" ? children : undefined));
  return (
    <span className={ui.chip}>
      {color ? <span className={ui.swatch} style={{ background: color }} aria-hidden="true" /> : null}
      {key && define(key) ? <InfoTip term={key}>{children}</InfoTip> : children}
    </span>
  );
}

/**
 * A choice among a few values, as one row of buttons. Not drawn when there is only one
 * value. With `title` it draws its own field label (with the glossary's line for `term`,
 * or the title itself), so the label goes with it when it hides.
 */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  title,
  term,
  guide,
}: {
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (v: T) => void;
  /** The group's accessible name. */
  label: string;
  /** A visible field label above the buttons. */
  title?: ReactNode;
  term?: string;
  /** The name a workflow step points at (data-guide, lib/guide.ts). */
  guide?: string;
}) {
  if (!hasChoice(options.map((o) => o.value))) return null; // one option is not a choice
  const group = (
    <div className={ui.segmented} role="radiogroup" aria-label={label} data-guide={title ? undefined : guide}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={`${ui.segment} ${o.value === value ? ui.segmentOn : ""}`}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
  if (!title) return group;
  const key = term ?? (typeof title === "string" ? title : undefined);
  return (
    <div className={ui.field} data-guide={guide}>
      <span className={ui.fieldLabel}>
        <span className={ui.labelled}>
          {title}
          {key && define(key) ? <InfoTip term={key} label={`About ${typeof title === "string" ? title : key}`} /> : null}
        </span>
      </span>
      {group}
    </div>
  );
}

export function Progress({ fraction, state }: { fraction: number; state?: string }) {
  const cls = state === "done" ? ui.progressDone : state === "failed" ? ui.progressFailed : "";
  return (
    <div className={ui.progress} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}>
      <div className={`${ui.progressBar} ${cls}`} style={{ width: `${Math.max(2, Math.min(100, fraction * 100))}%` }} />
    </div>
  );
}

/**
 * A labelled select for a filter or a selector. It is not drawn when there is nothing
 * to choose (one value or none, with no "All"), so a page never shows a filter that
 * cannot change anything: one commit, one data source. `all` adds an "All" option, and a
 * filter that is set stays drawn so it can be cleared.
 */
export function SelectField<T extends string | number>({
  label,
  value,
  options,
  onChange,
  render,
  all,
  term,
  dict,
  guide,
  disabled,
  id,
}: {
  label: string;
  value: T | "";
  options: readonly T[];
  onChange: (v: T | "") => void;
  render?: (v: T) => string;
  /** The text of an "All" option (value ""), when the filter can be cleared. */
  all?: string;
  /** The glossary term behind the label's info icon; the label itself by default. */
  term?: string;
  dict?: Glossary;
  /** The name a workflow step points at (data-guide, lib/guide.ts). */
  guide?: string;
  /** Kept in place but inert, for a control that does not apply to the current view. */
  disabled?: boolean;
  id?: string;
}) {
  // one value is not a choice, unless a filter is set to it: then "All" clears it
  if (!hasChoice(options) && !(all !== undefined && value !== "")) return null;
  const tip = define(term ?? label, dict);
  return (
    <label className={ui.field} data-guide={guide}>
      <span className={ui.fieldLabel}>
        <span className={ui.labelled}>
          {label}
          {tip ? <InfoTip term={term ?? label} dict={dict} label={`About ${label}`} /> : null}
        </span>
      </span>
      <select
        id={id}
        className={ui.select}
        disabled={disabled}
        value={String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return onChange("");
          const hit = options.find((o) => String(o) === raw);
          onChange(hit === undefined ? "" : hit);
        }}
      >
        {all !== undefined ? <option value="">{all}</option> : null}
        {options.map((o) => (
          <option key={String(o)} value={String(o)}>
            {render ? render(o) : String(o)}
          </option>
        ))}
      </select>
    </label>
  );
}
