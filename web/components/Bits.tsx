import type { ReactNode } from "react";
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
  children,
  className,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  right?: ReactNode;
  /** Where the numbers in this card came from: a file, a run, a module. */
  source?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`${ui.card} ${className ?? ""}`}>
      {title || right ? (
        <div className={ui.cardHead}>
          <div>
            {title ? <h2 className={ui.cardTitle}>{title}</h2> : null}
            {sub ? <p className={ui.cardSub}>{sub}</p> : null}
          </div>
          {right ? <div className={ui.row}>{right}</div> : null}
        </div>
      ) : null}
      {children}
      {source ? <Source>{source}</Source> : null}
    </section>
  );
}

export function Source({ children }: { children: ReactNode }) {
  return (
    <p className={ui.source}>
      <span>Source: </span>
      {children}
    </p>
  );
}

export function Tile({ label, value, note, accent }: { label: string; value: ReactNode; note?: ReactNode; accent?: boolean }) {
  return (
    <div className={`${ui.tile} ${accent ? ui.tileAccent : ""}`}>
      <span className={ui.tileLabel}>{label}</span>
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
  failed: ui.badgeFail,
  "NOT CLEARED": ui.badgeWarn,
  "NO CRITERION": ui.badgeWarn,
  running: ui.badgeWarn,
  queued: ui.badgeNeutral,
};

const VERDICT_ICON: Record<string, string> = {
  PASS: "✓",
  OK: "✓",
  done: "✓",
  FAIL: "✕",
  INVALID: "✕",
  "DIGEST MISMATCH": "✕",
  failed: "✕",
  "NOT CLEARED": "!",
  "NO CRITERION": "?",
  running: "…",
  queued: "…",
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

export function Chip({ children, color }: { children: ReactNode; color?: string }) {
  return (
    <span className={ui.chip}>
      {color ? <span className={ui.swatch} style={{ background: color }} aria-hidden="true" /> : null}
      {children}
    </span>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className={ui.segmented} role="radiogroup" aria-label={label}>
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
}

export function Progress({ fraction, state }: { fraction: number; state?: string }) {
  const cls = state === "done" ? ui.progressDone : state === "failed" ? ui.progressFailed : "";
  return (
    <div className={ui.progress} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)}>
      <div className={`${ui.progressBar} ${cls}`} style={{ width: `${Math.max(2, Math.min(100, fraction * 100))}%` }} />
    </div>
  );
}
