"use client";

import { useEffect, useState } from "react";
import ui from "@/components/ui.module.css";
import s from "./data.module.css";
import { Labelled } from "@/components/InfoTip";
import { guide } from "@/lib/guide";

/** The trading day on or before `value`, so a weekend or holiday picks the day before. */
export function snapTo(dates: string[], value: string): string {
  if (!value) return dates[dates.length - 1] ?? "";
  let pick = dates[0] ?? "";
  for (const d of dates) if (d <= value) pick = d;
  return pick;
}

/** One trading day: typed, picked or stepped; a weekend snaps to the Friday before. */
export function DateBar({ dates, date, onChange, children }: { dates: string[]; date: string; onChange: (d: string) => void; children?: React.ReactNode }) {
  const [typed, setTyped] = useState(date);
  useEffect(() => setTyped(date), [date]);
  const commit = (v: string) => {
    const d = snapTo(dates, v);
    setTyped(d);
    if (d && d !== date) onChange(d);
  };
  const step = (k: -1 | 1) => {
    const i = dates.indexOf(date);
    const next = dates[Math.max(0, Math.min(dates.length - 1, i + k))];
    if (next) onChange(next);
  };
  return (
    <div className={s.dateBar}>
      <div className={ui.field} {...guide("data-date")}>
        <span className={ui.fieldLabel}>
          <Labelled term="data.date">Trading day</Labelled>
        </span>
        <div className={ui.row} style={{ flexWrap: "nowrap" }}>
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => step(-1)} aria-label="Previous trading day" disabled={date === dates[0]}>
            ←
          </button>
          <input
            className={ui.input}
            type="date"
            aria-label="Trading day"
            value={typed}
            min={dates[0]}
            max={dates[dates.length - 1]}
            onChange={(e) => setTyped(e.target.value)}
            onBlur={() => commit(typed)}
            onKeyDown={(e) => e.key === "Enter" && commit(typed)}
          />
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => step(1)} aria-label="Next trading day" {...guide("data-next-day")} disabled={date === dates[dates.length - 1]}>
            →
          </button>
        </div>
      </div>
      {children}
    </div>
  );
}
