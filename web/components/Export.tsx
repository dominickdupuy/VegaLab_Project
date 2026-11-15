"use client";

import { type ReactNode, useEffect, useState } from "react";
import { useTokens } from "@/lib/chart";
import { download, pngWithCaption, type Row, slug, toCsv } from "@/lib/export";
import { notify } from "@/lib/activity";
import { guide } from "@/lib/guide";
import { InfoTip } from "./InfoTip";
import ui from "./ui.module.css";

export interface ChartHandle {
  /** The chart as a PNG data URL, or null before it has drawn. */
  png: () => string | null;
}

/**
 * Under a figure: the ways to take it away, as CSV, JSON or a PNG with its caption
 * (config hash, git SHA, data source: whatever produced it) printed on it. On the page
 * the caption sits behind an icon beside the buttons, not as a line of text. A pressed
 * button reads "Saved" for a moment.
 */
export function FigureFoot({ name, stamp, rows, json, chart }: { name: string; stamp?: ReactNode; rows?: () => Row[]; json?: () => unknown; chart?: () => ChartHandle | null }) {
  const t = useTokens();
  const base = slug(name);
  const caption = typeof stamp === "string" ? stamp : "";
  const [saved, setSaved] = useState<"csv" | "json" | "png" | null>(null);
  useEffect(() => {
    if (!saved) return;
    const id = window.setTimeout(() => setSaved(null), 1200);
    return () => window.clearTimeout(id);
  }, [saved]);
  const label = (format: "csv" | "json" | "png") => (saved === format ? "Saved" : format.toUpperCase());
  const png = async () => {
    const url = chart?.()?.png();
    if (!url) return;
    const blob = await pngWithCaption(url, caption, t.surface, t.muted, t.font);
    if (blob) {
      download(`${base}.png`, blob, "image/png");
      notify({ kind: "export", name, format: "png" });
      setSaved("png");
    }
  };
  return (
    <div className={ui.figFoot}>
      <span className={ui.exportBar} aria-label={`Export ${name}`} {...guide("export")}>
        {stamp ? <InfoTip icon="source" title="Where this figure comes from" label={`Where ${name} comes from`} text={<span className={ui.tipSource}>{stamp}</span>} /> : null}
        {rows ? (
          <button type="button" className={ui.exportBtn} onClick={() => {
              download(`${base}.csv`, toCsv(rows()), "text/csv");
              notify({ kind: "export", name, format: "csv" });
              setSaved("csv");
            }}>
            {label("csv")}
          </button>
        ) : null}
        {json || rows ? (
          <button type="button" className={ui.exportBtn} onClick={() => {
              download(`${base}.json`, JSON.stringify(json ? json() : rows?.(), null, 2), "application/json");
              notify({ kind: "export", name, format: "json" });
              setSaved("json");
            }}>
            {label("json")}
          </button>
        ) : null}
        {chart ? (
          <button type="button" className={ui.exportBtn} onClick={png}>
            {label("png")}
          </button>
        ) : null}
      </span>
      <span className={ui.srOnly} aria-live="polite">
        {saved ? `${saved.toUpperCase()} saved` : ""}
      </span>
    </div>
  );
}
