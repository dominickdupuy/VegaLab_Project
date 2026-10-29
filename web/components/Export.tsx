"use client";

import type { ReactNode } from "react";
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
 * the caption sits behind an icon beside the buttons, not as a line of text.
 */
export function FigureFoot({ name, stamp, rows, json, chart }: { name: string; stamp?: ReactNode; rows?: () => Row[]; json?: () => unknown; chart?: () => ChartHandle | null }) {
  const t = useTokens();
  const base = slug(name);
  const caption = typeof stamp === "string" ? stamp : "";
  const png = async () => {
    const url = chart?.()?.png();
    if (!url) return;
    const blob = await pngWithCaption(url, caption, t.surface, t.muted, t.font);
    if (blob) {
      download(`${base}.png`, blob, "image/png");
      notify({ kind: "export", name, format: "png" });
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
            }}>
            CSV
          </button>
        ) : null}
        {json || rows ? (
          <button type="button" className={ui.exportBtn} onClick={() => {
              download(`${base}.json`, JSON.stringify(json ? json() : rows?.(), null, 2), "application/json");
              notify({ kind: "export", name, format: "json" });
            }}>
            JSON
          </button>
        ) : null}
        {chart ? (
          <button type="button" className={ui.exportBtn} onClick={png}>
            PNG
          </button>
        ) : null}
      </span>
    </div>
  );
}
