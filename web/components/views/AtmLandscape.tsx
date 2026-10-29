"use client";

import { useMemo, useState } from "react";
import { Card, Segmented } from "@/components/Bits";
import { Chart3D, type Grid3D } from "@/components/gl/Chart3D";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Landscape } from "@/lib/api";
import { type Tokens, useTokens } from "@/lib/chart";
import { box, finiteRange, type GLOption, type Point3, surface, tenorAxis, vol, volRamp, zColour } from "@/lib/chart3d";
import { num } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * The ATM term structure through time, from panel/atm_term.parquet: date x tenor x
 * ATM IV. Sixteen years of days are averaged into at most a few hundred rows (the
 * stride is shown); a window of a few months is drawn day by day.
 */

const WINDOWS = [
  { value: "all", label: "All", start: undefined, end: undefined },
  { value: "2011", label: "2011", start: "2011-06-01", end: "2011-12-30" },
  { value: "2015", label: "2015-16", start: "2015-07-01", end: "2016-03-31" },
  { value: "2018", label: "2018", start: "2018-01-02", end: "2018-12-31" },
  { value: "2020", label: "2020", start: "2020-01-02", end: "2020-12-31" },
  { value: "2022", label: "2022", start: "2022-01-03", end: "2022-12-30" },
  { value: "custom", label: "Custom", start: undefined, end: undefined },
] as const;
type Window = (typeof WINDOWS)[number]["value"];

const DAY_MS = 86_400_000;

/** Days since the window's first date: small numbers, so nothing is lost in the GPU's floats. */
function dayAxis(l: Landscape): { x: number[]; t0: number } {
  const ms = l.date.map((d) => Date.parse(d));
  const t0 = ms[0] ?? 0;
  return { x: ms.map((m) => (m - t0) / DAY_MS), t0 };
}

function landscapeOption(t: Tokens, l: Landscape, pts: Point3[], z: [number, number]): GLOption {
  const { x, t0 } = dayAxis(l);
  const hi = x[x.length - 1] ?? 1;
  return {
    ...box(
      t,
      {
        name: "Date",
        min: 0,
        max: Math.max(1, hi),
        formatter: (v: number) => {
          // the axis's own end label lands beside the last round tick; the ticks carry the dates
          if (v === Math.max(1, hi) && v !== 0) return "";
          const d = new Date(t0 + v * DAY_MS).toISOString();
          return hi > 3 * 365 ? d.slice(0, 4) : hi > 120 ? d.slice(0, 7) : d.slice(5, 10);
        },
      },
      tenorAxis("Tenor", l.tenor_days),
      { name: "ATM IV (vol pts)", min: Math.floor(z[0]), max: Math.ceil(z[1]) },
      { width: 160, depth: 76, height: 64 },
    ),
    visualMap: [zColour(t, Math.floor(z[0]), Math.ceil(z[1]), volRamp(t), [0], "vol pts")],
    series: [surface(t, "main", pts, l.tenor_days.length, l.date.length, { wireframe: { show: l.date.length <= 160, lineStyle: { color: t.dark ? "rgba(232,238,247,0.16)" : "rgba(19,35,61,0.16)", width: 1 } } })],
  };
}

export function AtmLandscape() {
  const t = useTokens();
  const [win, setWin] = useState<Window>("all");
  const [custom, setCustom] = useState({ start: "2020-02-01", end: "2020-05-29" });
  const preset = WINDOWS.find((w) => w.value === win);
  const q = win === "custom" ? { ...custom, max_dates: 420 } : { start: preset?.start, end: preset?.end, max_dates: 420 };
  const land = useFetch((s) => api.landscape(q, s), JSON.stringify(q));
  const l = land.data;

  const built = useMemo(() => {
    if (!l || !l.date.length) return null;
    const xs = dayAxis(l).x;
    // row-major by tenor, so a row is one tenor's ATM IV through time
    const pts: Point3[] = [];
    l.tenor_days.forEach((_, j) => xs.forEach((x, i) => pts.push([x, j, vol(l.iv[i]?.[j])])));
    const z = finiteRange(pts.map((p) => p[2]));
    const nav: Grid3D = {
      rows: l.tenor_days.length,
      cols: l.date.length,
      describe: (r, c) => {
        const iv = l.iv[c]?.[r];
        if (iv === null || iv === undefined) return null;
        const when = l.stride > 1 ? `mean of ${l.stride} trading days to ${l.date[c]}` : l.date[c];
        return `${when} · ${l.tenor_days[r]}d · ATM IV ${(iv * 100).toFixed(2)} vol pts`;
      },
      at: (r, c) => {
        const p = pts[r * l.date.length + c];
        return p && Number.isFinite(p[2]) ? p : null;
      },
    };
    return { option: landscapeOption(t, l, pts, z), nav };
  }, [t, l]);

  return (
    <Card
      title="ATM implied vol by date and tenor"
      sub={
        l
          ? `${l.first ?? "–"} → ${l.last ?? "–"} · ${num(l.n_days)} trading days${l.stride > 1 ? ` · each row the mean of ${l.stride}` : " · every day"}`
          : "Date × tenor × ATM IV"
      }
    >
      <div className={ui.stack}>
        <div className={ui.filterBar}>
          <div className={ui.field}>
            <span className={ui.fieldLabel}>Window</span>
            <Segmented label="Date window" value={win} options={WINDOWS} onChange={setWin} />
          </div>
          {win === "custom" ? (
            <>
              <label className={ui.field}>
                <span className={ui.fieldLabel}>From</span>
                <input className={ui.input} type="date" value={custom.start} onChange={(e) => e.target.value && setCustom((c) => ({ ...c, start: e.target.value }))} />
              </label>
              <label className={ui.field}>
                <span className={ui.fieldLabel}>To</span>
                <input className={ui.input} type="date" value={custom.end} onChange={(e) => e.target.value && setCustom((c) => ({ ...c, end: e.target.value }))} />
              </label>
            </>
          ) : null}
        </div>
        {land.error ? <ErrorBox what="the term structure through time" error={land.error} onRetry={land.reload} /> : null}
        {land.loading && !l ? <Loading what="the term structure through time" /> : null}
        {l && !l.date.length ? <Empty title="No trading days in this window" /> : null}
        {built ? <Chart3D option={built.option} grid={built.nav} height={480} label="ATM implied volatility by date and tenor" camera={{ alpha: 22, beta: 24, distance: 215 }} /> : null}
      </div>
    </Card>
  );
}
