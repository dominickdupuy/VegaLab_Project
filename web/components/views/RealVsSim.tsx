"use client";

import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Reading } from "@/components/PageBits";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { Card, Segmented, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Chart3D, type Grid3D } from "@/components/gl/Chart3D";
import type { GLHover } from "@/components/gl/GLView";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import type { SigmaFrom, Simulated } from "@/lib/api";
import { dataApi, type SimulatedRead } from "@/lib/api_data";
import { axis, base, type Tokens } from "@/lib/chart";
import { box, divergingRamp, finiteRange, type GLOption, line, type Point3, points, surface, tenorAxis, tenorPos, vol, zColour } from "@/lib/chart3d";
import { esc, MINUS, num } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { type GridDay, describeCell, gridNav, gridPoints, ivSurfaceOption, type XMode, XModeToggle, xName, xOf } from "./IvSurface";

/**
 * The same day in the synthetic world: the chain optspread.market lists at its first
 * step, every contract at sigma + the planted premium, on the simulator's strikes and
 * tenors, beside the real surface. "Difference" draws real minus simulated over the
 * real grid, with the simulator's listed contracts outlined: what the synthetic world
 * leaves out (the skew, the term structure, the level on the day).
 */

const MODES = [
  { value: "side", label: "Side by side" },
  { value: "overlay", label: "Overlay" },
  { value: "diff", label: "Difference" },
] as const;
type Mode = (typeof MODES)[number]["value"];

const SIGMA = [
  { value: "config", label: "σ = config" },
  { value: "rv21", label: "σ = RV21" },
  { value: "atm", label: "matched to IV30" },
] as const;
const SIGMA_LABEL: Record<string, string> = Object.fromEntries(SIGMA.map((x) => [x.value, x.label]));

const pts1 = (x: number | null) => (x === null ? "–" : `${x >= 0 ? "+" : MINUS}${Math.abs(x * 100).toFixed(2)}`);

/**
 * The simulator beside the real surface, in 2D: the simulator prices every contract at
 * one level, so a 3D surface of it is a flat patch with nothing to read. Each of its
 * tenors is a line over its strikes, with the real IV at the same contracts beside it.
 */
function simSmileOption(t: Tokens, s: Simulated): EChartsOption {
  const colours = [t.s1, t.s3, t.s4];
  const series = s.tenor_trading_days.flatMap((td, r) => [
    {
      name: `Simulated · ${td} d`,
      type: "line" as const,
      data: s.strike.map((k, c) => [k, (s.iv[r]?.[c] ?? NaN) * 100]),
      symbol: "none",
      lineStyle: { color: t.s2, width: 2, type: r === 0 ? ("solid" as const) : ("dashed" as const) },
      itemStyle: { color: t.s2 },
    },
    {
      name: `Real · ${td} d`,
      type: "line" as const,
      data: s.strike.map((k, c) => [k, s.real[r]?.[c] === null || s.real[r]?.[c] === undefined ? null : (s.real[r]?.[c] ?? 0) * 100]),
      symbol: "circle",
      symbolSize: 4,
      connectNulls: false,
      lineStyle: { color: colours[r % colours.length], width: 1.6 },
      itemStyle: { color: colours[r % colours.length] },
    },
  ]);
  return {
    ...base(t),
    legend: { ...base(t).legend, type: "scroll" },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: [number, number | null] }[];
        const k = arr[0]?.value[0];
        return [`<b>strike ${num(k)}</b>`, ...arr.map((p) => `${esc(p.seriesName)}: ${p.value[1] === null || !Number.isFinite(p.value[1]) ? "not quoted" : p.value[1].toFixed(2)}`)].join("<br/>");
      },
    },
    grid: { ...base(t).grid, top: 44, left: 52, bottom: 44 },
    xAxis: { type: "value", name: "Strike", nameLocation: "middle", scale: true, ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "IV (vol pts)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 36 },
    series,
  };
}

/** The simulator's contracts as points: rows are its tenors, columns its strikes. */
function simPoints(s: Simulated, grid: number[], mode: XMode, z: (r: number, c: number) => number): Point3[] {
  const out: Point3[] = [];
  s.k.forEach((row, r) => row.forEach((k, c) => out.push([xOf(k, mode), tenorPos(s.tenor_days[r] ?? 0, grid), z(r, c)])));
  return out;
}

function describeContract(s: Simulated, r: number, c: number): string | null {
  const k = s.k[r]?.[c];
  const iv = s.iv[r]?.[c];
  if (k === undefined || iv === undefined) return null;
  const real = s.real[r]?.[c] ?? null;
  const td = s.tenor_trading_days[r] ?? 0;
  const cd = s.tenor_days[r] ?? 0;
  return (
    `strike ${num(s.strike[c])} · K/F ${Math.exp(k).toFixed(3)} · ${td} trading d (${cd.toFixed(1)}d) · sim ${(iv * 100).toFixed(2)}` +
    (real === null ? " · real not quoted" : ` · real ${(real * 100).toFixed(2)} · real − sim ${pts1(real - iv)} vol pts`)
  );
}

/** The outline of the simulator's listed contracts, drawn at height z. */
function windowOutline(s: Simulated, grid: number[], mode: XMode, z: number): Point3[] {
  const ks = s.k.flat();
  const x0 = xOf(Math.min(...ks), mode);
  const x1 = xOf(Math.max(...ks), mode);
  const y0 = tenorPos(s.tenor_days[0] ?? 0, grid);
  const y1 = tenorPos(s.tenor_days[s.tenor_days.length - 1] ?? 0, grid);
  return [
    [x0, y0, z],
    [x1, y0, z],
    [x1, y1, z],
    [x0, y1, z],
    [x0, y0, z],
  ];
}

function overlayOption(t: Tokens, s: Simulated, g: GridDay, mode: XMode, z: [number, number]): { option: GLOption; nav: Grid3D } {
  const real = gridPoints(g, mode);
  const sim = simPoints(s, g.tenor_days, mode, (r, c) => vol(s.iv[r]?.[c] ?? null));
  const base = ivSurfaceOption(t, g, mode, z, real);
  const option: GLOption = {
    ...base,
    series: [
      ...(base.series as GLOption[]),
      surface(t, "sim", sim, s.k.length, s.strike.length, {
        itemStyle: { color: t.s2, opacity: 0.8 },
        wireframe: { show: true, lineStyle: { color: t.s2, width: 1, opacity: 0.9 } },
      }),
      points("contracts", sim, t.s2, 5, { silent: false }),
    ],
  };
  return { option, nav: gridNav(g, real) };
}

function diffOption(t: Tokens, s: Simulated, g: GridDay, mode: XMode): { option: GLOption; nav: Grid3D } {
  const pts = gridPoints(g, mode, (iv) => (iv === null ? NaN : (iv - s.level) * 100));
  const bound = Math.max(1, Math.ceil(Math.max(...pts.map((p) => Math.abs(p[2])).filter(Number.isFinite), 0)));
  const xs = g.k.map((k) => xOf(k, mode));
  const [x0, x1] = [xs[0] ?? 0, xs[xs.length - 1] ?? 1];
  const top = g.tenor_days.length - 1;
  const zero: Point3[] = [
    [x0, 0, 0],
    [x1, 0, 0],
    [x0, top, 0],
    [x1, top, 0],
  ];
  const listed = windowOutline(s, g.tenor_days, mode, 0);
  const inside = (k: number, tenor: number) => {
    const ks = s.k.flat();
    return k >= Math.min(...ks) && k <= Math.max(...ks) && tenor >= (s.tenor_days[0] ?? 0) && tenor <= (s.tenor_days[s.tenor_days.length - 1] ?? 0);
  };
  const option: GLOption = {
    ...box(t, { name: xName(mode), min: x0, max: x1, formatter: (v: number) => v.toFixed(2) }, tenorAxis("Tenor", g.tenor_days), {
      name: "Real − sim (vol pts)",
      min: -bound,
      max: bound,
    }),
    visualMap: [zColour(t, -bound, bound, divergingRamp(t), [0], "vol pts")],
    series: [
      surface(t, "main", pts, g.tenor_days.length, g.k.length),
      surface(t, "zero", zero, 2, 2, { silent: true, wireframe: { show: false }, itemStyle: { color: t.s5, opacity: 0.18 } }),
      line("listed", listed, t.text, 2.5),
    ],
  };
  const nav = gridNav(g, pts, (r, c) => {
    const k = g.k[c] ?? 0;
    const tenor = g.tenor_days[r] ?? 0;
    const iv = g.iv[r]?.[c];
    if (iv === null || iv === undefined) return null;
    const where = inside(k, tenor) ? "inside the simulator's listed contracts" : "outside what the simulator lists";
    return `${describeCell(k, tenor, iv)} · sim ${(s.level * 100).toFixed(2)} · real − sim ${pts1(iv - s.level)} vol pts · ${where}`;
  });
  return { option, nav };
}

export type CompareView = Mode;
export const COMPARE_VIEWS = MODES.map((m) => m.value);

/**
 * Real against simulated on one day. The knobs are the page's (the URL keeps them);
 * this draws the comparison and its one-line reading of where the two differ most.
 */
export function RealVsSim({
  t,
  date,
  grid,
  gridded,
  mode,
  setMode,
  view,
  setView,
  premium,
  setPremium,
  sigmaFrom,
  setSigmaFrom,
}: {
  t: Tokens;
  date: string;
  grid: GridDay | undefined;
  /** The day is in the grid; until ``grid`` arrives it is loading, not missing. */
  gridded: boolean;
  mode: XMode;
  setMode: (m: XMode) => void;
  view: Mode;
  setView: (v: Mode) => void;
  premium: number;
  setPremium: (p: number) => void;
  sigmaFrom: SigmaFrom;
  setSigmaFrom: (s: SigmaFrom) => void;
}) {
  const [q, setQ] = useState(premium);
  useEffect(() => {
    const id = window.setTimeout(() => setQ(premium), 150);
    return () => window.clearTimeout(id);
  }, [premium]);
  const query = { date, premium: q, sigma_from: sigmaFrom };
  const sim = useFetch((s) => dataApi.simulated(query, s), JSON.stringify(query), Boolean(date && grid));
  const [last, setLast] = useState<SimulatedRead | undefined>();
  useEffect(() => {
    if (sim.data) setLast(sim.data);
  }, [sim.data]);
  const s = sim.data ?? (last?.date === date ? last : undefined);

  const z = useMemo<[number, number]>(() => {
    if (!grid || !s) return [0, 1];
    return finiteRange([...grid.iv.flat().map(vol), ...s.iv.flat().map(vol)]);
  }, [grid, s]);
  const real = useMemo(() => {
    if (!grid) return null;
    const pts = gridPoints(grid, mode);
    return { option: ivSurfaceOption(t, grid, mode, z, pts), nav: gridNav(grid, pts) };
  }, [t, grid, mode, z]);
  const overlay = useMemo(() => (s && grid && view === "overlay" ? overlayOption(t, s, grid, mode, z) : null), [t, s, grid, mode, z, view]);
  const diff = useMemo(() => (s && grid && view === "diff" ? diffOption(t, s, grid, mode) : null), [t, s, grid, mode, view]);
  const hoverContract = (h: GLHover) => {
    if (!s || h.seriesId !== "contracts") return null;
    const cols = s.strike.length;
    return describeContract(s, Math.floor(h.dataIndex / cols), h.dataIndex % cols);
  };
  const region = (name: string) => s?.regions.find((r) => r.name === name);
  const gap = (name: string) => {
    const r = region(name);
    return r && r.mean_diff !== null ? pts1(r.mean_diff / 100) : "–";
  };

  return (
    <div className={ui.stack}>
      <div className={ui.filterBar}>
        <div className={ui.field}>
          <span className={ui.fieldLabel}>
            <Labelled term="data.planted_premium">Planted premium</Labelled>
            <span>{premium.toFixed(2)} vol pts</span>
          </span>
          <input className={ui.range} type="range" aria-label="Planted premium in vol points" min={0} max={6} step={0.25} value={premium} onChange={(e) => setPremium(Number(e.target.value))} />
        </div>
        <Segmented label="Where the simulator's sigma comes from" title="Path vol" term="data.path_vol" guide="data-sigma-from" value={sigmaFrom} options={SIGMA} onChange={setSigmaFrom} />
      </div>
      <Reading loading={sim.loading}>{s?.reading}</Reading>
      {sim.error?.status === 422 ? (
        // the day has no such vol to set σ from (the panel's first days have no 21-day history): a fact, not a fault
        <Empty title={`No simulated surface with ${SIGMA_LABEL[sigmaFrom] ?? sigmaFrom} on ${date}`}>{sim.error.message}; pick another σ or another day.</Empty>
      ) : sim.error ? (
        <ErrorBox what="the simulated surface" error={sim.error} onRetry={sim.reload} />
      ) : null}
      {!gridded ? <Empty title="No gridded surface on this day" /> : null}
      {gridded && (!grid || !s) && !sim.error ? <Loading what="the simulated surface" /> : null}
      {grid && s ? (
        <>
          <div className={ui.grid4}>
            <Tile info="data.sim_level" label="Simulator IV" value={(s.level * 100).toFixed(2)} note={`σ ${(s.sigma * 100).toFixed(2)} + ${num(s.premium, 2)} planted · ${SIGMA.find((o) => o.value === s.sigma_from)?.label ?? s.sigma_from}`} />
            <Tile info="data.regions" label="Put wing gap" value={gap("put")} note={`real − sim, vol pts · ${region("put")?.n ?? 0} contracts`} />
            <Tile info="data.regions" label="Near the money gap" value={gap("atm")} note={`real − sim, vol pts · ${region("atm")?.n ?? 0} contracts`} />
            <Tile info="data.regions" label="Call wing gap" value={gap("call")} note={`real − sim, vol pts · ${region("call")?.n ?? 0} contracts`} />
          </div>
          <Card
            title={
              <>
                {view === "diff" ? "Real less simulated" : view === "overlay" ? "The simulator's level over the real surface" : "Real beside simulated"}
                <InfoTip term="data.cmp_view" />
              </>
            }
            right={
              <>
                <Segmented label="Comparison view" value={view} options={MODES} onChange={setView} />
                <XModeToggle value={mode} onChange={setMode} />
              </>
            }
          >
            {view === "side" && real ? (
              <div className={ui.grid2}>
                <div className={ui.stack}>
                  <p className={ui.groupTitle}>Real · {date}</p>
                  <Chart3D option={real.option} grid={real.nav} height={400} label={`Real implied volatility surface on ${date}`} />
                </div>
                <div className={ui.stack}>
                  <p className={ui.groupTitle}>Simulated, at its own contracts, with the real IV there</p>
                  <Chart option={simSmileOption(t, s)} height={400} label="Simulated implied volatility, one level at every strike and tenor, against the real IV at the same contracts" stamp={`${date} · σ ${(s.sigma * 100).toFixed(2)} + ${num(s.premium, 2)} vol pts`} />
                </div>
              </div>
            ) : null}
            {view === "overlay" && overlay ? <Chart3D option={overlay.option} grid={overlay.nav} describeHover={hoverContract} height={480} label="Real implied volatility surface with the simulated level laid over it as a reference plane" /> : null}
            {view === "diff" && diff ? <Chart3D option={diff.option} grid={diff.nav} height={480} label="Real minus simulated implied volatility over the real grid; the simulator's listed contracts outlined" /> : null}
          </Card>
        </>
      ) : null}
    </div>
  );
}
