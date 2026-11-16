/**
 * The detection floor's two charts, against the planted edge's size on a linear axis
 * (zero, the null, is on it): each arm's power curve with the 80% line and its floor
 * marked, and each arm's captured edge with its two-se bar. The curves are the server's
 * (optspread.alpha.floor, the study's machinery), so nothing here computes a noncentral t.
 */
import type { EChartsOption } from "echarts";
import { axis, base, type Tokens } from "@/lib/chart";

export interface FloorLine {
  name: string;
  color: string;
  /** (edge, power, lo, hi) every step from zero, interpolated by the server. */
  curve: [number, number, number, number][];
  /** each swept size's power at its measured noncentrality */
  knots: { x: number; power: number; detected: boolean }[];
  floor: number | null;
}

type CustomApi = { value: (i: number) => number; coord: (v: [number, number]) => [number, number] };

export function powerOption(t: Tokens, lines: FloorLine[], xName: string, fmtX: (v: number) => string): EChartsOption {
  const series = lines.flatMap((l) => [
    {
      type: "line" as const,
      name: l.name,
      data: l.curve.map((c) => [c[0], c[1]]),
      showSymbol: false,
      lineStyle: { width: 2, color: l.color },
      itemStyle: { color: l.color },
      z: 2,
      markLine:
        l.floor === null
          ? undefined
          : {
              silent: true,
              symbol: "none",
              lineStyle: { color: l.color, type: "dashed" as const, width: 1.5 },
              label: { show: true, formatter: `floor ${fmtX(l.floor)}`, color: l.color, fontFamily: t.font, fontSize: 11, position: "insideEndTop" as const },
              data: [{ xAxis: l.floor }],
            },
    },
    {
      type: "scatter" as const,
      name: `${l.name} at each size`,
      data: l.knots.map((k) => ({ value: [k.x, k.power], itemStyle: { color: k.detected ? l.color : t.surface, borderColor: l.color, borderWidth: 2 } })),
      symbolSize: 9,
      z: 3,
      tooltip: { show: false },
    },
  ]);
  return {
    ...base(t),
    animation: false,
    grid: { ...base(t).grid, left: 64, top: 40 },
    tooltip: { ...base(t).tooltip, trigger: "axis", valueFormatter: (v: unknown) => (typeof v === "number" ? `${Math.round(100 * v)}%` : "") },
    legend: { ...base(t).legend, data: lines.map((l) => l.name) },
    xAxis: { type: "value", min: 0, name: xName, nameLocation: "middle", ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: fmtX } },
    yAxis: { type: "value", min: 0, max: 1, name: "Power (chance of detecting it)", nameLocation: "middle", ...axis(t, { nameGap: 44 }), axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => `${Math.round(100 * v)}%` } },
    series: [
      ...series,
      {
        type: "line",
        data: [],
        markLine: { silent: true, symbol: "none", lineStyle: { color: t.muted, type: "dotted" }, label: { show: true, formatter: "80%", color: t.muted, position: "insideStartTop" }, data: [{ yAxis: 0.8 }] },
      },
    ],
  } as EChartsOption;
}

export interface CapturedLine {
  name: string;
  color: string;
  points: { x: number; y: number; se: number }[];
}

export function capturedOption(t: Tokens, lines: CapturedLine[], xName: string, fmtX: (v: number) => string, fmtY: (v: number) => string): EChartsOption {
  const series = lines.flatMap((l) => [
    {
      type: "line" as const,
      name: l.name,
      data: l.points.map((p) => [p.x, p.y]),
      symbol: "rect",
      symbolSize: 7,
      lineStyle: { width: 1.5, color: l.color },
      itemStyle: { color: l.color },
      z: 2,
    },
    {
      type: "custom" as const,
      name: `${l.name} ± 2 se`,
      renderItem: (_params: unknown, api: unknown) => {
        const a = api as CustomApi;
        const x = a.value(0);
        const lo = a.coord([x, a.value(1)]);
        const hi = a.coord([x, a.value(2)]);
        const style = { stroke: l.color, lineWidth: 1.5 };
        return {
          type: "group",
          children: [
            { type: "line", shape: { x1: lo[0], y1: lo[1], x2: hi[0], y2: hi[1] }, style },
            { type: "line", shape: { x1: lo[0] - 4, y1: lo[1], x2: lo[0] + 4, y2: lo[1] }, style },
            { type: "line", shape: { x1: hi[0] - 4, y1: hi[1], x2: hi[0] + 4, y2: hi[1] }, style },
          ],
        };
      },
      data: l.points.map((p) => [p.x, p.y - 2 * p.se, p.y + 2 * p.se]),
      encode: { x: 0, y: [1, 2] },
      z: 1,
    },
  ]);
  return {
    ...base(t),
    animation: false,
    grid: { ...base(t).grid, left: 72 },
    tooltip: { ...base(t).tooltip, trigger: "axis", valueFormatter: (v: unknown) => (typeof v === "number" ? fmtY(v) : "") },
    legend: { ...base(t).legend, data: lines.map((l) => l.name) },
    xAxis: { type: "value", min: 0, name: xName, nameLocation: "middle", ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: fmtX } },
    yAxis: { type: "value", name: "Captured, $ per episode", nameLocation: "middle", ...axis(t, { nameGap: 56 }), axisLabel: { ...axis(t).axisLabel, formatter: fmtY } },
    series: [...series, { type: "line", data: [], markLine: { silent: true, symbol: "none", lineStyle: { color: t.axis }, label: { show: false }, data: [{ yAxis: 0 }] } }],
  } as EChartsOption;
}
