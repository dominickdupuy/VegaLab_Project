/**
 * The Synthetic alpha page's two chart shapes, both a mean with a two-standard-error bar:
 * a forest (one row per label, the value along x) and a curve against a swept axis (the
 * value up y, the swept setting along a log x). The bars are drawn by a custom series, so
 * nothing outside the bundle EChartView already registers is needed.
 */
import type { EChartsOption } from "echarts";
import { axis, base, type Tokens } from "@/lib/chart";

export interface ForestItem {
  /** Rich-text label: `{p|...}{w|...}` for two aligned columns. */
  label: string;
  mean: number;
  se: number;
  color: string;
  tip: string;
}

type CustomApi = { value: (i: number) => number; coord: (v: [number, number]) => [number, number] };

function bar(color: string) {
  return (x1: number, y1: number, x2: number, y2: number) => ({ type: "line", shape: { x1, y1, x2, y2 }, style: { stroke: color, lineWidth: 1.5 } });
}

export function forestOption(t: Tokens, items: ForestItem[], xName: string, fmt: (v: number) => string): EChartsOption {
  return {
    ...base(t),
    animation: false, // a results chart has nothing to reveal, and a re-render must not restart it
    grid: { left: 300, right: 24, top: 8, bottom: 44 },
    tooltip: { ...base(t).tooltip, trigger: "item", formatter: (p: unknown) => items[(p as { dataIndex: number }).dataIndex]?.tip ?? "" },
    legend: { show: false },
    xAxis: { type: "value", name: xName, nameLocation: "middle", ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: fmt } },
    yAxis: {
      type: "category",
      inverse: true,
      data: items.map((i) => i.label),
      ...axis(t, { splitLine: { show: false } }),
      axisLabel: {
        margin: 10,
        rich: {
          p: { width: 160, align: "left", color: t.text, fontFamily: t.font, fontSize: 12.5 },
          w: { width: 120, align: "left", color: t.text2, fontFamily: t.font, fontSize: 12.5 },
        },
      },
    },
    series: [
      {
        type: "custom",
        name: "± 2 se",
        renderItem: (_params: unknown, api: unknown) => {
          const a = api as CustomApi;
          const i = a.value(0);
          const it = items[i];
          if (!it) return null;
          const lo = a.coord([it.mean - 2 * it.se, i]);
          const hi = a.coord([it.mean + 2 * it.se, i]);
          const line = bar(it.color);
          return { type: "group", children: [line(lo[0], lo[1], hi[0], hi[1]), line(lo[0], lo[1] - 4, lo[0], lo[1] + 4), line(hi[0], hi[1] - 4, hi[0], hi[1] + 4)] };
        },
        data: items.map((it, i) => [i, it.mean - 2 * it.se, it.mean + 2 * it.se]),
        encode: { x: [1, 2], y: 0 },
        z: 1,
      },
      {
        type: "scatter",
        name: "Mean",
        symbol: "rect",
        symbolSize: 8,
        data: items.map((it, i) => ({ value: [it.mean, i], itemStyle: { color: it.color } })),
        markLine: { silent: true, symbol: "none", lineStyle: { color: t.axis }, label: { show: false }, data: [{ xAxis: 0 }] },
        z: 2,
      },
    ],
  } as EChartsOption;
}

export interface CurvePoint {
  x: number;
  y: number | null;
  se: number | null;
}

export interface Curve {
  name: string;
  color: string;
  points: CurvePoint[];
}

/** One or more figures against a swept setting, each point with its two-se bar; x is log. */
export function curveOption(t: Tokens, curves: Curve[], xName: string, yName: string, fmt: (v: number) => string): EChartsOption {
  const series = curves.flatMap((c) => {
    const pts = c.points.filter((p): p is { x: number; y: number; se: number | null } => p.y !== null);
    const line = bar(c.color);
    return [
      {
        type: "line" as const,
        name: c.name,
        data: pts.map((p) => [p.x, p.y]),
        symbol: "rect",
        symbolSize: 7,
        lineStyle: { width: 1.5, color: c.color },
        itemStyle: { color: c.color },
        z: 2,
      },
      {
        type: "custom" as const,
        name: `${c.name} ± 2 se`,
        renderItem: (_params: unknown, api: unknown) => {
          const a = api as CustomApi;
          const x = a.value(0);
          const lo = a.coord([x, a.value(1)]);
          const hi = a.coord([x, a.value(2)]);
          return { type: "group", children: [line(lo[0], lo[1], hi[0], hi[1]), line(lo[0] - 4, lo[1], lo[0] + 4, lo[1]), line(hi[0] - 4, hi[1], hi[0] + 4, hi[1])] };
        },
        data: pts.map((p) => [p.x, p.y - 2 * (p.se ?? 0), p.y + 2 * (p.se ?? 0)]),
        encode: { x: 0, y: [1, 2] },
        z: 1,
      },
    ];
  });
  return {
    ...base(t),
    animation: false,
    grid: { ...base(t).grid, left: 72 },
    tooltip: { ...base(t).tooltip, trigger: "axis", valueFormatter: (v: unknown) => (typeof v === "number" ? fmt(v) : "") },
    legend: { ...base(t).legend, data: curves.map((c) => c.name) },
    xAxis: { type: "log", name: xName, nameLocation: "middle", ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => v.toLocaleString() } },
    yAxis: { type: "value", name: yName, nameLocation: "middle", ...axis(t, { nameGap: 56 }), axisLabel: { ...axis(t).axisLabel, formatter: fmt } },
    series: [...series, { type: "line", data: [], markLine: { silent: true, symbol: "none", lineStyle: { color: t.axis }, label: { show: false }, data: [{ yAxis: 0 }] } }],
  } as EChartsOption;
}
