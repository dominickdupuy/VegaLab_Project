"use client";

import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemReturn, EChartsOption } from "echarts";
import { FigureFoot } from "@/components/Export";
import ui from "@/components/ui.module.css";
import type { VrpEstimate, VrpMapping } from "@/lib/api";
import { axis, base, type Tokens } from "@/lib/chart";
import { MINUS, num, pct } from "@/lib/format";

/** The VRP mapping's pieces (docs/VRP_MAPPING.json), for the Real data page's first question. */

export const signedPts = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? "–" : `${x >= 0 ? "+" : MINUS}${Math.abs(x).toFixed(d)}`);

function whisker(color: string) {
  return (_: unknown, api: CustomSeriesRenderItemAPI): CustomSeriesRenderItemReturn => {
    const lo = api.coord([Number(api.value(1)), api.value(0)]);
    const hi = api.coord([Number(api.value(2)), api.value(0)]);
    const style = { stroke: color, lineWidth: 2 };
    const [x0, y, x1] = [lo[0] ?? 0, lo[1] ?? 0, hi[0] ?? 0];
    return {
      type: "group",
      children: [
        { type: "line", shape: { x1: x0, y1: y, x2: x1, y2: y }, style },
        { type: "line", shape: { x1: x0, y1: y - 5, x2: x0, y2: y + 5 }, style },
        { type: "line", shape: { x1: x1, y1: y - 5, x2: x1, y2: y + 5 }, style },
      ],
    };
  };
}

/** Estimates with their 95% intervals on the sweep's premium axis, the sweep's levels marked. */
export function axisOption(t: Tokens, rows: VrpEstimate[], sweep: number[], label: (r: VrpEstimate) => string = (r) => r.label): EChartsOption {
  const cats = rows.map(label);
  const lo = Math.min(-1, ...rows.map((r) => r.mean_low ?? 0));
  const hi = Math.max(4.5, ...rows.map((r) => r.mean_high ?? 0));
  return {
    ...base(t),
    legend: { show: false },
    tooltip: {
      ...base(t).tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const i = (p as { dataIndex: number }).dataIndex;
        const r = rows[i];
        return r
          ? `<b>${cats[i]}</b><br/>mean ${signedPts(r.mean)} vol pts<br/>95% [${signedPts(r.mean_low)}, ${signedPts(r.mean_high)}]<br/>median ${signedPts(r.median)} · positive ${pct(r.share_positive)}<br/>${num(r.n_days)} days`
          : "";
      },
    },
    grid: { left: 150, right: 24, top: 26, bottom: 44 },
    xAxis: { type: "value", name: "Premium (vol pts) · dashed: the sweep's levels", nameLocation: "middle", ...axis(t), min: Math.floor(lo), max: Math.ceil(hi), nameGap: 28 },
    yAxis: { type: "category", data: cats, inverse: true, ...axis(t), splitLine: { show: false } },
    series: [
      {
        name: "interval",
        type: "custom",
        data: rows.map((r, i) => [i, r.mean_low, r.mean_high]),
        encode: { y: 0, x: [1, 2] },
        renderItem: whisker(t.s1),
        silent: true,
        tooltip: { show: false },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: t.s5, width: 1, type: "dashed" },
          label: { color: t.muted, fontSize: 10, position: "start", formatter: (q: { value?: unknown }) => `${q.value}` }, // the y axis is inverted: start is the top
          data: sweep.map((x) => ({ xAxis: x })),
        },
        z: 2,
      },
      {
        name: "mean",
        type: "scatter",
        data: rows.map((r, i) => [r.mean, i]),
        symbolSize: (_: unknown, p: { dataIndex: number }) => (p.dataIndex === 0 ? 13 : 9),
        itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 },
        z: 3,
      },
    ],
  };
}

/** The headline under other definitions of realised vol. */
export function SensitivityTable({ m }: { m: VrpMapping }) {
  return (
    <div className={ui.stack}>
      <div className={ui.tableWrap}>
        <table className={ui.table}>
          <thead>
            <tr>
              <th>Definition</th>
              <th className={ui.num}>Mean</th>
              <th className={ui.num}>95% interval</th>
              <th className={ui.num}>Median</th>
              <th className={ui.num}>Positive</th>
            </tr>
          </thead>
          <tbody>
            {m.sensitivity.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td className={ui.num}>{signedPts(r.mean)}</td>
                <td className={ui.num}>
                  [{signedPts(r.mean_low)}, {signedPts(r.mean_high)}]
                </td>
                <td className={ui.num}>{signedPts(r.median)}</td>
                <td className={ui.num}>{pct(r.share_positive)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <FigureFoot name="vrp sensitivity" stamp={`VRP mapping · ${m.generated}`} rows={() => m.sensitivity.map((r) => ({ ...r }))} json={() => m} />
    </div>
  );
}
