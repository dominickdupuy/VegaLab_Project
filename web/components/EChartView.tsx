"use client";

import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import { useEffect, useRef } from "react";
import { optionSignature } from "@/lib/signature";

// only the pieces the dashboard draws, so the chunk carries nothing else
echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  HeatmapChart,
  CustomChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  MarkLineComponent,
  MarkAreaComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

export type ChartClick = (params: { seriesName?: string; dataIndex: number; name?: string; value?: unknown }) => void;
/** What Chart needs of the instance: a picture of it. */
export interface ChartInstance {
  getDataURL: (opts: { type?: "png"; pixelRatio?: number; backgroundColor?: string }) => string;
}

/** Every chart is drawn whole and still: no entrance, no morphing from the last option. */
function still(o: EChartsOption): EChartsOption {
  const list = Array.isArray(o.series) ? o.series : o.series ? [o.series] : [];
  return { ...o, animation: false, series: list.map((s) => ({ ...s, animation: false })) as EChartsOption["series"] };
}

/**
 * The ECharts binding itself, loaded on demand by Chart in Chart.tsx. Its own, so a
 * chart draws once per real change: the instance is made once, an option is set only
 * when what it draws differs from the last one set (`optionSignature`: the data, the
 * styles and the formatters' source), and it replaces the last one whole (notMerge), so
 * nothing morphs. A page that rebuilds an equal option on every render draws nothing new;
 * a click handler or a ready callback that changes identity is read through a ref.
 */
export default function EChartView({ option, height, onClick, onReady }: { option: EChartsOption; height: number; onClick?: ChartClick; onReady?: (chart: ChartInstance) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const drawn = useRef("");
  const click = useRef(onClick);
  click.current = onClick;
  const ready = useRef(onReady);
  ready.current = onReady;

  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const c = echarts.init(node, null, { renderer: "canvas" });
    chart.current = c;
    c.on("click", (p: unknown) => click.current?.(p as Parameters<ChartClick>[0]));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(node);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
      drawn.current = "";
    };
  }, []);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    const sig = optionSignature(option);
    if (sig === drawn.current) return;
    drawn.current = sig;
    c.setOption(still(option), { notMerge: true });
    // how many times this chart has been drawn, readable from the page (a redraw-loop check)
    if (el.current) el.current.dataset.draws = String(Number(el.current.dataset.draws ?? 0) + 1);
    ready.current?.(c);
  });

  return <div ref={el} style={{ height, width: "100%" }} />;
}
