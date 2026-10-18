"use client";

import dynamic from "next/dynamic";
import type { EChartsOption } from "echarts";
import type { ChartClick } from "./EChartView";
import { Skeleton } from "./States";
import ui from "./ui.module.css";

// ECharts loads in its own chunk when a chart first draws, so the pages do not wait for it
const EChartView = dynamic(() => import("./EChartView"), {
  ssr: false,
  loading: () => <Skeleton height={240} />,
});

export function Chart({
  option,
  height = 300,
  onClick,
  label,
}: {
  option: EChartsOption;
  height?: number;
  onClick?: ChartClick;
  /** What the chart shows, for a screen reader; the table beside it carries the numbers. */
  label: string;
}) {
  return (
    <div className={ui.chart} role="img" aria-label={label}>
      <EChartView option={option} height={height} onClick={onClick} />
    </div>
  );
}
