"use client";

import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
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
import EChartsCore from "echarts-for-react/lib/core";

// only the pieces the dashboard draws, so the chunk carries nothing else
echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  HeatmapChart,
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

/** The ECharts binding itself, loaded on demand by Chart in Chart.tsx. */
export default function EChartView({ option, height, onClick }: { option: EChartsOption; height: number; onClick?: ChartClick }) {
  return (
    <EChartsCore
      echarts={echarts}
      option={option}
      notMerge
      style={{ height, width: "100%" }}
      opts={{ renderer: "canvas" }}
      onEvents={onClick ? { click: onClick } : undefined}
    />
  );
}
