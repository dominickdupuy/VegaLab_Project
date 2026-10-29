"use client";

import dynamic from "next/dynamic";
import { type ReactNode, useRef, useState } from "react";
import type { EChartsOption } from "echarts";
import { useTokens } from "@/lib/chart";
import { type Row, rowsFromOption } from "@/lib/export";
import type { ChartClick, ChartInstance } from "./EChartView";
import { FigureFoot } from "./Export";
import { ChartSkeleton } from "./States";
import ui from "./ui.module.css";

// ECharts loads in its own chunk when a chart first draws, so the pages do not wait for it
const EChartView = dynamic(() => import("./EChartView"), {
  ssr: false,
  loading: () => null,
});

export function Chart({
  option,
  height = 300,
  onClick,
  label,
  stamp,
  rows,
  json,
  name,
}: {
  option: EChartsOption;
  height?: number;
  onClick?: ChartClick;
  /** What the chart shows, for a screen reader; the table beside it carries the numbers. */
  label: string;
  /** The figure's caption: the hash, commit and source that produced it. */
  stamp?: ReactNode;
  /** The rows CSV and JSON export; by default, the points the option draws. */
  rows?: () => Row[];
  /** What JSON export writes, when it is more than the rows. */
  json?: () => unknown;
  /** The export file name; the label by default. */
  name?: string;
}) {
  const t = useTokens();
  const inst = useRef<ChartInstance | null>(null);
  const [ready, setReady] = useState(false);
  return (
    <div>
      <div className={ui.chart} role="img" aria-label={label} style={{ position: "relative", minHeight: height }}>
        <EChartView
          option={option}
          height={height}
          onClick={onClick}
          onReady={(c) => {
            inst.current = c;
            if (!ready) setReady(true);
          }}
        />
        {ready ? null : (
          <div style={{ position: "absolute", inset: 0 }}>
            <ChartSkeleton height={height} />
          </div>
        )}
      </div>
      <FigureFoot
        name={name ?? label}
        stamp={stamp}
        rows={rows ?? (() => rowsFromOption(option as Record<string, unknown>))}
        json={json}
        chart={() => ({ png: () => inst.current?.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: t.surface }) ?? null })}
      />
    </div>
  );
}
