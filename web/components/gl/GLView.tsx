"use client";

import { useEffect, useRef } from "react";
import { TooltipComponent, VisualMapComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { Line3DChart, Scatter3DChart, SurfaceChart } from "echarts-gl/charts";
import { Grid3DComponent } from "echarts-gl/components";
import type { Camera, GLOption } from "@/lib/chart3d";
import { optionSignature } from "@/lib/signature";

// echarts-gl and only the parts the 3D views draw; this chunk loads on the pages that use it
echarts.use([TooltipComponent, VisualMapComponent, CanvasRenderer, Grid3DComponent, SurfaceChart, Line3DChart, Scatter3DChart]);

export interface GLHover {
  seriesId: string;
  dataIndex: number;
}

/**
 * The WebGL instance itself, loaded on demand by Chart3D. Its own binding rather than
 * echarts-for-react, because a 3D chart must keep the camera the reader dragged to:
 * a new option (a date stepped, a premium moved) merges into the scene and replaces
 * only the series, and the camera is set only on mount and on "Reset view".
 */
export default function GLView({
  option,
  overlay,
  camera,
  cameraKey,
  onHover,
  onReady,
}: {
  option: GLOption;
  /** Series laid over the option without replacing it: the keyboard cursor. */
  overlay: GLOption[];
  camera: Camera;
  cameraKey: number;
  onHover: (h: GLHover | null) => void;
  /** The instance, for a PNG of the scene. */
  onReady?: (chart: echarts.ECharts) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const hover = useRef(onHover);
  hover.current = onHover;
  const ready = useRef(onReady);
  ready.current = onReady;
  const first = useRef(true);
  const drawn = useRef("");

  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const c = echarts.init(node, null, { renderer: "canvas" });
    chart.current = c;
    ready.current?.(c);
    c.on("mousemove", (p: unknown) => {
      const q = p as { seriesId?: string; dataIndex?: number };
      if (q.seriesId !== undefined && typeof q.dataIndex === "number" && q.dataIndex >= 0) hover.current({ seriesId: q.seriesId, dataIndex: q.dataIndex });
    });
    c.on("mouseout", () => hover.current(null));
    c.on("globalout", () => hover.current(null));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(node);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
      first.current = true;
      drawn.current = "";
    };
  }, []);

  // set only when the scene differs from the last one set: a parent that rebuilds an equal
  // option (or a new empty overlay) on every render draws nothing new
  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    const sig = `${optionSignature(option)}|${overlay.map((o) => optionSignature(o)).join("|")}`;
    if (sig === drawn.current) return;
    drawn.current = sig;
    const grid3D = (option.grid3D ?? {}) as GLOption;
    const viewControl = (grid3D.viewControl ?? {}) as GLOption;
    const withCamera = first.current ? { ...option, grid3D: { ...grid3D, viewControl: { ...viewControl, ...camera } } } : option;
    first.current = false;
    c.setOption(withCamera, { replaceMerge: ["series", "visualMap"] });
    if (overlay.length) c.setOption({ series: overlay });
    if (el.current) el.current.dataset.draws = String(Number(el.current.dataset.draws ?? 0) + 1);
    // the camera is not a dependency on purpose: applied once, then left to the reader
  }, [option, overlay]);

  useEffect(() => {
    if (cameraKey === 0) return;
    chart.current?.setOption({ grid3D: { viewControl: { ...camera } } });
  }, [cameraKey]);

  return <div ref={el} style={{ width: "100%", height: "100%" }} />;
}
