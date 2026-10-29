"use client";

import dynamic from "next/dynamic";
import { type KeyboardEvent, type ReactNode, useEffect, useId, useMemo, useRef, useState } from "react";
import { CAMERA, type Camera, type GLOption, type Point3, points } from "@/lib/chart3d";
import { useTokens } from "@/lib/chart";
import { type Row, rowsFromOption } from "@/lib/export";
import { FigureFoot } from "../Export";
import type { GLHover } from "./GLView";
import { Skeleton } from "../States";
import ui from "../ui.module.css";

// echarts-gl is large: it loads in its own chunk, and only when a 3D chart first draws
const GLView = dynamic(() => import("./GLView"), { ssr: false, loading: () => <Skeleton height={240} /> });

/**
 * The grid a 3D view is drawn on, for the readout: what a point says, where it sits.
 * Row-major, the same order as the view's main surface series (id "main"), so a
 * hover's data index and an arrow key land on the same cell.
 */
export interface Grid3D {
  rows: number;
  cols: number;
  /** The readout for a cell, or null where nothing was quoted. */
  describe: (row: number, col: number) => string | null;
  /** Where the cell is drawn, for the keyboard cursor. */
  at: (row: number, col: number) => Point3 | null;
}

function webgl(): boolean {
  try {
    const c = document.createElement("canvas");
    return Boolean(c.getContext("webgl2") ?? c.getContext("webgl"));
  } catch {
    return false;
  }
}

/**
 * A 3D chart in a card: rotate by dragging, zoom with the wheel or a pinch, "Reset
 * view" to go back. The readout under it says what is under the pointer, in text and
 * live for a screen reader; with the chart focused, the arrow keys walk the grid and
 * move a marker, so every value is reachable without a mouse.
 */
export function Chart3D({
  option,
  grid,
  describeHover,
  label,
  height = 440,
  camera = CAMERA,
  hint = "Drag to rotate · scroll or pinch to zoom · arrow keys step through the grid",
  fallback,
  stamp,
  rows: exportRows,
}: {
  option: GLOption;
  grid?: Grid3D;
  /** The readout for a hover on a series other than "main". */
  describeHover?: (h: GLHover) => string | null;
  label: string;
  height?: number;
  camera?: Camera;
  hint?: string;
  /** Shown instead when the browser has no WebGL. */
  fallback?: ReactNode;
  /** The figure's caption: the file or design that produced it. */
  stamp?: ReactNode;
  /** The rows CSV and JSON export; by default, the points the option draws. */
  rows?: () => Row[];
}) {
  const t = useTokens();
  const [gl, setGl] = useState<boolean | null>(null);
  const [cameraKey, setCameraKey] = useState(0);
  const [readout, setReadout] = useState<string | null>(null);
  const [cursor, setCursor] = useState<[number, number] | null>(null);
  const hintId = useId();
  const inst = useRef<{ getDataURL: (o: Record<string, unknown>) => string } | null>(null);
  useEffect(() => setGl(webgl()), []);

  // a new grid (another date, another premium) keeps the cursor where it is, if it still fits
  const rows = grid?.rows ?? 0;
  const cols = grid?.cols ?? 0;
  useEffect(() => {
    setCursor((c) => (c && c[0] < rows && c[1] < cols ? c : null));
  }, [rows, cols]);
  useEffect(() => {
    if (cursor && grid) setReadout(grid.describe(cursor[0], cursor[1]) ?? "not quoted");
  }, [cursor, grid]);

  const overlay = useMemo(() => {
    const p = cursor && grid ? grid.at(cursor[0], cursor[1]) : null;
    return p ? [points("cursor", [p], t.s2, 14, { silent: true, itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2, opacity: 1 } })] : [];
  }, [cursor, grid, t]);

  const onHover = (h: GLHover | null) => {
    if (!h) return;
    if (h.seriesId === "main" && grid && grid.cols) {
      const r = Math.floor(h.dataIndex / grid.cols);
      const c = h.dataIndex % grid.cols;
      setReadout(grid.describe(r, c) ?? "not quoted");
    } else if (describeHover) {
      const text = describeHover(h);
      if (text) setReadout(text);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!grid || !grid.rows || !grid.cols) return;
    const move: Record<string, [number, number]> = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [1, 0], ArrowDown: [-1, 0] };
    const step = move[e.key];
    if (e.key === "Escape") {
      setCursor(null);
      return;
    }
    if (!step && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    setCursor((c) => {
      if (e.key === "Home") return [0, 0];
      if (e.key === "End") return [grid.rows - 1, grid.cols - 1];
      const [r, k] = c ?? [Math.floor(grid.rows / 2), Math.floor(grid.cols / 2)];
      const big = e.shiftKey ? 5 : 1;
      return [Math.max(0, Math.min(grid.rows - 1, r + (step?.[0] ?? 0) * big)), Math.max(0, Math.min(grid.cols - 1, k + (step?.[1] ?? 0) * big))];
    });
  };

  if (gl === false) return <>{fallback ?? <p className={ui.dim}>This view needs WebGL, which this browser does not offer.</p>}</>;
  return (
    <div className={ui.chart3d}>
      <div
        className={ui.chart3dCanvas}
        style={{ ["--h" as string]: `${height}px` }}
        role="img"
        aria-label={label}
        aria-describedby={hintId}
        tabIndex={0}
        onKeyDown={onKey}
        onBlur={() => setCursor(null)}
      >
        {gl ? <GLView option={option} overlay={overlay} camera={camera} cameraKey={cameraKey} onHover={onHover} onReady={(c) => (inst.current = c)} /> : <Skeleton height={height} />}
        <button type="button" className={`${ui.btn} ${ui.btnSmall} ${ui.chart3dReset}`} onClick={() => setCameraKey((k) => k + 1)}>
          Reset view
        </button>
      </div>
      <p className={ui.readout} aria-live="polite">
        {readout ?? <span id={hintId}>{hint}</span>}
      </p>
      {readout ? (
        <span id={hintId} className={ui.srOnly}>
          {hint}
        </span>
      ) : null}
      <FigureFoot
        name={label}
        stamp={stamp}
        rows={exportRows ?? (() => rowsFromOption(option))}
        chart={() => ({ png: () => inst.current?.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: t.surface }) ?? null })}
      />
    </div>
  );
}
