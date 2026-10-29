/**
 * The 3D views' shared pieces: the box, its axes, the colour ramps and the surface
 * series, styled from the same tokens as the 2D charts so both themes hold.
 *
 * Only data with two genuine input axes is drawn in 3D (strike x tenor, date x
 * tenor, premium x seeds); one-dimensional series stay on the 2D charts. Every view
 * builds a plain option object here and hands it to Chart3D, which owns the WebGL
 * instance, the camera and the readout, so a new view is one option builder.
 */

import type { Tokens } from "./chart";

/** An echarts option with the echarts-gl parts, which the echarts types do not know. */
export type GLOption = Record<string, unknown>;

export type Point3 = [number, number, number];

/** Where the camera starts; Chart3D puts it back there on "Reset view". */
export interface Camera {
  alpha: number; // elevation, degrees
  beta: number; // azimuth, degrees
  distance: number;
}

export const CAMERA: Camera = { alpha: 20, beta: 38, distance: 200 };

/** Sequential, one hue light to dark: the deck's blues. Dark mode runs the other way. */
export function volRamp(t: Tokens): string[] {
  return t.dark ? ["#1a2e4d", "#2f5a91", "#6b9ad8", "#cde2fb"] : ["#e6eef8", "#86b6ef", "#2a78d6", "#13233d"];
}

/** Diverging: blue below zero, orange above, a neutral grey between. */
export function divergingRamp(t: Tokens): string[] {
  return t.dark ? [t.s1, "#5a6b84", "#8b929e", "#a87a4e", t.s2] : [t.s1, "#8fa6c4", "#c9ccd2", "#e5b98c", t.s2];
}

export interface Axis3D {
  name: string;
  min?: number;
  max?: number;
  interval?: number;
  formatter?: (v: number) => string;
}

function axis3D(t: Tokens, a: Axis3D) {
  return {
    type: "value",
    name: a.name,
    min: a.min,
    max: a.max,
    interval: a.interval,
    nameGap: 26,
    nameTextStyle: { color: t.muted, fontFamily: t.font, fontSize: 12 },
    axisLine: { lineStyle: { color: t.axis, width: 1 } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontFamily: t.font, fontSize: 11, formatter: a.formatter },
    splitLine: { lineStyle: { color: t.grid, width: 1 } },
    axisPointer: { lineStyle: { color: t.accent, width: 1.5 }, label: { show: false } },
  };
}

/** The box and its three axes. The camera is not set here: Chart3D owns it. */
export function box(t: Tokens, x: Axis3D, y: Axis3D, z: Axis3D, size: { width?: number; depth?: number; height?: number } = {}): GLOption {
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: t.font, color: t.text2 },
    // an automatic axis prints its unrounded minimum beside the first round tick (-0.50
    // against -0.40): drop it
    xAxis3D: axis3D(t, {
      ...x,
      formatter: (v: number) => {
        if (x.interval === undefined && x.min !== undefined && Math.abs(v - x.min) < 1e-9) return "";
        return x.formatter ? x.formatter(v) : String(Number(v.toPrecision(6)));
      },
    }),
    yAxis3D: axis3D(t, y),
    // the z axis's lowest label sits on the box corner where the x and y labels meet, and an
    // automatic axis adds its unrounded maximum just above the last tick: drop both
    zAxis3D: axis3D(t, {
      ...z,
      formatter: (v: number) => {
        const at = (b: number | undefined) => b !== undefined && Math.abs(v - b) < 1e-9;
        if (at(z.min) || (z.interval === undefined && at(z.max))) return "";
        return z.formatter ? z.formatter(v) : String(Number(v.toPrecision(6)));
      },
    }),
    grid3D: {
      boxWidth: size.width ?? 120,
      boxDepth: size.depth ?? 96,
      boxHeight: size.height ?? 66,
      environment: "none",
      axisLine: { lineStyle: { color: t.axis } },
      splitArea: { show: false },
      light: {
        main: { intensity: 1.05, shadow: false, alpha: 40, beta: 30 },
        ambient: { intensity: 0.42 },
      },
      viewControl: {
        projection: "perspective",
        autoRotate: false,
        damping: 0.85,
        rotateSensitivity: 1.2,
        zoomSensitivity: 1.1,
        panSensitivity: 1,
        minDistance: 80,
        maxDistance: 400,
        animationDurationUpdate: 0,
      },
      postEffect: { enable: false },
      temporalSuperSampling: { enable: true },
    },
  };
}

/** A continuous colour bar keyed on z, applied to the named series only. */
export function zColour(t: Tokens, lo: number, hi: number, colours: string[], seriesIndex: number[], label: string, format: (v: number) => string = String): GLOption {
  return {
    type: "continuous",
    id: "z",
    dimension: 2,
    seriesIndex,
    min: lo,
    max: hi,
    calculable: false,
    realtime: false,
    orient: "vertical",
    right: 0,
    top: "middle",
    itemWidth: 10,
    itemHeight: 130,
    precision: 1,
    text: [`${format(hi)} ${label}`, format(lo)],
    textStyle: { color: t.muted, fontFamily: t.font, fontSize: 11 },
    inRange: { color: colours },
  };
}

/**
 * A surface over a rows x cols grid of points, row-major, holes as NaN. The shape is
 * given rather than guessed, so a row whose x differs (the simulator's log-moneyness
 * moves with each tenor's forward) still meshes.
 */
export function surface(t: Tokens, id: string, points: Point3[], rows: number, cols: number, extra: GLOption = {}): GLOption {
  return {
    id,
    type: "surface",
    data: points,
    dataShape: [rows, cols],
    shading: "lambert",
    silent: false,
    wireframe: { show: true, lineStyle: { color: t.dark ? "rgba(232,238,247,0.22)" : "rgba(19,35,61,0.22)", width: 1 } },
    itemStyle: { opacity: 1 },
    animation: false,
    ...extra,
  };
}

/** Points in 3D: contracts, a marked design, the keyboard cursor. */
export function points(id: string, data: Point3[], color: string, size: number, extra: GLOption = {}): GLOption {
  return {
    id,
    type: "scatter3D",
    data,
    symbolSize: size,
    itemStyle: { color, borderColor: "#ffffff", borderWidth: 0.5, opacity: 1 },
    animation: false,
    ...extra,
  };
}

export function line(id: string, data: Point3[], color: string, width: number, extra: GLOption = {}): GLOption {
  return { id, type: "line3D", data, lineStyle: { color, width, opacity: 1 }, animation: false, silent: true, ...extra };
}

/** The finite range of some numbers, padded so a flat surface still has height. */
export function finiteRange(values: Iterable<number | null | undefined>, pad = 0.04): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return [0, 1];
  const span = hi - lo || Math.max(Math.abs(hi), 1) * 0.1;
  return [lo - span * pad, hi + span * pad];
}

/**
 * A tenor's position on an axis spaced one step per grid tenor, so 7 to 30 days is
 * not squeezed into the first sliver of a 365-day axis. Linear between grid tenors.
 */
export function tenorPos(days: number, grid: number[]): number {
  if (!grid.length) return 0;
  if (days <= (grid[0] ?? 0)) return 0;
  for (let i = 1; i < grid.length; i++) {
    const a = grid[i - 1] ?? 0;
    const b = grid[i] ?? 0;
    if (days <= b) return i - 1 + (days - a) / (b - a);
  }
  return grid.length - 1;
}

/** Axis labels at the grid tenors only. */
export function tenorAxis(name: string, grid: number[]): Axis3D {
  return {
    name,
    min: 0,
    max: Math.max(1, grid.length - 1),
    interval: grid.length > 8 ? 2 : 1,
    formatter: (v: number) => (Number.isInteger(v) && grid[v] !== undefined ? `${grid[v]}d` : ""),
  };
}

export const vol = (x: number | null | undefined) => (x === null || x === undefined ? NaN : x * 100);
