"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Segmented } from "@/components/Bits";
import { Chart3D, type Grid3D } from "@/components/gl/Chart3D";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import type { Tokens } from "@/lib/chart";
import { box, finiteRange, type GLOption, type Point3, surface, tenorAxis, vol, volRamp, zColour } from "@/lib/chart3d";

/**
 * The real SPX implied-vol surface for one day in 3D, from panel/surface_grid.parquet:
 * log-moneyness (or K / F) x tenor x IV on the build's fixed grid, holes where nothing
 * was quoted. A play control animates it through consecutive trading days.
 */

export type XMode = "k" | "m";
export const X_MODES = [
  { value: "k", label: "ln(K/F)" },
  { value: "m", label: "K/F" },
] as const;

/** The axes and one day's values of the gridded surface. */
export interface GridDay {
  k: number[];
  tenor_days: number[];
  iv: (number | null)[][];
}

export const xOf = (k: number, mode: XMode) => (mode === "k" ? k : Math.exp(k));
export const xName = (mode: XMode) => (mode === "k" ? "ln(K/F)" : "K/F");

export function describeCell(k: number, tenor: number, iv: number | null | undefined, extra = ""): string | null {
  if (iv === null || iv === undefined || !Number.isFinite(iv)) return null;
  return `ln(K/F) ${k.toFixed(3)} · K/F ${Math.exp(k).toFixed(3)} · ${tenor}d · IV ${(iv * 100).toFixed(2)} vol pts${extra}`;
}

/** The grid's points, row-major by tenor, z in vol points. */
export function gridPoints(g: GridDay, mode: XMode, z: (iv: number | null, row: number, col: number) => number = (iv) => vol(iv)): Point3[] {
  const out: Point3[] = [];
  g.iv.forEach((row, i) => g.k.forEach((k, j) => out.push([xOf(k, mode), i, z(row[j] ?? null, i, j)])));
  return out;
}

export function gridNav(g: GridDay, pts: Point3[], describe?: (row: number, col: number) => string | null): Grid3D {
  const cols = g.k.length;
  return {
    rows: g.tenor_days.length,
    cols,
    describe: describe ?? ((r, c) => describeCell(g.k[c] ?? 0, g.tenor_days[r] ?? 0, g.iv[r]?.[c])),
    at: (r, c) => {
      const p = pts[r * cols + c];
      return p && Number.isFinite(p[2]) ? p : null;
    },
  };
}

export function ivSurfaceOption(t: Tokens, g: GridDay, mode: XMode, z: [number, number], pts: Point3[]): GLOption {
  const xs = g.k.map((k) => xOf(k, mode));
  return {
    ...box(
      t,
      { name: xName(mode), min: xs[0], max: xs[xs.length - 1], formatter: (v: number) => v.toFixed(2) },
      tenorAxis("Tenor", g.tenor_days),
      { name: "IV (vol pts)", min: Math.floor(z[0]), max: Math.ceil(z[1]) },
    ),
    visualMap: [zColour(t, Math.floor(z[0]), Math.ceil(z[1]), volRamp(t), [0], "vol pts")],
    tooltip: { show: false },
    series: [surface(t, "main", pts, g.tenor_days.length, g.k.length)],
  };
}

// -- playback ------------------------------------------------------------------------ //

const CHUNK = 63; // a quarter of trading days per request
const AHEAD = 20; // fetch the next chunk this many frames before the buffer runs out
export const SPEEDS = [
  { value: "1", label: "1×" },
  { value: "2", label: "2×" },
  { value: "4", label: "4×" },
] as const;
type Speed = (typeof SPEEDS)[number]["value"];

export interface Playback {
  playing: boolean;
  buffering: boolean;
  frameDate: string | null;
  frame: GridDay | null;
  range: [number, number] | null; // z held still while playing, so the motion reads
  speed: Speed;
  setSpeed: (s: Speed) => void;
  toggle: () => void;
  error: string | null;
}

/**
 * Plays the gridded surface forward from ``date`` through the trading days in
 * ``dates``. Frames come a quarter at a time from /api/data/grid/frames and are kept,
 * so a replay is instant. Pausing hands the frame's date back to the page.
 */
export function useGridPlayback(dates: string[], date: string, commit: (d: string) => void): Playback {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>("1");
  const [idx, setIdx] = useState(-1);
  const [buffering, setBuffering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [axes, setAxes] = useState<{ k: number[]; tenor_days: number[] } | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const [, setTick] = useState(0);
  const buf = useRef(new Map<string, (number | null)[][]>());
  const inflight = useRef(false);
  const idxRef = useRef(idx);
  idxRef.current = idx;

  const load = useCallback(async (start: string) => {
    if (inflight.current) return;
    inflight.current = true;
    setBuffering(true);
    try {
      const f = await api.gridFrames(start, CHUNK);
      for (const fr of f.frames) buf.current.set(fr.date, fr.iv);
      setAxes({ k: f.k, tenor_days: f.tenor_days });
      const [lo, hi] = finiteRange(f.frames.flatMap((fr) => fr.iv.flat().map(vol)), 0.02);
      setRange((r) => (r ? [Math.min(r[0], lo), Math.max(r[1], hi)] : [lo, hi]));
      setTick((n) => n + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPlaying(false);
    } finally {
      inflight.current = false;
      setBuffering(false);
    }
  }, []);

  // keep the buffer ahead of the frame being shown
  useEffect(() => {
    if (!playing || idx < 0) return;
    const missing = dates.slice(idx, idx + AHEAD).find((d) => !buf.current.has(d));
    if (missing) void load(missing);
  }, [playing, idx, dates, load]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const next = idxRef.current + 1;
      if (next >= dates.length) {
        setPlaying(false);
        return;
      }
      if (buf.current.has(dates[next] ?? "")) setIdx(next);
    }, 400 / Number(speed));
    return () => window.clearInterval(id);
  }, [playing, speed, dates]);

  // the last frame shown becomes the page's date when the playback stops
  const wasPlaying = useRef(false);
  useEffect(() => {
    if (wasPlaying.current && !playing) {
      const d = dates[idxRef.current];
      if (d) commit(d);
      setRange(null);
    }
    wasPlaying.current = playing;
  }, [playing, dates, commit]);

  const toggle = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    setError(null);
    let i = dates.findIndex((d) => d >= date);
    if (i < 0 || i >= dates.length - 1) i = 0; // at the end: start over
    setIdx(i);
    setPlaying(true);
  };

  const frameDate = playing ? (dates[idx] ?? null) : null;
  const iv = frameDate ? buf.current.get(frameDate) : undefined;
  return {
    playing,
    buffering,
    frameDate,
    frame: iv && axes ? { ...axes, iv } : null,
    range: playing ? range : null,
    speed,
    setSpeed,
    toggle,
    error,
  };
}

export function PlayControls({ p, disabled }: { p: Playback; disabled?: boolean }) {
  return (
    <div className={ui.toolbar}>
      <button
        type="button"
        className={`${ui.btn} ${ui.btnSmall} ${p.playing ? "" : ui.btnPrimary}`}
        onClick={p.toggle}
        disabled={disabled}
        aria-pressed={p.playing}
        aria-label={p.playing ? "Pause the playback" : "Play forward through the trading days"}
      >
        {p.playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <Segmented label="Playback speed" value={p.speed} options={SPEEDS} onChange={p.setSpeed} />
      {p.frameDate ? (
        <span className={ui.frameDate} aria-live="off">
          {p.frameDate}
          {p.buffering ? " · loading" : ""}
        </span>
      ) : null}
    </div>
  );
}

/** The surface itself: one day, or the frame being played. */
export function IvSurface3D({ t, g, mode, range, date }: { t: Tokens; g: GridDay; mode: XMode; range: [number, number] | null; date: string }) {
  const pts = useMemo(() => gridPoints(g, mode), [g, mode]);
  const z = useMemo(() => range ?? finiteRange(pts.map((p) => p[2])), [range, pts]);
  const option = useMemo(() => ivSurfaceOption(t, g, mode, z, pts), [t, g, mode, z, pts]);
  const nav = useMemo(() => gridNav(g, pts), [g, pts]);
  return <Chart3D option={option} grid={nav} height={460} label={`Implied volatility surface on ${date}: ${xName(mode)} by tenor by IV`} />;
}

export function XModeToggle({ value, onChange }: { value: XMode; onChange: (m: XMode) => void }) {
  return <Segmented label="Strike axis" value={value} options={X_MODES} onChange={onChange} />;
}
