"use client";

import { useEffect, useState } from "react";
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemReturn, EChartsOption, SeriesOption } from "echarts";
import { esc, num, usd } from "./format";
import { useTheme } from "./theme";

/**
 * Chart styling from the CSS tokens, so a chart follows the theme like everything
 * else. Colours are fixed by entity, never by rank: the fixed-delta seller is always
 * --s1, PPO always --s2, OptionMetrics always --s1 and Databento always --s2, so a
 * filter that hides a series never repaints the others. Status colours are not used
 * for series.
 */

export interface Tokens {
  text: string;
  text2: string;
  muted: string;
  grid: string;
  axis: string;
  surface: string;
  surface2: string;
  accent: string;
  pass: string;
  fail: string;
  warn: string;
  s1: string;
  s2: string;
  s3: string;
  s4: string;
  s5: string;
  font: string;
  dark: boolean;
}

const LIGHT: Tokens = {
  text: "#1f2533",
  text2: "#3a3f4a",
  muted: "#5a6272",
  grid: "#e3e8ef",
  axis: "#c7d0dc",
  surface: "#ffffff",
  surface2: "#e6eef8",
  accent: "#2a5c9a",
  pass: "#2e7d4f",
  fail: "#b3261e",
  warn: "#8a5a00",
  s1: "#2a5c9a",
  s2: "#d9822b",
  s3: "#1f8a84",
  s4: "#6b5bb5",
  s5: "#7a8394",
  font: "Calibri, Carlito, 'Segoe UI', system-ui, sans-serif",
  dark: false,
};

function readTokens(): Tokens {
  if (typeof document === "undefined") return LIGHT;
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    text: v("--text", LIGHT.text),
    text2: v("--text-2", LIGHT.text2),
    muted: v("--muted", LIGHT.muted),
    grid: v("--grid", LIGHT.grid),
    axis: v("--axis", LIGHT.axis),
    surface: v("--surface-solid", LIGHT.surface),
    surface2: v("--surface-2", LIGHT.surface2),
    accent: v("--accent", LIGHT.accent),
    pass: v("--pass", LIGHT.pass),
    fail: v("--fail", LIGHT.fail),
    warn: v("--warn", LIGHT.warn),
    s1: v("--s1", LIGHT.s1),
    s2: v("--s2", LIGHT.s2),
    s3: v("--s3", LIGHT.s3),
    s4: v("--s4", LIGHT.s4),
    s5: v("--s5", LIGHT.s5),
    font: v("--font", LIGHT.font),
    dark: document.documentElement.dataset.theme === "dark",
  };
}

/** The tokens for the current theme, re-read whenever it flips. */
export function useTokens(): Tokens {
  const [theme] = useTheme();
  const [tokens, setTokens] = useState<Tokens>(LIGHT);
  useEffect(() => setTokens(readTokens()), [theme]);
  return tokens;
}

const ARM_SLOT: Record<string, keyof Tokens> = {
  delta_seller: "s1",
  ppo: "s2",
  a2c: "s3",
  dqn: "s4",
  always_on: "s5",
  random: "s5",
  flat: "s5",
};

export function armColor(t: Tokens, arm: string): string {
  return t[ARM_SLOT[arm] ?? "s5"] as string;
}

export function sourceColor(t: Tokens, source: string): string {
  return source === "databento" ? t.s2 : source === "optionmetrics" ? t.s1 : t.s3;
}

export const ARM_LABEL: Record<string, string> = {
  delta_seller: "Fixed-delta seller",
  ppo: "PPO",
  a2c: "A2C",
  dqn: "DQN",
  always_on: "Always-on condor",
  random: "Random",
  flat: "Flat",
};

export const SOURCE_LABEL: Record<string, string> = {
  optionmetrics: "OptionMetrics",
  databento: "Databento OPRA",
  synthetic: "Synthetic",
};

/** The parts every chart shares: type, tooltip, grid, legend. */
export function base(t: Tokens) {
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: t.font, color: t.text2, fontSize: 12 },
    animationDuration: 300,
    grid: { left: 56, right: 18, top: 34, bottom: 40, containLabel: false },
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: t.surface,
      borderColor: t.axis,
      borderWidth: 1,
      textStyle: { color: t.text, fontFamily: t.font, fontSize: 12 },
      axisPointer: { type: "line" as const, lineStyle: { color: t.axis, width: 1 } },
      confine: true,
    },
    legend: {
      top: 0,
      right: 0,
      icon: "roundRect",
      itemWidth: 12,
      itemHeight: 8,
      textStyle: { color: t.text2, fontFamily: t.font, fontSize: 12 },
    },
  };
}

export function axis(t: Tokens, extra: Record<string, unknown> = {}) {
  return {
    axisLine: { lineStyle: { color: t.axis } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontFamily: t.font, fontSize: 11 },
    splitLine: { lineStyle: { color: t.grid } },
    nameTextStyle: { color: t.muted, fontFamily: t.font, fontSize: 11.5 },
    nameGap: 26,
    ...extra,
  };
}

/** A histogram's bars as [x0, x1, count] for a custom-width bar series. */
export function histBars(edges: number[], counts: number[]): { value: [number, number] }[] {
  return counts.map((c, i) => ({ value: [((edges[i] ?? 0) + (edges[i + 1] ?? 0)) / 2, c] as [number, number] }));
}

/** One point of a P&L-against-premium curve: a run's mean and its standard error. */
export interface PremiumPoint {
  premium: number;
  mean: number;
  se: number | null;
  label: string; // the run it came from, shown in the tooltip
}

function errorBar(color: string) {
  return (_: unknown, api: CustomSeriesRenderItemAPI): CustomSeriesRenderItemReturn => {
    const x = Number(api.value(0));
    const lo = api.coord([x, Number(api.value(1))]);
    const hi = api.coord([x, Number(api.value(2))]);
    const [x0, y0, y1] = [lo[0] ?? 0, lo[1] ?? 0, hi[1] ?? 0];
    const style = { stroke: color, lineWidth: 1.5 };
    return {
      type: "group",
      children: [
        { type: "line", shape: { x1: x0, y1: y0, x2: x0, y2: y1 }, style },
        { type: "line", shape: { x1: x0 - 4, y1: y0, x2: x0 + 4, y2: y0 }, style },
        { type: "line", shape: { x1: x0 - 4, y1: y1, x2: x0 + 4, y2: y1 }, style },
      ],
    };
  };
}

/**
 * Mean P&L against the planted premium for one arm: a line with markers in premium
 * order, and a +-1 se error bar on every point. Both series carry the arm's name, so
 * its legend entry toggles the two together.
 */
export function premiumCurve(t: Tokens, arm: string, points: PremiumPoint[]): SeriesOption[] {
  const name = ARM_LABEL[arm] ?? arm;
  const color = armColor(t, arm);
  const pts = [...points].sort((a, b) => a.premium - b.premium);
  return [
    {
      name,
      type: "line",
      data: pts.map((p) => ({ value: [p.premium, p.mean], point: p })),
      showSymbol: true,
      symbol: "circle",
      symbolSize: 8,
      lineStyle: { width: 2, color },
      itemStyle: { color, borderColor: t.surface, borderWidth: 2 },
      tooltip: {
        formatter: (q: unknown) => {
          const p = (q as { data: { point: PremiumPoint } }).data.point;
          return `<b>${esc(name)}</b> at ${num(p.premium, 2)} vol pts<br/>mean ${usd(p.mean)} ± ${usd(p.se)}<br/><span style="opacity:.7">${esc(p.label)}</span>`;
        },
      },
      z: 3,
    },
    {
      name,
      type: "custom",
      data: pts.filter((p) => p.se !== null).map((p) => [p.premium, p.mean - (p.se ?? 0), p.mean + (p.se ?? 0)]),
      encode: { x: 0, y: [1, 2] },
      renderItem: errorBar(color),
      silent: true,
      tooltip: { show: false },
      z: 2,
    },
  ];
}

/** The chart around a set of premium curves: value axes, an item tooltip, a legend. */
export function premiumChart(t: Tokens, curves: Record<string, PremiumPoint[]>): EChartsOption {
  const arms = Object.keys(curves);
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, trigger: "item" },
    legend: { ...base(t).legend, data: arms.map((a) => ARM_LABEL[a] ?? a) },
    xAxis: { type: "value", name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "Mean P&L per episode ($)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: arms.flatMap((arm) => premiumCurve(t, arm, curves[arm] ?? [])),
  };
}
