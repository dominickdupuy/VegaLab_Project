"use client";

import { useEffect, useState } from "react";
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
  always_on: "s3",
  random: "s4",
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
