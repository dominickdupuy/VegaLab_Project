"use client";

import { useEffect, useState } from "react";

/**
 * Reduced motion, for the code that animates outside CSS: ECharts and echarts-gl draw
 * on a canvas, and playback and rotation run on timers, so the CSS rule in
 * app/globals.css never reaches them. Every such animation asks here first.
 */
const QUERY = "(prefers-reduced-motion: reduce)";

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && Boolean(window.matchMedia?.(QUERY).matches);
}

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const m = window.matchMedia?.(QUERY);
    if (!m) return;
    setReduced(m.matches);
    const on = () => setReduced(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return reduced;
}
