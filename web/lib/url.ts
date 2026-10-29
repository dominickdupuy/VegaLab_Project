"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useRef } from "react";

/**
 * Every page keeps its knobs in the query string, so any view can be linked and a
 * workflow step can open it pre-filled. A page reads its parameters once, on mount,
 * and writes them back as they change with history.replaceState: no navigation, no
 * scroll, and parameters it does not own (a workflow's `wf` and `step`) survive.
 */

export type QueryValue = string | number | boolean | null | undefined;

/** Merge `update` into the current query; null, undefined or "" removes a key. */
export function setQuery(update: Record<string, QueryValue>): void {
  if (typeof window === "undefined") return;
  const q = new URLSearchParams(window.location.search);
  for (const [k, v] of Object.entries(update)) {
    if (v === null || v === undefined || v === "" || v === false) q.delete(k);
    else q.set(k, v === true ? "1" : String(v));
  }
  const s = q.toString();
  const url = `${window.location.pathname}${s ? `?${s}` : ""}${window.location.hash}`;
  // null, not the current state: Next.js copies its own state over, and only then tells
  // useSearchParams, so a component reading the URL sees the write (one passing Next's
  // state through is taken for Next's own call and left unsynced)
  if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) window.history.replaceState(null, "", url);
}

/**
 * Write `values` to the query whenever they change, leaving out any equal to its
 * default so a link carries only what was changed. Debounced, for sliders.
 */
export function useQuerySync(values: Record<string, QueryValue>, defaults: Record<string, QueryValue> = {}, ms = 150): void {
  const key = JSON.stringify(values);
  const first = useRef(true);
  useEffect(() => {
    // the first render only reflects what the URL already said
    if (first.current) {
      first.current = false;
      return;
    }
    const id = window.setTimeout(() => {
      const out: Record<string, QueryValue> = {};
      for (const [k, v] of Object.entries(values)) out[k] = defaults[k] !== undefined && String(defaults[k]) === String(v) ? null : v;
      setQuery(out);
    }, ms);
    return () => window.clearTimeout(id);
  }, [key]);
}

/** Readers for one parameter, with the page's default when absent or malformed. */
export function useQuery() {
  const p = useSearchParams();
  return {
    params: p,
    str: (k: string, d = ""): string => p.get(k) ?? d,
    num: (k: string, d: number, lo = -Infinity, hi = Infinity): number => {
      const raw = p.get(k);
      const v = raw === null || raw.trim() === "" ? NaN : Number(raw);
      return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d;
    },
    int: (k: string, d: number, lo = -Infinity, hi = Infinity): number => {
      const raw = p.get(k);
      const v = raw === null || raw.trim() === "" ? NaN : Math.round(Number(raw));
      return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d;
    },
    bool: (k: string): boolean => {
      const raw = p.get(k);
      return raw === "1" || raw === "true";
    },
    oneOf: <T extends string>(k: string, options: readonly T[], d: T): T => {
      const raw = p.get(k);
      return raw !== null && (options as readonly string[]).includes(raw) ? (raw as T) : d;
    },
  };
}

/** A link to `href` that keeps the workflow the reader is in, if any. */
export function keepWorkflow(href: string, params: URLSearchParams): string {
  const wf = params.get("wf");
  const step = params.get("step");
  if (!wf) return href;
  const [path, hash = ""] = href.split("#");
  const sep = (path ?? "").includes("?") ? "&" : "?";
  return `${path}${sep}wf=${encodeURIComponent(wf)}${step ? `&step=${encodeURIComponent(step)}` : ""}${hash ? `#${hash}` : ""}`;
}

/** Scroll to the URL's #anchor once the section it names has rendered. */
export function useHashScroll(ready: boolean): void {
  const done = useRef(false);
  useEffect(() => {
    if (!ready || done.current || typeof window === "undefined") return;
    const id = window.location.hash.slice(1);
    if (!id) return;
    const el = document.getElementById(id);
    if (el) {
      done.current = true;
      el.scrollIntoView({ block: "start" });
    }
  });
}

/** Scroll to the section a `section=` parameter names, once it has rendered. */
export function useSectionScroll(section: string, ready: boolean): void {
  const done = useRef("");
  useEffect(() => {
    if (!ready || !section || done.current === section) return;
    const el = document.getElementById(section);
    if (el) {
      done.current = section;
      el.scrollIntoView({ block: "start" });
    }
  });
}
