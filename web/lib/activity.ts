"use client";

import { useSyncExternalStore } from "react";

/**
 * What the reader has done on the site this visit, as the workflows see it: every
 * answer the API gave (lib/api.ts records each one) and the few actions that never
 * reach the server (an export). A workflow step checks its action against this log, so
 * it works on any page that uses the shared client, however that page is laid out, and
 * on the server's real answers rather than on what a page chose to report.
 *
 * This tab holds it, newest last, capped, and nothing is sent anywhere. The actions (a
 * POST: a launch, the leak detector, a replay) and the exports are kept in the tab's
 * sessionStorage too, so a reload mid-step still knows the run it launched; a GET
 * answer lives in memory only, since the reloaded page asks for it again.
 */

export interface Activity {
  /** "api" for a server answer; anything else is a UI action ("export"). */
  kind: string;
  at: number;
  method?: "GET" | "POST";
  /** The route without its query: "/api/runs/detail". */
  path?: string;
  /** The query, first value per key. */
  query?: Record<string, string>;
  body?: unknown;
  data?: unknown;
  [extra: string]: unknown;
}

const CAP = 200;
// answers no step reads, and big: logged without their payload
const LIGHT = ["/api/data/grid/frames", "/api/data/landscape", "/api/power/surface", "/api/data/daily", "/api/data/coverage"];
const KEY = "optspread.activity";
const KEEP = 40; // actions kept across a reload
const KEEP_CHARS = 400_000;

// POSTs that only ask, sent as the reader types: like a GET, no step reads them
const ASKS = ["/api/launch/preview"];
const kept = (a: Activity) => !(a.kind === "api" && (a.method === "GET" || ASKS.includes(a.path ?? "")));

function restore(): Activity[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(KEY);
    const xs: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(xs) ? (xs as Activity[]).filter((a) => a && typeof a.kind === "string" && typeof a.at === "number") : [];
  } catch {
    return [];
  }
}

function persist(): void {
  try {
    let xs = log.filter(kept).slice(-KEEP);
    let raw = JSON.stringify(xs);
    while (raw.length > KEEP_CHARS && xs.length) {
      xs = xs.slice(1);
      raw = JSON.stringify(xs);
    }
    window.sessionStorage.setItem(KEY, raw);
  } catch {
    /* nowhere to keep it: the actions last as long as the page */
  }
}

let log: Activity[] = restore();
let version = 0;
const listeners = new Set<() => void>();

function emit(): void {
  version += 1;
  for (const l of listeners) l();
}

/** Add an action to the log. */
export function notify(a: { kind: string; at?: number; [extra: string]: unknown }): void {
  const entry: Activity = { ...a, kind: a.kind, at: a.at ?? Date.now() };
  const next = [...log, entry];
  if (next.length > CAP) {
    // a page that polls must not push the reader's own actions out: the oldest answer goes first
    const i = next.findIndex((x) => !kept(x));
    next.splice(i >= 0 ? i : 0, 1);
  }
  log = next;
  if (kept(entry)) persist();
  emit();
}

/** Called by lib/api.ts for every answer the server gave. */
export function recordApi(method: "GET" | "POST", url: string, body: unknown, data: unknown): void {
  const [path = "", qs = ""] = url.split("?");
  if (path === "/api/health") return; // the header's heartbeat is not an action
  const query: Record<string, string> = {};
  for (const [k, v] of new URLSearchParams(qs)) if (!(k in query)) query[k] = v;
  notify({ kind: "api", method, path, query, body, data: LIGHT.includes(path) ? undefined : data });
}

export function activity(): readonly Activity[] {
  return log;
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** The log's version: a component that reads it re-renders on every new entry. */
export function useActivity(): number {
  return useSyncExternalStore(
    subscribe,
    () => version,
    () => 0,
  );
}

/** "GET /api/power" or "POST /api/leaks/demo": the key a step names a call by. */
export type Call = `${"GET" | "POST"} /api/${string}`;

/** Every answer to `call` since `since` (ms), oldest first; a trailing * matches a prefix. */
export function calls(call: Call, since = 0): Activity[] {
  const [method, route = ""] = call.split(" ");
  const prefix = route.endsWith("*") ? route.slice(0, -1) : null;
  return log.filter((a) => a.kind === "api" && a.at >= since && a.method === method && (prefix !== null ? (a.path ?? "").startsWith(prefix) : a.path === route));
}

/** The newest answer to `call` since `since`, if any. */
export function lastCall(call: Call, since = 0): Activity | undefined {
  const xs = calls(call, since);
  return xs[xs.length - 1];
}
