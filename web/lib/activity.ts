"use client";

import { useSyncExternalStore } from "react";

/**
 * What the reader has done on the site this visit, as the workflows see it: every
 * answer the API gave (lib/api.ts records each one) and the few actions that never
 * reach the server (an export). A workflow step checks its action against this log, so
 * it works on any page that uses the shared client, however that page is laid out, and
 * on the server's real answers rather than on what a page chose to report.
 *
 * Only this tab's memory holds it, newest last, capped. Nothing is sent anywhere.
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
let log: Activity[] = [];
let version = 0;
const listeners = new Set<() => void>();

function emit(): void {
  version += 1;
  for (const l of listeners) l();
}

/** Add an action to the log. */
export function notify(a: { kind: string; at?: number; [extra: string]: unknown }): void {
  const entry: Activity = { ...a, kind: a.kind, at: a.at ?? Date.now() };
  log = [...log.slice(-(CAP - 1)), entry];
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
