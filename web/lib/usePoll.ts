"use client";

import { useEffect, useRef } from "react";
import type { Loadable } from "./useFetch";

/**
 * Ask again every `ms` while `active`, but never while the last ask is still on its way.
 *
 * useFetch's reload aborts the request in flight and sends a new one. On a timer that
 * is a trap: a server slower than the beat is cut off and asked again on every beat,
 * so it never answers. A beat that finds the last request still out waits for the next.
 */
export function usePoll(loads: Loadable<unknown>[], active: boolean, ms = 1000): void {
  const ref = useRef(loads);
  ref.current = loads;
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => {
      for (const l of ref.current) if (!l.loading) l.reload();
    }, ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
}
