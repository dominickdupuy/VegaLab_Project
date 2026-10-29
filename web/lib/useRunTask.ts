"use client";

import { useEffect, useState } from "react";
import { ApiError } from "./api";
import { type Loadable, useFetch } from "./useFetch";
import { usePoll } from "./usePoll";

/**
 * A run's background view (capital, attribution, the action mix): load it, start it once
 * when it is missing and can be computed, and poll while it computes. Nothing to click.
 * `canStart` false holds the start (not the load) back: the views below the fold wait for
 * the one above it, since the server computes one at a time.
 */
export function useRunTask<T>(
  key: string,
  load: (signal: AbortSignal) => Promise<T>,
  start: () => Promise<unknown>,
  needsStart: (t: T) => boolean,
  busy: (t: T) => boolean,
  enabled = true,
  canStart = true,
): Loadable<T> & { startError: ApiError | undefined } {
  const res = useFetch(load, key, enabled);
  const [startedFor, setStartedFor] = useState("");
  const [startError, setStartError] = useState<ApiError | undefined>();
  const data = res.data;
  const { reload } = res;
  useEffect(() => {
    if (!data || !canStart || startedFor === key || !needsStart(data)) return;
    setStartedFor(key);
    setStartError(undefined);
    start().then(
      () => reload(),
      (e: unknown) => {
        setStartError(e instanceof ApiError ? e : new ApiError(0, String(e)));
        reload();
      },
    );
    // needsStart and start are read when the data arrives, not tracked
  }, [data, key, startedFor, reload, canStart]);
  usePoll([res], Boolean(data && busy(data)) || (startedFor === key && !data), 1000);
  return { ...res, startError };
}
