"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export interface Loadable<T> {
  data: T | undefined;
  error: ApiError | undefined;
  loading: boolean;
  /** Fetch again, keeping the current data on screen while it loads. */
  reload: () => void;
}

/**
 * What each loader last answered, by its source and key, for this tab: a view revisited,
 * or a choice toggled back, shows its answer at once and asks the server again behind it
 * (the GET layer keeps the same object when the answer is unchanged, so nothing redraws).
 */
const answered = new Map<string, unknown>();
const MAX_ANSWERS = 300;

function remember(id: string, data: unknown): void {
  answered.delete(id);
  answered.set(id, data);
  while (answered.size > MAX_ANSWERS) {
    const oldest = answered.keys().next().value;
    if (oldest === undefined) break;
    answered.delete(oldest);
  }
}

/**
 * Load something from the server when the component mounts and whenever `key`
 * changes. A slower earlier request can never overwrite a newer one. An empty key
 * with `enabled` false loads nothing, for a view still waiting on a choice. What this
 * loader answered for this key before is on screen at once while it is asked again.
 */
export function useFetch<T>(load: (signal: AbortSignal) => Promise<T>, key: string = "", enabled = true): Loadable<T> {
  // the loader's source tells two call sites with the same key apart
  const id = `${String(load)}\u0000${key}`;
  const recalled = enabled ? (answered.get(id) as T | undefined) : undefined;
  const [state, setState] = useState<{ data?: T; error?: ApiError; loading: boolean; key: string }>(() => ({ data: recalled, loading: enabled, key }));
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (!enabled) {
      setState({ loading: false, key });
      return;
    }
    const ctl = new AbortController();
    const before = answered.get(id) as T | undefined;
    setState((s) => (s.key === key ? { ...s, loading: true } : { data: before, loading: true, key }));
    loadRef
      .current(ctl.signal)
      .then((data) => {
        remember(id, data);
        setState((s) => (s.key === key && s.data === data && !s.error && !s.loading ? s : { data, loading: false, key }));
      })
      .catch((err: unknown) => {
        if (ctl.signal.aborted) return;
        const error = err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err));
        setState({ error, loading: false, key });
      });
    return () => ctl.abort();
  }, [key, tick, enabled]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const fresh = state.key === key;
  return {
    data: fresh ? state.data : recalled,
    error: fresh ? state.error : undefined,
    loading: enabled && (state.loading || !fresh),
    reload,
  };
}
