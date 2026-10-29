"use client";

import { useCallback, useSyncExternalStore } from "react";

/**
 * The theme: "light" (the default, the deck's own look) or "dark" (derived from its
 * navy). It lives on <html data-theme>, set before the first paint by the script in
 * app/layout.tsx and remembered under this key. Charts watch the attribute and
 * redraw from the tokens.
 */

export type Theme = "light" | "dark";
export const THEME_KEY = "optspread.theme";

export function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

export function setTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Nowhere to remember it; the choice still holds for this visit.
  }
}

/**
 * Flip the theme as a circle spreading from `from` (the toggle), where the browser has
 * the View Transitions API and motion is welcome; a plain switch otherwise.
 */
export function flipTheme(theme: Theme, from?: HTMLElement | null): void {
  const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
  const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (!doc.startViewTransition || calm) {
    setTheme(theme);
    return;
  }
  const r = from?.getBoundingClientRect();
  const x = r ? r.left + r.width / 2 : window.innerWidth;
  const y = r ? r.top + r.height / 2 : 0;
  const root = document.documentElement.style;
  root.setProperty("--flip-x", `${x}px`);
  root.setProperty("--flip-y", `${y}px`);
  root.setProperty("--flip-r", `${Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y))}px`);
  doc.startViewTransition(() => setTheme(theme));
}

// One watcher on <html data-theme> for the whole page, however many components ask.
const listeners = new Set<() => void>();
let watcher: MutationObserver | null = null;

function subscribe(on: () => void): () => void {
  listeners.add(on);
  if (!watcher && typeof MutationObserver !== "undefined") {
    watcher = new MutationObserver(() => listeners.forEach((l) => l()));
    watcher.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }
  return () => {
    listeners.delete(on);
    if (!listeners.size && watcher) {
      watcher.disconnect();
      watcher = null;
    }
  };
}

/** Subscribe to theme flips: for a store that derives from the theme (the chart tokens). */
export const subscribeTheme = subscribe;

/**
 * The current theme, kept in step with the attribute however it changes. Read from the
 * attribute on the first client render (light while the static page is prerendered), so
 * a component mounted after hydration never draws one frame in the other theme.
 */
export function useTheme(): [Theme, (next: Theme) => void] {
  const theme = useSyncExternalStore(subscribe, readTheme, () => "light" as Theme);
  const set = useCallback((next: Theme) => setTheme(next), []);
  return [theme, set];
}
