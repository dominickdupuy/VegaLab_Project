"use client";

import { useCallback, useEffect, useState } from "react";

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

/** The current theme, kept in step with the attribute however it changes. */
export function useTheme(): [Theme, (next: Theme) => void] {
  const [theme, setState] = useState<Theme>("light");
  useEffect(() => {
    setState(readTheme());
    const watcher = new MutationObserver(() => setState(readTheme()));
    watcher.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => watcher.disconnect();
  }, []);
  const set = useCallback((next: Theme) => setTheme(next), []);
  return [theme, set];
}
