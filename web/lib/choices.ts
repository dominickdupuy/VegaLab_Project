/**
 * The one rule for filters and selectors: a control is shown only when it can change
 * something. A filter over one commit, one data source or one arm is hidden, not drawn
 * disabled. Pages use `hasChoice` for controls they draw themselves; Segmented and
 * SelectField (components/Bits.tsx) apply it on their own.
 */
export function hasChoice(options: readonly unknown[] | undefined | null): boolean {
  return new Set(options ?? []).size > 1;
}

/** The options worth offering: distinct, in order. */
export function distinct<T>(options: readonly T[]): T[] {
  return [...new Set(options)];
}
