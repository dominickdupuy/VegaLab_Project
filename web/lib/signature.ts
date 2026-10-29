/**
 * What a chart option draws, as a string: equal for two options that would draw the same
 * picture, so a chart redraws only when that changes. Functions (formatters, a custom
 * series' renderItem) count by their source; whatever they read that changes the picture
 * changes the data beside them too. Remembered per option object, so a parent that keeps
 * its option in a memo pays for the string once.
 */
const seen = new WeakMap<object, string>();

function replacer(_key: string, value: unknown): unknown {
  return typeof value === "function" ? `ƒ${String(value)}` : value;
}

export function optionSignature(option: object): string {
  const hit = seen.get(option);
  if (hit !== undefined) return hit;
  const sig = JSON.stringify(option, replacer) ?? "";
  seen.set(option, sig);
  return sig;
}
