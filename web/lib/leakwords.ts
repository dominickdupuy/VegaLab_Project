/**
 * The leak detector's numbers in plain words, for the Leak detector page and its workflow:
 * a planted share of tomorrow's move as "a tenth", a p-value as the chance it is luck.
 */

const SHARES: [number, string][] = [
  [1, "all"],
  [0.75, "three quarters"],
  [0.5, "half"],
  [0.25, "a quarter"],
  [0.2, "a fifth"],
  [0.1, "a tenth"],
  [0.05, "a twentieth"],
  [0.025, "a fortieth"],
];

/** A share of tomorrow's move in words: 0.1 is "a tenth"; one with no short name as a percent. */
export function shareWord(eps: number): string {
  const hit = SHARES.find(([v]) => Math.abs(v - eps) < 1e-9);
  return hit ? hit[1] : `${Number((eps * 100).toFixed(1))}%`;
}

/** The share and its number: "a tenth (0.1)". */
export function shareOf(eps: number): string {
  return `${shareWord(eps)} (${Number(eps.toFixed(3))})`;
}

/** A p-value as the chance the signal is luck: "under 1 in a million", "1 in 400", "3%". */
export function luck(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "not tested";
  if (p < 1e-6) return "under 1 in a million";
  if (p < 0.01) return `1 in ${Math.round(1 / p).toLocaleString("en-US")}`;
  return `${Number((p * 100).toFixed(p < 0.1 ? 1 : 0))}%`;
}

/** Several shares as one phrase: "all, half and a tenth". */
export function shareWords(eps: number[]): string {
  const w = eps.map(shareWord);
  return w.length < 2 ? (w[0] ?? "none") : `${w.slice(0, -1).join(", ")} and ${w[w.length - 1]}`;
}

/** What each field the trader sees is, in a few words (optspread.env's observation). */
const FIELDS: Record<string, string> = {
  time_frac: "how far into the episode",
  has_position: "whether a position is open",
  action_frac: "the structure held",
  days_held_frac: "days the position has been held",
  dte_frac: "days to expiry",
  equity_frac: "account equity",
  unrealized_frac: "open profit or loss",
  margin_frac: "margin in use",
  atm_iv: "at-the-money implied vol",
  realized_vol_20d: "20-day realised vol",
  ret_1d: "today's return",
  delta_frac: "the book's delta",
  vega_frac: "the book's vega",
  theta_frac: "the book's theta",
};

/** A field in words: "today's return"; a field with no gloss keeps its name. */
export function fieldWord(name: string): string {
  return FIELDS[name] ?? name;
}

/** A field in words with its name after it, for a chart or a tooltip: "today's return (ret_1d)". */
export function fieldOf(name: string): string {
  return FIELDS[name] ? `${FIELDS[name]} (${name})` : name;
}
