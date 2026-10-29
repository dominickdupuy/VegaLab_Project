/** Number formatting, in one place so every page reads the same. Null prints as an en dash. */

const DASH = "–";
/** The typographic minus, U+2212: every negative number on a page prints with it. */
export const MINUS = "−";

const minus = (text: string) => (text.startsWith("-") ? `${MINUS}${text.slice(1)}` : text);

export function num(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return minus(x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }));
}

/** A planted premium as recorded, up to two decimals: 0.25 -> "0.25", 1 -> "1". */
export function prem(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return minus(x.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 2 }));
}

/** A signed number: "+0.64", "−0.34". */
export function signed(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return `${x >= 0 ? "+" : MINUS}${Math.abs(x).toFixed(digits)}`;
}

/** Dollars with a sign, no cents: "−$1,049". */
export function usd(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return `${x < 0 && Math.abs(x) >= 0.5 ? MINUS : ""}$${Math.abs(x).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return minus(`${(x * 100).toFixed(digits)}%`);
}

/** A decimal vol as vol points: 0.183 -> "18.3". */
export function volPts(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return minus((x * 100).toFixed(digits));
}

export function sci(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  if (x === 0) return "0";
  if (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e6) return minus(x.toExponential(2));
  return minus(x.toPrecision(3));
}

/** A p-value: scientific below 0.001, and one that underflowed to 0 as a bound. */
export function pval(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  if (x === 0) return "< 1e-300";
  return sci(x);
}

export function hours(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  if (x < 1 / 60) return `${Math.round(x * 3600)} s`;
  if (x < 1) return `${Math.round(x * 60)} min`;
  if (x < 48) return `${x.toFixed(1)} h`;
  return `${(x / 24).toFixed(1)} d`;
}

/** Text for an ECharts tooltip, which renders HTML: anything read from a run is escaped. */
export function esc(text: string | number | null | undefined): string {
  return String(text ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

export function short(hash: string, n = 8): string {
  return hash.length > n ? hash.slice(0, n) : hash;
}

export function when(iso: string): string {
  if (!iso) return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
