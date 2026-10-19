/** Number formatting, in one place so every page reads the same. Null prints as an en dash. */

const DASH = "–";

export function num(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Dollars with a sign, no cents: "-1,049". */
export function usd(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return `${x < 0 ? "-" : ""}$${Math.abs(x).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function pct(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return `${(x * 100).toFixed(digits)}%`;
}

/** A decimal vol as vol points: 0.183 -> "18.3". */
export function volPts(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return (x * 100).toFixed(digits);
}

export function sci(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e6)) return x.toExponential(2);
  return x.toPrecision(3);
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
