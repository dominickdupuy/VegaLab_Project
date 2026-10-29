/**
 * Getting a figure's numbers out: CSV and JSON of what the chart draws, PNG of the
 * chart with its caption burnt in, so an exported figure still says where it came from.
 */

export type Row = Record<string, unknown>;

function cell(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = typeof v === "object" ? JSON.stringify(v) : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(rows: Row[]): string {
  const cols: string[] = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  return [cols.join(","), ...rows.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n") + "\n";
}

export function download(name: string, body: BlobPart, type: string): void {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function slug(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "figure"
  );
}

type Datum = unknown;

function valueOf(d: Datum): unknown {
  return d && typeof d === "object" && !Array.isArray(d) && "value" in (d as Record<string, unknown>) ? (d as { value: unknown }).value : d;
}

const AXES = ["x", "y", "z", "w"];

/**
 * A chart's data as rows, read off its option: one row per point, the series named,
 * a category axis resolved to its label. What a chart without explicit rows exports.
 */
export function rowsFromOption(option: Record<string, unknown>): Row[] {
  const series = ([] as unknown[]).concat(option.series ?? []) as Record<string, unknown>[];
  const xAxis = ([] as unknown[]).concat(option.xAxis ?? []) as Record<string, unknown>[];
  const rows: Row[] = [];
  series.forEach((s, si) => {
    const data = Array.isArray(s.data) ? (s.data as Datum[]) : [];
    const name = typeof s.name === "string" && s.name ? s.name : `series ${si + 1}`;
    const axis = xAxis[Number(s.xAxisIndex ?? 0)];
    const cats = axis && axis.type === "category" && Array.isArray(axis.data) ? (axis.data as unknown[]) : null;
    data.forEach((d, i) => {
      const v = valueOf(d);
      if (v === null || v === undefined) return;
      const row: Row = { series: name };
      if (Array.isArray(v)) v.forEach((x, j) => (row[AXES[j] ?? `v${j}`] = x));
      else {
        row.x = cats ? cats[i] : i;
        row.y = v;
      }
      rows.push(row);
    });
  });
  return rows;
}

/** A PNG of the chart canvas with the caption written under it. */
export async function pngWithCaption(dataUrl: string, caption: string, background: string, color: string, font: string): Promise<Blob | null> {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const pad = caption ? 30 : 0;
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height + pad;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  if (caption) {
    ctx.fillStyle = color;
    ctx.font = `20px ${font}`;
    ctx.fillText(caption, 16, img.height + 21, canvas.width - 32);
  }
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b), "image/png"));
}
