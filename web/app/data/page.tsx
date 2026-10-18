"use client";

import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, Chip, PageHead, Segmented, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Empty, ErrorBox, Loading, PanelNotBuilt } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type ApiError, type Coverage, type Daily, type Seam, type Smile, type Surface, type Term } from "@/lib/api";
import { SOURCE_LABEL, axis, base, sourceColor, type Tokens, useTokens } from "@/lib/chart";
import { num, pct, volPts } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const ROOTS = [
  { value: "all", label: "SPX + SPXW" },
  { value: "SPX", label: "SPX" },
  { value: "SPXW", label: "SPXW" },
] as const;
type Roots = (typeof ROOTS)[number]["value"];

const vp = (x: number | null) => (x === null ? null : x * 100);

// ---------------------------------------------------------------- coverage

function coverageOption(t: Tokens, c: Coverage): EChartsOption {
  const overlap = c.overlap;
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => num(v as number) },
    xAxis: { type: "time", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Contracts quoted", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70, bottom: 64 },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, height: 18, bottom: 10, borderColor: t.axis, textStyle: { color: t.muted } },
    ],
    series: [
      ...c.strips.map((s, i) => ({
        name: SOURCE_LABEL[s.source] ?? s.source,
        type: "line" as const,
        showSymbol: false,
        sampling: "lttb" as const,
        lineStyle: { width: 1.5, color: sourceColor(t, s.source) },
        itemStyle: { color: sourceColor(t, s.source) },
        data: s.date.map((d, j) => [d, s.missing[j] ? null : s.n_contracts[j]]),
        markArea:
          i === 0 && overlap
            ? {
                silent: true,
                itemStyle: { color: t.dark ? "rgba(141,180,234,0.12)" : "rgba(42,92,154,0.08)" },
                label: { color: t.text2, fontSize: 11, formatter: "overlap: the seam" },
                data: [[{ xAxis: overlap[0] }, { xAxis: overlap[1] }]] as never,
              }
            : undefined,
      })),
      {
        name: "Missing day",
        type: "scatter" as const,
        symbol: "triangle",
        symbolSize: 9,
        itemStyle: { color: t.fail },
        data: c.missing.map(([d]) => [d, 0]),
        tooltip: { trigger: "item" as const, formatter: (p: unknown) => `missing: ${String((p as { value: [string] }).value[0])}` },
      },
    ],
  };
}

function CoverageSection({ c }: { c: Coverage }) {
  const t = useTokens();
  return (
    <>
      <div className={ui.grid4}>
        {c.spans.map((s) => (
          <Tile
            key={s.source}
            accent
            label={SOURCE_LABEL[s.source] ?? s.source}
            value={`${num(s.n_days)} days`}
            note={
              <>
                {s.first} → {s.last} · median {num(s.median_contracts)} contracts/day
              </>
            }
          />
        ))}
        <Tile label="Overlap" value={c.overlap ? `${c.overlap[2]} days` : "none"} note={c.overlap ? `${c.overlap[0]} → ${c.overlap[1]}, both sources` : "no shared trading days"} />
        <Tile label="Missing days" value={num(c.missing.length)} note="listed, never interpolated" />
      </div>
      <div className={ui.grid2}>
        <Card title="Coverage by source" sub="Contracts quoted per trading day. The shaded band is the Jul–Aug 2025 overlap where the seam is validated; triangles mark missing days." source={<code>{c.source}</code>}>
          <Chart option={coverageOption(t, c)} height={300} label="Contracts quoted per trading day by source" />
        </Card>
        <Card title="Missing trading days" sub="Days the build could not fill. Listed as they are: nothing is smoothed over." source={<code>{c.source}</code>}>
          {c.missing.length ? (
            <div className={ui.tableWrap} style={{ maxHeight: 300, overflow: "auto" }}>
              <table className={ui.table}>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {c.missing.map(([d, s]) => (
                    <tr key={`${d}-${s}`}>
                      <td className="mono">{d}</td>
                      <td>
                        <Chip color={sourceColor(t, s)}>{SOURCE_LABEL[s] ?? s}</Chip>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty title="No missing trading days" />
          )}
        </Card>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- one date

function smileOption(t: Tokens, s: Smile): EChartsOption {
  const ours = s.moneyness.map((m, i) => [m, vp(s.iv[i] ?? null), s.strike[i]]);
  const vendor = s.moneyness.map((m, i) => [m, vp(s.iv_vendor[i] ?? null), s.strike[i]]);
  return {
    ...base(t),
    tooltip: {
      ...base(t).tooltip,
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: [number, number | null, number] }[];
        const head = arr[0] ? `K/F ${arr[0].value[0].toFixed(3)} · strike ${num(arr[0].value[2])}` : "";
        return [head, ...arr.map((p) => `${p.seriesName}: <b>${p.value[1] === null ? "–" : p.value[1].toFixed(2)}</b>`)].join("<br/>");
      },
    },
    xAxis: { type: "value", name: "Moneyness K / F", nameLocation: "middle", scale: true, ...axis(t) },
    yAxis: { type: "value", name: "Implied vol (vol pts)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 40 },
    series: [
      {
        name: "Ours (optspread.pricing)",
        type: "line",
        data: ours,
        symbolSize: 5,
        lineStyle: { width: 2, color: t.s1 },
        itemStyle: { color: t.s1 },
        markLine: { symbol: "none", silent: true, lineStyle: { color: t.s5, width: 1, type: "solid" }, label: { color: t.muted, formatter: "F" }, data: [{ xAxis: 1 }] },
      },
      ...(s.has_vendor
        ? [{ name: "OptionMetrics vendor IV", type: "line" as const, data: vendor, showSymbol: false, lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2 } }]
        : []),
    ],
  };
}

function surfaceOption(t: Tokens, s: Surface): EChartsOption {
  const data: [number, number, number | null][] = [];
  s.iv.forEach((row, j) => row.forEach((v, i) => data.push([i, j, v === null ? null : v * 100])));
  const vals = data.map((d) => d[2]).filter((v): v is number => v !== null);
  const lo = vals.length ? Math.min(...vals) : 0;
  const hi = vals.length ? Math.max(...vals) : 1;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: {
      ...base(t).tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const v = (p as { value: [number, number, number | null] }).value;
        return `K/F ${(s.moneyness[v[0]] ?? 0).toFixed(3)} · ${s.dte[v[1]]} DTE<br/>IV <b>${v[2] === null ? "not quoted" : v[2].toFixed(2)}</b>`;
      },
    },
    grid: { left: 64, right: 80, top: 16, bottom: 44 },
    xAxis: { type: "category", data: s.moneyness.map((m) => m.toFixed(2)), name: "Moneyness K / F", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "category", data: s.dte.map(String), name: "Days to expiry", nameLocation: "middle", ...axis(t), nameGap: 44, splitLine: { show: false } },
    visualMap: {
      min: Math.floor(lo),
      max: Math.ceil(hi),
      calculable: true,
      orient: "vertical",
      right: 0,
      top: "middle",
      itemHeight: 160,
      text: ["vol pts", ""],
      textStyle: { color: t.muted },
      inRange: { color: t.dark ? ["#1a2e4d", "#2f5a91", "#6b9ad8", "#cde2fb"] : ["#e6eef8", "#86b6ef", "#2a78d6", "#13233d"] },
    },
    series: [{ type: "heatmap", data: data.filter((d) => d[2] !== null), progressive: 0, emphasis: { itemStyle: { borderColor: t.text, borderWidth: 1 } } }],
  };
}

function termOption(t: Tokens, ts: Term): EChartsOption {
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, formatter: (p: unknown) => { const q = (p as { value: [number, number]; dataIndex: number }[])[0]; return q ? `${q.value[0]} DTE · ${ts.expiry[q.dataIndex]}<br/>ATM IV <b>${q.value[1].toFixed(2)}</b>` : ""; } },
    xAxis: { type: "value", name: "Days to expiry", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "ATM implied vol (vol pts)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 40 },
    series: [{ type: "line", data: ts.dte.map((d, i) => [d, vp(ts.atm_iv[i] ?? null)]), symbolSize: 7, lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 } }],
  };
}

function DateSection({ dates, root }: { dates: string[]; root: string }) {
  const t = useTokens();
  const [date, setDate] = useState(dates[dates.length - 1] ?? "");
  const [typed, setTyped] = useState(date);
  const [source, setSource] = useState<string>("");
  const [roots, setRoots] = useState<Roots>("all");
  const [expiry, setExpiry] = useState<string>("");
  const [maxDte, setMaxDte] = useState(365);
  const q = { date, source: source || undefined, roots: roots === "all" ? undefined : [roots] };
  const qk = JSON.stringify(q);
  const chain = useFetch((s) => api.chain(q, s), qk, Boolean(date));
  useEffect(() => setExpiry(""), [qk]);
  const exp = expiry || chain.data?.default_expiry || "";
  const smile = useFetch((s) => api.smile({ ...q, expiry: exp }, s), `${qk}|${exp}`, Boolean(date && exp));
  const surface = useFetch((s) => api.surface({ ...q, max_dte: maxDte }, s), `${qk}|${maxDte}`, Boolean(date));
  const term = useFetch((s) => api.term(q, s), qk, Boolean(date));

  const snap = (value: string) => {
    if (!dates.length) return;
    // the nearest trading day on or before the one typed, so a weekend picks the Friday
    let pick = dates[0] ?? "";
    for (const d of dates) if (d <= value) pick = d;
    setDate(pick);
    setTyped(pick);
  };
  const step = (k: -1 | 1) => {
    const i = dates.indexOf(date);
    const next = dates[Math.max(0, Math.min(dates.length - 1, i + k))];
    if (next) {
      setDate(next);
      setTyped(next);
    }
  };

  const err = (e: ApiError | undefined, what: string) => (e ? <ErrorBox what={what} error={e} /> : null);
  return (
    <>
      <Card title="One trading day" sub="Smile, surface and term structure for a date. Only that year's partition is read, and only that date's rows.">
        <div className={ui.filterBar}>
          <label className={ui.field}>
            <span className={ui.fieldLabel}>Date</span>
            <div className={ui.row} style={{ flexWrap: "nowrap" }}>
              <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => step(-1)} aria-label="Previous trading day">
                ←
              </button>
              <input className={ui.input} type="date" value={typed} min={dates[0]} max={dates[dates.length - 1]} onChange={(e) => setTyped(e.target.value)} onBlur={() => snap(typed)} onKeyDown={(e) => e.key === "Enter" && snap(typed)} />
              <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => step(1)} aria-label="Next trading day">
                →
              </button>
            </div>
          </label>
          <div className={ui.field}>
            <span className={ui.fieldLabel}>Source</span>
            <Segmented
              label="Source"
              value={source || (chain.data?.sources.includes("optionmetrics") ? "optionmetrics" : chain.data?.sources[0] ?? "")}
              options={(chain.data?.sources ?? []).map((s) => ({ value: s, label: SOURCE_LABEL[s] ?? s }))}
              onChange={setSource}
            />
          </div>
          <div className={ui.field}>
            <span className={ui.fieldLabel}>Roots</span>
            <Segmented label="Roots" value={roots} options={ROOTS} onChange={setRoots} />
          </div>
          <label className={ui.field}>
            <span className={ui.fieldLabel}>Expiry for the smile</span>
            <select className={ui.select} value={exp} onChange={(e) => setExpiry(e.target.value)}>
              {(chain.data?.expiries ?? []).map((e) => (
                <option key={e.expiry} value={e.expiry}>
                  {e.expiry} · {e.dte} DTE · {e.n_quotes} quotes
                </option>
              ))}
            </select>
          </label>
          <label className={ui.field}>
            <span className={ui.fieldLabel}>
              Surface out to <span>{maxDte} DTE</span>
            </span>
            <input className={ui.range} type="range" min={30} max={730} step={5} value={maxDte} onChange={(e) => setMaxDte(Number(e.target.value))} />
          </label>
        </div>
      </Card>
      {err(chain.error, "the chain")}
      <div className={ui.grid2}>
        <Card
          title={smile.data ? `Smile · ${smile.data.expiry} (${smile.data.dte} DTE)` : "Smile"}
          sub={smile.data ? `OTM quotes, ${smile.data.data_source}; forward ${num(smile.data.forward, 1)} from put-call parity` : "Out-of-the-money quotes against K / F"}
          source={smile.data ? <code>{smile.data.source}</code> : undefined}
        >
          {err(smile.error, "the smile")}
          {smile.loading && !smile.data ? <Loading what="the smile" /> : null}
          {smile.data ? <Chart option={smileOption(t, smile.data)} height={320} label="Implied volatility smile against moneyness" /> : null}
        </Card>
        <Card title="ATM term structure" sub={term.data ? `ATM IV per expiry, interpolated at K / F = 1; ${term.data.data_source}` : "ATM IV per expiry"} source={term.data ? <code>{term.data.source}</code> : undefined}>
          {err(term.error, "the term structure")}
          {term.loading && !term.data ? <Loading what="the term structure" /> : null}
          {term.data ? term.data.dte.length ? <Chart option={termOption(t, term.data)} height={320} label="ATM implied volatility against days to expiry" /> : <Empty title="No expiry straddles the forward" /> : null}
        </Card>
      </div>
      <Card
        title="IV surface"
        sub="Moneyness × days to expiry. Interpolated along strike within each expiry only; blank where nothing was quoted, never extrapolated."
        source={surface.data ? <code>{surface.data.source}</code> : <code>{root}</code>}
      >
        {err(surface.error, "the surface")}
        {surface.loading && !surface.data ? <Loading what="the surface" /> : null}
        {surface.data ? surface.data.dte.length ? <Chart option={surfaceOption(t, surface.data)} height={Math.max(300, 22 * surface.data.dte.length + 70)} label="Implied volatility surface heatmap" /> : <Empty title="No expiry in range has enough quotes" /> : null}
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- IV30, RV and VRP

interface Stitched {
  date: string[];
  iv30: (number | null)[];
  rv21: (number | null)[];
  rvFwd: (number | null)[];
  vrp: (number | null)[];
  source: string[];
}

/** One series 2010 to now: OptionMetrics where it quotes, Databento after it ends. */
function stitch(d: Daily): Stitched {
  const by = new Map<string, { iv: number | null; rv: number | null; fwd: number | null; vrp: number | null; src: string }>();
  const order = [...d.series].sort((a, b) => (a.data_source === "optionmetrics" ? 1 : 0) - (b.data_source === "optionmetrics" ? 1 : 0));
  for (const s of order) s.date.forEach((dt, i) => by.set(dt, { iv: s.iv30[i] ?? null, rv: s.rv21[i] ?? null, fwd: s.rv21_fwd[i] ?? null, vrp: s.vrp30[i] ?? null, src: s.data_source }));
  const dates = [...by.keys()].sort();
  const pick = <K extends "iv" | "rv" | "fwd" | "vrp">(k: K) => dates.map((dt) => vp(by.get(dt)?.[k] ?? null));
  return { date: dates, iv30: pick("iv"), rv21: pick("rv"), rvFwd: pick("fwd"), vrp: pick("vrp"), source: dates.map((dt) => by.get(dt)?.src ?? "") };
}

function vrpOption(t: Tokens, s: Stitched, cut: string | null): EChartsOption {
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["IV30 ATM", "RV21, forward", "RV21, trailing"] },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "–") },
    grid: [
      { left: 60, right: 18, top: 34, height: "50%" },
      { left: 60, right: 18, top: "68%", bottom: 64 },
    ],
    xAxis: [
      { type: "category", data: s.date, gridIndex: 0, ...axis(t), axisLabel: { show: false }, splitLine: { show: false } },
      { type: "category", data: s.date, gridIndex: 1, ...axis(t), splitLine: { show: false } },
    ],
    yAxis: [
      { type: "value", gridIndex: 0, name: "Vol pts", nameLocation: "middle", ...axis(t), nameGap: 38 },
      { type: "value", gridIndex: 1, name: "VRP30", nameLocation: "middle", ...axis(t), nameGap: 38 },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1] },
      { type: "slider", xAxisIndex: [0, 1], height: 18, bottom: 10, borderColor: t.axis, textStyle: { color: t.muted } },
    ],
    series: [
      {
        name: "IV30 ATM",
        type: "line",
        data: s.iv30,
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 1.5, color: t.s1 },
        itemStyle: { color: t.s1 },
        markLine: cut
          ? { symbol: "none", silent: true, lineStyle: { color: t.s5, width: 1, type: "solid" }, label: { color: t.muted, formatter: "source cut-over" }, data: [{ xAxis: cut }] }
          : undefined,
      },
      { name: "RV21, forward", type: "line", data: s.rvFwd, showSymbol: false, sampling: "lttb", lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2 } },
      { name: "RV21, trailing", type: "line", data: s.rv21, showSymbol: false, sampling: "lttb", lineStyle: { width: 1, color: t.s5, opacity: 0.8 }, itemStyle: { color: t.s5 } },
      {
        name: "VRP30 = IV30 − forward RV21",
        type: "bar",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: s.vrp.map((v) => (v === null ? null : { value: v, itemStyle: { color: v >= 0 ? t.s1 : t.s2 } })),
        barCategoryGap: 0,
        large: true,
      },
    ],
  };
}

function VrpSection({ d }: { d: Daily }) {
  const t = useTokens();
  const s = useMemo(() => stitch(d), [d]);
  const cut = useMemo(() => {
    const i = s.source.findIndex((x, j) => j > 0 && x !== s.source[j - 1]);
    return i > 0 ? (s.date[i] ?? null) : null;
  }, [s]);
  return (
    <>
      <div className={ui.grid4}>
        {Object.entries(d.stats).map(([src, st]) => (
          <Tile
            key={src}
            accent
            label={`Mean VRP30 · ${SOURCE_LABEL[src] ?? src}`}
            value={`${num(st.mean_vol_pts, 2)} vol pts`}
            note={
              <>
                median {num(st.median_vol_pts, 2)} · positive {pct(st.share_positive)} of {num(st.n_days)} days · {st.first} → {st.last}
              </>
            }
          />
        ))}
      </div>
      <Card
        title="IV30 against realised vol, and the variance risk premium"
        sub="Top: 30-day ATM implied vol against trailing and forward 21-day realised vol. Bottom: VRP30, implied minus the realised vol that followed. Drag the brush to zoom."
        source={<code>{d.source}</code>}
      >
        {s.date.length ? <Chart option={vrpOption(t, s, cut)} height={460} label="IV30 and realised volatility over time, with the variance risk premium below" /> : <Empty title="No daily rows" />}
        <p className={ui.callout} style={{ marginTop: 12 }}>
          This is the external anchor of spec 4.5 and 5.5: the premium the market actually paid, in vol points, beside the premium the sweep plants (0, 1, 2 and 4 vol points).
          The mapping between the two is an argument with stated assumptions, made in M4, not this subtraction.
        </p>
      </Card>
    </>
  );
}

// ---------------------------------------------------------------- the seam

const SEAM_LABEL: Record<string, string> = {
  n_matched: "Contracts matched",
  iv_diff_median_volpts: "IV diff, median (vol pts)",
  iv_diff_p95_abs_volpts: "|IV diff|, p95 (vol pts)",
  mid_diff_median_rel: "Mid diff, median (relative)",
  forward_diff_median_bp: "Forward diff, median (bp)",
  ours_vs_vendor_median_volpts: "Ours − vendor IV, median (vol pts)",
};

function seamOption(t: Tokens, s: Seam): EChartsOption {
  const series = ["iv_diff_median_volpts", "iv_diff_p95_abs_volpts", "ours_vs_vendor_median_volpts"].filter((k) => s.stats[k]);
  const colors = [t.s1, t.s2, t.s3];
  return {
    ...base(t),
    xAxis: { type: "category", data: s.date, ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Vol pts", nameLocation: "middle", ...axis(t), nameGap: 36 },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "–") },
    series: series.map((k, i) => ({ name: SEAM_LABEL[k] ?? k, type: "line" as const, data: s.stats[k] ?? [], symbolSize: 5, lineStyle: { width: 1.8, color: colors[i] }, itemStyle: { color: colors[i] } })),
  };
}

function SeamSection({ s }: { s: Seam }) {
  const t = useTokens();
  return (
    <Card
      title="The seam: OptionMetrics against Databento"
      sub="Same contracts, same days, both sources, over the overlap. The cut-over is only as good as this agreement."
      source={
        <>
          <code>{s.source}</code>
          {s.iv30_date.length ? " · IV30 difference from spx_daily.parquet" : ""}
        </>
      }
    >
      {s.available ? (
        <div className={ui.stack}>
          <div className={ui.grid3}>
            {Object.entries(s.medians).map(([k, v]) => (
              <Tile key={k} label={SEAM_LABEL[k] ?? k} value={k === "n_matched" ? num(v) : k === "mid_diff_median_rel" ? pct(v, 2) : num(v, 3)} note={`median over ${s.date.length} days`} />
            ))}
          </div>
          <Chart option={seamOption(t, s)} height={260} label="Per-day seam statistics over the overlap" />
          {s.iv30_date.length ? (
            <p className={ui.dim} style={{ fontSize: 13 }}>
              IV30 {SOURCE_LABEL[s.iv30_sources[1] ?? ""] ?? s.iv30_sources[1]} minus {SOURCE_LABEL[s.iv30_sources[0] ?? ""] ?? s.iv30_sources[0]} on {s.iv30_date.length} shared days:
              median {num(median(s.iv30_diff_vol_pts), 3)} vol pts.
            </p>
          ) : null}
        </div>
      ) : (
        <Empty title="Seam report not written yet">
          <p>
            The build writes <code>panel/seam_overlap.parquet</code> once both sources cover the overlap.
          </p>
        </Empty>
      )}
    </Card>
  );
}

function median(xs: (number | null)[]): number | null {
  const v = xs.filter((x): x is number => x !== null).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? (v[m] ?? null) : ((v[m - 1] ?? 0) + (v[m] ?? 0)) / 2;
}

// ---------------------------------------------------------------- the page

export default function DataPage() {
  const status = useFetch((s) => api.dataStatus(s), "status");
  const built = Boolean(status.data?.built);
  const coverage = useFetch((s) => api.coverage(s), "coverage", built);
  const dates = useFetch((s) => api.dates(undefined, s), "dates", built);
  const daily = useFetch((s) => api.daily(undefined, undefined, s), "daily", built);
  const seam = useFetch((s) => api.seam(s), "seam", built);

  return (
    <>
      <PageHead
        eyebrow="Real SPX data"
        title="The stitched SPX panel"
        lede="WRDS OptionMetrics from 2010 to its August 2025 ceiling, Databento OPRA end-of-day quotes from July 2025 on, on one schema: forwards from put-call parity, IV and greeks recomputed with optspread.pricing on both halves, the vendor IV kept beside ours."
      />
      {status.error ? <ErrorBox what="the panel status" error={status.error} onRetry={status.reload} /> : null}
      {status.loading ? <Loading what="the panel status" /> : null}
      {status.data && !built ? <PanelNotBuilt missing={status.data.missing} root={status.data.root} /> : null}
      {built ? (
        <>
          {coverage.error ? <ErrorBox what="coverage" error={coverage.error} onRetry={coverage.reload} /> : null}
          {coverage.data ? <CoverageSection c={coverage.data} /> : <Loading what="coverage" />}
          {dates.data && dates.data.length ? <DateSection dates={dates.data} root={status.data?.root ?? ""} /> : null}
          {daily.error ? <ErrorBox what="the daily series" error={daily.error} onRetry={daily.reload} /> : null}
          {daily.data ? <VrpSection d={daily.data} /> : <Loading what="the daily series" />}
          {seam.data ? <SeamSection s={seam.data} /> : null}
        </>
      ) : null}
    </>
  );
}
