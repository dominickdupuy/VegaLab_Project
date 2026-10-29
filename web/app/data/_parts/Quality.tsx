"use client";

import { useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, Chip, Segmented, SelectField, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Coverage, type DataStatus, type Seam, type Smile } from "@/lib/api";
import { dataApi, type Quality } from "@/lib/api_data";
import { SOURCE_LABEL, axis, base, sourceColor, type Tokens, useTokens } from "@/lib/chart";
import { esc, num } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { useQuery, useQuerySync } from "@/lib/url";
import s from "./data.module.css";
import { Reading } from "@/components/PageBits";
import { Labelled } from "@/components/InfoTip";

const vp = (x: number | null | undefined) => (x === null || x === undefined ? null : x * 100);
const bp = (x: number | null) => (x === null ? "–" : x.toFixed(1));

const ROOTS = [
  { value: "all", label: "SPX + SPXW" },
  { value: "SPX", label: "SPX" },
  { value: "SPXW", label: "SPXW" },
] as const;
type Roots = (typeof ROOTS)[number]["value"];

function coverageOption(t: Tokens, c: Coverage): EChartsOption {
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => num(v as number) },
    xAxis: { type: "time", ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: undefined }, splitLine: { show: false } },
    yAxis: { type: "value", name: "Contracts quoted", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70, bottom: 64 },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, height: 18, bottom: 10, borderColor: t.axis, textStyle: { color: t.muted } },
    ],
    series: [
      ...c.strips.map((st) => ({
        name: SOURCE_LABEL[st.source] ?? st.source,
        type: "line" as const,
        showSymbol: false,
        sampling: "lttb" as const,
        lineStyle: { width: 1.5, color: sourceColor(t, st.source) },
        itemStyle: { color: sourceColor(t, st.source) },
        data: st.date.map((d, j) => [d, st.missing[j] ? null : st.n_contracts[j]]),
      })),
      ...(c.missing.length
        ? [
            {
              name: "Missing day",
              type: "scatter" as const,
              symbol: "triangle",
              symbolSize: 9,
              itemStyle: { color: t.fail },
              data: c.missing.map(([d]) => [d, 0]),
              tooltip: { trigger: "item" as const, formatter: (p: unknown) => `missing: ${esc((p as { value: [string] }).value[0])}` },
            },
          ]
        : []),
    ],
  };
}

const SEAM_LABEL: Record<string, string> = {
  iv_diff_median_volpts: "IV gap, median",
  iv_diff_p95_abs_volpts: "IV gap, 95th percentile",
  ours_vs_vendor_median_volpts: "Ours less vendor IV, median",
};

function seamOption(t: Tokens, sm: Seam): EChartsOption {
  const keys = Object.keys(SEAM_LABEL).filter((k) => sm.stats[k]);
  const colors = [t.s1, t.s2, t.s3];
  return {
    ...base(t),
    legend: { ...base(t).legend, data: keys.map((k) => SEAM_LABEL[k] ?? k) },
    xAxis: { type: "category", data: sm.date, ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Vol pts", nameLocation: "middle", ...axis(t), nameGap: 40 },
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "–") },
    series: keys.map((k, i) => ({ name: SEAM_LABEL[k] ?? k, type: "line" as const, data: sm.stats[k] ?? [], symbolSize: 5, lineStyle: { width: 1.8, color: colors[i] }, itemStyle: { color: colors[i] } })),
  };
}

function smileOption(t: Tokens, sm: Smile): EChartsOption {
  const ours = sm.moneyness.map((m, i) => [m, vp(sm.iv[i]), sm.strike[i]]);
  const vendor = sm.moneyness.map((m, i) => [m, vp(sm.iv_vendor[i]), sm.strike[i]]);
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["Ours", ...(sm.has_vendor ? ["OptionMetrics"] : [])] },
    tooltip: {
      ...base(t).tooltip,
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: [number, number | null, number] }[];
        const head = arr[0] ? `K/F ${arr[0].value[0].toFixed(3)} · strike ${num(arr[0].value[2])}` : "";
        return [head, ...arr.map((p) => `${esc(p.seriesName)}: <b>${p.value[1] === null ? "–" : p.value[1].toFixed(2)}</b>`)].join("<br/>");
      },
    },
    grid: { left: 52, right: 20, top: 40, bottom: 44 },
    xAxis: { type: "value", name: "K / F", nameLocation: "middle", scale: true, ...axis(t), nameGap: 28 },
    yAxis: { type: "value", name: "IV (vol pts)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 36 },
    series: [
      { name: "Ours", type: "line", data: ours, symbolSize: 5, lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1 } },
      ...(sm.has_vendor ? [{ name: "OptionMetrics", type: "line" as const, data: vendor, showSymbol: false, lineStyle: { width: 1.5, color: t.s2, type: "dashed" as const }, itemStyle: { color: t.s2 } }] : []),
    ],
  };
}

/** One expiry's quotes on the day as sent: our IV against the vendor's. Needs the chain partitions. */
function QuotedSmile({ date, status }: { date: string; status: DataStatus }) {
  const t = useTokens();
  const q = useQuery();
  const [source, setSource] = useState(() => q.str("vendor") || q.str("source"));
  const [roots, setRoots] = useState<Roots>(() => q.oneOf("roots", ["all", "SPX", "SPXW"] as const, "all"));
  const [expiry, setExpiry] = useState(() => q.str("expiry"));
  useQuerySync({ vendor: source, roots, expiry }, { roots: "all" });
  const year = Number(date.slice(0, 4));
  const has = status.years.includes(year);
  const cq = { date, source: source || undefined, roots: roots === "all" ? undefined : [roots] };
  const key = JSON.stringify(cq);
  const chain = useFetch((sig) => api.chain(cq, sig), key, has && Boolean(date));
  const [firstKey] = useState(key); // the expiry from the URL holds until the day or the roots change
  useEffect(() => {
    if (key !== firstKey) setExpiry("");
  }, [key, firstKey]);
  const exp = expiry || chain.data?.default_expiry || "";
  const smile = useFetch((sig) => api.smile({ ...cq, expiry: exp }, sig), `${key}|${exp}`, has && Boolean(exp));
  const sources = chain.data?.sources ?? [];
  if (!has) return <Empty title={`The ${year} chain partition is not in this data folder`} />;
  return (
    <div className={ui.stack}>
      <div className={ui.filterBar}>
        {sources.length > 1 ? (
          <Segmented label="Vendor" title="Vendor" term="data.vendor" value={source || (sources.includes("optionmetrics") ? "optionmetrics" : sources[0] ?? "")} options={sources.map((x) => ({ value: x, label: SOURCE_LABEL[x] ?? x }))} onChange={setSource} />
        ) : null}
        <Segmented label="Roots" title="Roots" term="data.roots" value={roots} options={ROOTS} onChange={setRoots} />
        <SelectField
          label="Expiry"
          term="data.expiry"
          value={exp}
          options={(chain.data?.expiries ?? []).map((e) => e.expiry)}
          render={(x) => {
            const e = chain.data?.expiries.find((y) => y.expiry === x);
            return e ? `${e.expiry} · ${e.dte} DTE · ${e.n_quotes} quotes` : x;
          }}
          onChange={(v) => v && setExpiry(v)}
        />
      </div>
      {chain.error ? <ErrorBox what="the chain" error={chain.error} onRetry={chain.reload} /> : null}
      {smile.error ? <ErrorBox what="the smile" error={smile.error} onRetry={smile.reload} /> : null}
      {(chain.loading || smile.loading) && !smile.data ? <Loading what="the quoted smile (one year of the chain)" /> : null}
      {smile.data ? <Chart option={smileOption(t, smile.data)} height={300} label="Implied volatility smile as quoted, ours against the vendor's" stamp={`${smile.data.date} · ${smile.data.expiry} · ${SOURCE_LABEL[smile.data.data_source] ?? smile.data.data_source}`} /> : null}
    </div>
  );
}

/** Coverage, the vendor seam, FRED and the build's version: small, at the bottom, folded. */
export function QualityBody({ date, status }: { date: string; status: DataStatus }) {
  const t = useTokens();
  const quality = useFetch((sig) => dataApi.quality(sig), "quality");
  const coverage = useFetch((sig) => api.coverage(sig), "coverage", status.files.some((f) => f.path.endsWith("coverage.parquet") && f.present));
  const seam = useFetch((sig) => api.seam(sig), "seam");
  const qd: Quality | undefined = quality.data;
  const c = coverage.data;
  const sm = seam.data;
  // the seam's headline, as docs/DATA.md states it: each day's 95th-percentile IV gap, averaged over the days
  const p95 = sm?.p95_mean;
  return (
    <div className={ui.stack}>
      <Reading loading={quality.loading}>{qd?.reading}</Reading>
      {quality.error ? <ErrorBox what="the data quality checks" error={quality.error} onRetry={quality.reload} /> : null}
      <div className={ui.grid4}>
        {(c?.spans ?? []).map((sp) => (
          <Tile key={sp.source} info="data.coverage" label={SOURCE_LABEL[sp.source] ?? sp.source} value={`${num(sp.n_days)} days`} note={`${sp.first} to ${sp.last} · median ${num(sp.median_contracts)} contracts a day`} />
        ))}
        <Tile info="data.seam" label="Vendor seam" value={p95 !== null && p95 !== undefined ? `${p95.toFixed(3)} vol pts` : "–"} note={sm?.available ? `95th-percentile IV gap on the average day · worst ${sm.p95_worst !== null ? sm.p95_worst.toFixed(3) : "–"} · ${sm.date.length} shared days` : "no seam report"} />
        <Tile
          info="data.panel_version"
          label="Panel version"
          value={qd?.panel_version ?? "–"}
          note={qd ? (qd.of_record === true ? "files match DATA_MANIFEST.json" : qd.of_record === false ? "files differ from DATA_MANIFEST.json" : "not checked against the manifest") : undefined}
        />
      </div>
      <div className={ui.grid2}>
        <Card title="Contracts quoted per day" info="data.coverage">
          {coverage.error ? <ErrorBox what="coverage" error={coverage.error} onRetry={coverage.reload} /> : null}
          {c ? <Chart option={coverageOption(t, c)} height={260} label="Contracts quoted per trading day by vendor" stamp="SPX panel coverage" /> : coverage.loading ? <Loading what="coverage" /> : null}
          {c ? (
            c.missing.length ? (
              <div className={ui.tableWrap} style={{ maxHeight: 200, overflow: "auto" }}>
                <table className={ui.table}>
                  <thead>
                    <tr>
                      <th>Missing day</th>
                      <th>Vendor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {c.missing.map(([d, src]) => (
                      <tr key={`${d}-${src}`}>
                        <td className="mono">{d}</td>
                        <td>
                          <Chip color={sourceColor(t, src)}>{SOURCE_LABEL[src] ?? src}</Chip>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className={ui.dim}>No trading day is missing.</p>
            )
          ) : null}
        </Card>
        <Card title="The vendor seam" info="data.seam">
          {sm?.available ? <Chart option={seamOption(t, sm)} height={260} label="Per-day IV differences between the two vendors over the days both quote" stamp="SPX panel seam report" /> : sm ? <Empty title="No seam report in this build" /> : <Loading what="the seam" />}
        </Card>
      </div>
      <div className={ui.grid2}>
        <Card title="Against FRED (basis points)" info="data.fred">
          {qd && !qd.fred_available ? <Empty title="No FRED check in this build" /> : null}
          {qd?.fred.length ? (
            <div className={ui.tableWrap} style={{ maxHeight: 280, overflow: "auto" }}>
              <table className={ui.table}>
                <thead>
                  <tr>
                    <th>Year</th>
                    <th>Vendor</th>
                    <th className={ui.num}>Rate less bills</th>
                    <th className={ui.num}>Rate less SOFR</th>
                    <th className={ui.num}>Spot less S&amp;P 500</th>
                  </tr>
                </thead>
                <tbody>
                  {qd.fred.map((r) => (
                    <tr key={`${r.year}-${r.source}`}>
                      <td>{r.year}</td>
                      <td>{SOURCE_LABEL[r.source] ?? r.source}</td>
                      <td className={ui.num}>{bp(r.rate_minus_bill_bp)}</td>
                      <td className={ui.num}>{bp(r.rate_minus_sofr_bp)}</td>
                      <td className={ui.num}>{bp(r.spot_vs_sp500_bp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          {qd?.checked.length ? (
            <p className={ui.dim} style={{ fontSize: 13, marginTop: 10 }}>
              {qd.checked.map((ck) => (
                <span key={ck.path} style={{ marginRight: 14 }}>
                  <span className={ck.match ? s.ok : ck.match === false ? s.bad : undefined}>{ck.match ? "✓" : ck.match === false ? "✕" : "?"}</span> {ck.path.replace("panel/", "")} {ck.actual ? <code>{ck.actual.slice(0, 12)}</code> : null}
                </span>
              ))}
            </p>
          ) : null}
        </Card>
        <Card title={`Quoted smile on ${date}`} info="data.quoted_smile">
          <QuotedSmile date={date} status={status} />
        </Card>
      </div>
    </div>
  );
}
