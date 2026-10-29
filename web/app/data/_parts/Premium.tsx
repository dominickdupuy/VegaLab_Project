"use client";

import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, Segmented, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { axisOption, SensitivityTable, signedPts } from "@/components/views/VrpMapping";
import { api } from "@/lib/api";
import { dataApi, type Premium } from "@/lib/api_data";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { num, pct } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { useQuery, useQuerySync } from "@/lib/url";
import s from "./data.module.css";
import { Fold, Reading } from "@/components/PageBits";
import { Labelled } from "@/components/InfoTip";

const SHOWS = [
  { value: "premium", label: "Premium" },
  { value: "levels", label: "Implied and realised" },
] as const;
type Show = (typeof SHOWS)[number]["value"];

const BYS = [
  { value: "period", label: "Periods" },
  { value: "year", label: "Years" },
] as const;
type By = (typeof BYS)[number]["value"];

const fix = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? "–" : x.toFixed(d));

function zoom(t: Tokens) {
  return [
    { type: "inside" as const, xAxisIndex: 0 },
    { type: "slider" as const, xAxisIndex: 0, height: 18, bottom: 10, borderColor: t.axis, textStyle: { color: t.muted } },
  ];
}

/** The daily premium, light, with its quarter's mean drawn over it and the long-run mean marked. */
function premiumOption(t: Tokens, p: Premium): EChartsOption {
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["Daily premium", `${p.smooth_days}-day mean`] },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => (typeof v === "number" ? `${signedPts(v)} vol pts` : "–") },
    grid: { left: 56, right: 24, top: 40, bottom: 64 },
    xAxis: { type: "category", data: p.date, ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Vol pts", nameLocation: "middle", ...axis(t), nameGap: 38 },
    dataZoom: zoom(t),
    series: [
      {
        name: "Daily premium",
        type: "line",
        data: p.premium,
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 1, color: t.s5, opacity: 0.55 },
        itemStyle: { color: t.s5 },
        markLine: {
          symbol: "none",
          silent: true,
          data: [
            { yAxis: 0, lineStyle: { color: t.axis, width: 1, type: "solid" }, label: { show: false } },
            ...(p.mean !== null
              ? [{ yAxis: p.mean, lineStyle: { color: t.s2, width: 1.5, type: "dashed" as const }, label: { color: t.text2, formatter: `mean ${signedPts(p.mean)}`, position: "insideEndTop" as const } }]
              : []),
          ],
        },
      },
      {
        name: `${p.smooth_days}-day mean`,
        type: "line",
        data: p.smooth,
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 2, color: t.s1 },
        itemStyle: { color: t.s1 },
      },
    ],
  };
}

/** IV30 against the realised vol that followed and the one before: one axis, vol points. */
function levelsOption(t: Tokens, p: Premium): EChartsOption {
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["IV30", "Realised, next 21 days", "Realised, last 21 days"] },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "–") },
    grid: { left: 56, right: 24, top: 40, bottom: 64 },
    xAxis: { type: "category", data: p.date, ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Vol pts", nameLocation: "middle", ...axis(t), nameGap: 38 },
    dataZoom: zoom(t),
    series: [
      { name: "IV30", type: "line", data: p.iv30, showSymbol: false, sampling: "lttb", lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1 } },
      { name: "Realised, next 21 days", type: "line", data: p.rv_fwd, showSymbol: false, sampling: "lttb", lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2 } },
      { name: "Realised, last 21 days", type: "line", data: p.rv_trailing, showSymbol: false, sampling: "lttb", lineStyle: { width: 1, color: t.s5, opacity: 0.8 }, itemStyle: { color: t.s5 } },
    ],
  };
}

export function PremiumView() {
  const t = useTokens();
  const q = useQuery();
  const [show, setShow] = useState<Show>(() => q.oneOf("show", ["premium", "levels"] as const, "premium"));
  const [by, setBy] = useState<By>(() => q.oneOf("by", ["period", "year"] as const, "period"));
  useQuerySync({ show, by }, { show: "premium", by: "period" });
  const prem = useFetch((sig) => dataApi.premium(sig), "premium");
  const vrp = useFetch((sig) => api.vrp(sig), "vrp");
  const p = prem.data;
  const m = vrp.data?.mapping ?? null;
  const h = m?.headline;
  const option = useMemo(() => (p ? (show === "premium" ? premiumOption(t, p) : levelsOption(t, p)) : null), [t, p, show]);
  const rows = m ? (by === "period" ? [m.headline, ...m.periods] : [m.headline, ...m.years]) : [];
  const now = p?.now;

  return (
    <div className={s.view}>
      <Reading loading={prem.loading}>{p?.reading}</Reading>
      {prem.error ? <ErrorBox what="the premium" error={prem.error} onRetry={prem.reload} /> : null}
      <div className={ui.grid4}>
        <Tile
          info="data.premium_mean"
          label="Mean premium"
          value={`${signedPts(p?.mean ?? h?.mean)} vol pts`}
          note={h ? `95% [${signedPts(h.mean_low)}, ${signedPts(h.mean_high)}] · ${num(h.n_days)} days` : p?.mean_low === null ? "no interval: docs/VRP_MAPPING.json not found" : undefined}
        />
        <Tile
          info="data.premium_recent"
          label="Latest quarter"
          value={now?.recent !== null && now?.recent !== undefined ? `${signedPts(now.recent)} vol pts` : "–"}
          note={now?.measured_date ? `to ${now.measured_date} · above ${pct(now.recent_share_below)} of quarters` : undefined}
        />
        <Tile
          info="data.premium_priced"
          label="Priced today"
          value={now?.priced !== null && now?.priced !== undefined ? `${signedPts(now.priced)} vol pts` : "–"}
          note={now ? `${now.date} · IV30 ${fix(now.iv30)} against ${fix(now.rv_trailing)} realised` : undefined}
        />
        <Tile
          info="data.days_positive"
          label="Days positive"
          value={h ? pct(h.share_positive) : "–"}
          note={h ? `median day ${signedPts(h.median)} vol pts` : undefined}
        />
      </div>

      <Card
        title={show === "premium" ? "The premium, day by day" : "Implied against realised volatility"} info={`data.${show === "premium" ? "premium_daily" : "iv30"}`}
        right={
          <Segmented label="Chart" title="Show" term="data.chart_view" value={show} options={SHOWS} onChange={setShow} />
        }
      >
        {prem.loading && !p ? <Loading what="the premium" /> : null}
        {p && option ? (
          p.date.length ? (
            <Chart
              option={option}
              height={380}
              label={show === "premium" ? "The daily volatility premium with its quarterly mean" : "30-day implied volatility against realised volatility before and after"}
              stamp={`SPX · ${p.date[0]} to ${p.date[p.date.length - 1]}`}
              rows={() => p.date.map((d, i) => ({ date: d, premium: p.premium[i], smooth: p.smooth[i], iv30: p.iv30[i], rv_fwd: p.rv_fwd[i], rv_trailing: p.rv_trailing[i] }))}
            />
          ) : (
            <Empty title="No daily rows" />
          )
        ) : null}
      </Card>

      <Card
        title="Across the sample, on the sweep's axis" info="data.subperiods"
        right={
          <Segmented label="Split by" title="Split by" term="data.by" value={by} options={BYS} onChange={setBy} />
        }
      >
        {vrp.error ? <ErrorBox what="the VRP mapping" error={vrp.error} onRetry={vrp.reload} /> : null}
        {vrp.data && !m ? <Empty title={vrp.data.error || "docs/VRP_MAPPING.json not found"} /> : null}
        {m ? (
          <Chart
            option={axisOption(t, rows, m.sweep_axis, (r) => (r === m.headline ? "All days" : r.label.replace(/ \(windows to [^)]*\)/, "")))}
            height={70 + (by === "year" ? 24 : 34) * rows.length}
            label="The premium's mean and 95% interval in each part of the sample, against the sweep's planted premiums"
            stamp={`VRP mapping · ${m.generated}`}
            rows={() => rows.map((r) => ({ ...r }))}
          />
        ) : null}
      </Card>

      {m ? (
        <Fold title="How it is measured" term="data.sensitivity" sub="the headline under other definitions of realised vol">
          <SensitivityTable m={m} />
        </Fold>
      ) : null}
    </div>
  );
}
