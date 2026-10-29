"use client";

import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, SelectField, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { AtmLandscape } from "@/components/views/AtmLandscape";
import { type GridDay, IvSurface3D, PlayControls, type Playback, type XMode, XModeToggle, xName, xOf } from "@/components/views/IvSurface";
import { signedPts } from "@/components/views/VrpMapping";
import type { Grid } from "@/lib/api";
import { dataApi } from "@/lib/api_data";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { pct } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { useQuery, useQuerySync } from "@/lib/url";
import s from "./data.module.css";
import { DateBar } from "./DateBar";
import { Fold, Reading } from "@/components/PageBits";
import { Labelled } from "@/components/InfoTip";

const vp = (x: number | null | undefined) => (x === null || x === undefined ? null : x * 100);
const fix1 = (x: number | null | undefined) => (x === null || x === undefined ? "–" : x.toFixed(1));

/** IV across the grid's strikes at one tenor. */
function smileOption(t: Tokens, g: GridDay, row: number, mode: XMode): EChartsOption {
  const tenor = g.tenor_days[row] ?? 0;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      formatter: (ps: unknown) => {
        const q = (ps as { value: [number, number | null] }[])[0];
        if (!q) return "";
        const k = mode === "k" ? q.value[0] : Math.log(q.value[0]);
        return `ln(K/F) ${k.toFixed(3)} · K/F ${Math.exp(k).toFixed(3)} · ${tenor}d<br/>IV <b>${q.value[1] === null ? "not quoted" : q.value[1].toFixed(2)}</b>`;
      },
    },
    grid: { left: 52, right: 20, top: 20, bottom: 44 },
    xAxis: { type: "value", name: xName(mode), nameLocation: "middle", scale: true, ...axis(t), nameGap: 28 },
    yAxis: { type: "value", name: "IV (vol pts)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 36 },
    series: [
      {
        name: `${tenor}-day smile`,
        type: "line",
        data: g.k.map((k, i) => [xOf(k, mode), vp(g.iv[row]?.[i])]),
        connectNulls: false,
        symbolSize: 5,
        lineStyle: { width: 2, color: t.s1 },
        itemStyle: { color: t.s1 },
        markLine: { symbol: "none", silent: true, lineStyle: { color: t.s5, width: 1, type: "solid" }, label: { color: t.muted, formatter: "F" }, data: [{ xAxis: mode === "k" ? 0 : 1 }] },
      },
    ],
  };
}

/** ATM IV and the put about 10% below across the grid's tenors. */
function termOption(t: Tokens, g: GridDay, putK: number): EChartsOption {
  const col = (k: number) => g.k.findIndex((x) => Math.abs(x - k) < 1e-6);
  const atm = col(0);
  const put = col(putK);
  const at = (c: number) => g.tenor_days.map((d, r) => [d, c < 0 ? null : vp(g.iv[r]?.[c])]);
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["At the money", "Put 10% below"] },
    tooltip: { ...base(t).tooltip, trigger: "axis", valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "not quoted") },
    grid: { left: 52, right: 20, top: 40, bottom: 44 },
    xAxis: { type: "value", name: "Tenor (calendar days)", nameLocation: "middle", ...axis(t), nameGap: 28 },
    yAxis: { type: "value", name: "IV (vol pts)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 36 },
    series: [
      { name: "At the money", type: "line", data: at(atm), symbolSize: 7, lineStyle: { width: 2, color: t.s1 }, itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 } },
      { name: "Put 10% below", type: "line", data: at(put), symbolSize: 6, lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2 } },
    ],
  };
}

export function SurfaceView({
  dates,
  date,
  setDate,
  grid,
  gridError,
  play,
  xMode,
  setXMode,
}: {
  dates: string[];
  date: string;
  setDate: (d: string) => void;
  grid: Grid | undefined;
  gridError?: React.ReactNode;
  play: Playback;
  xMode: XMode;
  setXMode: (m: XMode) => void;
}) {
  const t = useTokens();
  const q = useQuery();
  const [tenor, setTenor] = useState(() => q.int("tenor", 30, 1, 365));
  useQuerySync({ tenor }, { tenor: 30 });
  const shape = useFetch((sig) => dataApi.shape(date, sig), `shape|${date}`, Boolean(date));
  const sh = shape.data;
  // while playing, the smile and term structure follow the frame
  const g: GridDay | undefined = play.frame ?? grid;
  const shown = play.frameDate ?? date;
  const row = g ? Math.max(0, g.tenor_days.indexOf(tenor) >= 0 ? g.tenor_days.indexOf(tenor) : g.tenor_days.findIndex((d) => d >= tenor)) : 0;
  const smile = useMemo(() => (g ? smileOption(t, g, row, xMode) : null), [t, g, row, xMode]);
  const term = useMemo(() => (g ? termOption(t, g, sh?.put_k ?? -0.1) : null), [t, g, sh?.put_k]);

  return (
    <div className={s.view}>
      <DateBar dates={dates} date={date} onChange={setDate}>
        <div className={ui.field}>
          <span className={ui.fieldLabel}>
            <Labelled term="data.playback">Playback</Labelled>
          </span>
          <PlayControls p={play} disabled={!dates.length} />
        </div>
      </DateBar>
      {play.error ? <p className={ui.fieldError}>{play.error}</p> : null}
      <Reading loading={shape.loading}>{play.playing ? `Playing from ${date}; pause to read the day's shape.` : sh?.reading}</Reading>
      {shape.error ? <ErrorBox what="the surface's shape" error={shape.error} onRetry={shape.reload} /> : null}
      <div className={ui.grid4}>
        <Tile info="data.atm30" label="30-day ATM vol" value={fix1(sh?.atm)} note={sh?.atm_share_below !== null && sh?.atm_share_below !== undefined ? `above ${pct(sh.atm_share_below)} of days` : undefined} />
        <Tile info="data.skew" label="Put skew" value={sh?.skew !== null && sh?.skew !== undefined ? signedPts(sh.skew, 1) : "–"} note={sh ? `10% below the forward · typical ${signedPts(sh.skew_median, 1)}` : undefined} />
        <Tile info="data.term_slope" label="Term slope" value={sh?.term !== null && sh?.term !== undefined ? signedPts(sh.term, 1) : "–"} note={sh ? `a year out less 30 days · typical ${signedPts(sh.term_median, 1)}` : undefined} />
        <Tile info="data.call_wing" label="Call wing" value={sh?.call !== null && sh?.call !== undefined ? signedPts(sh.call, 1) : "–"} note="5% above the forward, less ATM" />
      </div>

      <Card title={`The surface on ${shown}`} info="data.surface" right={<XModeToggle value={xMode} onChange={setXMode} />}>
        <div className={ui.stack}>
          {gridError}
          {!g && !gridError ? <Loading what="the surface" /> : null}
          {g ? <IvSurface3D t={t} g={g} mode={xMode} range={play.range} date={shown} /> : null}
        </div>
      </Card>

      <div className={ui.grid2}>
        <Card
          title={`Smile at ${g?.tenor_days[row] ?? tenor} days`}
          info="data.smile"
          right={
            g ? <SelectField label="Tenor" term="data.smile_tenor" value={g.tenor_days[row] ?? tenor} options={g.tenor_days} render={(d) => `${d} days`} onChange={(v) => v !== "" && setTenor(v)} /> : null
          }
        >
          {smile && g ? <Chart option={smile} height={300} label={`Implied volatility across strikes at ${g.tenor_days[row]} days on ${shown}`} stamp={`SPX · ${shown}`} /> : <Empty title="No surface on this day" />}
        </Card>
        <Card title="Term structure" info="data.term">
          {term && g ? <Chart option={term} height={300} label={`At-the-money implied volatility across tenors on ${shown}`} stamp={`SPX · ${shown}`} /> : <Empty title="No surface on this day" />}
        </Card>
      </div>

      <Fold title="The term structure through time" term="data.landscape" sub="at-the-money vol by date and tenor, 2010 on">
        <AtmLandscape />
      </Fold>
    </div>
  );
}
