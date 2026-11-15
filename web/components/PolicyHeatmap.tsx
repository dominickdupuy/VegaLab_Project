"use client";

import { useState } from "react";
import type { EChartsOption } from "echarts";
import { STRUCT_LABEL, structureColor } from "@/components/ActionMix";
import { Card, Segmented } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { ErrorBox } from "@/components/States";
import ui from "@/components/ui.module.css";
import type { RunDetail } from "@/lib/api";
import { type PolicyHeatmap, runsApi } from "@/lib/api_runs";
import { axis, base, runStamp, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, pct } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const LOG_FILE = "action_log.json";

/** The steps in cell (time bin, vol bin), and how many of them chose action `a`. */
function cell(h: PolicyHeatmap, ti: number, vi: number, a: number): { n: number; k: number } {
  const counts = h.counts[ti]?.[vi] ?? [];
  return { n: counts.reduce((s, c) => s + c, 0), k: counts[a] ?? 0 };
}

function range(edges: number[], i: number, f: (v: number) => string): string {
  const lo = edges[i] ?? 0;
  const hi = edges[i + 1] ?? lo;
  return lo === hi ? f(lo) : `${f(lo)}–${f(hi)}`;
}

/** One action's share of the steps in each (ATM vol, time in episode) cell; a cell no step reached is left blank. */
function heatOption(t: Tokens, h: PolicyHeatmap, a: number): EChartsOption {
  const nt = h.time_frac_edges.length - 1;
  const nv = Math.max(1, h.atm_iv_edges.length - 1);
  const name = STRUCT_LABEL[h.action_names[a] ?? ""] ?? h.action_names[a] ?? "";
  const data: [number, number, number, number][] = [];
  for (let ti = 0; ti < nt; ti++)
    for (let vi = 0; vi < nv; vi++) {
      const { n, k } = cell(h, ti, vi, a);
      if (n > 0) data.push([vi, ti, k / n, n]);
    }
  return {
    ...base(t),
    grid: { left: 70, right: 18, top: 16, bottom: 56, containLabel: false },
    tooltip: {
      ...base(t).tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const [vi, ti, share, n] = (p as { data: [number, number, number, number] }).data;
        return `ATM vol ${esc(range(h.atm_iv_edges, vi, (v) => pct(v, 1)))}, ${esc(range(h.time_frac_edges, ti, (v) => pct(v)))} through the episode<br/><b>${pct(share)}</b> of ${num(n)} steps chose ${esc(name)}`;
      },
    },
    xAxis: { type: "category", data: Array.from({ length: nv }, (_, i) => range(h.atm_iv_edges, i, (v) => pct(v, 1))), name: "ATM implied vol the policy saw", nameLocation: "middle", ...axis(t, { nameGap: 34, splitLine: { show: false } }) },
    yAxis: { type: "category", data: Array.from({ length: nt }, (_, i) => range(h.time_frac_edges, i, (v) => pct(v))), name: "Through the episode", nameLocation: "middle", ...axis(t, { nameGap: 56, splitLine: { show: false } }) },
    visualMap: { min: 0, max: 1, dimension: 2, show: false, inRange: { color: [t.surface2, structureColor(t, h.action_names[a] ?? "")] } },
    series: [{ type: "heatmap", data, itemStyle: { borderColor: t.surface, borderWidth: 1 }, emphasis: { itemStyle: { borderColor: t.text, borderWidth: 1 } } }],
  };
}

/** The action chosen most often other than flat, else flat: where a reader looks first. */
function busiest(h: PolicyHeatmap): number {
  const totals = h.action_names.map((_, a) => h.counts.reduce((s, row) => s + row.reduce((r, c) => r + (c[a] ?? 0), 0), 0));
  let best = 0;
  for (let a = 1; a < totals.length; a++) if ((totals[a] ?? 0) > (best === 0 ? 0 : (totals[best] ?? 0))) best = a;
  return best;
}

/**
 * Where a launched run's policy acted: the share of steps that chose each action, by how
 * far through the episode and the ATM vol it saw. Read from the action log a dashboard job
 * writes beside its equity paths; a run without one (a sweep cell, a command-line run) has
 * no card.
 */
export function PolicyHeatmapCard({ d }: { d: RunDetail }) {
  const t = useTokens();
  const logged = d.artifacts.some((a) => a.path === LOG_FILE);
  const heat = useFetch((s) => runsApi.policy(d.row.key, s), `policy|${d.row.key}`, logged);
  const h = heat.data;
  const [picked, setPicked] = useState<string>("");
  if (!logged) return null;
  if (heat.error) return <ErrorBox what="the policy heatmap" error={heat.error} onRetry={heat.reload} />;
  if (!h || !h.available) return null;
  const chosen = h.action_names.filter((_, a) => h.counts.some((row) => row.some((c) => (c[a] ?? 0) > 0)));
  const a = picked && h.action_names.includes(picked) ? h.action_names.indexOf(picked) : busiest(h);
  const label = (n: string) => STRUCT_LABEL[n] ?? n;
  const oneVol = h.atm_iv_edges.length === 2 && h.atm_iv_edges[0] === h.atm_iv_edges[1];
  return (
    <Card
      title="Where it acted"
      info="policy_heatmap"
      sub={`The share of steps that chose ${label(h.action_names[a] ?? "")}, by how far through the episode and the at-the-money vol the policy saw: ${num(h.n_steps)} steps over ${num(h.n_episodes)} episodes.${oneVol ? " The vol never moved in this world (no vol of vol), so it reads by time alone." : ""}`}
      source={`From ${LOG_FILE}, hashed in the run's manifest.`}
    >
      <div className={ui.stack}>
        {chosen.length > 1 ? (
          <Segmented label="Action" value={h.action_names[a] ?? ""} options={chosen.map((n) => ({ value: n, label: label(n) }))} onChange={setPicked} />
        ) : null}
        <Chart
          option={heatOption(t, h, a)}
          height={oneVol ? 300 : 340}
          label={`Share of steps choosing ${label(h.action_names[a] ?? "")}, by time in the episode and ATM implied vol`}
          stamp={runStamp(d.row)}
          rows={() =>
            h.counts.flatMap((row, ti) =>
              row.map((c, vi) => ({
                time_from: h.time_frac_edges[ti],
                time_to: h.time_frac_edges[ti + 1],
                atm_iv_from: h.atm_iv_edges[vi],
                atm_iv_to: h.atm_iv_edges[vi + 1],
                ...Object.fromEntries(h.action_names.map((n, i) => [n, c[i] ?? 0])),
              })),
            )
          }
          json={() => h}
        />
      </div>
    </Card>
  );
}
