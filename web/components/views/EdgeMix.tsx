"use client";

import type { EChartsOption } from "echarts";
import { STRUCT_LABEL, structureColor } from "@/components/ActionMix";
import { Card, Chip } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import lab from "@/components/lab.module.css";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { ArmPhrase, Reading } from "@/components/PageBits";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type GridActionMix, STRUCTURES } from "@/lib/api";
import { ARM_LABEL, armColor, axis, base, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, pct, prem } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * The Attribution page's action mix: for each arm, the share of steps it holds each
 * structure at every premium, one chart per arm, and a sentence on how its choice
 * moves with the premium. The run detail keeps its own (ActionMix.RunMixCard).
 */

type Pooled = GridActionMix["pooled"][number];

function topOf(r: Pooled): { s: string; share: number } {
  const m = r.summary.held_share;
  const s = [...STRUCTURES].sort((a, b) => (m[b] ?? 0) - (m[a] ?? 0))[0] ?? "FLAT";
  return { s, share: m[s] ?? 0 };
}

const IN: Record<string, string> = { IRON_CONDOR: "in an iron condor", SHORT_STRANGLE: "in a short strangle", PUT_CREDIT_SPREAD: "in a put credit spread" };
/** "flat 69% of the time", "in a short strangle 78% of the time". */
const state = (x: { s: string; share: number }) => `${x.s === "FLAT" ? "flat" : (IN[x.s] ?? `in ${x.s}`)} ${pct(x.share)} of the time`;

/** "The fixed-delta seller is in an iron condor 98% of the time at every premium." */
function mixSentence(arm: string, rows: Pooled[]): string {
  const lo = rows[0];
  const hi = rows[rows.length - 1];
  if (!lo || !hi) return "";
  const a = topOf(lo);
  const steady = rows.every((r) => topOf(r).s === a.s && Math.abs(topOf(r).share - a.share) < 0.03);
  if (steady) return `${ArmPhrase(arm)} is ${state(a)} at every premium.`;
  return `${ArmPhrase(arm)} is ${state(a)} at ${prem(lo.premium)} vol pts and ${state(topOf(hi))} at ${prem(hi.premium)}.`;
}

function armMixOption(t: Tokens, rows: Pooled[]): EChartsOption {
  const cats = rows.map((r) => prem(r.premium));
  return {
    ...base(t),
    legend: { ...base(t).legend, data: STRUCTURES.map((n) => STRUCT_LABEL[n] ?? n) },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: number; dataIndex: number }[];
        const r = rows[arr[0]?.dataIndex ?? 0];
        return [`<b>${esc(cats[arr[0]?.dataIndex ?? 0])} vol pts</b>`, ...arr.filter((q) => q.value > 0.0005).map((q) => `${esc(q.seriesName)}: ${pct(q.value, 1)}`), r ? `opens ${num(r.summary.opens_per_episode, 2)} per episode` : ""].join("<br/>");
      },
    },
    grid: { ...base(t).grid, left: 48, right: 12, top: 44, bottom: 40 },
    xAxis: { type: "category", data: cats, name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", min: 0, max: 1, ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    series: STRUCTURES.map((n) => ({
      id: n,
      name: STRUCT_LABEL[n] ?? n,
      type: "bar" as const,
      stack: "held",
      barCategoryGap: "28%",
      itemStyle: { color: structureColor(t, n), borderColor: t.surface, borderWidth: 0.5 },
      emphasis: { focus: "series" as const },
      data: rows.map((r) => r.summary.held_share[n] ?? 0),
    })),
  };
}

export function EdgeMix({ grid, command, episodesPerCell }: { grid: string; command: string; episodesPerCell?: number }) {
  const t = useTokens();
  const mix = useFetch((s) => api.gridMix(grid, undefined, s), `gridmix|${grid}`, Boolean(grid));
  const g = mix.data?.grid;
  const byArm = (arm: string) => (g?.pooled ?? []).filter((p) => p.arm === arm).sort((a, b) => a.premium - b.premium);
  const stamp = g ? `grid ${g.grid_hash} · ${g.n_seeds} blocks · ${g.source} · ${g.reproduces ? "reproduces" : g.agrees ? "agrees" : "differs"}${mix.data ? ` · actionmix/${mix.data.path.split("/").pop()}` : ""}` : "";
  const perCell = g?.cells[0]?.summary.n_episodes;
  const read = perCell === undefined ? "" : episodesPerCell && perCell < episodesPerCell ? `the first ${num(perCell)} of ${num(episodesPerCell)} episodes of each cell` : `${num(perCell)} episodes of each cell`;
  const rows = g ? g.arms.flatMap(byArm) : [];
  return (
    <div id="mix">
      <Card title="What each arm trades" info="attribution.mix" source={g ? `Pooled over ${g.n_seeds} blocks per arm and premium, from ${read}. ${stamp}` : undefined}>
        {mix.loading && !mix.data ? <Loading what="the action mix" /> : null}
        {mix.error?.status === 404 ? (
          <Empty title="No action mix for this grid">
            <p>
              <code>{command}</code>
            </p>
          </Empty>
        ) : mix.error ? (
          <ErrorBox what="the action mix" error={mix.error} onRetry={mix.reload} />
        ) : null}
        {g ? (
          <div className={ui.stack}>
            <Reading>{g.arms.map((arm) => mixSentence(arm, byArm(arm))).join(" ")}</Reading>
            <div className={ui.grid2}>
              {g.arms.map((arm) => (
                <div key={arm}>
                  <p className={lab.armHead}>
                    <span className={lab.armDot} style={{ background: armColor(t, arm) }} aria-hidden="true" />
                    {ARM_LABEL[arm] ?? arm}
                  </p>
                  <Chart option={armMixOption(t, byArm(arm))} height={260} label={`Share of steps holding each structure at each premium, ${ARM_LABEL[arm] ?? arm}`} stamp={stamp} rows={() => byArm(arm).map((p) => ({ arm: p.arm, premium: p.premium, ...Object.fromEntries(STRUCTURES.map((s) => [`held_${s}`, p.summary.held_share[s] ?? 0])) }))} />
                </div>
              ))}
            </div>
            <details className={lab.subFold}>
              <summary>Opens, refusals and holding days</summary>
              <div className={ui.tableWrap}>
                <table className={ui.table}>
                  <thead>
                    <tr>
                      <th>Arm</th>
                      <th className={ui.num}>Premium</th>
                      <th className={ui.num}>Held</th>
                      <th>Mostly</th>
                      <th className={ui.num}>Opens/ep</th>
                      <th className={ui.num}>Refused/ep</th>
                      <th className={ui.num}>Days held p50</th>
                      <th>Re-simulation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((p) => {
                      const m = p.summary;
                      const top = STRUCTURES.filter((s) => s !== "FLAT").sort((a, b) => (m.held_share[b] ?? 0) - (m.held_share[a] ?? 0))[0] ?? "FLAT";
                      return (
                        <tr key={`${p.arm}-${p.premium}`}>
                          <td>{ARM_LABEL[p.arm] ?? p.arm}</td>
                          <td className={ui.num}>{num(p.premium, 2)}</td>
                          <td className={ui.num}>{pct(1 - (m.held_share.FLAT ?? 0), 1)}</td>
                          <td>{(m.held_share[top] ?? 0) > 0 ? <Chip color={structureColor(t, top)}>{STRUCT_LABEL[top]}</Chip> : <span className={ui.dim}>flat</span>}</td>
                          <td className={ui.num}>{num(m.opens_per_episode, 2)}</td>
                          <td className={ui.num}>{num(m.refused_per_episode, 2)}</td>
                          <td className={ui.num}>{num(m.holding_days?.p50, 0)}</td>
                          <td className={ui.dim}>{p.reproduces ? "bit for bit" : p.agrees ? "within 1e-9" : "differs"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <FigureFoot name={`action mix ${g.grid_hash}`} stamp={stamp} rows={() => rows.map((p) => ({ arm: p.arm, premium: p.premium, opens_per_episode: p.summary.opens_per_episode, refused_per_episode: p.summary.refused_per_episode, refused_share: p.summary.refused_share, holding_days_p50: p.summary.holding_days?.p50 ?? null, ...Object.fromEntries(STRUCTURES.map((s) => [`held_${s}`, p.summary.held_share[s] ?? 0])) }))} json={() => g} />
            </details>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
