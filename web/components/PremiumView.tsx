"use client";

import { useMemo, useState } from "react";
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams, CustomSeriesRenderItemReturn, EChartsOption } from "echarts";
import { Card, Segmented } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { Empty } from "@/components/States";
import ui from "@/components/ui.module.css";
import type { PremiumContrast, PremiumGroup, Premiums, World } from "@/lib/api_runs";
import { ARM_LABEL, armColor, axis, base, type Tokens, useTokens } from "@/lib/chart";
import { esc, prem, usd } from "@/lib/format";

const ARM_ORDER = ["flat", "always_on", "random", "delta_seller", "ppo"];

/** A series: one arm, or a learned arm at one training budget. */
interface Series {
  id: string;
  arm: string;
  variant: string;
  name: string;
}

function seriesOf(rows: { arm: string; variant: string }[]): Series[] {
  const seen = new Map<string, Series>();
  for (const r of rows) {
    const id = `${r.arm}|${r.variant}`;
    if (!seen.has(id)) seen.set(id, { id, arm: r.arm, variant: r.variant, name: `${ARM_LABEL[r.arm] ?? r.arm}${r.variant ? ` · ${r.variant}` : ""}` });
  }
  return [...seen.values()].sort((a, b) => ARM_ORDER.indexOf(a.arm) - ARM_ORDER.indexOf(b.arm) || a.variant.localeCompare(b.variant));
}

/** Variants of one arm get lighter shades of its colour, so the arm still reads as one. */
function seriesColor(t: Tokens, s: Series, all: Series[]): string {
  const same = all.filter((x) => x.arm === s.arm);
  const i = same.indexOf(s);
  const c = armColor(t, s.arm);
  return i <= 0 ? c : `color-mix(in srgb, ${c} ${Math.max(40, 100 - 20 * i)}%, ${t.surface})`;
}

function boxItem(slot: number, slots: number, color: string) {
  return (_: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI): CustomSeriesRenderItemReturn => {
    const x = Number(api.value(0));
    const band = (api.size?.([1, 0]) as number[] | undefined)?.[0] ?? 40;
    const span = band * 0.8;
    const cx = (api.coord([x, 0])[0] ?? 0) - span / 2 + ((slot + 0.5) * span) / slots;
    const w = Math.max(6, Math.min(26, (span / slots) * 0.62));
    const y = (i: number) => api.coord([x, Number(api.value(i))])[1] ?? 0;
    const [p5 = 0, p25 = 0, p50 = 0, p75 = 0, p95 = 0, mean = 0, lo = 0, hi = 0] = [1, 2, 3, 4, 5, 6, 7, 8].map(y);
    const thin = { stroke: color, lineWidth: 1 };
    const kids: CustomSeriesRenderItemReturn[] = [
      { type: "line", shape: { x1: cx, y1: p5, x2: cx, y2: p25 }, style: thin },
      { type: "line", shape: { x1: cx, y1: p75, x2: cx, y2: p95 }, style: thin },
      { type: "line", shape: { x1: cx - w / 4, y1: p5, x2: cx + w / 4, y2: p5 }, style: thin },
      { type: "line", shape: { x1: cx - w / 4, y1: p95, x2: cx + w / 4, y2: p95 }, style: thin },
      { type: "rect", shape: { x: cx - w / 2, y: Math.min(p25, p75), width: w, height: Math.abs(p75 - p25) || 1 }, style: { fill: color, opacity: 0.22, stroke: color, lineWidth: 1 } },
      { type: "line", shape: { x1: cx - w / 2, y1: p50, x2: cx + w / 2, y2: p50 }, style: { stroke: color, lineWidth: 1.5 } },
    ];
    if (Number.isFinite(Number(api.value(7))) && Number.isFinite(Number(api.value(8))))
      kids.push({ type: "line", shape: { x1: cx, y1: lo, x2: cx, y2: hi }, style: { stroke: color, lineWidth: 3.5, opacity: 0.9 } });
    kids.push({ type: "circle", shape: { cx, cy: mean, r: 3.6 }, style: { fill: "#fff", stroke: color, lineWidth: 2 } });
    return { type: "group", children: kids } as CustomSeriesRenderItemReturn;
  };
}

function boxOption(t: Tokens, groups: PremiumGroup[], all: Series[], real: boolean): EChartsOption {
  const premiums = [...new Set(groups.map((g) => g.premium))].sort((a, b) => a - b);
  const cats = premiums.map((p) => (real ? "Real SPX" : `${prem(p)}`));
  return {
    ...base(t),
    legend: { ...base(t).legend, data: all.map((s) => s.name) },
    tooltip: { ...base(t).tooltip, trigger: "item" },
    grid: { ...base(t).grid, left: 74, top: 40 },
    xAxis: { type: "category", data: cats, name: real ? "" : "Planted premium (vol pts)", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "P&L per episode ($)", nameLocation: "middle", scale: true, ...axis(t), nameGap: 58 },
    series: all.map((s, slot) => {
      const color = seriesColor(t, s, all);
      const rows = groups.filter((g) => `${g.arm}|${g.variant}` === s.id);
      return {
        name: s.name,
        type: "custom" as const,
        itemStyle: { color },
        renderItem: boxItem(slot, all.length, color),
        encode: { x: 0, y: [1, 5] },
        data: rows.map((g) => ({
          value: [premiums.indexOf(g.premium), g.p5, g.p25, g.p50, g.p75, g.p95, g.mean, g.ci_low, g.ci_high],
          g,
        })),
        tooltip: {
          formatter: (p: unknown) => {
            const g = (p as { data: { g: PremiumGroup } }).data.g;
            return [
              `<b>${esc(s.name)}</b>${real ? " on real SPX" : ` at ${esc(prem(g.premium))} vol pts`}`,
              `mean ${usd(g.mean)} (95% ${usd(g.ci_low)} to ${usd(g.ci_high)})`,
              `median ${usd(g.p50)} · middle half ${usd(g.p25)} to ${usd(g.p75)}`,
              `5th to 95th ${usd(g.p5)} to ${usd(g.p95)}`,
              `<span style="opacity:.7">${g.n_episodes} episodes in ${g.n_runs} run${g.n_runs === 1 ? "" : "s"}; interval over ${g.interval}</span>`,
            ].join("<br/>");
          },
        },
      };
    }),
  };
}

function contrastOption(t: Tokens, rows: PremiumContrast[], all: Series[]): EChartsOption {
  const shown = all.filter((s) => rows.some((r) => `${r.arm}|${r.variant}` === s.id && r.estimate !== null));
  return {
    ...base(t),
    legend: { ...base(t).legend, data: shown.map((s) => s.name) },
    tooltip: { ...base(t).tooltip, trigger: "item" },
    grid: { ...base(t).grid, left: 74, top: 40 },
    xAxis: { type: "value", name: "Planted premium (vol pts)", nameLocation: "middle", min: 0, ...axis(t) },
    yAxis: { type: "value", name: "Gain over zero premium ($/ep)", nameLocation: "middle", ...axis(t), nameGap: 58 },
    series: shown.flatMap((s) => {
      const color = seriesColor(t, s, all);
      const pts = rows.filter((r) => `${r.arm}|${r.variant}` === s.id && r.estimate !== null).sort((a, b) => a.premium - b.premium);
      return [
        {
          name: s.name,
          type: "line" as const,
          data: [{ value: [0, 0], r: null }, ...pts.map((r) => ({ value: [r.premium, r.estimate], r }))],
          symbol: "circle",
          symbolSize: 8,
          lineStyle: { width: 2, color },
          itemStyle: { color, borderColor: t.surface, borderWidth: 2 },
          tooltip: {
            formatter: (p: unknown) => {
              const r = (p as { data: { r: PremiumContrast | null } }).data.r;
              if (!r) return `<b>${esc(s.name)}</b><br/>zero premium: the reference`;
              return [
                `<b>${esc(s.name)}</b> at ${esc(prem(r.premium))} vol pts less itself at 0`,
                `${usd(r.estimate)} per episode (95% ${usd(r.ci_low)} to ${usd(r.ci_high)})`,
                r.detected ? "detected: the one-sided bound is above zero" : "not detected",
                `<span style="opacity:.7">${r.n_pairs} paired episodes, ${r.n_blocks} run pair${r.n_blocks === 1 ? "" : "s"}; interval over ${r.interval}</span>`,
              ].join("<br/>");
            },
          },
          markLine: { symbol: "none", silent: true, lineStyle: { color: t.text2, width: 1 }, label: { show: false }, data: [{ yAxis: 0 }] },
          z: 3,
        },
        {
          name: s.name,
          type: "custom" as const,
          silent: true,
          tooltip: { show: false },
          data: pts.filter((r) => r.ci_low !== null && r.ci_high !== null).map((r) => [r.premium, r.ci_low, r.ci_high]),
          encode: { x: 0, y: [1, 2] },
          renderItem: (_: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI): CustomSeriesRenderItemReturn => {
            const x = Number(api.value(0));
            const a = api.coord([x, Number(api.value(1))]);
            const b = api.coord([x, Number(api.value(2))]);
            const st = { stroke: color, lineWidth: 1.5 };
            const x0 = a[0] ?? 0;
            return {
              type: "group",
              children: [
                { type: "line", shape: { x1: x0, y1: a[1] ?? 0, x2: x0, y2: b[1] ?? 0 }, style: st },
                { type: "line", shape: { x1: x0 - 4, y1: a[1] ?? 0, x2: x0 + 4, y2: a[1] ?? 0 }, style: st },
                { type: "line", shape: { x1: x0 - 4, y1: b[1] ?? 0, x2: x0 + 4, y2: b[1] ?? 0 }, style: st },
              ],
            } as CustomSeriesRenderItemReturn;
          },
          z: 2,
        },
      ];
    }),
  };
}

/** One sentence: what each arm gains at the largest premium it ran, against itself at zero. */
function sentence(rows: PremiumContrast[], all: Series[]): string {
  const parts: string[] = [];
  for (const s of all) {
    const pts = rows.filter((r) => `${r.arm}|${r.variant}` === s.id && r.estimate !== null).sort((a, b) => b.premium - a.premium);
    const top = pts[0];
    if (!top) continue;
    if (top.estimate === 0 && top.ci_low === 0) {
      parts.push(`${s.name} gains nothing`);
      continue;
    }
    parts.push(`${s.name} ${usd(top.estimate)} at ${prem(top.premium)} vol pts (95% ${usd(top.ci_low)} to ${usd(top.ci_high)}${top.detected ? "" : ", not detected"})`);
  }
  if (!parts.length) return "";
  return `Against itself at zero premium, on the same paths: ${parts.join("; ")}.`;
}

function worldName(w: World): string {
  if (w.source !== "synthetic") return w.label ? `Real SPX · ${w.label}` : "Real SPX";
  return w.label ? w.label : "Default world";
}

export function PremiumView({ data }: { data: Premiums }) {
  const t = useTokens();
  const worlds = data.worlds.filter((w) => data.groups.some((g) => g.world === w.id));
  const [pick, setPick] = useState<string>("");
  const world = worlds.find((w) => w.id === pick) ?? worlds[0];
  const groups = useMemo(() => data.groups.filter((g) => g.world === world?.id), [data, world]);
  const contrasts = useMemo(() => data.contrasts.filter((c) => c.world === world?.id), [data, world]);
  const all = useMemo(() => seriesOf(groups), [groups]);
  if (!world) return null;
  const real = world.source !== "synthetic";
  const line = real ? "" : sentence(contrasts, all);
  const options = worlds.slice(0, 8).map((w) => ({ value: w.id, label: `${worldName(w)} (${w.n_runs})` }));
  const hasContrast = contrasts.some((c) => c.estimate !== null);
  return (
    <Card
      title="What each arm earns as the premium grows"
      info="box"
      sub={line || (real ? "Real history: no premium is planted, so each arm is one box." : "Per arm and premium: the spread of episode P&L, the mean and its 95% interval.")}
      source={`${data.n_runs} OK runs read from their run.json and manifest${data.n_skipped ? `; ${data.n_skipped} did not read` : ""}. Runs pool only within one world: the same config but for the arm, seeds, episode count and premium.`}
    >
      <div className={ui.stack}>
        {options.length > 1 ? (
          <div className={ui.row}>
            <span className={ui.fieldLabel}>
              <Labelled term="world">World</Labelled>
            </span>
            <Segmented label="World" value={world.id} options={options} onChange={setPick} />
          </div>
        ) : null}
        <Chart
          option={boxOption(t, groups, all, real)}
          height={340}
          label="P&L per episode by planted premium, one box per arm"
          rows={() => groups.map((g) => ({ arm: g.arm, variant: g.variant, premium: g.premium, n_runs: g.n_runs, n_episodes: g.n_episodes, mean: g.mean, ci_low: g.ci_low, ci_high: g.ci_high, p5: g.p5, p25: g.p25, p50: g.p50, p75: g.p75, p95: g.p95, interval: g.interval }))}
        />
        {!real ? (
          <>
            <p className={ui.groupTitle}>
              Each arm against itself at zero premium <InfoTip term="paired_contrast" />
            </p>
            {hasContrast ? (
              <Chart
                option={contrastOption(t, contrasts, all)}
                height={300}
                label="Paired gain over zero premium, per arm, with 95% intervals"
                rows={() => contrasts.map((c) => ({ arm: c.arm, variant: c.variant, premium: c.premium, estimate: c.estimate, ci_low: c.ci_low, ci_high: c.ci_high, lower_bound: c.lower_bound, detected: c.detected, n_pairs: c.n_pairs, n_blocks: c.n_blocks, interval: c.interval, reason: c.reason }))}
              />
            ) : (
              <Empty title="No pairs in this world">
                <p>A contrast needs the same arm at zero premium on the same seeds.</p>
              </Empty>
            )}
            {contrasts.some((c) => c.interval === "episodes") ? (
              <p className={ui.dim} style={{ fontSize: 12.5 }}>
                Where only one run pair exists the interval is over its episodes <InfoTip term="contrast_episodes" />; over several pairs it is the pre-registered rule <InfoTip term="contrast_runs" />.
              </p>
            ) : null}
          </>
        ) : null}
      </div>
    </Card>
  );
}
