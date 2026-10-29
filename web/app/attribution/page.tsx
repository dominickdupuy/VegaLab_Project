"use client";

import Link from "next/link";
import { Suspense, useState } from "react";
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemReturn, EChartsOption } from "echarts";
import { STRUCT_LABEL } from "@/components/ActionMix";
import { Card, Chip, PageHead, Segmented, SelectField, Source, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import lab from "@/components/lab.module.css";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { ArmPhrase, Fold, Reading } from "@/components/PageBits";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import { EdgeMix } from "@/components/views/EdgeMix";
import ui from "@/components/ui.module.css";
import { api, type ArmEdge, type Edge, type EdgeDetail, type EdgeList, type GridEdge } from "@/lib/api";
import { ARM_LABEL, armColor, axis, base, componentColor, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, pct, prem, usd } from "@/lib/format";
import { guide } from "@/lib/guide";
import { useFetch } from "@/lib/useFetch";
import { useHashScroll, useQuery, useQuerySync } from "@/lib/url";

const LABEL: Record<string, string> = { delta: "Delta", gamma: "Gamma", vega: "Vega", theta: "Theta", carry: "Carry", cost: "Cost", residual: "Residual", pnl: "P&L edge" };
// a cell's re-simulation against run.json: bit for bit, within 1e-9, or with episodes beyond it
const VERDICT: Record<string, string> = { reproduces: "IDENTICAL", agrees: "AGREES", differs: "EPISODES DIFFER" };

/** A component whose one-sided interval excludes zero, either way. */
function carries(e: Edge): boolean {
  return (e.lower_bound ?? 0) > 0 || (e.upper_bound ?? 0) < 0;
}

function carriers(a: ArmEdge): string[] {
  return Object.entries(a.components)
    .filter(([, e]) => carries(e))
    .sort((x, y) => Math.abs(y[1].estimate ?? 0) - Math.abs(x[1].estimate ?? 0))
    .map(([c, e]) => `${LABEL[c] ?? c} ${(e.estimate ?? 0) > 0 ? "+" : "−"}`);
}

/** "1 of 500 episodes differs", "3 of 1,000 episodes differ in 2 cells". */
function episodesText(bad: number, n: number, cells: number): string {
  return `${num(bad)} of ${num(n)} episodes ${bad === 1 ? "differs" : "differ"}${cells > 1 ? ` in ${cells} cells` : ""}`;
}

function stampOf(g: GridEdge): string {
  return `grid ${g.grid_hash} · S = ${g.n_seeds} × N = ${num(g.n_episodes)}${g.first ? ` (first ${g.first})` : ""} · ${g.source} · ${g.reproduction}`;
}

/** A whisker from the one-sided lower bound to estimate + t·se, horizontal or vertical. */
function whisker(color: string, horizontal: boolean) {
  return (_: unknown, a: CustomSeriesRenderItemAPI): CustomSeriesRenderItemReturn => {
    const cat = a.value(0);
    const lo = Number(a.value(1));
    const hi = Number(a.value(2));
    const p0 = horizontal ? a.coord([lo, cat]) : a.coord([cat, lo]);
    const p1 = horizontal ? a.coord([hi, cat]) : a.coord([cat, hi]);
    const [x0, y0, x1, y1] = [p0[0] ?? 0, p0[1] ?? 0, p1[0] ?? 0, p1[1] ?? 0];
    const style = { stroke: color, lineWidth: 1.5 };
    const cap = (x: number, y: number) => (horizontal ? { type: "line" as const, shape: { x1: x, y1: y - 5, x2: x, y2: y + 5 }, style } : { type: "line" as const, shape: { x1: x - 5, y1: y, x2: x + 5, y2: y }, style });
    return { type: "group", children: [{ type: "line", shape: { x1: x0, y1: y0, x2: x1, y2: y1 }, style }, cap(x0, y0), cap(x1, y1)] };
  };
}

function edgeTip(name: string, e: Edge | undefined): string {
  if (!e) return esc(name);
  return `<b>${esc(name)}</b><br/>edge ${usd(e.estimate)} (se ${usd(e.se)})<br/>95% one-sided: ≥ ${usd(e.lower_bound)} · upper ${usd(e.upper_bound)}<br/>t ${num(e.t_stat, 2)}${carries(e) ? " · excludes 0" : ""}`;
}

/** A component's part in the sentence, by its sign: "more theta collected (+$454)". */
const PHRASE: Record<string, [string, string]> = {
  theta: ["more theta collected", "less theta collected"],
  gamma: ["gamma it no longer pays", "extra gamma paid"],
  delta: ["directional gains", "directional losses"],
  vega: ["vega gains", "vega losses"],
  carry: ["carry earned", "carry paid"],
  cost: ["lower trading costs", "higher trading costs"],
};

const signedUsd = (x: number) => `${x >= 0 ? "+" : ""}${usd(x)}`;

/** The edge of one arm at one premium in a sentence: what carries it, what takes it back. */
function edgeSentence(a: ArmEdge, comps: string[]): string {
  const total = a.pnl.estimate ?? 0;
  const parts = comps.filter((c) => c !== "residual" && PHRASE[c]).map((c) => ({ c, x: a.components[c]?.estimate ?? 0 }));
  const pos = parts.filter((p) => p.x > 0).sort((x, y) => y.x - x.x);
  const neg = parts.filter((p) => p.x < 0).sort((x, y) => x.x - y.x);
  const posSum = pos.reduce((s, p) => s + p.x, 0);
  const negSum = -neg.reduce((s, p) => s + p.x, 0);
  const say = (p: { c: string; x: number }) => `${PHRASE[p.c]?.[p.x >= 0 ? 0 : 1]} (${signedUsd(p.x)})`;
  // the parts that matter: the largest, and any other above a sixth of the gross on its side
  const main = pos.filter((p, i) => i === 0 || p.x >= posSum / 6).slice(0, 2);
  const back = neg.filter((p, i) => (i === 0 && -p.x >= Math.abs(total) / 6) || -p.x >= Math.max(posSum, Math.abs(total)) / 6).slice(0, 2);
  const premium = `${prem(a.premium)} vol pt${a.premium === 1 ? "" : "s"}`;
  const head = `${ArmPhrase(a.arm)}'s edge at ${premium} is ${usd(total)} per episode${a.pnl_saved.rejected ? "" : ", not detected"}`;
  if (!main.length) return `${head}, with nothing adding to it.`;
  const verb = negSum > posSum / 2 ? "most of it given back as" : "partly offset by";
  return `${head}: mostly ${main.map(say).join(" plus ")}${back.length ? `, ${verb} ${back.map(say).join(" and ")}` : ""}.`;
}

/** One arm's edge at one premium: each component with its interval, the P&L edge on top. */
function edgeOption(t: Tokens, a: ArmEdge, comps: string[]): EChartsOption {
  const cats = ["pnl", ...comps];
  const get = (c: string) => (c === "pnl" ? a.pnl : a.components[c]);
  const pnl = a.pnl.estimate;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "item", formatter: (p: unknown) => { const q = p as { dataIndex: number }; const c = cats[q.dataIndex] ?? ""; return edgeTip(LABEL[c] ?? c, get(c)); } },
    grid: { left: 150, right: 24, top: 12, bottom: 44, containLabel: false },
    yAxis: {
      type: "category",
      inverse: true,
      data: cats.map((c) => LABEL[c] ?? c),
      ...axis(t),
      axisLine: { show: false },
      splitLine: { show: false },
      axisLabel: {
        color: t.text2,
        fontFamily: t.font,
        fontSize: 12.5,
        formatter: (_: string, i: number) => {
          const c = cats[i] ?? "";
          const v = get(c)?.estimate;
          const tag = c === "pnl" ? "edge" : "name";
          return `{${tag}|${LABEL[c] ?? c}}  {val|${v === null || v === undefined ? "–" : signedUsd(v)}}`;
        },
        rich: {
          name: { color: t.text2, fontSize: 12.5, fontFamily: t.font },
          edge: { color: t.text, fontSize: 12.5, fontWeight: 700, fontFamily: t.font },
          val: { color: t.muted, fontSize: 12, fontFamily: t.font, width: 58, align: "right" },
        },
      },
    },
    xAxis: { type: "value", name: "$ per episode, against the same arm at 0", nameLocation: "middle", ...axis(t), nameGap: 28 },
    series: [
      {
        id: "components",
        type: "bar",
        barWidth: "56%",
        data: cats.map((c) => ({
          value: get(c)?.estimate ?? null,
          itemStyle: c === "pnl" ? { color: t.text, opacity: 0.88, borderRadius: 2 } : { color: componentColor(t, c), borderRadius: 2, borderColor: c === "residual" ? t.text2 : undefined, borderWidth: c === "residual" ? 0.6 : 0 },
        })),
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          data: [
            { xAxis: 0, lineStyle: { color: t.axis, type: "solid", width: 1 } },
            ...(pnl !== null ? [{ xAxis: pnl, lineStyle: { color: t.text, type: "dashed" as const, width: 1, opacity: 0.55 } }] : []),
          ],
        },
      },
      {
        id: "intervals",
        type: "custom",
        renderItem: whisker(t.text2, true),
        encode: { x: [1, 2], y: 0 },
        data: cats.map((c, i) => [i, get(c)?.lower_bound ?? null, get(c)?.upper_bound ?? null]).filter((r) => r[1] !== null && r[2] !== null),
        tooltip: { show: false },
        silent: true,
        z: 4,
      },
    ],
  };
}

/** The same split at every premium: components stacked by sign, the P&L edge with its whisker. */
function premiumsOption(t: Tokens, edges: ArmEdge[], comps: string[]): EChartsOption {
  const cats = edges.map((e) => num(e.premium, 2));
  return {
    ...base(t),
    legend: { ...base(t).legend, data: [...comps.map((c) => LABEL[c] ?? c), "P&L edge"] },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: number | null; dataIndex: number }[];
        const e = edges[arr[0]?.dataIndex ?? 0];
        return [`<b>${esc(cats[arr[0]?.dataIndex ?? 0])} vol pts</b>`, ...comps.map((c) => `${esc(LABEL[c] ?? c)}: ${usd(e?.components[c]?.estimate)}`), `<b>P&L edge ${usd(e?.pnl.estimate)}</b> (≥ ${usd(e?.pnl.lower_bound)})`].join("<br/>");
      },
    },
    grid: { ...base(t).grid, left: 74, top: 44 },
    xAxis: { type: "category", data: cats, name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Edge ($ per episode)", nameLocation: "middle", ...axis(t), nameGap: 58 },
    series: [
      ...comps.map((c) => ({
        id: c,
        name: LABEL[c] ?? c,
        type: "bar" as const,
        stack: "edge",
        barCategoryGap: "35%",
        data: edges.map((e) => e.components[c]?.estimate ?? null),
        itemStyle: { color: componentColor(t, c), borderColor: c === "residual" ? t.text2 : undefined, borderWidth: c === "residual" ? 0.6 : 0 },
      })),
      { id: "pnl", name: "P&L edge", type: "scatter" as const, data: edges.map((e) => e.pnl.estimate), symbol: "diamond", symbolSize: 11, itemStyle: { color: t.text, borderColor: t.surface, borderWidth: 1.5 }, z: 5 },
      {
        id: "pnl-interval",
        name: "P&L edge",
        type: "custom" as const,
        renderItem: whisker(t.text, false),
        encode: { x: 0, y: [1, 2] },
        data: edges.map((e, i) => [i, e.pnl.lower_bound, e.pnl.upper_bound]).filter((r) => r[1] !== null && r[2] !== null),
        tooltip: { show: false },
        silent: true,
        z: 4,
      },
    ],
  };
}

/** The edge split by the structure held over each step: components stacked, P&L marked. */
function structuresOption(t: Tokens, a: ArmEdge, structs: string[], comps: string[]): EChartsOption {
  const cats = [...structs].reverse();
  return {
    ...base(t),
    legend: { ...base(t).legend, data: [...comps.map((c) => LABEL[c] ?? c), "P&L edge"] },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (ps: unknown) => {
        const arr = ps as { dataIndex: number }[];
        const s = cats[arr[0]?.dataIndex ?? 0] ?? "";
        const b = a.by_structure[s] ?? {};
        return [`<b>${esc(STRUCT_LABEL[s] ?? s)}</b>`, ...comps.map((c) => `${esc(LABEL[c] ?? c)}: ${usd(b[c]?.estimate)}`), `<b>P&L edge ${usd(b.pnl?.estimate)}</b> (se ${usd(b.pnl?.se)})`].join("<br/>");
      },
    },
    grid: { ...base(t).grid, left: 118, top: 44, bottom: 44 },
    yAxis: { type: "category", data: cats.map((s) => STRUCT_LABEL[s] ?? s), ...axis(t), splitLine: { show: false }, axisLabel: { ...axis(t).axisLabel, color: t.text2, fontSize: 12 } },
    xAxis: { type: "value", name: "Edge ($ per episode)", nameLocation: "middle", ...axis(t), nameGap: 28 },
    series: [
      ...comps.map((c) => ({
        id: c,
        name: LABEL[c] ?? c,
        type: "bar" as const,
        stack: "edge",
        barWidth: "55%",
        data: cats.map((s) => a.by_structure[s]?.[c]?.estimate ?? null),
        itemStyle: { color: componentColor(t, c), borderColor: c === "residual" ? t.text2 : undefined, borderWidth: c === "residual" ? 0.6 : 0 },
      })),
      { id: "pnl", name: "P&L edge", type: "scatter" as const, data: cats.map((s) => a.by_structure[s]?.pnl?.estimate ?? null), symbol: "diamond", symbolSize: 11, itemStyle: { color: t.text, borderColor: t.surface, borderWidth: 1.5 }, z: 5 },
      {
        id: "pnl-interval",
        name: "P&L edge",
        type: "custom" as const,
        renderItem: whisker(t.text, true),
        encode: { x: [1, 2], y: 0 },
        data: cats.map((s, i) => [i, a.by_structure[s]?.pnl?.lower_bound ?? null, a.by_structure[s]?.pnl?.upper_bound ?? null]).filter((r) => r[1] !== null && r[2] !== null),
        tooltip: { show: false },
        silent: true,
        z: 4,
      },
    ],
  };
}

function EdgeCellText({ e }: { e: Edge | undefined }) {
  if (!e) return <>–</>;
  return <span style={{ fontWeight: carries(e) ? 600 : undefined }}>{usd(e.estimate)}</span>;
}

/** Components that are zero everywhere in the grid (vega and carry when implied vol is held) are left out. */
function shownComponents(g: GridEdge): { shown: string[]; zero: string[] } {
  const live = (c: string) => g.edges.some((e) => Math.abs(e.components[c]?.estimate ?? 0) >= 0.5);
  return { shown: g.components.filter((c) => c === "residual" || live(c)), zero: g.components.filter((c) => c !== "residual" && !live(c)) };
}

/** One arm's card: the sentence, then each component of its edge with its interval. */
function ArmEdgeCard({ g, a, comps, stamp }: { g: GridEdge; a: ArmEdge; comps: string[]; stamp: string }) {
  const t = useTokens();
  return (
    <Card
      title={
        <span className={lab.armHead}>
          <span className={lab.armDot} style={{ background: armColor(t, a.arm) }} aria-hidden="true" />
          {ARM_LABEL[a.arm] ?? a.arm}
        </span>
      }
      right={<Chip color={a.pnl_saved.rejected ? t.pass : t.s5}>{a.pnl_saved.rejected ? "detected" : "not detected"}</Chip>}
    >
      <div className={ui.stack}>
        <Reading>{edgeSentence(a, comps)}</Reading>
        <Chart
          option={edgeOption(t, a, comps)}
          height={44 + 34 * (comps.length + 1)}
          label={`Edge by component, ${ARM_LABEL[a.arm] ?? a.arm} at premium ${a.premium}`}
          stamp={stamp}
          rows={() => ["pnl", ...comps].map((c) => ({ arm: a.arm, premium: a.premium, component: c, ...(c === "pnl" ? a.pnl : a.components[c]), block_means: undefined }))}
          json={() => a}
          name={`edge ${a.arm} ${a.premium} ${g.grid_hash}`}
        />
      </div>
    </Card>
  );
}

function Detail({ d, arm, premium, setArm, setPremium }: { d: EdgeDetail; arm: string; premium: number; setArm: (a: string) => void; setPremium: (p: number) => void }) {
  const t = useTokens();
  const g = d.edge;
  useHashScroll(true);
  const { shown: comps, zero } = shownComponents(g);
  const at = (x: string) => g.edges.find((e) => e.arm === x && e.premium === premium) ?? g.edges.find((e) => e.arm === x);
  const a = at(arm) ?? g.edges[0];
  const stamp = stampOf(g);
  // held share of each structure at p and at 0, the mean over the arm's blocks
  const held = (p: number, s: string) => {
    const cs = g.cells.filter((c) => c.arm === a?.arm && c.premium === p);
    return cs.length ? cs.reduce((acc, c) => acc + (c.held_share[s] ?? 0), 0) / cs.length : null;
  };
  const [cellArm, setCellArm] = useState("");
  const cells = g.cells.filter((c) => !cellArm || c.arm === cellArm);
  const counts = { reproduces: 0, agrees: 0, differs: 0 } as Record<string, number>;
  for (const c of g.cells) counts[c.reproduction] = (counts[c.reproduction] ?? 0) + 1;
  const episodesLine = (c: (typeof g.cells)[number]) => {
    const n = d.episode_counts[c.run_id];
    return n ? episodesText(n.n_differ, n.n_episodes, 1) : "an episode differs";
  };
  if (!a) return <Empty title="No premium above zero in this grid" />;
  const armTabs = g.arms.map((x) => ({ value: x, label: ARM_LABEL[x] ?? x }));
  return (
    <>
      <div className={ui.grid2} {...guide("attribution-edge")}>
        {g.arms.map((x) => {
          const e = at(x);
          return e ? <ArmEdgeCard key={x} g={g} a={e} comps={comps} stamp={stamp} /> : null;
        })}
      </div>
      <p className={ui.dim} style={{ fontSize: 12.5, marginTop: -4 }}>
        <Labelled term="attribution.whisker">Bars: each greek&apos;s share of the edge, the arm at {prem(premium)} vol pts against itself at 0, over {g.n_seeds} seed blocks.</Labelled>{" "}
        Dashed line: the P&L edge the bars sum to.{zero.length ? ` ${zero.map((c) => LABEL[c] ?? c).join(" and ")}: zero throughout, not shown.` : ""} Each chart has its own scale.
      </p>

      <Card title="How the split changes with the premium" info="attribution.acrossPremiums" sub="Click a premium to read it above.">
        <div className={ui.grid2}>
          {g.arms.map((x) => {
            const armEdges = g.edges.filter((e) => e.arm === x).sort((p, q) => p.premium - q.premium);
            return (
              <div key={x}>
                <p className={lab.armHead}>
                  <span className={lab.armDot} style={{ background: armColor(t, x) }} aria-hidden="true" />
                  {ARM_LABEL[x] ?? x}
                </p>
                <Chart
                  option={premiumsOption(t, armEdges, comps)}
                  height={300}
                  label={`Edge by component at each premium, ${x}`}
                  stamp={stamp}
                  onClick={(p) => {
                    const e = armEdges[p.dataIndex];
                    if (e) setPremium(e.premium);
                  }}
                  rows={() => armEdges.map((e) => ({ arm: e.arm, premium: e.premium, pnl: e.pnl.estimate, pnl_se: e.pnl.se, pnl_lower: e.pnl.lower_bound, ...Object.fromEntries(comps.flatMap((c) => [[c, e.components[c]?.estimate ?? null], [`${c}_se`, e.components[c]?.se ?? null]])) }))}
                />
              </div>
            );
          })}
        </div>
      </Card>

      {g.arms.length > 1 ? (
        <div className={lab.bar}>
          <Segmented label="Arm for the details" title="Details for" term="attribution.arm" guide="attribution-arm" value={a.arm} options={armTabs} onChange={setArm} />
        </div>
      ) : null}

      <Fold id="structures" title="By structure held" term="attribution.structures" sub={`${ARM_LABEL[a.arm] ?? a.arm} at ${prem(a.premium)} vol pts: each step's terms charged to the structure held over its move.`}>
        <Chart option={structuresOption(t, a, g.structures, comps)} height={280} label={`Edge by structure held, ${a.arm} at premium ${a.premium}`} stamp={stamp} rows={() => g.structures.flatMap((s) => ["pnl", ...comps].map((c) => ({ structure: s, component: c, estimate: a.by_structure[s]?.[c]?.estimate ?? null, se: a.by_structure[s]?.[c]?.se ?? null })))} />
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Structure</th>
                <th className={ui.num}>Held at {prem(a.premium)}</th>
                <th className={ui.num}>Held at 0</th>
                <th className={ui.num}>P&L edge</th>
                <th className={ui.num}>SE</th>
                <th className={ui.num}>Lower</th>
                <th className={ui.num}>Theta</th>
                <th className={ui.num}>Gamma</th>
                <th className={ui.num}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {g.structures.map((s) => {
                const b = a.by_structure[s] ?? {};
                return (
                  <tr key={s}>
                    <td>{STRUCT_LABEL[s] ?? s}</td>
                    <td className={ui.num}>{pct(held(a.premium, s), 1)}</td>
                    <td className={ui.num}>{pct(held(0, s), 1)}</td>
                    <td className={ui.num}>
                      <EdgeCellText e={b.pnl} />
                    </td>
                    <td className={ui.num}>{usd(b.pnl?.se)}</td>
                    <td className={ui.num}>{usd(b.pnl?.lower_bound)}</td>
                    <td className={ui.num}>
                      <EdgeCellText e={b.theta} />
                    </td>
                    <td className={ui.num}>
                      <EdgeCellText e={b.gamma} />
                    </td>
                    <td className={ui.num}>
                      <EdgeCellText e={b.cost} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className={ui.dim} style={{ fontSize: 12.5 }}>
          Structures sum to the P&L edge within {usd(Math.abs(a.structure_gap ?? 0))}; components within {usd(Math.abs(a.identity_gap ?? 0))}. Carried by: {carriers(a).join(", ") || "none"}; residual {pct(a.residual_edge_share, 2)} of the edge.
        </p>
      </Fold>

      <Fold title="Every arm and premium" term="attribution.table" sub={`One-sided t on ${g.n_seeds} block means, critical ${num(g.critical, 3)}. Bold: the interval excludes 0. Click a row to read it above.`}>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Arm</th>
                <th className={ui.num}>Premium</th>
                <th className={ui.num}>P&L edge</th>
                <th className={ui.num}>SE</th>
                <th className={ui.num}>Lower</th>
                <th className={ui.num}>Re-simulated</th>
                {comps.map((c) => (
                  <th key={c} className={ui.num}>
                    {LABEL[c]}
                  </th>
                ))}
                <th className={ui.num}>Residual / edge</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {g.edges.map((e) => (
                <tr
                  key={`${e.arm}-${e.premium}`}
                  className={`${ui.clickRow} ${e.arm === a.arm && e.premium === a.premium ? ui.rowSelected : ""}`}
                  onClick={() => {
                    setArm(e.arm);
                    setPremium(e.premium);
                  }}
                >
                  <td>{ARM_LABEL[e.arm] ?? e.arm}</td>
                  <td className={ui.num}>{num(e.premium, 2)}</td>
                  <td className={ui.num}>
                    <EdgeCellText e={e.pnl_saved} />
                  </td>
                  <td className={ui.num}>{usd(e.pnl_saved.se)}</td>
                  <td className={ui.num}>{usd(e.pnl_saved.lower_bound)}</td>
                  <td className={ui.num} title={`se ${usd(e.pnl.se)} · ≥ ${usd(e.pnl.lower_bound)}`}>
                    <EdgeCellText e={e.pnl} />
                  </td>
                  {comps.map((c) => (
                    <td key={c} className={ui.num} title={`se ${usd(e.components[c]?.se)}`}>
                      <EdgeCellText e={e.components[c]} />
                    </td>
                  ))}
                  <td className={ui.num}>{pct(e.residual_edge_share, 2)}</td>
                  <td>
                    <Chip color={e.pnl_saved.rejected ? t.pass : t.s5}>{e.pnl_saved.rejected ? "detected" : "not detected"}</Chip>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={ui.dim} style={{ fontSize: 12.5 }}>
          P&L edge: from run.json, as Sweep and the analysis read it. Re-simulated and the components: from the attribution&apos;s own re-simulation, which they sum to.
        </p>
        <FigureFoot name={`edge ${g.grid_hash}`} stamp={stamp} rows={() => g.edges.flatMap((e) => ["pnl", ...comps].map((c) => { const x = c === "pnl" ? e.pnl : e.components[c]; return { arm: e.arm, premium: e.premium, component: c, estimate: x?.estimate ?? null, se: x?.se ?? null, lower_bound: x?.lower_bound ?? null, upper_bound: x?.upper_bound ?? null, rejected: x?.rejected ?? null }; }))} json={() => g.edges} />
      </Fold>

      <Fold
        id="cells"
        title="Cells and reproduction"
        term="attribution.reproduction"
        sub={`${g.cells.length} runs re-simulated: ${counts.reproduces} bit for bit, ${counts.agrees} within 1e-9, ${counts.differs} with an episode beyond · max |ΔP&L| ${usd(g.max_abs_pnl_diff)}`}
      >
        {g.arms.length > 1 ? (
          <div className={lab.bar}>
            <span className={ui.fieldLabel}>Arm</span>
            <Segmented label="Arm for the cells" value={cellArm} options={[{ value: "", label: "All" }, ...armTabs]} onChange={setCellArm} />
          </div>
        ) : null}
        <div className={ui.tableWrap} style={{ maxHeight: 520, overflow: "auto" }}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Arm</th>
                <th className={ui.num}>Premium</th>
                <th className={ui.num}>Block</th>
                <th className={ui.num}>Mean P&L</th>
                <th className={ui.num}>Residual share</th>
                <th className={ui.num}>Max |ΔP&L|</th>
                <th>Reproduction</th>
                <th>Run</th>
              </tr>
            </thead>
            <tbody>
              {cells.map((c) => {
                const key = d.run_keys[c.run_id];
                return (
                  <tr key={c.run_id}>
                    <td>{ARM_LABEL[c.arm] ?? c.arm}</td>
                    <td className={ui.num}>{num(c.premium, 2)}</td>
                    <td className={ui.num}>{c.block}</td>
                    <td className={ui.num}>{usd(c.mean.pnl)}</td>
                    <td className={ui.num}>{pct(c.residual_share, 2)}</td>
                    <td className={ui.num}>{c.max_abs_pnl_diff !== null && c.max_abs_pnl_diff < 0.01 ? c.max_abs_pnl_diff.toExponential(1) : usd(c.max_abs_pnl_diff)}</td>
                    <td>
                      <Verdict value={VERDICT[c.reproduction] ?? c.reproduction} /> {c.reproduction === "differs" ? <span className={ui.dim}>{episodesLine(c)}</span> : null}
                    </td>
                    <td>
                      {key ? (
                        <Link href={`/runs/?key=${encodeURIComponent(key)}#attribution`} className="mono">
                          {c.run_id.slice(0, 16)}
                        </Link>
                      ) : (
                        <span className={`${ui.dim} mono`} title="not under this runs root">
                          {c.run_id.slice(0, 16)} · absent
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <FigureFoot name={`edge cells ${g.grid_hash}`} stamp={`${stamp} · ${d.folder}${d.name}`} rows={() => cells.map((c) => ({ arm: c.arm, premium: c.premium, block: c.block, run_id: c.run_id, run_key: d.run_keys[c.run_id] ?? "", reproduction: c.reproduction, max_abs_pnl_diff: c.max_abs_pnl_diff, residual_share: c.residual_share, gross: c.gross, ...c.mean }))} />
      </Fold>
    </>
  );
}

export default function AttributionPage() {
  return (
    <Suspense fallback={<Loading what="the attribution" />}>
      <Attribution />
    </Suspense>
  );
}

function NoEdge({ list }: { list: EdgeList }) {
  const missing = list.grids.filter((g) => !g.has_edge);
  return (
    <Pending title="No edge attribution under the attribution root">
      {missing.length ? (
        <ul>
          {missing.map((g) => (
            <li key={g.grid_hash}>
              <code>{g.edge_command}</code>
            </li>
          ))}
        </ul>
      ) : (
        <p>
          <code>{list.command}</code>
        </p>
      )}
      <p className={ui.dim}>
        Written to <code>$OPTSPREAD_ATTRIBUTION/{list.folder}</code> (default <code>attribution/{list.folder}</code>).
      </p>
    </Pending>
  );
}

function Attribution() {
  const q = useQuery();
  const list = useFetch((s) => api.edges(s), "edges");
  const [chosen, setChosen] = useState(() => q.str("file"));
  const [arm, setArm] = useState(() => q.str("arm"));
  const [premium, setPremium] = useState(() => q.num("p", NaN));
  const files = (list.data?.files ?? []).filter((f) => !f.error);
  const file = files.find((f) => f.name === chosen) ?? files.find((f) => f.grid_hash === chosen) ?? files[0];
  const detail = useFetch((s) => api.edge(file?.grid_hash ?? "", file?.first ?? null, s), file?.name ?? "", Boolean(file));
  const g = detail.data?.edge;
  const armNow = g?.arms.includes(arm) ? arm : (g?.arms[0] ?? "");
  const pos = (g?.premiums ?? file?.premiums ?? []).filter((p) => p > 0);
  // one vol point by default: the size of the real premium, and where the seller's edge reads cleanly
  const premNow = pos.includes(premium) ? premium : pos.includes(1) ? 1 : (pos[pos.length - 1] ?? NaN);
  useQuerySync({ file: chosen || null, arm: arm || null, p: Number.isFinite(premium) ? premium : null });
  // the action mix is its own file: shown for the grid read, or the newest grid under the root
  const grid = list.data?.grids.find((x) => x.grid_hash === file?.grid_hash) ?? (file ? undefined : list.data?.grids[0]);
  const mixGrid = file?.grid_hash ?? grid?.grid_hash ?? "";
  const mixCommand = detail.data?.actionmix_command ?? grid?.actionmix_command ?? 'make actionmix ARGS="--grid <runs root>/grid-<hash>.json --workers 2"';
  return (
    <>
      <PageHead eyebrow="Attribution" title="Where does the edge come from?" />
      {files.length ? (
        <Card>
          <div className={lab.bar}>
            <div className={lab.barItem}>
              <Segmented label="Planted premium" title="Planted premium, vol pts" term="premium" guide="attribution-premium" value={String(premNow)} options={pos.map((p) => ({ value: String(p), label: prem(p) }))} onChange={(v) => setPremium(Number(v))} />
            </div>
            <div className={lab.barItem}>
              <SelectField
                id="attribution-grid"
                label="Grid"
                term="attribution.grid"
                guide="attribution-grid"
                value={file?.name ?? ""}
                options={files.map((f) => f.name)}
                render={(n) => {
                  const f = files.find((x) => x.name === n);
                  return f ? `${f.grid_hash}${f.first ? ` · first ${f.first}` : ""} · ${f.source}` : n;
                }}
                onChange={(v) => v && setChosen(v)}
              />
            </div>
            {g ? (
              <span className={lab.barNote}>
                <Labelled term="attribution.edge">
                  Edge: each arm at the premium minus itself at 0, {g.n_seeds} seed blocks of {num(g.n_episodes)} episodes
                </Labelled>
                <Source>{`${stampOf(g)} · ${detail.data?.folder ?? ""}${detail.data?.name ?? ""}`}</Source>
              </span>
            ) : null}
          </div>
        </Card>
      ) : null}
      {list.error ? <ErrorBox what="the edge files" error={list.error} onRetry={list.reload} /> : null}
      {list.loading && !list.data ? <Loading what="the edge files" /> : null}
      {list.data && !files.length ? <NoEdge list={list.data} /> : null}
      {list.data?.files.some((f) => f.error) ? (
        <p className={ui.fieldError}>
          {list.data.files
            .filter((f) => f.error)
            .map((f) => `${f.name}: ${f.error}`)
            .join("; ")}
        </p>
      ) : null}
      {detail.error ? <ErrorBox what={`the edge of ${file?.grid_hash}`} error={detail.error} onRetry={detail.reload} /> : null}
      {file && detail.loading && !detail.data ? <Loading what={`the edge of ${file.grid_hash}`} /> : null}
      {detail.data ? <Detail d={detail.data} arm={armNow} premium={premNow} setArm={setArm} setPremium={setPremium} /> : null}
      {mixGrid ? <EdgeMix grid={mixGrid} command={mixCommand} episodesPerCell={g?.n_episodes} /> : null}
      {detail.data?.grid_key ? (
        <p className={ui.dim} style={{ fontSize: 12.5 }}>
          The P&L contrast itself: <Link href={`/sweep/?key=${encodeURIComponent(detail.data.grid_key)}&view=contrasts`}>Sweep · contrasts</Link>.
        </p>
      ) : null}
    </>
  );
}
