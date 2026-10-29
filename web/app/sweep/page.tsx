"use client";

import Link from "next/link";
import { Suspense, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, Chip, PageHead, Segmented, SelectField, Source } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import lab from "@/components/lab.module.css";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { armPhrase, Fold, Reading } from "@/components/PageBits";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { M4View } from "@/components/views/M4View";
import { api, type GridDetail, type Sweep } from "@/lib/api";
import { ARM_LABEL, armColor, axis, base, type PremiumPoint, premiumChart, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, prem, short, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { useQuery, useQuerySync } from "@/lib/url";

const VIEWS = [
  { value: "cells", label: "Means" },
  { value: "contrasts", label: "Contrasts" },
  { value: "power", label: "Power" },
] as const;
type View = (typeof VIEWS)[number]["value"];

function oracleCurves(s: Sweep): Record<string, PremiumPoint[]> {
  return Object.fromEntries(
    Object.entries(s.oracle).map(([arm, pts]) => [arm, pts.map((p) => ({ premium: p.premium, mean: p.mean_pnl, se: p.se_pnl, label: s.kind === "grid" ? `${s.summary.n_seeds} blocks` : p.run_id }))]),
  );
}

function sweepStamp(s: Sweep): string {
  const g = s.grid;
  return g ? `grid ${g.grid_hash} · git ${g.git_shas.map((x) => short(x, 10)).join(", ") || "?"} · ${g.n_seeds} × ${num(g.n_episodes)} · synthetic` : `sweep ${s.summary.sweep_hash} · seeds ${s.summary.seed}… · synthetic`;
}

/** Each block's contrast mean at each premium, with the grand mean: tau^2 made visible. */
function blocksOption(t: Tokens, g: GridDetail, arms: string[]): EChartsOption {
  return {
    ...base(t),
    legend: { ...base(t).legend, data: arms.map((a) => ARM_LABEL[a] ?? a) },
    tooltip: { ...base(t).tooltip, trigger: "item", formatter: (p: unknown) => { const q = p as { seriesName: string; value: [number, number] }; return `${esc(q.seriesName)}<br/>${q.value[0]} vol pts · ${usd(q.value[1])}`; } },
    xAxis: { type: "value", name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "Contrast, arm at p − arm at 0 ($)", nameLocation: "middle", ...axis(t), nameGap: 58, scale: true },
    grid: { ...base(t).grid, left: 76 },
    series: arms.flatMap((arm) => {
      const color = armColor(t, arm);
      const cs = (g.contrasts[arm] ?? []).filter((c) => c.estimate !== null);
      return [
        { name: ARM_LABEL[arm] ?? arm, type: "scatter" as const, data: cs.flatMap((c) => c.block_means.filter((b): b is number => b !== null).map((b) => [c.premium, b])), symbolSize: 7, itemStyle: { color, opacity: 0.55 } },
        { name: ARM_LABEL[arm] ?? arm, type: "line" as const, data: cs.map((c) => [c.premium, c.estimate]), symbolSize: 9, lineStyle: { color, width: 2 }, itemStyle: { color, borderColor: t.surface, borderWidth: 2 }, z: 3 },
      ];
    }),
  };
}

function GridCells({ s, arm }: { s: Sweep; arm: string }) {
  const g = s.grid as GridDetail;
  const cells = g.cells.filter((c) => !arm || c.arm === arm);
  return (
    <>
      {g.missing.length ? (
        <Pending title={`${g.missing.length} cell(s) not under this runs root`}>
          <p>{g.missing.slice(0, 4).join("; ")}{g.missing.length > 4 ? "; …" : ""}</p>
        </Pending>
      ) : null}
      <div className={ui.tableWrap} style={{ maxHeight: 520, overflow: "auto" }}>
        <table className={ui.table}>
          <thead>
            <tr>
              <th>Arm</th>
              <th className={ui.num}>Premium</th>
              <th className={ui.num}>Block</th>
              <th className={ui.num}>Mean P&L</th>
              <th className={ui.num}>SE</th>
              <th className={ui.num}>Cost/ep</th>
              <th className={ui.num}>Trades/ep</th>
              <th>Run</th>
            </tr>
          </thead>
          <tbody>
            {cells.map((c) => (
              <tr key={c.run_id}>
                <td>{ARM_LABEL[c.arm] ?? c.arm}</td>
                <td className={ui.num}>{num(c.premium, 2)}</td>
                <td className={ui.num}>{c.block}</td>
                <td className={ui.num}>{usd(c.mean_pnl)}</td>
                <td className={ui.num}>{usd(c.se_pnl)}</td>
                <td className={ui.num}>{usd(c.mean_cost)}</td>
                <td className={ui.num}>{num(c.mean_trades, 1)}</td>
                <td>
                  {c.present ? (
                    <Link href={`/runs/?key=${encodeURIComponent(c.key)}`} className="mono">
                      {short(c.run_id, 16)}
                    </Link>
                  ) : (
                    <span className={`${ui.dim} mono`}>{short(c.run_id, 16)} · missing</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <FigureFoot name={`cells ${g.grid_hash}`} stamp={`${sweepStamp(s)} · ${s.source}`} rows={() => cells.map((c) => ({ ...c }))} />
    </>
  );
}

function ScriptedCells({ s }: { s: Sweep }) {
  const rows = Object.entries(s.oracle).flatMap(([arm, pts]) => pts.map((p) => ({ arm, ...p })));
  return (
    <>
      <div className={ui.tableWrap}>
        <table className={ui.table}>
          <thead>
            <tr>
              <th>Arm</th>
              <th className={ui.num}>Premium</th>
              <th className={ui.num}>Mean P&L</th>
              <th className={ui.num}>SE</th>
              <th className={ui.num}>Lift</th>
              <th>Run</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={`${p.arm}-${p.premium}`}>
                <td>{ARM_LABEL[p.arm] ?? p.arm}</td>
                <td className={ui.num}>{prem(p.premium)}</td>
                <td className={ui.num}>{usd(p.mean_pnl)}</td>
                <td className={ui.num}>{usd(p.se_pnl)}</td>
                <td className={ui.num}>{usd(p.lift)}</td>
                <td>
                  <Link href={`/runs/?q=${p.run_id}`} className="mono">
                    {short(p.run_id, 14)}…
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <FigureFoot name={`cells ${s.summary.sweep_hash}`} stamp={sweepStamp(s)} rows={() => rows} />
    </>
  );
}

/** The smallest premium from which every higher premium is detected, per arm, as a phrase. */
function detectedFrom(g: GridDetail, arm: string): string {
  const name = armPhrase(arm);
  const cs = (g.contrasts[arm] ?? []).filter((c) => c.estimate !== null && c.premium > 0).sort((a, b) => a.premium - b.premium);
  if (!cs.length) return `${name}: no contrast`;
  let from: number | null = null;
  for (let k = cs.length - 1; k >= 0; k--) {
    if (cs[k]?.detected) from = cs[k]?.premium ?? null;
    else break;
  }
  if (from === null) return `${name}'s edge at no premium`;
  return from === cs[0]?.premium ? `${name}'s edge at every premium` : `${name}'s edge from ${prem(from)} vol pt${from === 1 ? "" : "s"} up`;
}

function Contrasts({ s, arm }: { s: Sweep; arm: string }) {
  const t = useTokens();
  const g = s.grid;
  if (!g)
    return (
      <Card title="Contrasts">
        <Empty title="Contrasts need seed blocks">
          <p>A scripted sweep is one block on shared seeds; an orchestrated grid has S of them.</p>
        </Empty>
      </Card>
    );
  const arms = arm ? [arm] : g.arms;
  const rows = arms.flatMap((a) => (g.contrasts[a] ?? []).map((c) => ({ arm: a, ...c })));
  const phrases = arms.map((a) => detectedFrom(g, a));
  return (
    <>
      <Card title="Each arm against itself at zero premium" info="sweep.contrasts">
        <div className={ui.stack}>
          <Reading note={`Each dot is one of ${g.n_seeds} seed blocks of ${num(g.n_episodes)} episodes; how far they scatter is the seed-to-seed noise that sets the floor.`}>
            The pre-registered test detects {phrases.join(", and ")}.
          </Reading>
          {rows.length ? (
            <Chart option={blocksOption(t, g, arms)} height={360} label="Per-block contrast means against planted premium, by arm" stamp={sweepStamp(s)} rows={() => rows.flatMap((c) => c.block_means.map((b, i) => ({ arm: c.arm, premium: c.premium, block: i, contrast_mean: b })))} />
          ) : (
            <Empty title="No contrasts: the grid needs a 0 premium and two blocks" />
          )}
        </div>
      </Card>
      <Fold title="The test at each premium" term="sweep.event" sub={`One-sided t on the block means, ${g.n_seeds - 1} df.`}>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Arm</th>
                <th className={ui.num}>Premium</th>
                <th className={ui.num}>Estimate</th>
                <th className={ui.num}>SE</th>
                <th className={ui.num}>t</th>
                <th className={ui.num}>Critical</th>
                <th className={ui.num}>Lower bound</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={`${c.arm}-${c.premium}`}>
                  <td>{ARM_LABEL[c.arm] ?? c.arm}</td>
                  <td className={ui.num}>{num(c.premium, 2)}</td>
                  <td className={ui.num}>{usd(c.estimate)}</td>
                  <td className={ui.num}>{usd(c.se)}</td>
                  <td className={ui.num}>{num(c.t, 2)}</td>
                  <td className={ui.num}>{num(c.critical, 3)}</td>
                  <td className={ui.num}>{usd(c.lower_bound)}</td>
                  <td>{c.reason ? <span className={ui.dim}>{c.reason}</span> : <Chip color={c.detected ? t.pass : t.s5}>{c.detected ? "detected" : "not detected"}</Chip>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <FigureFoot name={`contrasts ${g.grid_hash}`} stamp={sweepStamp(s)} rows={() => rows.map(({ block_means, ...c }) => ({ ...c, block_means: block_means.join(" ") }))} />
      </Fold>
    </>
  );
}

/** The means view's reading: how much each arm's P&L rises from no premium to the most. */
function meansReading(s: Sweep, arms: string[]) {
  const ends = arms
    .map((a) => {
      const pts = [...(s.oracle[a] ?? [])].sort((x, y) => x.premium - y.premium);
      return { a, lo: pts[0], hi: pts[pts.length - 1] };
    })
    .filter((x) => x.lo && x.hi && x.hi.premium > x.lo.premium);
  if (!ends.length) return null;
  const top = Math.max(...ends.map((x) => x.hi?.premium ?? 0));
  const bottom = Math.min(...ends.map((x) => x.lo?.premium ?? 0));
  const rise = ends.map((x) => `${armPhrase(x.a)}'s by ${usd((x.hi?.mean_pnl ?? 0) - (x.lo?.mean_pnl ?? 0))}`);
  const atZero = ends.map((x) => `${armPhrase(x.a)} ${usd(x.lo?.mean_pnl)}`);
  return {
    main: (
      <>
        From {prem(bottom)} to {prem(top)} vol pts of planted premium, mean P&L per episode rises: {rise.join(", ")}.
      </>
    ),
    note: `At ${prem(bottom)} vol pts: ${atZero.join(", ")} per episode${s.grid?.costs ? ", after costs" : ""}.`,
  };
}

function SweepView({ s, view, arm }: { s: Sweep; view: View; arm: string }) {
  const t = useTokens();
  const g = s.grid;
  const curves = oracleCurves(s);
  const shown = arm ? Object.fromEntries(Object.entries(curves).filter(([a]) => a === arm)) : curves;
  const read = meansReading(s, Object.keys(shown));
  if (view === "contrasts") return <Contrasts s={s} arm={arm} />;
  return (
    <>
      <Card title="Mean P&L against planted premium" info="sweep.means">
        <div className={ui.stack}>
          {read ? <Reading note={read.note}>{read.main}</Reading> : null}
          <Chart option={premiumChart(t, shown)} height={360} label="Mean P&L against planted premium, by arm" stamp={sweepStamp(s)} rows={() => Object.entries(s.oracle).flatMap(([a, pts]) => pts.map((p) => ({ arm: a, ...p })))} />
        </div>
      </Card>
      <Fold id="cells" title="Cells" term="sweep.cells" sub={g ? `${g.cells.length} runs${g.missing.length ? `, ${g.missing.length} missing under this runs root` : ""}` : "Lift: against the first premium, same arm."}>
        {g ? <GridCells s={s} arm={arm} /> : <ScriptedCells s={s} />}
      </Fold>
    </>
  );
}

export default function SweepPage() {
  return (
    <Suspense fallback={<Loading what="sweeps" />}>
      <Sweeps />
    </Suspense>
  );
}

/** What the grid is, in one line: its cells and design. */
function designLine(s: Sweep): string {
  const g = s.grid;
  const n = s.summary;
  return g
    ? `${num(n.n_cells)} runs: ${n.arms.length} arms × ${n.premiums.length} premiums × ${g.n_seeds} seed blocks of ${num(g.n_episodes)} episodes · costs ${g.costs ? "on" : "off"}`
    : `${num(n.n_cells)} runs: ${n.arms.length} arms × ${n.premiums.length} premiums on shared seeds`;
}

function Sweeps() {
  const q = useQuery();
  const list = useFetch((s) => api.sweeps(s), "sweeps");
  const [chosen, setChosen] = useState(() => q.str("key"));
  const [view, setView] = useState<View>(() => q.oneOf("view", ["cells", "contrasts", "power"] as const, "cells"));
  const [arm, setArm] = useState(() => q.str("arm"));
  const study = useFetch((s) => api.study(s), "study");
  const sweeps = list.data?.sweeps ?? [];
  // the confirmatory grid by default (the pilot and any other grid stay selectable),
  // else the last orchestrated grid, else the last scripted sweep
  const confirmatory = study.data ? sweeps.find((s) => s.kind === "grid" && s.sweep_hash === study.data?.grid_hash) : undefined;
  const fallback = confirmatory?.key ?? [...sweeps].reverse().find((s) => s.kind === "grid")?.key ?? sweeps[sweeps.length - 1]?.key ?? "";
  const key = sweeps.some((s) => s.key === chosen) ? chosen : study.loading && !study.data ? "" : fallback;
  useQuerySync({ key: chosen || null, view, arm }, { view: "cells" });
  const sweep = useFetch((s) => api.sweep(key, s), key, Boolean(key));
  const arms = sweeps.find((s) => s.key === key)?.arms ?? [];
  // the bar keeps every control in place: one that does not apply to a view is disabled, not removed
  const power = view === "power";
  const armNow = arms.includes(arm) ? arm : "";

  return (
    <>
      <PageHead eyebrow="Sweep" title="Does the edge show up in the grid?" />
      <Card>
        <div className={lab.bar}>
          <div className={lab.barItem}>
            <Segmented label="View" title="View" term="sweep.view" guide="sweep-view" value={view} options={VIEWS} onChange={setView} />
          </div>
          <div className={lab.barItem}>
            <SelectField id="sweep-arm" label="Arm" term={power ? "sweep.sweepPower" : "sweep.arm"} guide="sweep-arm" all="All arms" disabled={power} value={power ? "" : armNow} options={arms} render={(a) => ARM_LABEL[a] ?? a} onChange={setArm} />
          </div>
          <div className={lab.barItem}>
            <SelectField
              id="sweep-grid"
              label="Grid"
              term={power ? "sweep.sweepPower" : "sweep.sweep"}
              guide="sweep-grid"
              disabled={power}
              value={power ? (confirmatory?.key ?? key) : key}
              options={sweeps.map((s) => s.key)}
              render={(k) => {
                const s = sweeps.find((x) => x.key === k);
                return s ? `${s.key === confirmatory?.key ? "Confirmatory" : s.key.includes("pilot") ? "Pilot" : "Grid"} · ${s.sweep_hash.slice(0, 8)} · ${s.n_cells} runs` : k;
              }}
              onChange={(v) => v && setChosen(v)}
            />
          </div>
          {sweep.data && !power ? (
            <span className={lab.barNote}>
              {designLine(sweep.data)}
              <Source>{sweepStamp(sweep.data)}</Source>
            </span>
          ) : null}
        </div>
      </Card>
      {power ? (
        <M4View />
      ) : (
        <>
          {list.error ? <ErrorBox what="sweeps" error={list.error} onRetry={list.reload} /> : null}
          {list.loading ? <Loading what="sweeps" /> : null}
          {list.data && !sweeps.length ? (
            <Empty title="No sweep under this runs root">
              <p>
                <code>make sweep</code> writes <code>grid-&lt;hash&gt;.json</code> and its cells; serve that root with <code>OPTSPREAD_RUNS</code>.
              </p>
            </Empty>
          ) : null}
          {sweep.error ? <ErrorBox what="the sweep" error={sweep.error} onRetry={sweep.reload} /> : null}
          {key && !sweep.data && !sweep.error ? <Loading what="the sweep" /> : null}
          {sweep.data ? <SweepView s={sweep.data} view={view} arm={armNow} /> : null}
        </>
      )}
    </>
  );
}
