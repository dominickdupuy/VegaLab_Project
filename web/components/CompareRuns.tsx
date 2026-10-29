"use client";

import Link from "next/link";
import type { EChartsOption } from "echarts";
import { Card, Chip, PageHead, Tile, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { mixSentence, STRUCT_LABEL, structureColor, topStructure, useRunMix } from "@/components/ActionMix";
import { attributionSentence, componentsOption, equityOption, perEpisode, useAttribution, useCapital } from "@/components/RunExtras";
import { InfoTip } from "@/components/InfoTip";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { Fold, Reading } from "@/components/PageBits";
import { COMPONENTS, type MixSummary, type RunDetail, STRUCTURES } from "@/lib/api";
import { type PairedCompare, runsApi } from "@/lib/api_runs";
import { ARM_LABEL, armColor, axis, base, histBars, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, pct, prem, short, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

export function histOption(t: Tokens, edges: number[], counts: number[], color: string, name: string, xName = "P&L per episode ($)"): EChartsOption {
  const width = edges.length > 1 ? (edges[1] ?? 0) - (edges[0] ?? 0) : 1;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: {
      ...base(t).tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const v = (p as { value: [number, number] }).value;
        return `${usd(v[0] - width / 2)} to ${usd(v[0] + width / 2)}<br/><b>${v[1]}</b> episode${v[1] === 1 ? "" : "s"}`;
      },
    },
    xAxis: { type: "value", name: xName, nameLocation: "middle", scale: true, ...axis(t) },
    yAxis: { type: "value", name: "Episodes", nameLocation: "middle", minInterval: 1, ...axis(t), nameGap: 36 },
    series: [
      {
        name,
        type: "bar",
        barWidth: "92%",
        barCategoryGap: 0,
        itemStyle: { color, borderRadius: [3, 3, 0, 0] },
        data: histBars(edges, counts),
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: t.text2, type: "solid", width: 1 },
          label: { color: t.muted, formatter: "0" },
          data: [{ xAxis: 0 }],
        },
      },
    ],
  };
}

const COMPONENT_LABEL: Record<string, string> = { delta: "delta", gamma: "gamma", vega: "vega", theta: "theta", carry: "carry", cost: "cost", residual: "the residual" };

function name(d: RunDetail, label: string): string {
  if (label) return label;
  const where = d.row.source !== "synthetic" ? "real SPX" : `${prem(d.row.premium)} vol pts`;
  return `${ARM_LABEL[d.row.policy] ?? d.row.policy} · ${where}`;
}

function diffOption(t: Tokens, c: PairedCompare): EChartsOption {
  const lo = c.diff_ci_low;
  const hi = c.diff_ci_high;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, formatter: (ps: unknown) => { const i = (ps as { dataIndex: number }[])[0]?.dataIndex ?? 0; const x = c.diffs[i]; return x ? `<b>seed ${x.seed}</b><br/>A ${usd(x.a)} · B ${usd(x.b)}<br/>B − A <b>${usd(x.diff)}</b>` : ""; } },
    grid: { ...base(t).grid, left: 74, top: 20 },
    xAxis: { type: "category", data: c.diffs.map((x) => x.seed), name: "Episode seed (the same path in both)", nameLocation: "middle", ...axis(t), axisLabel: { ...axis(t).axisLabel, show: c.diffs.length <= 60 }, splitLine: { show: false } },
    yAxis: { type: "value", name: "B − A ($)", nameLocation: "middle", ...axis(t), nameGap: 58 },
    series: [
      {
        type: "bar",
        data: c.diffs.map((x) => ({ value: x.diff, itemStyle: { color: x.diff >= 0 ? t.s2 : t.s1, opacity: 0.75 } })),
        barCategoryGap: c.diffs.length > 100 ? "0%" : "20%",
        markLine: {
          symbol: "none",
          silent: true,
          data: [
            { yAxis: 0, lineStyle: { color: t.text2, width: 1, type: "solid" }, label: { show: false } },
            ...(c.diff_mean !== null ? [{ yAxis: c.diff_mean, lineStyle: { color: t.text, width: 2, type: "solid" as const }, label: { formatter: `mean ${usd(c.diff_mean)}`, color: t.text, position: "insideEndTop" as const } }] : []),
          ],
        },
        markArea: lo !== null && hi !== null ? { silent: true, itemStyle: { color: t.text, opacity: 0.08 }, data: [[{ yAxis: lo }, { yAxis: hi }]] } : undefined,
      },
    ],
  };
}

function heldOption(t: Tokens, rows: { name: string; m: MixSummary }[]): EChartsOption {
  return {
    ...base(t),
    legend: { ...base(t).legend, data: STRUCTURES.map((s) => STRUCT_LABEL[s] ?? s) },
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, formatter: (ps: unknown) => { const arr = ps as { seriesName: string; value: number; name: string }[]; return [`<b>${esc(arr[0]?.name)}</b>`, ...arr.map((p) => `${esc(p.seriesName)}: ${pct(p.value, 1)}`)].join("<br/>"); } },
    grid: { ...base(t).grid, left: 40, right: 20, top: 34, bottom: 30 },
    yAxis: { type: "category", data: rows.map((r) => r.name), inverse: true, ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: (v: string) => (v.length > 26 ? `${v.slice(0, 25)}…` : v) }, splitLine: { show: false } },
    xAxis: { type: "value", min: 0, max: 1, ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    series: STRUCTURES.map((s) => ({ name: STRUCT_LABEL[s] ?? s, type: "bar" as const, stack: "held", data: rows.map((r) => r.m.held_share[s] ?? 0), itemStyle: { color: structureColor(t, s) }, barCategoryGap: "40%" })),
  };
}

function MixRow({ label, m }: { label: string; m: MixSummary | undefined }) {
  const top = m ? topStructure(m) : undefined;
  const x = top && m ? m.structures[top] : undefined;
  return (
    <tr>
      <td>{label}</td>
      <td className={ui.num}>{m ? pct(1 - (m.held_share.FLAT ?? 0), 0) : "…"}</td>
      <td>{m ? (top ? STRUCT_LABEL[top] : "flat") : "…"}</td>
      <td className={ui.num}>{m ? num(m.opens_per_episode, 1) : "…"}</td>
      <td className={ui.num}>{x ? num(x.mean_tenor_days, 0) : "–"}</td>
      <td className={ui.num}>{x ? num(x.mean_short_abs_delta, 2) : "–"}</td>
      <td className={ui.num}>{m?.holding_days ? num(m.holding_days.p50, 0) : "–"}</td>
      <td className={ui.num}>{m ? num(m.n_refused) : "…"}</td>
    </tr>
  );
}

function Paired({ c }: { c: PairedCompare }) {
  const t = useTokens();
  const na = name(c.a, c.a_label);
  const nb = name(c.b, c.b_label);
  const capA = useCapital(c.a);
  const capB = useCapital(c.b);
  const attA = useAttribution(c.a.row.key);
  const attB = useAttribution(c.b.row.key);
  const mixA = useRunMix(c.a.row.key);
  const mixB = useRunMix(c.b.row.key);
  const ca = capA.data?.state === "done" ? capA.data : undefined;
  const cb = capB.data?.state === "done" ? capB.data : undefined;
  const pathsA = c.a.equity?.paths ?? ca?.paths ?? [];
  const pathsB = c.b.equity?.paths ?? cb?.paths ?? [];
  const ra = attA.data?.result ?? undefined;
  const rb = attB.data?.result ?? undefined;
  const ma = mixA.data?.mix?.summary;
  const mb = mixB.data?.mix?.summary;
  const colorA = t.s1;
  const colorB = t.s2;
  // the component that carries the gap
  let gap = "";
  if (ra && rb) {
    const pa = perEpisode(ra);
    const pb = perEpisode(rb);
    const by = COMPONENTS.map((k) => ({ k, d: (pb[k] ?? 0) - (pa[k] ?? 0) })).sort((x, y) => Math.abs(y.d) - Math.abs(x.d));
    const top = by[0];
    if (top) gap = `Most of the gap is ${COMPONENT_LABEL[top.k] ?? top.k}: B ${top.d >= 0 ? "gains" : "loses"} ${usd(Math.abs(top.d))} per episode there against A${by[1] && Math.abs(by[1].d) > 0.25 * Math.abs(top.d) ? `, then ${COMPONENT_LABEL[by[1].k] ?? by[1].k} (${by[1].d >= 0 ? "+" : "−"}${usd(Math.abs(by[1].d))})` : ""}.`;
  }
  const waiting = (x: { data?: { state: string; can_compute: boolean } }) => !x.data || x.data.state === "computing" || (x.data.state === "absent" && x.data.can_compute);
  const sharedNote = c.shared_seeds < Math.min(c.a.episodes.length, c.b.episodes.length) ? ` of ${Math.min(c.a.episodes.length, c.b.episodes.length)}` : "";
  return (
    <>
      <PageHead
        eyebrow={c.question ? "Compare · answering" : "Compare, paired on shared seeds"}
        title={c.question || `${nb} against ${na}`}
        right={
          <Link href="/runs/" className={`${ui.btn} ${ui.btnSmall}`}>
            ← All runs
          </Link>
        }
      />
      <Reading>{c.verdict}</Reading>
      <div className={ui.grid4}>
        <Tile
          label="Paired difference, B − A"
          info="paired_contrast"
          value={`${c.diff_mean !== null && c.diff_mean > 0 ? "+" : ""}${usd(c.diff_mean)}`}
          note={
            <>
              per episode, ± {c.diff_ci_low !== null && c.diff_ci_high !== null ? usd((c.diff_ci_high - c.diff_ci_low) / 2) : "–"} (95%){" "}
              {c.diff_ci_low !== null && c.diff_ci_high !== null ? (c.diff_ci_low > 0 || c.diff_ci_high < 0 ? "· clear" : "· within noise") : ""}
            </>
          }
        />
        <Tile label="B ahead on" value={pct(c.b_better_share, 0)} note="of the shared seeds" />
        <Tile label="Shared seeds" value={`${num(c.shared_seeds)}${sharedNote}`} note="the same paths in both" info="paired_seeds" />
        <Tile label="What differs" value={num(c.config_diff.length)} note={c.config_diff.slice(0, 3).map((x) => x.key.split(".").pop()).join(", ") || "identical configs"} />
      </div>
      <Card
        title="P&L difference by seed"
      info="paired_seeds"
        sub={`B − A on each shared path: the market's luck is the same in both, so it cancels. The line is the mean, the shaded band its 95% interval (${usd(c.diff_ci_low)} to ${usd(c.diff_ci_high)}).`}
      >
        {c.diffs.length ? (
          <div className={ui.grid2}>
            <Chart option={diffOption(t, c)} height={280} label="Per-seed difference B minus A" rows={() => c.diffs.map((x) => ({ ...x }))} />
            {c.diff_histogram.counts.length ? <Chart option={histOption(t, c.diff_histogram.edges, c.diff_histogram.counts, t.s4, "differences", "B − A per seed ($)")} height={280} label="Histogram of paired differences" /> : null}
          </div>
        ) : (
          <Empty title="No shared seeds">
            <p>Runs pair only on the seeds both ran.</p>
          </Empty>
        )}
      </Card>
      <p className={ui.groupTitle} style={{ margin: 0 }}>
        Each run on its own: its mean is noisy at this sample size; the paired difference above is the precise number
      </p>
      <div className={ui.grid2}>
        {(
          [
            ["A", c.a, na, colorA],
            ["B", c.b, nb, colorB],
          ] as const
        ).map(([k, d, n, col]) => (
          <Card
            key={k}
            title={
              <>
                <span className={ui.swatch} style={{ background: col, display: "inline-block", marginRight: 6 }} aria-hidden="true" />
                {k} · <Link href={`/runs/?key=${encodeURIComponent(d.row.key)}`}>{n}</Link>
              </>
            }
            right={<Verdict value={d.row.status} />}
          >
            <p className={ui.dim} style={{ margin: 0, fontSize: 13 }}>
              <Chip color={armColor(t, d.row.policy)}>{ARM_LABEL[d.row.policy] ?? d.row.policy}</Chip> mean {usd(d.risk.mean)} (se {usd(d.risk.se)}) · CVaR 5% {usd(d.risk.cvar5)} · {num(d.risk.turnover, 1)} trades/ep · <span className="mono">{short(d.row.config_hash, 10)}-s{d.row.seed}</span>
            </p>
          </Card>
        ))}
      </div>
      <Card title="Equity paths, overlaid" sub="The band is the 10th to 90th percentile of every episode, the line the mean; faint lines are single episodes.">
        {pathsA.length && pathsB.length ? (
          <Chart option={equityOption(t, pathsA, ca?.equity_band ?? null, colorA, `A · ${na}`, { paths: pathsB, band: cb?.equity_band ?? null, color: colorB, name: `B · ${nb}` })} height={320} label="Equity paths of both runs overlaid" />
        ) : waiting(capA) || waiting(capB) ? (
          <Loading what="both runs' equity paths by re-simulation" />
        ) : (
          <Pending title="Equity paths are not available for both runs">{capA.data?.reason || capB.data?.reason}</Pending>
        )}
      </Card>
      <Card
        title="Where each run's P&L comes from"
      info="attribution"
        sub={gap || undefined}
      >
        {ra && rb ? (
          <div className={ui.stack}>
            <Chart option={componentsOption(t, [{ name: `A · ${na}`, color: colorA, v: perEpisode(ra) }, { name: `B · ${nb}`, color: colorB, v: perEpisode(rb) }])} height={300} label="Mean P&L per episode by component, both runs" />
            <div className={ui.grid2}>
              <p className={ui.dim} style={{ margin: 0 }}>
                A: {attributionSentence(ra)}
              </p>
              <p className={ui.dim} style={{ margin: 0 }}>
                B: {attributionSentence(rb)}
              </p>
            </div>
          </div>
        ) : attA.data?.state === "absent" && !attA.data.can_compute ? (
          <Pending title="Not attributed here">{attA.data.reason}.</Pending>
        ) : attB.data?.state === "absent" && !attB.data.can_compute ? (
          <Pending title="Not attributed here">{attB.data.reason}.</Pending>
        ) : attA.data?.state === "failed" || attB.data?.state === "failed" ? (
          <Pending title="The attribution failed">{attA.data?.reason || attB.data?.reason}</Pending>
        ) : (
          <Loading what="both attributions" />
        )}
      </Card>
      <Card
        title="What each traded"
      info="action_mix"
        sub={ma && mb ? `A: ${mixSentence(ma)} B: ${mixSentence(mb)}` : undefined}
      >
        {ma || mb ? (
          <div className={ui.stack}>
            {ma && mb ? (
              <Chart
                option={heldOption(t, [
                  { name: `A · ${na}`, m: ma },
                  { name: `B · ${nb}`, m: mb },
                ])}
                height={150}
                label="Share of days each structure was held, both runs"
              />
            ) : null}
            <div className={ui.tableWrap}>
              <table className={ui.table}>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th className={ui.num}>Held</th>
                    <th>Mostly</th>
                    <th className={ui.num}>Opens/ep</th>
                    <th className={ui.num}>Tenor (d)</th>
                    <th className={ui.num}>|Δ| short</th>
                    <th className={ui.num}>Days held p50</th>
                    <th className={ui.num}>Refused</th>
                  </tr>
                </thead>
                <tbody>
                  <MixRow label={`A · ${na}`} m={ma} />
                  <MixRow label={`B · ${nb}`} m={mb} />
                </tbody>
              </table>
            </div>
          </div>
        ) : mixA.data?.state === "absent" && !mixA.data.can_compute ? (
          <Pending title="Not re-simulated here">{mixA.data.reason}.</Pending>
        ) : (
          <Loading what="both action mixes" />
        )}
      </Card>
      <Fold title={`Every config field that differs (${c.config_diff.length})`}>
        <div className={ui.tableWrap}>
          {c.config_diff.length ? (
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>A</th>
                  <th>B</th>
                </tr>
              </thead>
              <tbody>
                {c.config_diff.map((r) => (
                  <tr key={r.key}>
                    <td className="mono">{r.key}</td>
                    <td>{JSON.stringify(r.a)}</td>
                    <td>{JSON.stringify(r.b)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty title="Identical configs" />
          )}
        </div>
      </Fold>
    </>
  );
}

/** Two runs on the same seeds: the paired difference, then where it comes from. */
export function CompareRuns({ a, b, group }: { a: string; b: string; group?: string }) {
  const cmp = useFetch((s) => runsApi.compare(a, b, group, s), `${a}|${b}|${group ?? ""}`);
  if (cmp.error) return <ErrorBox what="the comparison" error={cmp.error} onRetry={cmp.reload} />;
  if (!cmp.data) return <Loading what="the comparison" />;
  return <Paired key={`${a}|${b}`} c={cmp.data} />;
}
