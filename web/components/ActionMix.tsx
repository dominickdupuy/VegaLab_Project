"use client";

import type { EChartsOption } from "echarts";
import { Card, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, type MixSummary, type RunDetail, STRUCTURES } from "@/lib/api";
import { type MixDetail, type MixView, runsApi } from "@/lib/api_runs";
import { useRunTask } from "@/lib/useRunTask";
import { ARM_LABEL, axis, base, runStamp, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, pct, usd } from "@/lib/format";

export const STRUCT_LABEL: Record<string, string> = {
  FLAT: "Flat",
  IRON_CONDOR: "Iron condor",
  SHORT_STRANGLE: "Short strangle",
  PUT_CREDIT_SPREAD: "Put credit spread",
};

/** One colour per structure, fixed by name like the arms' and the components'. */
export function structureColor(t: Tokens, s: string): string {
  const map: Record<string, string> = { FLAT: t.s5, IRON_CONDOR: t.s1, SHORT_STRANGLE: t.s4, PUT_CREDIT_SPREAD: t.s3 };
  return map[s] ?? t.s5;
}

/** Share of the episodes holding each structure over the move from each day. */
function byDayOption(t: Tokens, m: MixSummary, names: readonly string[]): EChartsOption {
  const days = m.by_day.map((d) => d.day);
  return {
    ...base(t),
    legend: { ...base(t).legend, data: names.map((n) => STRUCT_LABEL[n] ?? n) },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: number; axisValue: number; dataIndex: number }[];
        const d = m.by_day[arr[0]?.dataIndex ?? 0];
        return [`<b>Day ${esc(arr[0]?.axisValue)}</b>`, ...arr.map((p) => `${esc(p.seriesName)}: ${pct(p.value, 1)}`), d ? `opens ${d.opens} · refused ${d.refused}` : ""].join("<br/>");
      },
    },
    grid: { ...base(t).grid, left: 52, top: 40 },
    xAxis: { type: "category", data: days, name: "Day of episode", nameLocation: "middle", boundaryGap: false, ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", min: 0, max: 1, name: "Share held", nameLocation: "middle", ...axis(t), nameGap: 38, axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    series: names.map((n, id) => ({
      name: STRUCT_LABEL[n] ?? n,
      type: "line" as const,
      stack: "held",
      areaStyle: { opacity: 0.85 },
      symbol: "none",
      lineStyle: { width: 0 },
      itemStyle: { color: structureColor(t, n) },
      data: m.by_day.map((d) => {
        const total = d.held.reduce((a, b) => a + b, 0);
        return total ? (d.held[id] ?? 0) / total : 0;
      }),
    })),
  };
}

function MixTiles({ m }: { m: MixSummary }) {
  const held = 1 - (m.held_share.FLAT ?? 0);
  const top = STRUCTURES.filter((s) => s !== "FLAT").sort((a, b) => (m.held_share[b] ?? 0) - (m.held_share[a] ?? 0))[0];
  return (
    <div className={ui.grid4}>
      <Tile label="Held" value={pct(held, 1)} note={top && (m.held_share[top] ?? 0) > 0 ? `mostly ${STRUCT_LABEL[top]} (${pct(m.held_share[top], 1)})` : "flat throughout"} />
      <Tile label="Opens per episode" value={num(m.opens_per_episode, 2)} note={`${num(m.n_opens)} over ${num(m.n_episodes)} episodes`} />
      <Tile label="Refused opens" value={num(m.n_refused)} note={`${num(m.refused_per_episode, 2)} per episode · ${pct(m.refused_share, 1)} of attempts`} />
      <Tile label="Holding days" value={m.holding_days ? num(m.holding_days.p50, 0) : "–"} note={m.holding_days ? `median · p10 ${num(m.holding_days.p10, 0)} · p90 ${num(m.holding_days.p90, 0)} · max ${num(m.holding_days.max, 0)}` : "nothing opened"} />
    </div>
  );
}

function StructureTable({ m }: { m: MixSummary }) {
  const rows = STRUCTURES.filter((s) => s !== "FLAT").map((s) => ({ s, x: m.structures[s] }));
  return (
    <div className={ui.tableWrap}>
      <table className={ui.table}>
        <thead>
          <tr>
            <th>Structure</th>
            <th className={ui.num}>Held</th>
            <th className={ui.num}>Opens</th>
            <th className={ui.num}>Tenor (d)</th>
            <th className={ui.num}>|Δ| short</th>
            <th className={ui.num}>|Δ| wing</th>
            <th className={ui.num}>Credit</th>
            <th className={ui.num}>Margin</th>
            <th className={ui.num}>Days held</th>
            <th>Exits</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, x }) => (
            <tr key={s}>
              <td>
                {STRUCT_LABEL[s] ?? s} <InfoTip term={`structure.${s}`} />
              </td>
              <td className={ui.num}>{pct(m.held_share[s], 1)}</td>
              <td className={ui.num}>{num(x?.n_opens ?? 0)}</td>
              <td className={ui.num}>{num(x?.mean_tenor_days, 1)}</td>
              <td className={ui.num}>{num(x?.mean_short_abs_delta, 3)}</td>
              <td className={ui.num}>{num(x?.mean_long_abs_delta, 3)}</td>
              <td className={ui.num}>{usd(x?.mean_credit)}</td>
              <td className={ui.num}>{usd(x?.mean_margin)}</td>
              <td className={ui.num}>{num(x?.mean_holding_days, 1)}</td>
              <td className={ui.dim}>{x && Object.keys(x.exits).length ? Object.entries(x.exits).map(([k, n]) => `${k} ${n}`).join(" · ") : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const EXIT_LABEL: Record<string, string> = { policy: "closed by the policy", expiry: "settled at expiry", margin: "liquidated on margin", horizon: "open at the episode's end" };

/** The structure a summary holds most, or undefined when it holds nothing. */
export function topStructure(m: MixSummary): string | undefined {
  const top = STRUCTURES.filter((s) => s !== "FLAT").sort((a, b) => (m.held_share[b] ?? 0) - (m.held_share[a] ?? 0))[0];
  return top && (m.held_share[top] ?? 0) > 0 ? top : undefined;
}

/** One sentence: what the policy holds, what it opens at what delta and tenor, for how long, and how it leaves. */
export function mixSentence(m: MixSummary): string {
  const top = topStructure(m);
  if (!top) return "Stays flat throughout: it opens nothing.";
  const held = STRUCTURES.filter((s) => s !== "FLAT" && (m.held_share[s] ?? 0) > 0.05);
  const x = m.structures[top];
  const what =
    held.length > 1
      ? `Spreads its days over ${held.map((s) => `${(STRUCT_LABEL[s] ?? s).toLowerCase()} ${pct(m.held_share[s], 0)}`).join(", ")} (flat ${pct(m.held_share.FLAT, 0)})`
      : `Holds ${STRUCT_LABEL[top]?.toLowerCase() ?? top} ${pct(m.held_share[top], 0)} of days`;
  const open = x ? `, opening ${num(m.opens_per_episode, 1)} per episode at about ${num(x.mean_tenor_days, 0)} days to expiry and ${num(x.mean_short_abs_delta, 2)} short delta` : "";
  const exits = Object.entries(m.exits).sort((a, b) => b[1] - a[1]);
  const total = exits.reduce((a, [, n]) => a + n, 0);
  const lead = exits[0];
  const leave = m.holding_days ? `; each is held a median ${num(m.holding_days.p50, 0)} days` : "";
  const how = lead && total ? ` and is mostly ${EXIT_LABEL[lead[0]] ?? lead[0]} (${pct(lead[1] / total, 0)})` : "";
  const refused = m.n_refused ? ` ${num(m.n_refused)} opens were refused for margin.` : "";
  return `${what}${open}${leave}${how}.${refused}`;
}

/** The run's action mix: the full one (every holding) re-simulated on open when needed. */
export function useRunMix(key: string, enabled = true, canStart = true) {
  return useRunTask<MixView>(
    `mixview|${key}`,
    (s) => runsApi.mix(key, s),
    () => runsApi.computeMix(key),
    (m) => (m.state === "absent" || m.state === "lean") && m.can_compute,
    (m) => m.state === "computing",
    enabled,
    canStart,
  );
}

function daysHeldOption(t: Tokens, d: MixDetail): EChartsOption {
  const names = STRUCTURES.filter((s) => s !== "FLAT" && (d.days_held[s] ?? []).some((n) => n > 0));
  const len = Math.max(0, ...names.map((s) => d.days_held[s]?.length ?? 0));
  return {
    ...base(t),
    legend: { ...base(t).legend, data: names.map((n) => STRUCT_LABEL[n] ?? n) },
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, formatter: (ps: unknown) => { const arr = ps as { seriesName: string; value: number; name: string }[]; return [`<b>Held ${esc(arr[0]?.name)} days</b>`, ...arr.filter((p) => p.value).map((p) => `${esc(p.seriesName)}: ${p.value}`)].join("<br/>"); } },
    grid: { ...base(t).grid, left: 52, top: 36 },
    xAxis: { type: "category", data: Array.from({ length: len }, (_, i) => i), name: "Days held", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Holdings", nameLocation: "middle", minInterval: 1, ...axis(t), nameGap: 38 },
    series: names.map((n) => ({ name: STRUCT_LABEL[n] ?? n, type: "bar" as const, stack: "days", data: d.days_held[n] ?? [], itemStyle: { color: structureColor(t, n) }, barCategoryGap: "15%" })),
  };
}

/** One episode, day by day: a cell per day coloured by what was held, a tick where it traded. */
function Timeline({ ex }: { ex: NonNullable<MixDetail["example"]> }) {
  const t = useTokens();
  return (
    <div className={ui.stack} style={{ gap: 6 }}>
      <div style={{ display: "flex", gap: 1, alignItems: "flex-end" }} role="img" aria-label={`Episode ${ex.seed}, what was held each day`}>
        {ex.held.map((h, i) => {
          const ev = ex.events[i] ?? "";
          const mark = ev === "open" || ev === "switch" || ev === "close" || ev === "refused";
          return (
            <div key={i} title={`day ${i}: ${STRUCT_LABEL[h] ?? h}${ev && ev !== "hold" && ev !== "idle" ? ` (${ev})` : ""}`} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <span style={{ height: 6, width: 2, background: mark ? (ev === "refused" ? t.fail : t.text2) : "transparent" }} />
              <span style={{ height: 22, width: "100%", borderRadius: 2, background: h === "FLAT" ? "var(--surface-2)" : structureColor(t, h), opacity: h === "FLAT" ? 1 : 0.85 }} />
            </div>
          );
        })}
      </div>
      <div className={ui.spread} style={{ fontSize: 12 }}>
        <span className={ui.dim}>day 0</span>
        <span className={ui.dim}>
          ticks: opens, closes and rolls · {ex.holdings.length} holding{ex.holdings.length === 1 ? "" : "s"}:{" "}
          {ex.holdings.map((h) => `${STRUCT_LABEL[h.structure] ?? h.structure} day ${h.open_day}→${h.exit_day} (${h.exit})`).join("; ")}
        </span>
        <span className={ui.dim}>day {ex.held.length - 1}</span>
      </div>
    </div>
  );
}

function ExitBar({ m }: { m: MixSummary }) {
  const t = useTokens();
  const rows = Object.entries(m.exits).filter(([, n]) => n > 0);
  const total = rows.reduce((a, [, n]) => a + n, 0);
  if (!total) return null;
  const colors: Record<string, string> = { policy: t.s1, expiry: t.s3, margin: t.fail, horizon: t.s5 };
  return (
    <div className={ui.stack} style={{ gap: 6 }}>
      <div style={{ display: "flex", height: 14, borderRadius: 4, overflow: "hidden" }} role="img" aria-label="How holdings left the book">
        {rows.map(([k, n]) => (
          <div key={k} title={`${EXIT_LABEL[k] ?? k}: ${n}`} style={{ width: `${(n / total) * 100}%`, background: colors[k] ?? t.s5 }} />
        ))}
      </div>
      <div className={ui.row} style={{ fontSize: 12.5, gap: 12 }}>
        {rows.map(([k, n]) => (
          <span key={k}>
            <span className={ui.swatch} style={{ background: colors[k] ?? t.s5, display: "inline-block", marginRight: 4 }} aria-hidden="true" />
            {EXIT_LABEL[k] ?? k} {pct(n / total, 0)} <InfoTip term={`exit.${k}`} />
          </span>
        ))}
      </div>
    </div>
  );
}

/** On the run detail: what this run held over the episode, what it opened and how it left. */
export function RunMixCard({ d, wait = false }: { d: RunDetail; wait?: boolean }) {
  const t = useTokens();
  const mix = useRunMix(d.row.key, true, !wait);
  const v = mix.data;
  const r = v?.mix;
  const stamp = runStamp(d.row);
  const waiting = !v || v.state === "computing" || ((v.state === "absent" || v.state === "lean") && v.can_compute);
  return (
    <div id="actionmix">
      <Card
        title="What it traded"
      info="action_mix"
        sub={r ? mixSentence(r.summary) : undefined}
        source={r ? `${num(r.summary.n_episodes)} episodes re-simulated; ${r.reproduces ? "reproduces run.json bit for bit" : r.agrees ? "agrees with run.json within 1e-9 relative" : "differs from run.json beyond 1e-9"}.` : undefined}
      >
        {mix.error ? <ErrorBox what="the action mix" error={mix.error} onRetry={mix.reload} /> : null}
        {mix.startError ? <ErrorBox what="the action mix" error={mix.startError} /> : null}
        {!r && waiting ? <Loading what={`the action mix by re-simulation${v?.seconds ? ` (${num(v.seconds, 0)} s)` : ""}`} /> : null}
        {v?.state === "absent" && !v.can_compute ? <Pending title="Not re-simulated here">{v.reason}.</Pending> : null}
        {v?.state === "failed" && !r ? <ErrorBox what="the action mix" error={new ApiError(500, v.reason)} /> : null}
        {r ? (
          <div className={ui.stack}>
            <MixTiles m={r.summary} />
            {r.summary.by_day.length ? (
              <>
                <p className={ui.groupTitle}>
                  Held over the episode <InfoTip term="held_share" />
                </p>
                <Chart
                  option={byDayOption(t, r.summary, r.action_names)}
                  height={240}
                  label="Share of episodes holding each structure, by day of episode"
                  stamp={stamp}
                  rows={() => r.summary.by_day.map((x) => ({ day: x.day, opens: x.opens, refused: x.refused, ...Object.fromEntries(r.action_names.map((n, i) => [`held_${n}`, x.held[i] ?? 0])) }))}
                  json={() => r.summary}
                />
              </>
            ) : null}
            <p className={ui.groupTitle}>What it opens, and how each left the book</p>
            <StructureTable m={r.summary} />
            <ExitBar m={r.summary} />
            {v?.detail ? (
              <div className={ui.grid2}>
                <div>
                  <p className={ui.groupTitle}>
                    How long each was held <InfoTip term="holding_days" />
                  </p>
                  <Chart option={daysHeldOption(t, v.detail)} height={230} label="Holdings by days held, per structure" stamp={stamp} />
                </div>
                {v.detail.example ? (
                  <div>
                    <p className={ui.groupTitle}>
                      A typical episode (seed {v.detail.example.seed}, the median P&L {usd(v.detail.example.pnl)})
                    </p>
                    <Timeline ex={v.detail.example} />
                  </div>
                ) : null}
              </div>
            ) : waiting ? (
              <Loading what={`holding periods and a typical episode${v?.seconds ? ` (${num(v.seconds, 0)} s)` : ""}`} />
            ) : null}
          </div>
        ) : null}
      </Card>
    </div>
  );
}
