"use client";

import { useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, SelectField, Source, Tile, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import lab from "@/components/lab.module.css";
import { ApiError, api, type Attribution, type AttributionResult, COMPONENTS, type Replay, type RunDetail, type Terms } from "@/lib/api";
import { type Band, type CapitalPaths, runsApi } from "@/lib/api_runs";
import { armColor, axis, base, componentColor, runStamp, type Tokens, useTokens } from "@/lib/chart";
import { esc, hours, num, pct, sci, usd } from "@/lib/format";
import { guide } from "@/lib/guide";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import { useRunTask } from "@/lib/useRunTask";

const LABEL: Record<string, string> = {
  delta: "Delta",
  gamma: "Gamma",
  vega: "Vega",
  theta: "Theta",
  carry: "Carry",
  cost: "Cost",
  residual: "Residual",
};

// ---------------------------------------------------------------- replay

/** The badge a replay earns: bit for bit, within the tolerance, or not, and for a PPO
 * retrain off its own platform, not comparable rather than a failure. */
export function replayBadge(r: Replay): string {
  if (r.verdict === "DIFFERS" && !r.comparable) return "NOT COMPARABLE";
  return r.verdict || "DIFFERS";
}

function replayLine(r: Replay): string {
  if (r.verdict === "AGREES") return `within ${sci(r.rtol)} relative, largest ${sci(r.max_rel_diff)}`;
  if (r.verdict === "IDENTICAL") return "bit for bit";
  return r.note;
}

/** Rerun the run from its manifest and compare run.json: bit for bit, or for a scripted
 * run within the seller's 1e-9 relative. A PPO run retrains first, so it asks before. */
export function ReplayControl({ d }: { d: RunDetail }) {
  const key = d.row.key;
  const status = useFetch((s) => api.replay(key, s), `replay|${key}`);
  const [err, setErr] = useState<ApiError | undefined>();
  const [asking, setAsking] = useState(false);
  usePoll([status], status.data?.state === "running", 800);
  const r = status.data;
  const start = async () => {
    setErr(undefined);
    setAsking(false);
    try {
      await api.startReplay(key);
      status.reload();
    } catch (e) {
      setErr(e instanceof ApiError ? e : new ApiError(0, String(e)));
    }
  };
  const busy = r?.state === "running";
  const cost = r?.retrains ? `retrains ${num(r.train_steps)} steps first, about ${hours(r.estimate_hours)} on one core` : "";
  return (
    <span className={ui.row} style={{ gap: 8 }}>
      {r?.state === "done" ? (
        <span title={`${r.note}\n${r.reason}`}>
          <Verdict value={replayBadge(r)} /> <span className={ui.dim}>{replayLine(r)}</span>
        </span>
      ) : null}
      {r?.state === "failed" ? <span className={ui.fieldError}>replay failed: {r.reason}</span> : null}
      {err ? <span className={ui.fieldError}>{err.message}</span> : null}
      {asking && r ? (
        <span className={ui.confirm} role="group" aria-label="Confirm the retrain">
          <span>
            Replay {cost}. {r.note}
          </span>
          <button type="button" className={`${ui.btn} ${ui.btnSmall} ${ui.btnPrimary}`} onClick={start} {...guide("runs-replay")}>
            Retrain and replay
          </button>
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => setAsking(false)}>
            Cancel
          </button>
        </span>
      ) : (
        <button
          type="button"
          className={`${ui.btn} ${ui.btnSmall}`}
          onClick={r?.retrains ? () => setAsking(true) : start}
          {...guide("runs-replay")}
          disabled={busy || !r?.can_replay}
          title={r?.can_replay ? (r.retrains ? `Rerun from the manifest: ${cost}` : "Rerun from the manifest and compare run.json: bit for bit, or within 1e-9 relative") : r?.reason}
        >
          {busy ? `Replaying… ${num(r?.seconds ?? 0, 0)} s` : r?.retrains ? `Replay (${hours(r.estimate_hours)})` : "Replay"}
        </button>
      )}
    </span>
  );
}

/** The replay's verdict and its reason, once one has run. */
export function ReplayResult({ d }: { d: RunDetail }) {
  const key = d.row.key;
  const status = useFetch((s) => api.replay(key, s), `replay-line|${key}`);
  usePoll([status], status.data?.state === "running", 1000);
  const r = status.data;
  if (!r || r.state !== "done") return null;
  return (
    <Card
      title="Replay"
      info="replay"
      right={
        <>
          <Verdict value={replayBadge(r)} />
          <InfoTip term={replayBadge(r)} />
          <Source>{`optspread.replay.replay_run: rerun from the manifest, run.json compared ${r.rtol === null ? "byte for byte" : `byte for byte, then every number within ${sci(r.rtol)} relative`}`}</Source>
        </>
      }
    >
      <p className={ui.breakWords}>
        <b>{replayBadge(r)}</b> · {r.note} · {num(r.seconds, 1)} s
      </p>
      <p className={`${ui.breakWords} ${ui.dim}`} style={{ fontSize: 12.5 }}>
        {r.reason}
      </p>
    </Card>
  );
}

// ---------------------------------------------------------------- capital and equity

function capitalLine(c: CapitalPaths): string {
  if (c.reproduces) return `${c.n} episodes re-simulated, every P&L equal to run.json bit for bit`;
  if (c.agrees) return `${c.n} episodes re-simulated, every P&L within ${sci(c.rtol)} relative of run.json (largest |Δ| ${usd2(c.max_abs_diff)})`;
  return `${c.n} episodes re-simulated · ${c.n_differ} of ${c.n} episode${c.n === 1 ? "" : "s"} ${c.n_differ === 1 ? "differs" : "differ"} from run.json beyond ${sci(c.rtol)} (largest |Δ| ${usd2(c.max_abs_diff)})`;
}

/** The run's re-simulation: started on open when it can run here, polled until done. */
export function useCapital(d: RunDetail) {
  const key = d.row.key;
  return useRunTask<CapitalPaths>(
    `capital|${key}`,
    (s) => runsApi.capital(key, s),
    () => api.computeCapital(key),
    (c) => c.state === "absent" && c.can_compute,
    (c) => c.state === "computing",
  );
}

/** Paths and their band by day: the band is every episode, the lines a sample. */
export function equityOption(t: Tokens, paths: number[][], band: Band | null, color: string, name = "mean", extra: { paths: number[][]; band: Band | null; color: string; name: string } | null = null): EChartsOption {
  const sets = [{ paths, band, color, name }, ...(extra ? [extra] : [])];
  const len = Math.max(1, ...sets.flatMap((x) => [...x.paths.map((p) => p.length), x.band?.mean.length ?? 0]));
  const days = Array.from({ length: len }, (_, i) => i);
  const meanOf = (x: (typeof sets)[number]) =>
    x.band?.mean ??
    days.map((i) => {
      const vals = x.paths.map((p) => p[i]).filter((v): v is number => v !== undefined);
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    });
  return {
    ...base(t),
    legend: { ...base(t).legend, show: Boolean(extra), data: sets.map((x) => x.name) },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      formatter: (ps: unknown) => {
        const i = (ps as { dataIndex: number }[])[0]?.dataIndex ?? 0;
        return [`<b>Day ${i}</b>`, ...sets.map((x) => {
          const m = meanOf(x)[i];
          const lo = x.band?.p10[i];
          const hi = x.band?.p90[i];
          return `${esc(x.name)}: mean ${usd(m)}${lo !== undefined && hi !== undefined ? ` · 10–90% ${usd(lo)} to ${usd(hi)}` : ""}`;
        })].join("<br/>");
      },
    },
    xAxis: { type: "category", data: days, name: "Trading day of the episode", nameLocation: "middle", boundaryGap: false, ...axis(t), axisLabel: { ...axis(t).axisLabel, interval: (i: number) => i % 5 === 0 }, splitLine: { show: false } },
    yAxis: { type: "value", name: "Cumulative P&L ($)", nameLocation: "middle", ...axis(t), nameGap: 58 },
    grid: { ...base(t).grid, left: 74, top: extra ? 36 : 20 },
    series: sets.flatMap((x, k) => [
      ...(x.band
        ? [
            { name: `${x.name} p10`, type: "line" as const, data: x.band.p10, stack: `band${k}`, symbol: "none", lineStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
            {
              name: `${x.name} band`,
              type: "line" as const,
              data: x.band.p90.map((v, i) => (v === null || x.band?.p10[i] === null ? null : v - (x.band?.p10[i] ?? 0))),
              stack: `band${k}`,
              symbol: "none",
              lineStyle: { opacity: 0 },
              areaStyle: { color: x.color, opacity: 0.14 },
              silent: true,
              tooltip: { show: false },
            },
          ]
        : []),
      ...x.paths.slice(0, extra ? 20 : 60).map((p, i) => ({
        name: `${x.name} path ${i}`,
        type: "line" as const,
        data: p,
        showSymbol: false,
        silent: true,
        tooltip: { show: false },
        lineStyle: { width: 1, color: x.color, opacity: extra ? 0.12 : 0.16 },
      })),
      { name: x.name, type: "line" as const, data: meanOf(x), showSymbol: false, itemStyle: { color: extra ? x.color : t.text }, lineStyle: { width: 2.5, color: extra ? x.color : t.text } },
    ]),
  };
}

/** Equity paths: recorded by a launched run, else re-simulated from the manifest. */
export function EquityCard({ d, cap }: { d: RunDetail; cap: ReturnType<typeof useCapital> }) {
  const t = useTokens();
  const c = cap.data;
  const color = armColor(t, d.row.policy);
  const recorded = d.equity;
  const paths = recorded?.paths ?? (c?.state === "done" ? c.paths : []);
  const band = c?.state === "done" ? c.equity_band : null;
  const stamp = runStamp(d.row);
  const final = band?.mean[band.mean.length - 1];
  const sub = band
    ? `On a typical day the book sits between ${usd(band.p10[Math.floor(band.p10.length / 2)])} and ${usd(band.p90[Math.floor(band.p90.length / 2)])} by mid-episode and ends at ${usd(final)} on average.`
    : recorded
      ? `${recorded.paths.length} of ${recorded.n_recorded} recorded paths, and the mean.`
      : undefined;
  return (
    <Card
      title="Equity paths"
      info="equity_path"
      sub={sub}
      source={recorded ? `Recorded by the launch in ${recorded.file}; the band is from the re-simulation when it has run.` : "Re-simulated from the manifest: this run did not record its paths. Cached under the attribution root, never in the run."}
    >
      {paths.length ? (
        <Chart option={equityOption(t, paths, band, color)} height={300} label="Cumulative P&L paths per episode, the 10-90% band and the mean" stamp={stamp} rows={() => (band ? band.mean.map((m, i) => ({ day: i, mean: m, p10: band.p10[i], p50: band.p50[i], p90: band.p90[i] })) : paths.map((p, i) => ({ path: i, final: p[p.length - 1] })))} />
      ) : cap.error ? (
        <ErrorBox what="the equity paths" error={cap.error} onRetry={cap.reload} />
      ) : !c || c.state === "computing" || (c.state === "absent" && c.can_compute) ? (
        <Loading what={`the equity paths by re-simulation${c?.seconds ? ` (${num(c.seconds, 0)} s)` : ""}`} />
      ) : c.state === "failed" ? (
        <ErrorBox what="the re-simulation" error={new ApiError(500, c.reason)} />
      ) : (
        <Pending title="Not re-simulated here">{c.reason}.</Pending>
      )}
    </Card>
  );
}

function marginOption(t: Tokens, band: Band, color: string): EChartsOption {
  const days = band.mean.map((_, i) => i);
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["mean", "90th percentile"] },
    tooltip: { ...base(t).tooltip, trigger: "axis", formatter: (ps: unknown) => { const i = (ps as { dataIndex: number }[])[0]?.dataIndex ?? 0; return `<b>Day ${i}</b><br/>mean ${usd(band.mean[i])}<br/>90th ${usd(band.p90[i])}`; } },
    xAxis: { type: "category", data: days, name: "Trading day of the episode", nameLocation: "middle", boundaryGap: false, ...axis(t), axisLabel: { ...axis(t).axisLabel, interval: (i: number) => i % 5 === 0 }, splitLine: { show: false } },
    yAxis: { type: "value", name: "Margin posted ($)", nameLocation: "middle", min: 0, ...axis(t), nameGap: 58 },
    grid: { ...base(t).grid, left: 74, top: 36 },
    series: [
      { name: "mean", type: "line", data: band.mean, showSymbol: false, areaStyle: { color, opacity: 0.18 }, lineStyle: { color, width: 2 }, itemStyle: { color } },
      { name: "90th percentile", type: "line", data: band.p90, showSymbol: false, lineStyle: { color, width: 1, type: "dashed" }, itemStyle: { color } },
    ],
  };
}

/** Drawdown within each episode and return on what the book posted: cash and peak margin. */
export function CapitalCard({ d, cap }: { d: RunDetail; cap: ReturnType<typeof useCapital> }) {
  const t = useTokens();
  const c = cap.data;
  const done = c?.state === "done" ? c : undefined;
  const sentence = done
    ? done.mean_peak_margin
      ? `The book ties up ${usd(done.mean_peak_margin)} of margin at its peak and returns ${pct(done.mean_return_on_margin, 1)} on it per episode; the worst fall inside an episode is ${usd(done.worst_drawdown)} (${pct(done.worst_drawdown_on_cash, 2)} of cash).`
      : "The book never posts margin: it holds nothing."
    : undefined;
  return (
    <Card
      title="On posted capital"
      info="margin"
      sub={sentence}
      source={done ? `${capitalLine(done)}. ${done.method}.` : undefined}
    >
      {cap.error ? <ErrorBox what="the capital view" error={cap.error} onRetry={cap.reload} /> : null}
      {cap.startError ? <ErrorBox what="the re-simulation" error={cap.startError} /> : null}
      {!c || c.state === "computing" || (c.state === "absent" && c.can_compute) ? <Loading what={`the episodes${c?.seconds ? ` (${num(c.seconds, 0)} s)` : ""}`} /> : null}
      {c?.state === "absent" && !c.can_compute ? <Pending title="Not re-simulated here">{c.reason}.</Pending> : null}
      {c?.state === "failed" ? <ErrorBox what="the re-simulation" error={new ApiError(500, c.reason)} /> : null}
      {done ? (
        <div className={ui.stack}>
          <div className={ui.grid4}>
            <Tile label="Return on initial cash" value={pct(done.mean_return_on_cash, 3)} note={<>mean per episode · cash {usd(done.initial_cash)}</>} info="return_on_cash" />
            <Tile label="Return on peak margin" value={pct(done.mean_return_on_margin, 2)} note="total P&L / total peak margin" info="return_on_margin" />
            <Tile label="Peak margin" value={usd(done.mean_peak_margin)} note={<>mean · max {usd(done.max_peak_margin)}</>} info="peak_margin" />
            <Tile label="Drawdown in an episode" value={usd(done.worst_drawdown)} note={<>worst · median {usd(done.median_drawdown)}</>} info="drawdown_episode" />
          </div>
          {done.margin_band && done.mean_peak_margin ? (
            <Chart option={marginOption(t, done.margin_band, armColor(t, d.row.policy))} height={240} label="Margin posted by day of the episode: the mean and the 90th percentile" stamp={runStamp(d.row)} rows={() => (done.margin_band ? done.margin_band.mean.map((m, i) => ({ day: i, mean: m, p90: done.margin_band?.p90[i] ?? null })) : [])} />
          ) : null}
        </div>
      ) : null}
    </Card>
  );
}

/** Dollars to the cent, or a tiny difference in scientific notation. */
function usd2(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return Math.abs(x) < 0.01 ? `$${sci(x)}` : usd(x);
}

/** How a run's attribution compares with run.json: bit for bit, within 1e-9, or how many episodes differ. */
export function attributionLine(r: AttributionResult): string {
  if (r.reproduces) return "reproduces run.json bit for bit";
  if (r.agrees) return "agrees with run.json within 1e-9 relative";
  const bad = r.episodes.filter((e) => e.agrees === false).length;
  if (r.agrees === false && bad) return `${bad} of ${r.episodes.length} episode${r.episodes.length === 1 ? "" : "s"} ${bad === 1 ? "differs" : "differ"} from run.json`;
  if (r.reproduces === false) return "not bit for bit; no tolerance check in this file";
  return "";
}

// ---------------------------------------------------------------- attribution

function stackOption(t: Tokens, cats: (string | number)[], rows: Record<string, number | null>[], xName: string, pnl: (number | null)[]): EChartsOption {
  return {
    ...base(t),
    legend: { ...base(t).legend, data: [...COMPONENTS.map((c) => LABEL[c] ?? c), "P&L"] },
    tooltip: {
      ...base(t).tooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (ps: unknown) => {
        const arr = ps as { seriesName: string; value: number | null; name: string }[];
        return [`<b>${esc(xName)} ${esc(arr[0]?.name)}</b>`, ...arr.map((p) => `${esc(p.seriesName)}: ${usd(p.value)}`)].join("<br/>");
      },
    },
    grid: { ...base(t).grid, left: 74, top: 44 },
    xAxis: { type: "category", data: cats, name: xName, nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "$", nameLocation: "middle", ...axis(t), nameGap: 58 },
    series: [
      ...COMPONENTS.map((c) => ({
        name: LABEL[c] ?? c,
        type: "bar" as const,
        stack: "terms",
        data: rows.map((r) => r[c] ?? null),
        itemStyle: { color: componentColor(t, c), borderColor: c === "residual" ? t.text2 : undefined, borderWidth: c === "residual" ? 0.6 : 0 },
        barCategoryGap: "25%",
      })),
      { name: "P&L", type: "line" as const, data: pnl, symbol: "circle", symbolSize: 5, lineStyle: { width: 0 }, itemStyle: { color: t.text }, z: 5 },
    ],
  };
}

/** Per-episode means of each component: the totals over the episode count. */
export function perEpisode(r: AttributionResult): Terms {
  const n = Math.max(1, r.episodes.length);
  const out = {} as Terms;
  for (const [k, v] of Object.entries(r.totals)) out[k as keyof Terms] = v === null ? null : v / n;
  return out;
}

/** Where the P&L comes from, one bar per component; with `other`, two runs side by side. */
export function componentsOption(t: Tokens, terms: { name: string; color: string; v: Terms }[]): EChartsOption {
  const cats = [...COMPONENTS.map((c) => LABEL[c] ?? c), "P&L"];
  const keys = [...COMPONENTS, "pnl"] as const;
  return {
    ...base(t),
    legend: { ...base(t).legend, show: terms.length > 1, data: terms.map((x) => x.name) },
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, formatter: (ps: unknown) => { const arr = ps as { seriesName: string; value: number | null; name: string }[]; return [`<b>${esc(arr[0]?.name)}</b> per episode`, ...arr.map((p) => `${esc(p.seriesName)}: ${usd(p.value)}`)].join("<br/>"); } },
    grid: { ...base(t).grid, left: 92, top: terms.length > 1 ? 34 : 14, right: terms.length > 1 ? 24 : 72 },
    yAxis: { type: "category", data: cats, inverse: true, ...axis(t), splitLine: { show: false } },
    xAxis: { type: "value", name: "$ per episode", nameLocation: "middle", ...axis(t) },
    series: terms.map((x) => ({
      name: x.name,
      type: "bar" as const,
      // the series' own colour, so the legend's swatch matches its bars
      itemStyle: { color: x.color },
      data: keys.map((k, i) => ({ value: x.v[k] ?? null, itemStyle: { color: terms.length > 1 ? x.color : k === "pnl" ? t.text : componentColor(t, COMPONENTS[i] ?? "residual") } })),
      barGap: "10%",
      barCategoryGap: "30%",
      label: { show: terms.length === 1, position: "right", color: t.muted, fontSize: 11, formatter: (p: { value?: unknown }) => usd(typeof p.value === "number" ? p.value : null) },
    })),
  };
}

/** One sentence: the component that earned most, the one that cost most, and the residual. */
export function attributionSentence(r: AttributionResult): string {
  const m = perEpisode(r);
  const parts = COMPONENTS.filter((c) => c !== "residual").map((c) => ({ c, v: m[c] ?? 0 }));
  const best = [...parts].sort((a, b) => b.v - a.v)[0];
  const worst = [...parts].sort((a, b) => a.v - b.v)[0];
  if (!best || !worst) return "";
  const pnl = m.pnl ?? 0;
  const lead = best.v > 0 ? `${LABEL[best.c]} earned ${usd(best.v)} per episode` : "Nothing earned on its own";
  const drag = worst.v < 0 ? `, ${(LABEL[worst.c] ?? worst.c).toLowerCase()} cost ${usd(-worst.v)}` : "";
  return `${lead}${drag}, for ${usd(pnl)} net; the greeks leave ${pct(r.residual_share, 1)} of the gross unexplained.`;
}

/** The run's attribution, cached under the attribution root; computed on open (once
 * `canStart`: after the capital re-simulation above it). */
export function useAttribution(key: string, seed?: number, canStart = true) {
  return useRunTask<Attribution>(
    `attr|${key}|${seed ?? ""}`,
    (s) => api.attribution(key, seed, s),
    () => api.attribute(key),
    (a) => a.state === "absent" && a.can_compute,
    (a) => a.state === "computing",
    true,
    canStart,
  );
}

/** Whether a run's capital re-simulation is still to come: the views below it wait. */
export function capitalPending(cap: ReturnType<typeof useCapital>): boolean {
  const c = cap.data;
  return !c || c.state === "computing" || (c.state === "absent" && c.can_compute);
}

export function AttributionCard({ d, wait = false }: { d: RunDetail; wait?: boolean }) {
  const t = useTokens();
  const key = d.row.key;
  const [seed, setSeed] = useState<number | undefined>();
  const att = useAttribution(key, seed, !wait);
  const [last, setLast] = useState<Attribution | undefined>();
  useEffect(() => {
    if (att.data) setLast(att.data);
  }, [att.data]);
  const a = att.data ?? last;
  const r = a?.result;
  const stamp = runStamp(d.row);
  return (
    <div id="attribution">
      <Card
        title="Where the P&L comes from"
      info="attribution"
        sub={r ? attributionSentence(r) : "Each day's P&L split into the greeks, carry, cost and a residual."}
        source={a ? `${r ? `${r.episodes.length} episodes re-simulated; ${attributionLine(r)}. ` : ""}Cached at ${a.file}.` : undefined}
      >
        {att.error && !a ? <ErrorBox what="the attribution" error={att.error} onRetry={att.reload} /> : null}
        {att.startError ? <ErrorBox what="the attribution" error={att.startError} /> : null}
        {!a || (a.state === "absent" && a.can_compute) ? <Loading what="the attribution" /> : null}
        {a?.state === "absent" && !a.can_compute ? <Pending title="Not attributed here">{a.reason}.</Pending> : null}
        {a?.state === "computing" ? <Loading what={`the attribution (${num(a.seconds ?? 0, 0)} s)`} /> : null}
        {a?.state === "failed" ? <ErrorBox what="the attribution" error={new ApiError(500, a.reason)} /> : null}
        {r ? (
          <div className={ui.stack}>
            <Chart option={componentsOption(t, [{ name: "this run", color: armColor(t, d.row.policy), v: perEpisode(r) }])} height={260} label="Mean P&L per episode by component" stamp={stamp} rows={() => Object.entries(perEpisode(r)).map(([k, v]) => ({ component: k, per_episode: v }))} json={() => r} />
            <div className={ui.grid4}>
              {(["theta", "gamma", "vega", "delta"] as const).map((c) => (
                <Tile key={c} label={LABEL[c] ?? c} value={usd(perEpisode(r)[c])} note={<>per episode</>} info={`attr.${c}`} />
              ))}
            </div>
            <details className={lab.subFold}>
              <summary {...guide("runs-by-day")}>By episode and by day</summary>
              <div className={ui.stack}>
                <Chart
                  option={stackOption(
                    t,
                    r.episodes.map((e) => e.seed),
                    r.episodes.map((e) => e.totals),
                    "Episode seed",
                    r.episodes.map((e) => e.totals.pnl),
                  )}
                  height={300}
                  label="Attribution per episode: components stacked, P&L marked"
                  onClick={(p) => {
                    const e = r.episodes[p.dataIndex];
                    if (e) setSeed(e.seed);
                  }}
                  stamp={stamp}
                  rows={() => r.episodes.map((e) => ({ seed: e.seed, ...e.totals, residual_share: e.residual_share }))}
                />
                <div className={ui.spread}>
                  <p className={ui.groupTitle}>Episode {a?.seed ?? "–"}, day by day</p>
                  <SelectField label="Episode" guide="runs-episode" value={a?.seed ?? ""} options={r.episodes.map((e) => e.seed)} render={(x) => `seed ${x} · ${usd(r.episodes.find((e) => e.seed === x)?.totals.pnl)}`} onChange={(v) => v !== "" && setSeed(v)} />
                </div>
                {a && a.steps.length ? (
                  <Chart
                    option={stackOption(
                      t,
                      a.steps.map((s) => s.day),
                      a.steps.map((s) => s.terms),
                      "Day",
                      a.steps.map((s) => s.terms.pnl),
                    )}
                    height={280}
                    label={`Attribution per step of episode ${a.seed}`}
                    stamp={`${stamp} · seed ${a.seed}`}
                    rows={() => a.steps.map((s) => ({ day: s.day, spot: s.spot, spot_next: s.spot_next, action_id: s.action_id, n_legs: s.n_legs, ...s.terms }))}
                  />
                ) : (
                  <Empty title="No steps recorded for this episode" />
                )}
              </div>
            </details>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
