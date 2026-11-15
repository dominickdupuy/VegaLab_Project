"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Labelled } from "@/components/InfoTip";
import { Reading } from "@/components/PageBits";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type SynthParams, type SynthPaths } from "@/lib/api";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { num, pct } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { useQuery, useQuerySync } from "@/lib/url";

/**
 * The simulated market before a run is launched on it: sample paths of the index, the
 * at-the-money vol a policy would see along each, and the smile half way through, for
 * the path vol and the research knobs (drift, vol of vol, skew). Path i is the episode
 * on seed seed + i, so a run launched from here with the same first seed plays them.
 */

const DEFAULTS: SynthParams = { sigma: 0.2, alpha_bps: 0, vol_of_vol: 0, skew_slope: 0, n_days: 63, n_paths: 5, seed: 0 };
const KEYS = Object.keys(DEFAULTS) as (keyof SynthParams)[];
// the knobs a launch carries over; n_paths is this page's own
const LAUNCH_KEYS: (keyof SynthParams)[] = ["sigma", "alpha_bps", "vol_of_vol", "skew_slope", "n_days", "seed"];
const PATH_SLOTS: (keyof Tokens)[] = ["s1", "s2", "s3", "s4", "s6", "s7", "s5"];

const pathColor = (t: Tokens, i: number) => String(t[PATH_SLOTS[i % PATH_SLOTS.length] ?? "s1"]);

function Slider({ id, label, term, value, min, max, step, onChange, show }: { id: string; label: string; term?: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; show?: string }) {
  return (
    <div className={ui.field}>
      <span className={ui.fieldLabel}>
        <Labelled term={term}>
          <label htmlFor={id}>{label}</label>
        </Labelled>
        <span>{show ?? num(value)}</span>
      </span>
      <input id={id} className={ui.range} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

function pathsOption(t: Tokens, d: SynthPaths, ys: number[][], name: string, fmt: (v: number) => string): EChartsOption {
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => fmt(v as number) },
    grid: { ...base(t).grid, left: 64 },
    xAxis: { type: "value", min: 0, max: d.params.n_days, name: "Trading day", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", scale: true, name, nameLocation: "middle", ...axis(t, { nameGap: 48 }), axisLabel: { ...axis(t).axisLabel, formatter: fmt } },
    series: ys.map((path, i) => ({
      name: `Seed ${d.params.seed + i}`,
      type: "line" as const,
      data: path.map((v, day) => [day, v]),
      showSymbol: false,
      lineStyle: { width: 1.5, color: pathColor(t, i) },
      itemStyle: { color: pathColor(t, i) },
    })),
  };
}

function smileOption(t: Tokens, d: SynthPaths): EChartsOption {
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => pct(v as number, 1) },
    grid: { ...base(t).grid, left: 64 },
    xAxis: { type: "value", scale: true, name: "ln(strike / spot)", nameLocation: "middle", ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => num(v, 2) } },
    yAxis: { type: "value", scale: true, name: "Implied vol", nameLocation: "middle", ...axis(t, { nameGap: 48 }), axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v, 0) } },
    series: d.smile_ivs.map((ivs, i) => ({
      name: `Seed ${d.params.seed + i}`,
      type: "line" as const,
      data: ivs.map((v, j) => [d.smile_log_moneyness[i]?.[j] ?? 0, v]),
      showSymbol: false,
      lineStyle: { width: 1.5, color: pathColor(t, i) },
      itemStyle: { color: pathColor(t, i) },
    })),
  };
}

/** The market in one sentence: how far the paths went, what the vol did, the smile's tilt. */
function reading(d: SynthPaths): string {
  const p = d.params;
  const ends = d.spots.map((s) => (s[s.length - 1] ?? 0) / (s[0] || 1) - 1);
  const mean = ends.reduce((a, b) => a + b, 0) / Math.max(1, ends.length);
  const vols = d.atm_ivs.flat();
  const lo = Math.min(...vols);
  const hi = Math.max(...vols);
  const drift = p.alpha_bps ? `a drift of ${num(p.alpha_bps)} bps a day (${pct(p.alpha_bps / 10_000 * p.n_days, 1)} over the path)` : "no drift";
  const vol = p.vol_of_vol ? `the at-the-money vol wanders between ${pct(lo, 1)} and ${pct(hi, 1)}` : `the at-the-money vol stays at ${pct(p.sigma, 0)}`;
  const smile = p.skew_slope ? (p.skew_slope > 0 ? "downside strikes are dearer (an equity skew)" : "upside strikes are dearer") : "the smile is flat";
  return `With ${drift}, the ${num(p.n_paths)} paths end ${pct(mean, 1)} from where they start on average; ${vol}, and ${smile}.`;
}

function initial(q: ReturnType<typeof useQuery>): SynthParams {
  return {
    sigma: q.num("sigma", DEFAULTS.sigma, 0.05, 1),
    alpha_bps: q.num("alpha_bps", DEFAULTS.alpha_bps, 0, 200),
    vol_of_vol: q.num("vol_of_vol", DEFAULTS.vol_of_vol, 0, 2),
    skew_slope: q.num("skew_slope", DEFAULTS.skew_slope, -1, 1),
    n_days: q.int("n_days", DEFAULTS.n_days, 5, 252),
    n_paths: q.int("n_paths", DEFAULTS.n_paths, 1, 20),
    seed: q.int("seed", DEFAULTS.seed, 0, 9999),
  };
}

export default function SynthPage() {
  return (
    <Suspense fallback={<Loading what="the synthetic market" />}>
      <Synth />
    </Suspense>
  );
}

function Synth() {
  const t = useTokens();
  const query = useQuery();
  const [p, setP] = useState<SynthParams>(() => initial(query));
  const [q, setQ] = useState<SynthParams>(p);
  useEffect(() => {
    const id = window.setTimeout(() => setQ(p), 150);
    return () => window.clearTimeout(id);
  }, [p]);
  useQuerySync(Object.fromEntries(KEYS.map((k) => [k, p[k]])), Object.fromEntries(KEYS.map((k) => [k, DEFAULTS[k]])));
  const paths = useFetch((s) => api.synth(q, s), JSON.stringify(q));
  const [last, setLast] = useState<SynthPaths | undefined>();
  useEffect(() => {
    if (paths.data) setLast(paths.data);
  }, [paths.data]);
  const d = paths.data ?? last;
  const set = (k: keyof SynthParams) => (v: number) => setP((x) => ({ ...x, [k]: v }));
  const launch = new URLSearchParams(LAUNCH_KEYS.filter((k) => p[k] !== DEFAULTS[k]).map((k) => [k, String(p[k])]));
  const launchHref = `/launch/${launch.toString() ? `?${launch.toString()}` : ""}`;

  return (
    <>
      <PageHead
        eyebrow="Synth data"
        title="What does the simulated market look like?"
        lede="Set the path vol and the research knobs and see the paths a run would trade, before you launch one. At zero drift, vol of vol and skew this is the world the study ran in."
        right={
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => setP(DEFAULTS)}>
            Reset
          </button>
        }
      />
      <div className={ui.split}>
        <div className={`${ui.stack} ${ui.sticky}`}>
          <Card title="The market">
            <div className={ui.stack}>
              <Slider id="synth-sigma" label="Path volatility" term="sigma" value={p.sigma} min={0.05} max={1} step={0.01} onChange={set("sigma")} show={pct(p.sigma, 0)} />
              <Slider id="synth-alpha" label="Drift (bps per day)" term="alpha_bps" value={p.alpha_bps} min={0} max={200} step={5} onChange={set("alpha_bps")} />
              <Slider id="synth-vov" label="Vol of vol" term="vol_of_vol" value={p.vol_of_vol} min={0} max={2} step={0.05} onChange={set("vol_of_vol")} show={num(p.vol_of_vol, 2)} />
              <Slider id="synth-skew" label="Skew slope" term="skew_slope" value={p.skew_slope} min={-1} max={1} step={0.05} onChange={set("skew_slope")} show={num(p.skew_slope, 2)} />
              <Slider id="synth-days" label="Days per path" term="n_days" value={p.n_days} min={5} max={252} step={1} onChange={set("n_days")} />
              <Slider id="synth-paths" label="Paths shown" value={p.n_paths} min={1} max={20} step={1} onChange={set("n_paths")} />
              <Slider id="synth-seed" label="First seed" term="seed_first" value={p.seed} min={0} max={9999} step={1} onChange={set("seed")} />
              <Link href={launchHref} className={`${ui.btn} ${ui.btnPrimary}`} style={{ textAlign: "center" }}>
                Launch a run on this market →
              </Link>
            </div>
          </Card>
        </div>
        <div className={ui.stack}>
          {paths.error ? <ErrorBox what="the synthetic paths" error={paths.error} onRetry={paths.reload} /> : null}
          {!d ? <Loading what="the synthetic paths" /> : null}
          {d ? (
            <>
              <Card>
                <Reading>{reading(d)}</Reading>
              </Card>
              <Card title="Index paths" sub={`${num(d.params.n_paths)} paths of ${num(d.params.n_days)} trading days, from ${num(d.spots[0]?.[0] ?? 0)}.`} source={d.source}>
                <Chart option={pathsOption(t, d, d.spots, "Index level", (v) => num(v))} height={280} label="Simulated index level by trading day, one line per seed" rows={() => d.spots.flatMap((s, i) => s.map((v, day) => ({ seed: d.params.seed + i, day, spot: v })))} />
              </Card>
              <div className={ui.grid2}>
                <Card title="At-the-money vol" sub="The near tenor at spot: the atm_iv a policy observes each day.">
                  <Chart option={pathsOption(t, d, d.atm_ivs, "Implied vol", (v) => pct(v, 1))} height={260} label="Near-tenor at-the-money implied vol by trading day, one line per seed" rows={() => d.atm_ivs.flatMap((s, i) => s.map((v, day) => ({ seed: d.params.seed + i, day, atm_iv: v })))} />
                </Card>
                <Card title={`The smile on day ${d.smile_day}`} sub="The near tenor's listed strikes, half way through each path.">
                  <Chart option={smileOption(t, d)} height={260} label="Near-tenor implied vol against log-moneyness on the middle day, one line per seed" rows={() => d.smile_ivs.flatMap((s, i) => s.map((v, j) => ({ seed: d.params.seed + i, log_moneyness: d.smile_log_moneyness[i]?.[j] ?? null, iv: v })))} />
                </Card>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </>
  );
}
